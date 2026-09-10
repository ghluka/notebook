//! The researcher endpoint.
//!
//! Phase 0 is retrieval-then-answer: search the index with the user's question,
//! put the hits in front of the model, ask for a cited answer. Phase 2 replaces
//! the fixed retrieval step with a tool loop over `search_sources`,
//! `read_document`, `list_sources` and `ask_source`, with the same prompt and the
//! same citation contract, but the model decides what to look at.

use axum::Json;
use axum::extract::State;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::db::{self, SearchHit};
use crate::error::{AppError, AppResult};
use crate::llm::{ChatRequest, Effort, Message};
use crate::models;
use crate::state::AppState;

/// Failure modes this prompt exists to prevent, all seen in the wild:
///
/// - copying the citation template literally, `[source_title="x.pdf", ...]`
/// - opening every answer with "From the provided excerpts"
/// - treating "what did I just ask?" as a question about the sources, and
///   answering that the excerpts do not contain it
/// - guessing which file would have held a missing answer
const SYSTEM: &str = "\
You are the researcher in a notebook of the user's own sources, in an ongoing \
conversation with them.\n\n\
Two kinds of message reach you, and they are answered differently.\n\n\
A question about the sources is answered from the excerpts below and nothing \
else. Never answer one from your own knowledge of the subject.\n\n\
A question about this conversation is answered from the conversation itself: \
what the user asked earlier, what you replied, what you meant, what to do next. \
Follow-ups like \"why?\", \"what did I just ask\", \"explain that again\" and \
\"why that file?\" are of this kind. The excerpts are search results for the \
latest message and are often irrelevant to these; ignore them and just answer. \
Never tell the user their own question is missing from the excerpts.\n\n\
Answer directly, in the user's own terms. Do not open with a preamble: no \
\"From the provided excerpts\", no \"Based on the sources\", no restating the \
question.\n\n\
Cite a claim about the sources by copying the bracketed label printed above the \
excerpt you used, exactly as it appears, for example [08NumberTheoryII.pdf, \
p. 12]. Nothing else goes inside the brackets: no field names, no quotation \
marks, no equals signs. Cite only what you actually used, and never invent a \
label. Answers drawn from the conversation need no citation.\n\n\
When the excerpts do not answer a question about the sources, say so in one \
sentence and say what you searched for. Do not guess which file would have had \
it: you cannot know that, and naming one is a fabrication. Suggest a different \
wording, or ask the user to narrow the search to particular files.\n\n\
Use markdown, and LaTeX for mathematics.";

#[derive(Deserialize)]
pub struct ChatBody {
    pub message: String,
    /// Continues an existing conversation when present.
    #[serde(default)]
    pub conversation_id: Option<String>,
    /// Restrict retrieval to these sources.
    #[serde(default)]
    pub source_ids: Vec<String>,
    #[serde(default)]
    pub max_hits: Option<i64>,
    /// Overrides the researcher role for this turn (the prompt-bar picker).
    #[serde(default)]
    pub model_id: Option<String>,
    /// Overrides the stored thinking effort for this turn.
    #[serde(default)]
    pub effort: Option<String>,
}

#[derive(Serialize)]
pub struct Citation {
    pub source_id: String,
    pub source_title: String,
    pub chunk_id: String,
    pub locator: Option<String>,
}

#[derive(Serialize)]
pub struct ChatResponseBody {
    pub conversation_id: String,
    pub answer: String,
    pub citations: Vec<Citation>,
    pub model: String,
    pub model_display_name: String,
    pub effort: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    /// What the prompt bar's context meter needs.
    pub context_used: u32,
    pub context_window: i64,
    /// How many excerpts were put in front of the model, cited or not.
    pub excerpts_searched: usize,
}

pub async fn chat(
    State(state): State<AppState>,
    Json(body): Json<ChatBody>,
) -> AppResult<Json<ChatResponseBody>> {
    if body.message.trim().is_empty() {
        return Err(AppError::BadRequest("message is empty".into()));
    }

    // Created only after the model answers, so a stopped or failed turn does not
    // leave an empty conversation behind.
    let existing = match &body.conversation_id {
        Some(id) => Some(
            db::get_conversation(&state.db, &state.user_id, id)
                .await?
                .ok_or_else(|| AppError::NotFound(format!("conversation {id}")))?,
        ),
        None => None,
    };

    let limit = body.max_hits.unwrap_or(8).clamp(1, 30);
    let mut hits = Vec::new();
    if body.source_ids.is_empty() {
        hits = db::search_chunks(&state.db, &body.message, None, limit).await?;
    } else {
        for source_id in &body.source_ids {
            hits.extend(
                db::search_chunks(&state.db, &body.message, Some(source_id), limit).await?,
            );
        }
        hits.sort_by(|a, b| a.score.total_cmp(&b.score));
        hits.truncate(limit as usize);
    }

    // A compaction summary stands in for the turns it replaced, then whatever
    // has been said since, then the retrieved excerpts, then the question.
    let mut messages: Vec<Message> = Vec::new();
    if let Some(conversation) = &existing {
        if let Some(summary) = &conversation.summary {
            messages.push(Message::user(format!(
                "Notes from the earlier part of this conversation:\n\n{summary}"
            )));
            messages.push(Message::assistant("Understood, I have those notes."));
        }
        for m in db::active_messages(&state.db, &conversation.id).await? {
            match m.role.as_str() {
                "user" => messages.push(Message::user(m.content)),
                "assistant" => messages.push(Message::assistant(m.content)),
                _ => {}
            }
        }
    }
    messages.push(Message::user(format!(
        "{}\n\n---\n\nQuestion: {}",
        render_excerpts(&hits),
        body.message
    )));

    let resolved = match &body.model_id {
        Some(id) => models::resolve_model_id(&state.db, &state.user_id, id)
            .await?
            .ok_or_else(|| AppError::NotFound(format!("model {id}")))?,
        None => state.role_model("researcher").await?,
    };

    let effort = match &body.effort {
        Some(e) => Effort::parse(e)
            .ok_or_else(|| AppError::BadRequest("effort must be off|low|medium|high".into()))?,
        None => models::effort(&state.db, &state.user_id).await?,
    };

    let request = ChatRequest::new(&resolved.model.model_id, messages)
        .system(SYSTEM)
        .max_tokens(resolved.model.max_output_tokens as u32)
        .effort(effort);

    let client = resolved.client(&state.http)?;
    let response = crate::llm::chat_with_retry(client.as_ref(), &request).await.map_err(|e| {
        if e.is_rate_limited() {
            AppError::from_rate_limit(
                e,
                &resolved.model.id,
                &resolved.model.display_name,
                " while answering",
            )
        } else {
            AppError::Llm(e)
        }
    })?;

    let citations: Vec<Citation> = cited_hits(&response.text, &hits)
        .into_iter()
        .map(|h| Citation {
            source_id: h.source_id.clone(),
            source_title: h.source_title.clone(),
            chunk_id: h.chunk_id.clone(),
            locator: h.locator.clone(),
        })
        .collect();

    let conversation_id = match existing {
        Some(c) => c.id,
        None => {
            let title: String = body.message.chars().take(60).collect();
            db::create_conversation(&state.db, &state.user_id, &title).await?.id
        }
    };

    db::append_message(&state.db, &conversation_id, "user", &body.message, None, None, None)
        .await?;
    db::append_message(
        &state.db,
        &conversation_id,
        "assistant",
        &response.text,
        Some(&json!(&citations).to_string()),
        Some(&resolved.model.display_name),
        Some(db::Usage {
            input_tokens: response.usage.input_tokens,
            output_tokens: response.usage.output_tokens,
        }),
    )
    .await?;

    Ok(Json(ChatResponseBody {
        conversation_id,
        answer: response.text,
        citations,
        model: response.model,
        model_display_name: resolved.model.display_name.clone(),
        effort: effort.as_str().to_string(),
        input_tokens: response.usage.input_tokens,
        output_tokens: response.usage.output_tokens,
        context_used: response.usage.input_tokens + response.usage.output_tokens,
        context_window: resolved.model.context_window,
        excerpts_searched: hits.len(),
    }))
}

/// The label a model should copy to cite this excerpt.
fn citation_label(hit: &SearchHit) -> String {
    match hit.locator.as_deref() {
        Some(locator) => format!("[{}, {}]", hit.source_title, locator),
        None => format!("[{}]", hit.source_title),
    }
}

/// The excerpt block doubles as the citation example: each one is introduced by
/// the exact label the model should copy. Attribute syntax was tried first and
/// models echoed it into their prose, key names and all.
///
/// These are search results for the latest message, not context the model asked
/// for, so the header says as much: a follow-up about the conversation should
/// not be answered by reporting what the excerpts lack.
fn render_excerpts(hits: &[SearchHit]) -> String {
    if hits.is_empty() {
        return "No excerpts matched this message. If it is a question about the \
                sources, say you found nothing. If it is about our conversation, \
                just answer it."
            .to_string();
    }
    // Row ids are deliberately absent: citations are rebuilt server side from
    // these same hits, and models that see an id tend to cite it verbatim.
    let mut out = String::from(
        "Search results for the message below, in case they are useful. Ignore \
         them if the message is about our conversation rather than the sources.\n",
    );
    for h in hits {
        out.push_str(&format!("\n{}\n", citation_label(h)));
        if let Some(heading) = h.heading.as_deref().filter(|x| !x.is_empty()) {
            out.push_str(&format!("Section: {heading}\n"));
        }
        out.push_str(&format!("{}\n", h.snippet));
    }
    out
}

/// Which excerpts the answer actually leaned on.
///
/// Every retrieved excerpt used to be shown as a citation, so an answer saying
/// it found nothing still carried eight source pills. Only excerpts the model
/// cited are kept: by exact label, or failing that by file name, since a model
/// citing a file without a locator still means the top hit for that file.
fn cited_hits<'a>(answer: &str, hits: &'a [SearchHit]) -> Vec<&'a SearchHit> {
    let lower = answer.to_lowercase();
    let mut used: Vec<&SearchHit> = Vec::new();

    for hit in hits {
        if lower.contains(&citation_label(hit).to_lowercase()) {
            used.push(hit);
        }
    }
    if !used.is_empty() {
        return used;
    }

    for hit in hits {
        let by_name = format!("[{}", hit.source_title).to_lowercase();
        if lower.contains(&by_name) && !used.iter().any(|u| u.source_id == hit.source_id) {
            used.push(hit);
        }
    }
    used
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(title: &str, locator: &str, source: &str) -> SearchHit {
        SearchHit {
            chunk_id: format!("{source}-{locator}"),
            source_id: source.into(),
            source_title: title.into(),
            heading: None,
            locator: Some(locator.into()),
            ordinal: 0,
            snippet: "text".into(),
            score: 0.0,
        }
    }

    #[test]
    fn keeps_only_what_the_answer_cited() {
        let hits = vec![
            hit("08NumberTheoryII.pdf", "line 249", "s1"),
            hit("fuchs.pdf", "line 12", "s2"),
        ];
        let answer = "An LDE asks for integer solutions [08NumberTheoryII.pdf, line 249].";
        let used = cited_hits(answer, &hits);

        assert_eq!(used.len(), 1);
        assert_eq!(used[0].source_title, "08NumberTheoryII.pdf");
    }

    #[test]
    fn an_answer_citing_nothing_shows_nothing() {
        let hits = vec![hit("fuchs.pdf", "line 1", "s2")];
        let used = cited_hits("I could not find that in your sources.", &hits);

        assert!(used.is_empty(), "a nothing found answer must not carry citations");
    }

    #[test]
    fn a_file_cited_without_a_locator_still_resolves() {
        let hits = vec![
            hit("fuchs.pdf", "line 1", "s2"),
            hit("fuchs.pdf", "line 90", "s2"),
        ];
        let used = cited_hits("See [fuchs.pdf] for the proof.", &hits);

        // One pill for the file, not one per excerpt from it.
        assert_eq!(used.len(), 1);
        assert_eq!(used[0].locator.as_deref(), Some("line 1"));
    }
}
