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

const SYSTEM: &str = "\
You are the researcher in a notebook of user-supplied sources. You answer only \
from the excerpts provided in the conversation. Every factual claim carries an \
inline citation of the form [source_title, locator]. If the excerpts do not \
support an answer, say exactly what is missing and which source would need to \
be consulted. Never fill the gap from your own knowledge. Use markdown, and \
LaTeX for mathematics.";

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

    let response = resolved.client(&state.http)?.chat(&request).await?;

    let citations: Vec<Citation> = hits
        .iter()
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
    }))
}

fn render_excerpts(hits: &[SearchHit]) -> String {
    if hits.is_empty() {
        return "No excerpts matched the question. Say so rather than guessing.".to_string();
    }
    // The row id is deliberately absent: citations are rebuilt server side from
    // these same hits, and models that see an id tend to cite it verbatim.
    let mut out = String::from("Excerpts from the notebook:\n");
    for h in hits {
        out.push_str(&format!(
            "\n<excerpt title=\"{}\" locator=\"{}\"{}>\n{}\n</excerpt>\n",
            h.source_title,
            h.locator.as_deref().unwrap_or("unknown"),
            h.heading.as_deref().map(|x| format!(" heading=\"{x}\"")).unwrap_or_default(),
            h.snippet,
        ));
    }
    out
}
