//! The researcher endpoint.
//!
//! One question becomes a short agent loop: the excerpts a search already found
//! are handed over up front, and from there the model can search again, read a
//! source around a line, or list what is in the notebook, until it can answer.
//!
//! The loop exists because a single shot of ellipsised snippets makes a capable
//! model look stupid. It would be told an equation is defined on line 249 of a
//! file, be given 32 tokens either side, and report that no worked example
//! exists, while the example sat a paragraph below.

use axum::Json;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::db::{self, SearchHit};
use crate::error::{AppError, AppResult};
use crate::llm::{
    ChatRequest, ContentPart, Effort, LlmError, LlmProvider, Message, Role, StreamEvent, Tool,
    ToolCall,
};
use crate::models;
use crate::state::AppState;

/// How many rounds of tool calls before the model has to answer with what it
/// has. Enough to search, read, and search again.
const MAX_TOOL_ROUNDS: usize = 6;
/// Chunks pulled per search, before neighbours.
const HITS_PER_SEARCH: i64 = 8;
/// Chunks either side of a hit, so a passage is not cut in half.
const NEIGHBOUR_RADIUS: i64 = 1;
/// Cap on one `read_source` slice, so a long rendition is paged, not dumped.
const READ_MAX_CHARS: usize = 24_000;
/// Lines per `read_source` call when the model does not say.
const READ_DEFAULT_LINES: usize = 220;

const SYSTEM: &str = "\
You are the researcher in a notebook of the user's own sources, in an ongoing \
conversation with them.\n\n\
Two kinds of message reach you, and they are answered differently.\n\n\
A question about the sources is answered from those sources and nothing else. \
Never answer one from your own knowledge of the subject.\n\n\
A question about this conversation is answered from the conversation itself: \
what the user asked earlier, what you replied, what you meant, what to do next. \
Follow-ups like \"why?\", \"what did I just ask\" and \"explain that again\" are of \
this kind. Just answer them; never tell the user their own question is missing \
from the sources.\n\n\
You have tools, and you are expected to use them before giving up. The excerpts \
below were found by one keyword search on the user's words, which is a starting \
point and not a verdict. If they look close but incomplete, read the source \
around that line: worked examples, proofs and tables usually sit next to the \
sentence that names them. If the wording missed, search again with the words \
the field actually uses. Only after looking should you say something is not \
there.\n\n\
Answer directly, in the user's own terms. Do not open with a preamble: no \
\"From the provided excerpts\", no \"Based on the sources\", no restating the \
question.\n\n\
Cite a claim about the sources by copying the bracketed label printed above the \
excerpt or slice you used, exactly as it appears, for example \
[08NumberTheoryII.pdf, line 249]. Nothing else goes inside the brackets: no \
field names, no quotation marks, no equals signs. Put the citation after the \
sentence it supports, never inside the sentence as a word of it: the reader \
sees sources listed separately, so each sentence must read correctly with the \
bracket deleted. Answers drawn from the conversation need no citation.\n\n\
When the sources genuinely do not answer a question, say so in one sentence and \
say what you looked for. Do not guess which file would have had it. Suggest \
different wording, or ask the user to attach the right files.\n\n\
Use markdown, and LaTeX for mathematics.";

#[derive(Deserialize)]
pub struct ChatBody {
    pub message: String,
    /// Continues an existing conversation when present.
    #[serde(default)]
    pub conversation_id: Option<String>,
    /// Restrict retrieval to these sources, from the conversation attachments.
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
    /// The exact bracketed text the model wrote. The client strips these from
    /// the prose, since the source chips below the answer already say it.
    pub label: String,
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
    /// Tool calls the model made on the way to this answer.
    pub tool_calls: Vec<String>,
}

/// The tools, described the way they will be used.
fn researcher_tools(attached: &[String]) -> Vec<Tool> {
    let scope_note = if attached.is_empty() {
        "Searches the whole notebook."
    } else {
        "Searches the files attached to this conversation."
    };

    vec![
        Tool {
            name: "search_sources".into(),
            description: format!(
                "Keyword search over the sources. {scope_note} Returns whole \
                 passages with the label to cite them by. Use different words \
                 than last time: the terms the field uses, a definition, a \
                 theorem name."
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords, not a sentence."
                    }
                },
                "required": ["query"]
            }),
        },
        Tool {
            name: "read_source".into(),
            description: "Read a source around a line, in order, with line \
                 numbers. This is how you follow a citation: an excerpt from \
                 line 249 means the example, proof or table you want is \
                 probably within a page of line 249. Prefer this over giving \
                 up."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The file name, as printed in a label."
                    },
                    "from_line": {
                        "type": "integer",
                        "description": "First line to read. Start a page before \
                                        the line you saw."
                    },
                    "lines": {
                        "type": "integer",
                        "description": "How many lines to read, default 220."
                    }
                },
                "required": ["title"]
            }),
        },
        Tool {
            name: "list_sources".into(),
            description: "Every source in the notebook: name, kind, length and \
                 a short summary. Use it to decide where to look."
                .into(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
    ]
}

/// Search, expanded: whole chunks, with their neighbours, ordered as they read.
async fn search_expanded(
    state: &AppState,
    query: &str,
    attached: &[String],
    limit: i64,
) -> AppResult<Vec<SearchHit>> {
    let mut hits = Vec::new();
    if attached.is_empty() {
        hits = db::search_chunks(&state.db, query, None, limit).await?;
    } else {
        for source_id in attached {
            hits.extend(db::search_chunks(&state.db, query, Some(source_id), limit).await?);
        }
        hits.sort_by(|a, b| a.score.total_cmp(&b.score));
        hits.truncate(limit as usize);
    }

    // Pull in what sits either side of each hit, then read them in order.
    let mut expanded: Vec<SearchHit> = Vec::new();
    for hit in &hits {
        for near in
            db::chunks_around(&state.db, &hit.source_id, hit.ordinal, NEIGHBOUR_RADIUS).await?
        {
            if !expanded.iter().any(|e| e.chunk_id == near.chunk_id) {
                expanded.push(near);
            }
        }
    }
    expanded.sort_by(|a, b| {
        a.source_title.cmp(&b.source_title).then(a.ordinal.cmp(&b.ordinal))
    });
    Ok(expanded)
}

/// The label a model should copy to cite this excerpt.
fn citation_label(hit: &SearchHit) -> String {
    match hit.locator.as_deref() {
        Some(locator) => format!("[{}, {}]", hit.source_title, locator),
        None => format!("[{}]", hit.source_title),
    }
}

/// Excerpts as the model sees them: the label to cite, then the whole passage.
fn render_excerpts(hits: &[SearchHit], budget_chars: usize) -> String {
    if hits.is_empty() {
        return "Nothing matched that search.".to_string();
    }
    let mut out = String::new();
    for hit in hits {
        let mut block = format!("\n{}\n", citation_label(hit));
        if let Some(heading) = hit.heading.as_deref().filter(|h| !h.is_empty()) {
            block.push_str(&format!("Section: {heading}\n"));
        }
        block.push_str(&format!("{}\n", hit.content));
        if out.len() + block.len() > budget_chars {
            out.push_str("\n[More matches were left out to stay within context.]\n");
            break;
        }
        out.push_str(&block);
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
        if lower.contains(&citation_label(hit).to_lowercase())
            && !used.iter().any(|u| u.chunk_id == hit.chunk_id)
        {
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

/// Distinctive words and numbers of a passage: long enough to be specific,
/// so that finding several of them in an answer means the answer came from
/// there rather than from the model's own knowledge.
fn distinctive_tokens(text: &str) -> Vec<String> {
    let mut tokens: Vec<String> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| {
            let numeric = t.chars().all(|c| c.is_ascii_digit());
            (numeric && t.len() >= 3) || (!numeric && t.chars().count() >= 6)
        })
        .map(str::to_lowercase)
        .collect();
    tokens.sort();
    tokens.dedup();
    tokens
}

/// Passages the answer visibly reuses.
///
/// A model that answers well from an excerpt and forgets the label still owes
/// the reader a source. Rather than listing everything that was retrieved,
/// which is how an answer saying "I found nothing" ended up wearing eight
/// citations, this keeps only passages the answer actually echoes.
const OVERLAP_REQUIRED: usize = 4;

fn echoed_hits<'a>(answer: &str, hits: &'a [SearchHit]) -> Vec<&'a SearchHit> {
    let answer_tokens = distinctive_tokens(answer);
    if answer_tokens.is_empty() {
        return Vec::new();
    }

    let mut best: Vec<(&SearchHit, usize)> = Vec::new();
    for hit in hits {
        let shared = distinctive_tokens(&hit.content)
            .iter()
            .filter(|t| answer_tokens.binary_search(t).is_ok())
            .count();
        if shared < OVERLAP_REQUIRED {
            continue;
        }
        // One passage per file: the strongest overlap wins.
        match best.iter_mut().find(|(h, _)| h.source_id == hit.source_id) {
            Some(slot) if slot.1 < shared => *slot = (hit, shared),
            Some(_) => {}
            None => best.push((hit, shared)),
        }
    }
    best.sort_by(|a, b| b.1.cmp(&a.1));
    best.into_iter().map(|(hit, _)| hit).collect()
}

/// Run one tool call and describe the result the way an excerpt is described,
/// so anything the model reads can be cited the same way.
async fn run_tool(
    state: &AppState,
    call: &crate::llm::ToolCall,
    attached: &[String],
    seen: &mut Vec<SearchHit>,
    read: &mut Vec<(String, String)>,
    budget_chars: usize,
) -> AppResult<String> {
    match call.name.as_str() {
        "search_sources" => {
            let query = call.arguments["query"].as_str().unwrap_or_default();
            if query.trim().is_empty() {
                return Ok("No query given.".into());
            }
            let hits = search_expanded(state, query, attached, HITS_PER_SEARCH).await?;
            for hit in &hits {
                if !seen.iter().any(|s| s.chunk_id == hit.chunk_id) {
                    seen.push(hit.clone());
                }
            }
            Ok(format!(
                "Search for \"{query}\" found {} passages.\n{}",
                hits.len(),
                render_excerpts(&hits, budget_chars)
            ))
        }
        "read_source" => {
            let title = call.arguments["title"].as_str().unwrap_or_default();
            let from_line = call.arguments["from_line"].as_u64().unwrap_or(1).max(1) as usize;
            let lines = call.arguments["lines"].as_u64().unwrap_or(READ_DEFAULT_LINES as u64)
                as usize;

            let briefs = db::source_briefs(&state.db, &state.user_id).await?;
            let found = briefs
                .iter()
                .find(|b| b.title.eq_ignore_ascii_case(title.trim()))
                .or_else(|| {
                    briefs.iter().find(|b| {
                        b.title.to_lowercase().contains(&title.trim().to_lowercase())
                    })
                });
            let Some(brief) = found else {
                return Ok(format!(
                    "No source called \"{title}\". Call list_sources to see the names."
                ));
            };

            let slice = db::read_document_lines(
                &state.db,
                &state.user_id,
                &brief.id,
                from_line,
                lines.min(600),
                READ_MAX_CHARS.min(budget_chars),
            )
            .await?;
            let Some(slice) = slice else {
                return Ok(format!("\"{}\" has no rendition to read yet.", brief.title));
            };
            if !read.iter().any(|(id, _)| id == &brief.id) {
                read.push((brief.id.clone(), brief.title.clone()));
            }

            Ok(format!(
                "{}, lines {} to {} of {}. Cite passages from here as \
                 [{}, line N] using the numbers shown.\n\n{}",
                slice.title,
                slice.from_line,
                slice.to_line,
                slice.total_lines,
                slice.title,
                slice.text
            ))
        }
        "list_sources" => {
            let briefs = db::source_briefs(&state.db, &state.user_id).await?;
            if briefs.is_empty() {
                return Ok("The notebook is empty.".into());
            }
            let mut out = String::from("Sources in the notebook:\n");
            for b in briefs {
                out.push_str(&format!(
                    "\n{} ({}, {} lines, {})\n{}\n",
                    b.title,
                    b.kind,
                    b.total_lines,
                    b.status,
                    b.summary.as_deref().unwrap_or("no summary").chars().take(240)
                        .collect::<String>()
                ));
            }
            Ok(out)
        }
        other => Ok(format!("No tool called {other}.")),
    }
}

/// Everything a turn needs before the model is called: the same for a
/// one-shot answer and a streamed one, so a failure here is still a plain
/// HTTP error in both.
struct PreparedTurn {
    existing: Option<db::Conversation>,
    resolved: models::Resolved,
    effort: Effort,
    budget_chars: usize,
    seen: Vec<SearchHit>,
    messages: Vec<Message>,
    tools: Vec<Tool>,
}

async fn prepare_turn(
    state: &AppState,
    body: &ChatBody,
) -> AppResult<(PreparedTurn, Box<dyn LlmProvider>)> {
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

    // A large window is there to be used. Roughly four characters per token,
    // and a third of the window for source material leaves room for the
    // conversation, the tools and a long answer.
    let budget_chars = if resolved.model.context_window > 0 {
        ((resolved.model.context_window as usize) * 4 / 3).clamp(8_000, 400_000)
    } else {
        24_000
    };

    let limit = body.max_hits.unwrap_or(HITS_PER_SEARCH).clamp(1, 30);
    let seen = search_expanded(state, &body.message, &body.source_ids, limit).await?;

    // A compaction summary stands in for the turns it replaced, then whatever
    // has been said since, then the excerpts, then the question.
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

    let attachment_note = if body.source_ids.is_empty() {
        String::new()
    } else {
        format!(
            "The user has attached {} file(s) to this conversation; searches are \
             limited to them.\n\n",
            body.source_ids.len()
        )
    };
    messages.push(Message::user(format!(
        "{attachment_note}A keyword search on this message found these passages. \
         Search again or read a source if they are not enough.\n{}\n\n---\n\n\
         Question: {}",
        render_excerpts(&seen, budget_chars),
        body.message
    )));

    let client = resolved.client(&state.http)?;
    let tools =
        if resolved.model.supports_tools { researcher_tools(&body.source_ids) } else { vec![] };

    Ok((
        PreparedTurn { existing, resolved, effort, budget_chars, seen, messages, tools },
        client,
    ))
}

/// What one model round produced, however it arrived.
struct RoundOutcome {
    text: String,
    tool_calls: Vec<ToolCall>,
    input_tokens: u32,
    output_tokens: u32,
}

/// The client end of the SSE turn. A closed receiver means the person went
/// away, and the turn stops quietly: nothing is sent, nothing is persisted.
type TokenSink = futures::channel::mpsc::UnboundedSender<Result<Event, axum::Error>>;

fn emit_json(tx: &TokenSink, event: &'static str, value: serde_json::Value) {
    if tx.is_closed() {
        return;
    }
    match Event::default().event(event).json_data(value) {
        Ok(ev) => {
            let _ = tx.unbounded_send(Ok(ev));
        }
        // The payloads are strings and counters; this cannot realistically
        // happen, and an error event is the honest fallback if it does.
        Err(e) => send_error(tx, &AppError::Internal(anyhow::anyhow!(e))),
    }
}

fn send_error(tx: &TokenSink, e: &AppError) {
    if tx.is_closed() {
        return;
    }
    // `payload` carries the rate-limit kind and model alongside the prose, so
    // the client can treat it like the HTTP failure it would have been.
    let event = Event::default()
        .event("error")
        .json_data(e.payload())
        .unwrap_or_else(|_| Event::default().event("error").data("stream failed"));
    let _ = tx.unbounded_send(Ok(event));
}

/// One trip to the model. Without a sink this is the old one-shot call; with
/// one, tokens stream to the client as they arrive and the full text is still
/// returned for citations and persistence.
async fn run_round(
    client: &dyn LlmProvider,
    request: &ChatRequest,
    resolved: &models::Resolved,
    tokens: Option<&TokenSink>,
) -> AppResult<RoundOutcome> {
    let rate_limit = |e: LlmError| {
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
    };
    let Some(tx) = tokens else {
        let r = crate::llm::chat_with_retry(client, request).await.map_err(rate_limit)?;
        return Ok(RoundOutcome {
            text: r.text,
            tool_calls: r.tool_calls,
            input_tokens: r.usage.input_tokens,
            output_tokens: r.usage.output_tokens,
        });
    };

    match crate::llm::chat_stream_with_retry(client, request).await {
        Ok(stream) => {
            use futures::StreamExt;
            let mut stream = stream;
            let mut text = String::new();
            let mut done = crate::llm::StreamDone::default();
            while let Some(ev) = stream.next().await {
                if tx.is_closed() {
                    return Err(AppError::Internal(anyhow::anyhow!("client went away")));
                }
                match ev.map_err(rate_limit)? {
                    StreamEvent::Text(t) => {
                        text.push_str(&t);
                        emit_json(tx, "token", json!({ "t": t }));
                    }
                    StreamEvent::Done(d) => done = d,
                }
            }
            Ok(RoundOutcome {
                text,
                tool_calls: done.tool_calls,
                input_tokens: done.input_tokens,
                output_tokens: done.output_tokens,
            })
        }
        // Endpoints that never learned `stream` fail the handshake naming it.
        // Answer one-shot for that round and forward it as a single token, so
        // the turn still streams from the client's point of view.
        Err(e) if is_stream_unsupported(&e) => {
            tracing::info!("streaming refused, answering one-shot for this round");
            let r = crate::llm::chat_with_retry(client, request).await.map_err(rate_limit)?;
            emit_json(tx, "token", json!({ "t": r.text }));
            Ok(RoundOutcome {
                text: r.text,
                tool_calls: r.tool_calls,
                input_tokens: r.usage.input_tokens,
                output_tokens: r.usage.output_tokens,
            })
        }
        Err(e) => Err(rate_limit(e)),
    }
}

/// A 400 or 422 naming the `stream` parameter is a no, not a rate limit.
fn is_stream_unsupported(e: &LlmError) -> bool {
    match e {
        LlmError::Api { status, body, .. } => {
            matches!(status, 400 | 404 | 422) && body.to_ascii_lowercase().contains("stream")
        }
        _ => false,
    }
}

/// The assistant message to append before running tools, shared by both paths.
fn assistant_message(outcome: &RoundOutcome) -> Message {
    let mut content = Vec::new();
    if !outcome.text.is_empty() {
        content.push(ContentPart::text(outcome.text.clone()));
    }
    for call in &outcome.tool_calls {
        content.push(ContentPart::ToolUse {
            id: call.id.clone(),
            name: call.name.clone(),
            input: call.arguments.clone(),
        });
    }
    Message { role: Role::Assistant, content }
}

/// How a tool call reads in the turn's margin and meta line.
fn tool_log_line(call: &ToolCall) -> String {
    match call.name.as_str() {
        "search_sources" => {
            format!("searched {}", call.arguments["query"].as_str().unwrap_or_default())
        }
        "read_source" => format!(
            "read {} from line {}",
            call.arguments["title"].as_str().unwrap_or_default(),
            call.arguments["from_line"].as_u64().unwrap_or(1)
        ),
        other => other.to_string(),
    }
}

/// Citations, conversation and persistence: everything after the final text is
/// known. Runs only on success, so a stopped or failed turn still leaves no
/// trace either way it was asked.
/// Tokens summed over every round of the turn.
#[derive(Default)]
struct TurnUsage {
    input_tokens: u32,
    output_tokens: u32,
}

async fn finish_turn(
    state: &AppState,
    prep: &PreparedTurn,
    message: &str,
    text: &str,
    usage: TurnUsage,
    read_sources: Vec<(String, String)>,
    tool_log: Vec<String>,
) -> AppResult<ChatResponseBody> {
    let (input_tokens, output_tokens) = (usage.input_tokens, usage.output_tokens);
    let mut citations: Vec<Citation> = cited_hits(text, &prep.seen)
        .into_iter()
        .map(|h| Citation {
            source_id: h.source_id.clone(),
            source_title: h.source_title.clone(),
            chunk_id: h.chunk_id.clone(),
            locator: h.locator.clone(),
            label: citation_label(h),
        })
        .collect();

    // Smaller models answer well from a passage and forget the label. Fall back
    // to what the answer visibly reuses, then to what it deliberately opened.
    // An answer that reuses nothing gets no citations, which is the point.
    if citations.is_empty() {
        citations = echoed_hits(text, &prep.seen)
            .into_iter()
            .map(|h| Citation {
                source_id: h.source_id.clone(),
                source_title: h.source_title.clone(),
                chunk_id: h.chunk_id.clone(),
                locator: h.locator.clone(),
                label: String::new(),
            })
            .collect();
    }
    if citations.is_empty() {
        citations = read_sources
            .iter()
            .map(|(id, title)| Citation {
                source_id: id.clone(),
                source_title: title.clone(),
                chunk_id: String::new(),
                locator: None,
                label: String::new(),
            })
            .collect();
    }

    let conversation_id = match &prep.existing {
        Some(c) => c.id.clone(),
        None => {
            let title: String = message.chars().take(60).collect();
            db::create_conversation(&state.db, &state.user_id, &title).await?.id
        }
    };

    db::append_message(&state.db, &conversation_id, "user", message, None, None, None).await?;
    db::append_message(
        &state.db,
        &conversation_id,
        "assistant",
        text,
        Some(&json!(&citations).to_string()),
        Some(&prep.resolved.model.display_name),
        Some(db::Usage { input_tokens, output_tokens }),
    )
    .await?;

    Ok(ChatResponseBody {
        conversation_id,
        answer: text.to_string(),
        citations,
        model: prep.resolved.model.model_id.clone(),
        model_display_name: prep.resolved.model.display_name.clone(),
        effort: prep.effort.as_str().to_string(),
        input_tokens,
        output_tokens,
        context_used: input_tokens + output_tokens,
        context_window: prep.resolved.model.context_window,
        excerpts_searched: prep.seen.len(),
        tool_calls: tool_log,
    })
}

pub async fn chat(
    State(state): State<AppState>,
    Json(body): Json<ChatBody>,
) -> AppResult<Json<ChatResponseBody>> {
    if body.message.trim().is_empty() {
        return Err(AppError::BadRequest("message is empty".into()));
    }

    let (mut prep, client) = prepare_turn(&state, &body).await?;

    let mut input_tokens = 0;
    let mut output_tokens = 0;
    let mut tool_log: Vec<String> = Vec::new();
    let mut read_sources: Vec<(String, String)> = Vec::new();

    // The loop: answer, or ask for more and come back.
    let mut round = 0;
    let outcome = loop {
        let request = ChatRequest::new(&prep.resolved.model.model_id, prep.messages.clone())
            .system(SYSTEM)
            .tools(prep.tools.clone())
            .max_tokens(prep.resolved.model.max_output_tokens as u32)
            .effort(prep.effort);

        let outcome = run_round(client.as_ref(), &request, &prep.resolved, None).await?;
        input_tokens += outcome.input_tokens;
        output_tokens += outcome.output_tokens;

        if outcome.tool_calls.is_empty() || round >= MAX_TOOL_ROUNDS {
            break outcome;
        }

        prep.messages.push(assistant_message(&outcome));
        let mut results = Vec::new();
        for call in &outcome.tool_calls {
            tool_log.push(tool_log_line(call));
            let output = run_tool(
                &state,
                call,
                &body.source_ids,
                &mut prep.seen,
                &mut read_sources,
                prep.budget_chars,
            )
            .await?;
            results.push(ContentPart::ToolResult {
                tool_use_id: call.id.clone(),
                content: output,
                is_error: false,
            });
        }
        prep.messages.push(Message { role: Role::Tool, content: results });
        round += 1;
    };

    Ok(Json(
        finish_turn(
            &state,
            &prep,
            &body.message,
            &outcome.text,
            TurnUsage { input_tokens, output_tokens },
            read_sources,
            tool_log,
        )
        .await?,
    ))
}

/// The same turn as server-sent events: `token` carries prose as it arrives,
/// `tool` narrates each tool call, and `done` carries the whole
/// `ChatResponseBody` so the client finalizes exactly like a one-shot answer.
/// An `error` event ends a failed turn; like the one-shot path, nothing is
/// persisted unless the turn completes.
pub async fn stream(
    State(state): State<AppState>,
    Json(body): Json<ChatBody>,
) -> AppResult<Sse<impl futures::Stream<Item = Result<Event, axum::Error>>>> {
    if body.message.trim().is_empty() {
        return Err(AppError::BadRequest("message is empty".into()));
    }
    let (prep, client) = prepare_turn(&state, &body).await?;
    let (tx, rx) = futures::channel::mpsc::unbounded();
    tokio::spawn(run_stream_turn(state, body, prep, client, tx));
    Ok(Sse::new(rx).keep_alive(KeepAlive::new().interval(std::time::Duration::from_secs(15))))
}

async fn run_stream_turn(
    state: AppState,
    body: ChatBody,
    mut prep: PreparedTurn,
    client: Box<dyn LlmProvider>,
    tx: TokenSink,
) {
    let mut input_tokens = 0;
    let mut output_tokens = 0;
    let mut tool_log: Vec<String> = Vec::new();
    let mut read_sources: Vec<(String, String)> = Vec::new();
    let mut round = 0;

    loop {
        if tx.is_closed() {
            return;
        }
        let request = ChatRequest::new(&prep.resolved.model.model_id, prep.messages.clone())
            .system(SYSTEM)
            .tools(prep.tools.clone())
            .max_tokens(prep.resolved.model.max_output_tokens as u32)
            .effort(prep.effort);

        let outcome = match run_round(client.as_ref(), &request, &prep.resolved, Some(&tx)).await
        {
            Ok(o) => o,
            Err(e) => {
                send_error(&tx, &e);
                return;
            }
        };
        input_tokens += outcome.input_tokens;
        output_tokens += outcome.output_tokens;

        if outcome.tool_calls.is_empty() || round >= MAX_TOOL_ROUNDS {
            if tx.is_closed() {
                return;
            }
            match finish_turn(
                &state,
                &prep,
                &body.message,
                &outcome.text,
                TurnUsage { input_tokens, output_tokens },
                read_sources,
                tool_log,
            )
            .await
            {
                Ok(res) => emit_json(&tx, "done", json!(res)),
                Err(e) => send_error(&tx, &e),
            }
            return;
        }

        prep.messages.push(assistant_message(&outcome));
        let mut results = Vec::new();
        for call in &outcome.tool_calls {
            let line = tool_log_line(call);
            tool_log.push(line.clone());
            emit_json(&tx, "tool", json!({ "t": line }));
            let output = match run_tool(
                &state,
                call,
                &body.source_ids,
                &mut prep.seen,
                &mut read_sources,
                prep.budget_chars,
            )
            .await
            {
                Ok(o) => o,
                Err(e) => {
                    send_error(&tx, &e);
                    return;
                }
            };
            results.push(ContentPart::ToolResult {
                tool_use_id: call.id.clone(),
                content: output,
                is_error: false,
            });
        }
        prep.messages.push(Message { role: Role::Tool, content: results });
        round += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit_with(title: &str, locator: &str, source: &str, content: &str) -> SearchHit {
        let mut h = hit(title, locator, source);
        h.content = content.into();
        h
    }

    fn hit(title: &str, locator: &str, source: &str) -> SearchHit {
        SearchHit {
            chunk_id: format!("{source}-{locator}"),
            source_id: source.into(),
            source_title: title.into(),
            heading: None,
            locator: Some(locator.into()),
            ordinal: 0,
            snippet: "text".into(),
            content: "text".into(),
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
    fn an_uncited_answer_is_attributed_to_what_it_echoes() {
        let mut hit = hit("08NumberTheoryII.pdf", "line 69", "s1");
        hit.content = "Exercise 7. Find a solution to the linear Diophantine \
                       equation 1053x + 481y = 39 using the Euclidean algorithm."
            .into();
        let other = hit_with("11GroupTheory.pdf", "line 3", "s2", "Cosets partition a group.");

        let answer = "Solve the linear Diophantine equation 1053x + 481y = 39. \
                      Using the Euclidean algorithm, gcd(1053, 481) = 1.";
        let hits = vec![hit, other];
        let used = echoed_hits(answer, &hits);

        assert_eq!(used.len(), 1, "only the passage the answer reuses");
        assert_eq!(used[0].source_title, "08NumberTheoryII.pdf");
    }

    #[test]
    fn a_nothing_found_answer_echoes_nothing() {
        let mut hit = hit("fuchs.pdf", "line 1", "s2");
        hit.content = "Riemann surfaces and covering spaces, with monodromy.".into();
        let hits = vec![hit];
        let used = echoed_hits("I could not find that in your sources.", &hits);

        assert!(used.is_empty());
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

    #[test]
    fn only_a_stream_refusal_falls_back_to_one_shot() {
        use crate::llm::LlmError;
        let refused = LlmError::Api {
            status: 400,
            body: "Invalid content part type: stream must be boolean".into(),
            retry_after: None,
        };
        assert!(is_stream_unsupported(&refused));
        let plain_400 = LlmError::Api {
            status: 400,
            body: "model does not accept image parts".into(),
            retry_after: None,
        };
        assert!(!is_stream_unsupported(&plain_400));
        assert!(!is_stream_unsupported(&LlmError::Request("nope".into())));
    }

    fn scripted_resolved() -> models::Resolved {
        models::Resolved {
            model: models::Model {
                id: "m1".into(),
                provider_id: "p1".into(),
                model_id: "test-model".into(),
                display_name: "Test".into(),
                context_window: 32_768,
                max_output_tokens: 4096,
                supports_vision: false,
                supports_tools: true,
                supports_thinking: false,
                input_cost: None,
                output_cost: None,
                hidden: false,
                pinned: false,
                source: "manual".into(),
                last_seen_at: None,
                sort_order: 0,
                created_at: "now".into(),
            },
            provider: models::Provider {
                id: "p1".into(),
                owner_id: "local".into(),
                name: "Test".into(),
                api_style: "openai".into(),
                base_url: "http://localhost:1".into(),
                api_key: String::new(),
                created_at: "now".into(),
                updated_at: "now".into(),
            },
        }
    }

    struct ScriptedStream;

    #[async_trait::async_trait]
    impl crate::llm::LlmProvider for ScriptedStream {
        fn name(&self) -> &'static str {
            "scripted"
        }

        async fn chat(
            &self,
            _req: &crate::llm::ChatRequest,
        ) -> Result<crate::llm::ChatResponse, crate::llm::LlmError> {
            unreachable!("this test streams");
        }

        async fn chat_stream(
            &self,
            _req: &crate::llm::ChatRequest,
        ) -> Result<crate::llm::TokenStream, crate::llm::LlmError> {
            let events = vec![
                Ok(crate::llm::StreamEvent::Text("hel".into())),
                Ok(crate::llm::StreamEvent::Text("lo".into())),
                Ok(crate::llm::StreamEvent::Done(crate::llm::StreamDone {
                    tool_calls: Vec::new(),
                    input_tokens: 5,
                    output_tokens: 2,
                })),
            ];
            Ok(Box::pin(futures::stream::iter(events)))
        }
    }

    #[tokio::test]
    async fn streams_tokens_to_the_sink_while_accumulating() {
        let scripted = ScriptedStream;
        let resolved = scripted_resolved();
        let req = crate::llm::ChatRequest::new("test-model", vec![]);
        let (tx, mut rx) = futures::channel::mpsc::unbounded();

        let outcome = run_round(&scripted, &req, &resolved, Some(&tx)).await.unwrap();
        assert_eq!(outcome.text, "hello");
        assert_eq!((outcome.input_tokens, outcome.output_tokens), (5, 2));
        assert!(outcome.tool_calls.is_empty());

        // Both token events were forwarded before the round returned.
        assert!(rx.try_next().unwrap().is_some());
        assert!(rx.try_next().unwrap().is_some());
        assert!(rx.try_next().is_err(), "only the two tokens were sent");
    }
}
