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

const MAX_TOOL_ROUNDS: usize = 6;
const TITLE_MAX_CHARS: usize = 60;
const TITLE_MAX_TOKENS: u32 = 1024;

// the namer never answers the question, it only labels the thread
const NAMER_SYSTEM: &str = "\
You name conversations in a research notebook.\n\n\
Given the user's opening message, reply with a short title for the thread: \
the subject in three to six words, in sentence case, no trailing period. \
Name what the thread is about, do not answer it and do not echo the message \
back. Never use quotes, markdown, or a prefix like \"Title:\". \
Reply with the title alone.";

const HITS_PER_SEARCH: i64 = 8;
const NEIGHBOUR_RADIUS: i64 = 1;
const READ_MAX_CHARS: usize = 200_000;
const READ_MIN_CHARS: usize = 24_000;
const READ_DEFAULT_LINES: usize = 2000;
const READ_MAX_LINES: usize = 2000;

fn read_max_chars(budget_chars: usize) -> usize {
    (budget_chars / 4).clamp(READ_MIN_CHARS, READ_MAX_CHARS)
}

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
[notes.pdf, line 249]. Nothing else goes inside the brackets: no \
field names, no quotation marks, no equals signs. Every citation names its \
file, however many times that file has been cited already, and none goes \
inside mathematics: close the formula first, then cite. Put the citation after the \
sentence it supports, never inside the sentence as a word of it: the reader \
sees sources listed separately, so each sentence must read correctly with the \
bracket deleted: never \"as shown in [notes.pdf, line 3]\", which reads \"as \
shown in.\" once the mark is gone. Answers drawn from the conversation need no \
citation.\n\n\
When the sources genuinely do not answer a question, say so in one sentence and \
say what you looked for. Do not guess which file would have had it. Suggest \
different wording, or ask the user to attach the right files.\n\n\
You can do things with the sources, not only point at them: work an \
example, build a table, solve an exercise, set out a proof, or put what a \
source says in another form. The facts and definitions come from the sources \
and are cited; the working is yours to show. When the user asks you to make \
something, make it in the answer. Never reply that it can be made, or that it \
is shown somewhere.\n\n\
Use markdown, and LaTeX for mathematics. A table is a markdown table: a header \
row, a row of dashes under it, then one row per line.";

const ANSWER_NOW: &str = "You stopped without answering. Answer the question now, in \
prose, from what you have already found, and cite it. Do not search again. If the \
sources do not answer it, say so in one sentence.";

fn stalled(outcome: &RoundOutcome, round: usize) -> bool {
    outcome.text.trim().is_empty() && (outcome.tool_calls.is_empty() || round >= MAX_TOOL_ROUNDS)
}

#[derive(Deserialize)]
pub struct ChatBody {
    pub message: String,
    #[serde(default)]
    pub conversation_id: Option<String>,
    #[serde(default)]
    pub source_ids: Vec<String>,
    #[serde(default)]
    pub max_hits: Option<i64>,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub effort: Option<String>,
}

#[derive(Serialize)]
pub struct Citation {
    pub source_id: String,
    pub source_title: String,
    pub chunk_id: String,
    pub locator: Option<String>,
    // exact bracketed text the model wrote; the client strips it from prose
    pub label: String,
}

#[derive(Serialize)]
pub struct ChatResponseBody {
    pub conversation_id: String,
    pub message_id: String,
    pub user_message_id: String,
    pub answer: String,
    pub citations: Vec<Citation>,
    pub model: String,
    pub model_display_name: String,
    pub effort: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub context_used: u32,
    pub context_window: i64,
    pub excerpts_searched: usize,
    pub tool_calls: Vec<String>,
    pub thinking_ms: u64,
    pub trace: Vec<TraceStep>,
}

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
                 up. To summarize or explain a whole file, read it from line \
                 1 without a line limit and keep going until the reply shows \
                 the final line; short files come back whole in one call."
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
                                        the line you saw, or at 1 for the whole file."
                    },
                    "lines": {
                        "type": "integer",
                        "description": "How many lines to read. Leave out to \
                                        read to the end of what fits."
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

async fn search_expanded(
    state: &AppState,
    vault: &str,
    query: &str,
    attached: &[String],
    limit: i64,
) -> AppResult<Vec<SearchHit>> {
    let mut hits = Vec::new();
    if attached.is_empty() {
        hits = db::search_chunks(&state.db, vault, query, None, limit).await?;
    } else {
        for source_id in attached {
            hits.extend(
                db::search_chunks(&state.db, vault, query, Some(source_id), limit).await?,
            );
        }
        hits.sort_by(|a, b| a.score.total_cmp(&b.score));
        hits.truncate(limit as usize);
    }

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

fn citation_label(hit: &SearchHit) -> String {
    match hit.locator.as_deref() {
        Some(locator) => format!("[{}, {}]", hit.source_title, locator),
        None => format!("[{}]", hit.source_title),
    }
}

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

// look-alike brackets/spaces/hyphens folded for matching only; stored answer keeps original
fn plain_marks(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '\u{3010}' | '\u{3014}' | '\u{3016}' | '\u{FF3B}' => '[',
            '\u{3011}' | '\u{3015}' | '\u{3017}' | '\u{FF3D}' => ']',
            '\u{FF0C}' => ',',
            '\u{00A0}' | '\u{2007}' | '\u{2009}' | '\u{200A}' | '\u{202F}' | '\u{3000}' => ' ',
            '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}' | '\u{2212}' => '-',
            other => other,
        })
        .collect()
}

const FOLLOW_UP_WORDS: usize = 2;
const FOLLOW_UP_CONTEXT_CHARS: usize = 240;

// a thin follow-up is searched with the exchange before it; marks stay out
fn retrieval_query(message: &str, history: &[db::StoredMessage]) -> String {
    if db::search_words(message).len() >= FOLLOW_UP_WORDS {
        return message.to_string();
    }
    let last = |role: &str| history.iter().rev().find(|m| m.role == role);
    let mut query = message.to_string();
    for earlier in [last("assistant"), last("user")].into_iter().flatten() {
        query.push('\n');
        query.extend(without_marks(&earlier.content).chars().take(FOLLOW_UP_CONTEXT_CHARS));
    }
    query
}

fn without_marks(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut depth = 0usize;
    for c in plain_marks(text).chars() {
        match c {
            '[' => depth += 1,
            ']' => depth = depth.saturating_sub(1),
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    out
}

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

// catalogue titles carry extensions/number prefixes, model quotes the human name
fn match_source_by_title<'a>(
    briefs: &'a [db::SourceBrief],
    title: &str,
) -> Option<&'a db::SourceBrief> {
    let want = title.trim();
    if want.is_empty() {
        return None;
    }
    briefs
        .iter()
        .find(|b| b.title.eq_ignore_ascii_case(want))
        .or_else(|| {
            let lower = want.to_lowercase();
            briefs.iter().find(|b| b.title.to_lowercase().contains(&lower))
        })
        .or_else(|| {
            let flat: String =
                want.to_lowercase().chars().filter(|c| c.is_alphanumeric()).collect();
            if flat.is_empty() {
                return None;
            }
            briefs.iter().find(|b| {
                let own: String =
                    b.title.to_lowercase().chars().filter(|c| c.is_alphanumeric()).collect();
                own.contains(&flat) || flat.contains(&own)
            })
        })
}

async fn run_tool(
    state: &AppState,
    vault: &str,
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
            let hits = search_expanded(state, vault, query, attached, HITS_PER_SEARCH).await?;
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

            let briefs = db::source_briefs(&state.db, vault).await?;
            let found = match_source_by_title(&briefs, title);
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
                lines.min(READ_MAX_LINES),
                read_max_chars(budget_chars),
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
            let briefs = db::source_briefs(&state.db, vault).await?;
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

struct PreparedTurn {
    existing: Option<db::Conversation>,
    vault: String,
    resolved: models::Resolved,
    effort: Effort,
    budget_chars: usize,
    seen: Vec<SearchHit>,
    messages: Vec<Message>,
    tools: Vec<Tool>,
    attached: Vec<String>,
}

async fn prepare_turn(
    state: &AppState,
    body: &ChatBody,
) -> AppResult<(PreparedTurn, Box<dyn LlmProvider>)> {
    // created only after the model answers, so a stopped/failed turn leaves nothing
    let existing = match &body.conversation_id {
        Some(id) => Some(
            db::get_conversation(&state.db, &state.user_id, id)
                .await?
                .ok_or_else(|| AppError::NotFound(format!("conversation {id}")))?,
        ),
        None => None,
    };
    let vault = match existing.as_ref().and_then(|c| c.vault_id.clone()) {
        Some(vault) => vault,
        None => db::active_vault(&state.db, &state.user_id).await?.id,
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

    let budget_chars = if resolved.model.context_window > 0 {
        ((resolved.model.context_window as usize) * 4 / 3).clamp(8_000, 400_000)
    } else {
        24_000
    };

    let history = match &existing {
        Some(conversation) => db::active_messages(&state.db, &conversation.id).await?,
        None => Vec::new(),
    };

    let limit = body.max_hits.unwrap_or(HITS_PER_SEARCH).clamp(1, 30);
    let query = retrieval_query(&body.message, &history);
    let seen = search_expanded(state, &vault, &query, &body.source_ids, limit).await?;

    let mut messages: Vec<Message> = Vec::new();
    if let Some(conversation) = &existing {
        if let Some(summary) = &conversation.summary {
            messages.push(Message::user(format!(
                "Notes from the earlier part of this conversation:\n\n{summary}"
            )));
            messages.push(Message::assistant("Understood, I have those notes."));
        }
        for m in history {
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
            "The user attached {} file(s) to this question, and searches are \
             limited to them.\n\n",
            body.source_ids.len()
        )
    };
    // a keyword search never lists files, so the catalogue goes in up front
    let catalogue = render_catalogue(&db::source_briefs(&state.db, &vault).await?);
    messages.push(Message::user(format!(
        "{attachment_note}{catalogue}\n\nA keyword search on this message found \
         these passages. Search again or read a source if they are not \
         enough. If the question names a file by order or description instead \
         of title (\"the second reading\"), resolve it through the file list \
         above: read the index-like file first (a readings list, the \
         syllabus), then read the file it points to. Never declare anything \
         absent until you have consulted the file list and read the most \
         plausible file; one keyword search is never enough for that. When \
         asked to summarize, explain, or work through a source, read the \
         whole file first: every reply tells you \"lines X to Y of Z\"; keep \
         reading from the next line until Y reaches Z, and only then answer \
         from everything you read.\n{}\n\n---\n\n\
         Question: {}",
        render_excerpts(&seen, budget_chars),
        body.message
    )));

    let client = resolved.client(&state.http)?;
    let tools =
        if resolved.model.supports_tools { researcher_tools(&body.source_ids) } else { vec![] };

    Ok((
        PreparedTurn {
            existing,
            vault,
            resolved,
            effort,
            budget_chars,
            seen,
            messages,
            tools,
            attached: body.source_ids.clone(),
        },
        client,
    ))
}

fn render_catalogue(briefs: &[db::SourceBrief]) -> String {
    if briefs.is_empty() {
        return "Files in the notebook: none yet.".to_string();
    }
    let mut out = String::from("Files in the notebook:\n");
    for b in briefs {
        let summary: String =
            b.summary.as_deref().unwrap_or("no summary").chars().take(120).collect();
        out.push_str(&format!(
            "\n- {} ({}, {}, {} lines): {}",
            b.title, b.kind, b.status, b.total_lines, summary
        ));
    }
    out
}

struct RoundOutcome {
    thinking: String,
    thinking_ms: u64,
    text: String,
    tool_calls: Vec<ToolCall>,
    input_tokens: u32,
    output_tokens: u32,
}

// closed receiver means the person left; stop quietly, persist nothing
type TokenSink = futures::channel::mpsc::UnboundedSender<Result<Event, axum::Error>>;

fn emit_json(tx: &TokenSink, event: &'static str, value: serde_json::Value) {
    if tx.is_closed() {
        return;
    }
    match Event::default().event(event).json_data(value) {
        Ok(ev) => {
            let _ = tx.unbounded_send(Ok(ev));
        }
        Err(e) => send_error(tx, &AppError::Internal(anyhow::anyhow!(e))),
    }
}

fn send_error(tx: &TokenSink, e: &AppError) {
    if tx.is_closed() {
        return;
    }
    let event = Event::default()
        .event("error")
        .json_data(e.payload())
        .unwrap_or_else(|_| Event::default().event("error").data("stream failed"));
    let _ = tx.unbounded_send(Ok(event));
}

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
            thinking: r.thinking.unwrap_or_default(),
            thinking_ms: 0,
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
            let mut thinking = String::new();
            // timed from the first thought to the first word of the answer
            let mut thought_since: Option<std::time::Instant> = None;
            let mut thinking_ms = 0u64;
            let mut done = crate::llm::StreamDone::default();
            while let Some(ev) = stream.next().await {
                if tx.is_closed() {
                    return Err(AppError::Internal(anyhow::anyhow!("client went away")));
                }
                match ev.map_err(rate_limit)? {
                    StreamEvent::Thinking(t) => {
                        thought_since.get_or_insert_with(std::time::Instant::now);
                        thinking.push_str(&t);
                        emit_json(tx, "thinking", json!({ "t": t }));
                    }
                    StreamEvent::Text(t) => {
                        if let Some(since) = thought_since.take() {
                            thinking_ms += since.elapsed().as_millis() as u64;
                        }
                        text.push_str(&t);
                        emit_json(tx, "token", json!({ "t": t }));
                    }
                    StreamEvent::Done(d) => done = d,
                }
            }
            if let Some(since) = thought_since {
                thinking_ms += since.elapsed().as_millis() as u64;
            }
            Ok(RoundOutcome {
                text,
                thinking,
                thinking_ms,
                tool_calls: done.tool_calls,
                input_tokens: done.input_tokens,
                output_tokens: done.output_tokens,
            })
        }
        // endpoints that never learned stream fail the handshake; answer one-shot
        Err(e) if is_stream_unsupported(&e) => {
            tracing::info!("streaming refused, answering one-shot for this round");
            let r = crate::llm::chat_with_retry(client, request).await.map_err(rate_limit)?;
            if let Some(thought) = r.thinking.as_deref().filter(|t| !t.is_empty()) {
                emit_json(tx, "thinking", json!({ "t": thought }));
            }
            emit_json(tx, "token", json!({ "t": r.text }));
            Ok(RoundOutcome {
                text: r.text,
                thinking: r.thinking.unwrap_or_default(),
                thinking_ms: 0,
                tool_calls: r.tool_calls,
                input_tokens: r.usage.input_tokens,
                output_tokens: r.usage.output_tokens,
            })
        }
        Err(e) => Err(rate_limit(e)),
    }
}

fn is_stream_unsupported(e: &LlmError) -> bool {
    match e {
        LlmError::Api { status, body, .. } => {
            matches!(status, 400 | 404 | 422) && body.to_ascii_lowercase().contains("stream")
        }
        _ => false,
    }
}

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
            signature: call.signature.clone(),
        });
    }
    Message { role: Role::Assistant, content }
}

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
        "list_sources" => "listed the sources".to_string(),
        other => other.replace('_', " "),
    }
}

#[derive(Default)]
struct TurnUsage {
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Default)]
struct TurnTrace {
    read_sources: Vec<(String, String)>,
    tool_log: Vec<String>,
    thinking: String,
    thinking_ms: u64,
    steps: Vec<TraceStep>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TraceStep {
    Thinking { text: String, ms: u64 },
    Tool { line: String, found: String },
}

fn tool_found(call: &ToolCall, output: &str) -> String {
    match call.name.as_str() {
        "search_sources" => {
            let mut files: Vec<(String, Vec<String>)> = Vec::new();
            for line in output.lines().map(str::trim) {
                let Some(inner) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) else {
                    continue;
                };
                let Some((title, locator)) = inner.rsplit_once(", ") else { continue };
                let Some(number) = locator.strip_prefix("line ") else { continue };
                match files.iter_mut().find(|(t, _)| t == title) {
                    Some((_, numbers)) => numbers.push(number.to_string()),
                    None => files.push((title.to_string(), vec![number.to_string()])),
                }
            }
            if files.is_empty() {
                return "Nothing matched.".to_string();
            }
            files
                .into_iter()
                .map(|(title, numbers)| {
                    let noun = if numbers.len() == 1 { "line" } else { "lines" };
                    format!("{title}, {noun} {}", numbers.join(", "))
                })
                .collect::<Vec<_>>()
                .join("\n")
        }
        // "notes.pdf, lines 1 to 220 of 400. Cite passages from here as ..."
        "read_source" => output
            .lines()
            .next()
            .unwrap_or_default()
            .split(". Cite")
            .next()
            .unwrap_or_default()
            .to_string(),
        "list_sources" => match output.matches(" lines, ").count() {
            0 => output.lines().next().unwrap_or_default().to_string(),
            1 => "1 file".to_string(),
            n => format!("{n} files"),
        },
        _ => output.chars().take(200).collect(),
    }
}

/// a model's title can arrive quoted, prefixed or as a whole paragraph: keep the
/// first line, drop the decoration, and clamp to something a sidebar row can show
fn tidy_title(raw: &str) -> Option<String> {
    let mut lines = raw.trim().lines().map(str::trim).filter(|l| !l.is_empty());
    let mut line = lines.next()?;
    if line.ends_with(':') {
        if let Some(next) = lines.next() {
            line = next;
        }
    }
    let line = line.strip_prefix("Title:").unwrap_or(line).trim();
    let line = line.trim_matches(|c| matches!(c, '"' | '\'' | '`' | '*' | '#')).trim();
    let line = line.trim_end_matches(['.', ':']).trim();
    if line.is_empty() {
        return None;
    }
    let mut out: String = line.chars().take(TITLE_MAX_CHARS).collect();
    if line.chars().count() > TITLE_MAX_CHARS {
        if let Some(cut) = out.rfind(' ') {
            if cut >= TITLE_MAX_CHARS / 2 {
                out.truncate(cut);
            }
        }
    }
    Some(out.trim().to_string())
}

fn title_from_message(message: &str) -> String {
    message.chars().take(TITLE_MAX_CHARS).collect()
}

async fn generate_title(state: &AppState, message: &str, tx: Option<&TokenSink>) -> Option<String> {
    let resolved = models::resolve_namer(&state.db, &state.user_id).await.ok()??;
    let client = resolved.client(&state.http).ok()?;

    let prompt = message.chars().take(2000).collect::<String>();
    let request = ChatRequest::new(&resolved.model.model_id, vec![Message::user(&prompt)])
        .system(NAMER_SYSTEM)
        .max_tokens(TITLE_MAX_TOKENS);

    let mut title = String::new();
    match tx {
        Some(tx) => {
            let mut stream = crate::llm::chat_stream_with_retry(client.as_ref(), &request)
                .await
                .ok()?;
            while let Some(event) = futures::StreamExt::next(&mut stream).await {
                match event {
                    Ok(StreamEvent::Text(t)) => {
                        title.push_str(&t);
                        emit_json(tx, "title", json!({ "t": t }));
                    }
                    Ok(_) => {}
                    Err(_) => return None,
                }
                if title.contains('\n') || title.chars().count() > TITLE_MAX_CHARS * 2 {
                    break;
                }
            }
        }
        None => title = client.chat(&request).await.ok()?.text,
    }
    tidy_title(&title)
}

struct OpenTurn {
    conversation_id: String,
    user_message_id: String,
}

async fn begin_turn(state: &AppState, prep: &PreparedTurn, message: &str) -> AppResult<OpenTurn> {
    let conversation_id = match &prep.existing {
        Some(c) => c.id.clone(),
        None => {
            let title = title_from_message(message);
            db::create_conversation(&state.db, &state.user_id, &prep.vault, &title).await?.id
        }
    };

    let user_message_id =
        db::append_message(&state.db, &conversation_id, "user", message, None, None, None).await?;
    if !prep.attached.is_empty() {
        let mut files = Vec::new();
        for id in &prep.attached {
            if let Some(s) = db::get_source(&state.db, id).await? {
                files.push(json!({ "id": s.id, "title": s.title, "kind": s.kind }));
            }
        }
        db::set_message_attachments(&state.db, &user_message_id, &json!(files).to_string())
            .await?;
    }
    Ok(OpenTurn { conversation_id, user_message_id })
}

async fn apply_title(state: &AppState, conversation_id: &str, title: Option<String>) {
    let Some(title) = title else { return };
    let _ = db::rename_conversation(&state.db, &state.user_id, conversation_id, &title).await;
}

async fn finish_turn(
    state: &AppState,
    prep: &PreparedTurn,
    open: &OpenTurn,
    text: &str,
    usage: TurnUsage,
    trace: TurnTrace,
) -> AppResult<ChatResponseBody> {
    let (input_tokens, output_tokens) = (usage.input_tokens, usage.output_tokens);
    let mut citations: Vec<Citation> = cited_hits(&plain_marks(text), &prep.seen)
        .into_iter()
        .map(|h| Citation {
            source_id: h.source_id.clone(),
            source_title: h.source_title.clone(),
            chunk_id: h.chunk_id.clone(),
            locator: h.locator.clone(),
            label: citation_label(h),
        })
        .collect();

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
        citations = trace
            .read_sources
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

    let OpenTurn { conversation_id, user_message_id } = open;
    let message_id = db::append_message(
        &state.db,
        conversation_id,
        "assistant",
        text,
        Some(&json!(&citations).to_string()),
        Some(&prep.resolved.model.display_name),
        Some(db::Usage { input_tokens, output_tokens }),
    )
    .await?;
    if !trace.thinking.trim().is_empty() {
        db::set_message_thinking(&state.db, &message_id, &trace.thinking, trace.thinking_ms)
            .await?;
    }
    if !trace.steps.is_empty() {
        db::set_message_trace(&state.db, &message_id, &json!(&trace.steps).to_string()).await?;
    }

    Ok(ChatResponseBody {
        conversation_id: conversation_id.clone(),
        message_id,
        user_message_id: user_message_id.clone(),
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
        tool_calls: trace.tool_log,
        thinking_ms: trace.thinking_ms,
        trace: trace.steps,
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
    let open = begin_turn(&state, &prep, &body.message).await?;

    let mut input_tokens = 0;
    let mut output_tokens = 0;
    let mut tool_log: Vec<String> = Vec::new();
    let mut read_sources: Vec<(String, String)> = Vec::new();
    let mut thinking = String::new();
    let mut thinking_ms = 0u64;
    let mut steps: Vec<TraceStep> = Vec::new();

    let mut round = 0;
    let mut answered_now = false;
    let outcome = loop {
        let request = ChatRequest::new(&prep.resolved.model.model_id, prep.messages.clone())
            .system(SYSTEM)
            .tools(prep.tools.clone())
            .max_tokens(prep.resolved.model.max_output_tokens as u32)
            .effort(prep.effort);

        let outcome = run_round(client.as_ref(), &request, &prep.resolved, None).await?;
        input_tokens += outcome.input_tokens;
        output_tokens += outcome.output_tokens;
        if !outcome.thinking.is_empty() {
            if !thinking.is_empty() {
                thinking.push_str("\n\n");
            }
            thinking.push_str(&outcome.thinking);
            steps.push(TraceStep::Thinking {
                text: outcome.thinking.clone(),
                ms: outcome.thinking_ms,
            });
        }
        thinking_ms += outcome.thinking_ms;

        // worked then said nothing: ask once, dropping the round's output
        if stalled(&outcome, round) && !answered_now {
            answered_now = true;
            prep.messages.push(Message::user(ANSWER_NOW));
            continue;
        }

        if outcome.tool_calls.is_empty() || round >= MAX_TOOL_ROUNDS {
            break outcome;
        }

        prep.messages.push(assistant_message(&outcome));
        let mut results = Vec::new();
        for call in &outcome.tool_calls {
            tool_log.push(tool_log_line(call));
            let output = run_tool(
                &state,
                &prep.vault,
                call,
                &body.source_ids,
                &mut prep.seen,
                &mut read_sources,
                prep.budget_chars,
            )
            .await?;
            steps.push(TraceStep::Tool {
                line: tool_log_line(call),
                found: tool_found(call, &output),
            });
            results.push(ContentPart::ToolResult {
                tool_use_id: call.id.clone(),
                content: output,
                is_error: false,
            });
        }
        prep.messages.push(Message { role: Role::Tool, content: results });
        round += 1;
    };

    if prep.existing.is_none() {
        let title = generate_title(&state, &body.message, None).await;
        apply_title(&state, &open.conversation_id, title).await;
    }

    Ok(Json(
        finish_turn(
            &state,
            &prep,
            &open,
            &outcome.text,
            TurnUsage { input_tokens, output_tokens },
            TurnTrace { read_sources, tool_log, thinking, thinking_ms, steps },
        )
        .await?,
    ))
}

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
    let mut thinking = String::new();
    let mut thinking_ms = 0u64;
    let mut steps: Vec<TraceStep> = Vec::new();
    let mut round = 0;
    let mut answered_now = false;

    let fresh = prep.existing.is_none();
    let open = match begin_turn(&state, &prep, &body.message).await {
        Ok(o) => o,
        Err(e) => {
            send_error(&tx, &e);
            return;
        }
    };
    let will_name =
        fresh && models::resolve_namer(&state.db, &state.user_id).await.ok().flatten().is_some();

    emit_json(
        &tx,
        "chat",
        json!({ "conversation_id": open.conversation_id, "fresh": fresh, "naming": will_name }),
    );

    let naming = match will_name {
        false => None,
        true => {
            let (state, message, tx) = (state.clone(), body.message.clone(), tx.clone());
            let id = open.conversation_id.clone();
            Some(tokio::spawn(async move {
                let title = generate_title(&state, &message, Some(&tx)).await;
                apply_title(&state, &id, title.clone()).await;
                title
            }))
        }
    };

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
        if !outcome.thinking.is_empty() {
            if !thinking.is_empty() {
                thinking.push_str("\n\n");
            }
            thinking.push_str(&outcome.thinking);
            steps.push(TraceStep::Thinking {
                text: outcome.thinking.clone(),
                ms: outcome.thinking_ms,
            });
        }
        thinking_ms += outcome.thinking_ms;

        // worked then said nothing: ask once, dropping the round's output
        if stalled(&outcome, round) && !answered_now {
            answered_now = true;
            prep.messages.push(Message::user(ANSWER_NOW));
            continue;
        }

        if outcome.tool_calls.is_empty() || round >= MAX_TOOL_ROUNDS {
            if tx.is_closed() {
                return;
            }
            match finish_turn(
                &state,
                &prep,
                &open,
                &outcome.text,
                TurnUsage { input_tokens, output_tokens },
                TurnTrace { read_sources, tool_log, thinking, thinking_ms, steps },
            )
            .await
            {
                Ok(res) => emit_json(&tx, "done", json!(res)),
                Err(e) => send_error(&tx, &e),
            }
            if let Some(handle) = naming {
                let named = handle.await.ok().flatten();
                emit_json(&tx, "named", json!({ "title": named }));
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
                &prep.vault,
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
            let found = tool_found(call, &output);
            emit_json(&tx, "tool_done", json!({ "found": found }));
            steps.push(TraceStep::Tool { line, found });
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

    #[test]
    fn a_plain_title_survives_untouched() {
        assert_eq!(tidy_title("Derivatives of trig functions").as_deref(),
                   Some("Derivatives of trig functions"));
    }

    #[test]
    fn a_dressed_up_title_is_undressed() {
        for raw in [
            "\"Derivatives of trig functions\"",
            "Title: Derivatives of trig functions",
            "**Derivatives of trig functions**",
            "Derivatives of trig functions.",
            "  Derivatives of trig functions  ",
        ] {
            assert_eq!(tidy_title(raw).as_deref(), Some("Derivatives of trig functions"),
                       "{raw} should reduce to the bare title");
        }
    }

    #[test]
    fn a_chatty_namer_gives_up_the_title_under_its_preamble() {
        let raw = "Sure, here is a title:\n\nDerivatives of trig functions";
        assert_eq!(tidy_title(raw).as_deref(), Some("Derivatives of trig functions"));
    }

    #[test]
    fn a_title_that_merely_ends_in_a_colon_is_kept() {
        assert_eq!(tidy_title("Calculus:").as_deref(), Some("Calculus"));
    }

    #[test]
    fn a_long_title_is_clamped_on_a_word_boundary() {
        let raw = "The fundamental theorem of calculus and its consequences for integration";
        let got = tidy_title(raw).expect("title");
        assert!(got.chars().count() <= TITLE_MAX_CHARS, "{got} fits a sidebar row");
        assert!(raw.starts_with(&got), "{got} is a prefix of the original");
        assert!(!got.ends_with(' '), "{got} has no trailing space");
        assert!(raw[got.len()..].starts_with(' '), "{got} was cut between words");
    }

    #[test]
    fn nothing_usable_names_nothing() {
        for raw in ["", "   ", "\n\n", "\"\"", "."] {
            assert_eq!(tidy_title(raw), None, "{raw:?} yields no title");
        }
    }

    #[test]
    fn without_a_namer_the_message_still_names_the_thread() {
        let long = "a".repeat(200);
        assert_eq!(title_from_message(&long).chars().count(), TITLE_MAX_CHARS);
        assert_eq!(title_from_message("short one"), "short one");
    }

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
            hit("notes.pdf", "line 249", "s1"),
            hit("textbook.pdf", "line 12", "s2"),
        ];
        let answer = "A widget needs a flag [notes.pdf, line 249].";
        let used = cited_hits(answer, &hits);

        assert_eq!(used.len(), 1);
        assert_eq!(used[0].source_title, "notes.pdf");
    }

    #[test]
    fn an_uncited_answer_is_attributed_to_what_it_echoes() {
        let mut hit = hit("notes.pdf", "line 69", "s1");
        hit.content = "Exercise 7. Compute the monthly total from the itemised \
                       receipts using the running balance method."
            .into();
        let other = hit_with("other.pdf", "line 3", "s2", "Each group splits into disjoint sets.");

        let answer = "Compute the monthly total from the itemised receipts. \
                      Using the running balance method, the total is found.";
        let hits = vec![hit, other];
        let used = echoed_hits(answer, &hits);

        assert_eq!(used.len(), 1, "only the passage the answer reuses");
        assert_eq!(used[0].source_title, "notes.pdf");
    }

    #[test]
    fn a_nothing_found_answer_echoes_nothing() {
        let mut hit = hit("textbook.pdf", "line 1", "s2");
        hit.content = "Surface patches and covering maps, with monodromy.".into();
        let hits = vec![hit];
        let used = echoed_hits("I could not find that in your sources.", &hits);

        assert!(used.is_empty());
    }

    #[test]
    fn an_answer_citing_nothing_shows_nothing() {
        let hits = vec![hit("textbook.pdf", "line 1", "s2")];
        let used = cited_hits("I could not find that in your sources.", &hits);

        assert!(used.is_empty(), "a nothing found answer must not carry citations");
    }

    #[test]
    fn a_file_cited_without_a_locator_still_resolves() {
        let hits = vec![
            hit("textbook.pdf", "line 1", "s2"),
            hit("textbook.pdf", "line 90", "s2"),
        ];
        let used = cited_hits("See [textbook.pdf] for the proof.", &hits);

        assert_eq!(used.len(), 1);
        assert_eq!(used[0].locator.as_deref(), Some("line 1"));
    }

    #[test]
    fn citations_in_look_alike_brackets_still_resolve() {
        let hits = vec![
            hit("guide.pdf", "line 2", "s1"),
            hit("guide.pdf", "line 3", "s1"),
            hit("textbook.pdf", "line 9", "s2"),
        ];
        let answer = "Students double\u{2011}back and re\u{2011}read.\
                      \u{3010}guide.pdf, line 2\u{3011} Students fix on formulas.\
                      \u{3010}guide.pdf,\u{202F}line\u{202F}3\u{3011}";
        assert!(cited_hits(answer, &hits).is_empty(), "the raw marks match nothing");

        let used = cited_hits(&plain_marks(answer), &hits);
        let lines: Vec<_> = used.iter().filter_map(|h| h.locator.as_deref()).collect();
        assert_eq!(lines, vec!["line 2", "line 3"], "exact lines, not a word-overlap guess");

        assert_eq!(plain_marks("re\u{2011}read"), "re-read");
        assert_eq!(plain_marks("plain [a.pdf, line 1]"), "plain [a.pdf, line 1]");
    }

    fn call(name: &str) -> ToolCall {
        ToolCall { id: "c".into(), name: name.into(), arguments: json!({}), signature: None }
    }

    #[test]
    fn a_round_that_ends_in_nothing_is_a_stall() {
        let round = |text: &str, calls: usize| RoundOutcome {
            text: text.into(),
            thinking: "I should search for it.".into(),
            thinking_ms: 0,
            tool_calls: (0..calls).map(|_| call("search_sources")).collect(),
            input_tokens: 0,
            output_tokens: 0,
        };
        assert!(stalled(&round("", 0), 0), "thought, then nothing");
        assert!(stalled(&round("  \n", 0), 2), "whitespace is nothing too");
        assert!(!stalled(&round("", 1), 0), "a tool call is progress");
        assert!(stalled(&round("", 1), MAX_TOOL_ROUNDS), "a call with no budget left to run it");
        assert!(!stalled(&round("A set is a collection.", 0), 0));
        assert_eq!(tool_log_line(&call("list_sources")), "listed the sources");
    }

    #[test]
    fn a_tool_call_is_summed_up_by_what_it_found() {
        let hits = vec![
            hit("guide.pdf", "line 306", "s1"),
            hit("guide.pdf", "line 307", "s1"),
            hit("readings.md", "line 4", "s2"),
        ];
        let searched = render_excerpts(&hits, 100_000);
        assert_eq!(
            tool_found(&call("search_sources"), &searched),
            "guide.pdf, lines 306, 307\nreadings.md, line 4"
        );
        assert_eq!(tool_found(&call("search_sources"), "Nothing matched that search."),
                   "Nothing matched.");

        let read = "notes.pdf, lines 1 to 220 of 400. Cite passages from here as \
                    [notes.pdf, line N] using the numbers shown.\n\n1 | text";
        assert_eq!(tool_found(&call("read_source"), read), "notes.pdf, lines 1 to 220 of 400");
    }

    fn said(role: &str, content: &str) -> db::StoredMessage {
        db::StoredMessage {
            id: String::new(),
            conversation_id: String::new(),
            role: role.into(),
            content: content.into(),
            tool_calls: None,
            citations: None,
            compacted: false,
            model: None,
            input_tokens: None,
            output_tokens: None,
            thinking: None,
            thinking_ms: None,
            trace: None,
            attachments: None,
            created_at: String::new(),
        }
    }

    #[test]
    fn a_follow_up_that_names_nothing_searches_with_the_exchange_before_it() {
        let history = vec![
            said("user", "summarize the first reading"),
            said("assistant", "Reading styles differ.\u{3010}guide.pdf, line 2\u{3011}"),
            said("user", "what about the second reading?"),
            said("assistant", "The second reading is \u{201c}Maple and Oak\u{201d} for Week 02."),
        ];
        let query = retrieval_query("can you summarize it???", &history);
        assert!(query.contains("Maple and Oak"), "{query}");
        assert!(query.contains("second reading"), "{query}");
        assert!(!query.contains("guide"), "only the exchange just before: {query}");

        let named = "summarize the second reading";
        assert_eq!(retrieval_query(named, &history), named);
        assert_eq!(retrieval_query("what about the second reading?", &history),
                   "what about the second reading?");
        assert_eq!(retrieval_query("why?", &[]), "why?");
    }

    #[test]
    fn earlier_citations_do_not_steer_the_search() {
        let history = vec![
            said("user", "what about integrals"),
            said("assistant", "Euler's theorem. \u{3010}textbook.pdf, line 3804\u{3011} [textbook.pdf, line 3821]"),
        ];
        let query = retrieval_query("and?", &history);
        assert!(query.contains("Euler"), "{query}");
        assert!(!query.to_lowercase().contains("textbook"), "{query}");
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

        assert!(rx.try_next().unwrap().is_some());
        assert!(rx.try_next().unwrap().is_some());
        assert!(rx.try_next().is_err(), "only the two tokens were sent");
    }

    fn brief(title: &str) -> db::SourceBrief {
        db::SourceBrief {
            id: format!("id-{title}"),
            title: title.into(),
            kind: "pdf".into(),
            status: "ready".into(),
            summary: None,
            total_lines: 100,
        }
    }

    #[test]
    fn read_caps_scale_with_the_window_without_regressing_small_ones() {
        assert_eq!(read_max_chars(400_000), 100_000);
        assert_eq!(read_max_chars(32_768 * 4 / 3), READ_MIN_CHARS);
        assert_eq!(read_max_chars(10_000_000), READ_MAX_CHARS);
    }

    #[test]
    fn catalogue_lists_every_file_for_the_model() {
        let out = render_catalogue(&[brief("readings.md"), brief("guide.pdf")]);
        assert!(out.contains("readings.md (pdf, ready, 100 lines)"));
        assert!(out.contains("guide.pdf"));
        assert!(render_catalogue(&[]).contains("none yet"));
    }

    #[test]
    fn a_human_title_finds_its_numbered_file() {
        let briefs = vec![
            brief("02MapleAndOak.pdf"),
            brief("03BirchTheory.pdf"),
            brief("readings.md"),
        ];
        assert_eq!(
            match_source_by_title(&briefs, "Maple and Oak").unwrap().title,
            "02MapleAndOak.pdf"
        );
        assert_eq!(
            match_source_by_title(&briefs, "readings.md").unwrap().title,
            "readings.md"
        );
        assert_eq!(
            match_source_by_title(&briefs, "birch").unwrap().title,
            "03BirchTheory.pdf"
        );
        assert!(match_source_by_title(&briefs, "no such file").is_none());
        assert!(match_source_by_title(&briefs, "   ").is_none());
    }
}
