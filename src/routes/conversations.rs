//! Saved conversations: list them, reopen one, rename, delete, compact.

use axum::Json;
use axum::extract::{Path, State};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::db;
use crate::error::{AppError, AppResult};
use crate::llm::{ChatRequest, Message};
use crate::state::AppState;

const COMPACT_SYSTEM: &str = "\
You are compacting a research conversation so it can continue with a shorter \
context. Write a dense summary of what has happened: the questions asked, the \
findings and the sources they came from with their locators, decisions taken, \
and anything still open. Keep every fact and citation that a later answer might \
need, drop the phrasing and the pleasantries. Write it as notes for yourself, \
not as a reply to anyone.";

pub async fn list(State(state): State<AppState>) -> AppResult<Json<Value>> {
    let conversations = db::list_conversations(&state.db, &state.user_id).await?;
    Ok(Json(json!({ "conversations": conversations })))
}

/// `GET /api/conversations/{id}` returns the whole transcript, compacted turns
/// included, because the history should still read in full after a compaction.
pub async fn get(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> AppResult<Json<Value>> {
    let conversation = load(&state, &id).await?;
    let messages = db::conversation_messages(&state.db, &conversation.id).await?;

    // What the context meter should read on reopening: the last turn's usage.
    let last_usage = messages
        .iter()
        .rev()
        .find(|m| m.role == "assistant" && m.input_tokens.is_some())
        .map(|m| m.input_tokens.unwrap_or(0) + m.output_tokens.unwrap_or(0));

    Ok(Json(json!({
        "conversation": conversation,
        "messages": messages,
        "context_used": last_usage,
    })))
}

#[derive(Deserialize)]
pub struct RenameBody {
    pub title: String,
}

pub async fn rename(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<RenameBody>,
) -> AppResult<Json<Value>> {
    let title = body.title.trim();
    if title.is_empty() {
        return Err(AppError::BadRequest("title is empty".into()));
    }
    load(&state, &id).await?;
    db::rename_conversation(&state.db, &state.user_id, &id, title).await?;
    Ok(Json(json!({ "id": id, "title": title })))
}

pub async fn delete(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> AppResult<Json<Value>> {
    db::delete_conversation(&state.db, &state.user_id, &id).await?;
    Ok(Json(json!({ "deleted": id })))
}

/// `POST /api/conversations/{id}/compact` folds the transcript into a summary
/// and marks those turns compacted, so the next prompt carries the summary
/// instead of the whole exchange. An already compacted conversation compacts
/// again, summarising its previous summary along with what followed.
pub async fn compact(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> AppResult<Json<Value>> {
    let conversation = load(&state, &id).await?;
    let active = db::active_messages(&state.db, &conversation.id).await?;

    if active.is_empty() {
        return Err(AppError::BadRequest(
            "nothing to compact: this conversation has no uncompacted turns".into(),
        ));
    }

    let mut transcript = String::new();
    if let Some(previous) = &conversation.summary {
        transcript.push_str("Summary of everything before this point:\n");
        transcript.push_str(previous);
        transcript.push_str("\n\n");
    }
    transcript.push_str("Transcript to fold in:\n");
    for m in &active {
        transcript.push_str(&format!("\n[{}]\n{}\n", m.role, m.content));
    }

    let resolved = state.role_model("researcher").await?;
    let request = ChatRequest::new(&resolved.model.model_id, vec![Message::user(transcript)])
        .system(COMPACT_SYSTEM)
        .max_tokens(resolved.model.max_output_tokens.min(4096) as u32);

    let client = resolved.client(&state.http)?;
    let response = crate::llm::chat_with_retry(client.as_ref(), &request).await.map_err(|e| {
        if e.is_rate_limited() {
            AppError::from_rate_limit(
                e,
                &resolved.model.id,
                &resolved.model.display_name,
                " while compacting",
            )
        } else {
            AppError::Llm(e)
        }
    })?;
    let summary = response.text.trim().to_string();
    if summary.is_empty() {
        return Err(AppError::Unsupported(
            "the model returned an empty summary; nothing was compacted".into(),
        ));
    }

    let compacted = db::compact_conversation(&state.db, &state.user_id, &id, &summary).await?;

    Ok(Json(json!({
        "conversation_id": id,
        "summary": summary,
        "compacted_messages": compacted,
        "model": resolved.model.display_name,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    })))
}

async fn load(state: &AppState, id: &str) -> AppResult<db::Conversation> {
    db::get_conversation(&state.db, &state.user_id, id)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("conversation {id}")))
}
