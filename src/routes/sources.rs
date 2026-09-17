//! Source library: upload, organise, inspect, search, delete.

use axum::Json;
use axum::body::Body;
use axum::extract::{Multipart, Path, Query, State};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::analyzer;
use crate::db::{self, NewSource, Source};
use crate::error::{AppError, AppResult};
use crate::state::AppState;

#[derive(Serialize)]
pub struct Uploaded {
    source: Source,
    stored_new_blob: bool,
}

// one bad file shouldn't sink the rest of a drop
#[derive(Serialize)]
pub struct Rejected {
    filename: Option<String>,
    reason: String,
}

pub async fn upload(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> AppResult<Json<Value>> {
    struct Incoming {
        filename: Option<String>,
        media_type: Option<String>,
        bytes: Vec<u8>,
    }

    let mut title: Option<String> = None;
    let mut folder_id: Option<String> = None;
    let mut files: Vec<Incoming> = Vec::new();

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::BadRequest(format!("malformed multipart: {e}")))?
    {
        match field.name().unwrap_or_default() {
            "title" => title = field.text().await.ok().filter(|t| !t.trim().is_empty()),
            "folder_id" => folder_id = field.text().await.ok().filter(|t| !t.trim().is_empty()),
            "file" | "files" | "files[]" => {
                let filename = field.file_name().map(str::to_string);
                let media_type = field.content_type().map(str::to_string);
                let bytes = field
                    .bytes()
                    .await
                    .map_err(|e| AppError::BadRequest(format!("could not read file: {e}")))?
                    .to_vec();
                files.push(Incoming { filename, media_type, bytes });
            }
            _ => {}
        }
    }

    if files.is_empty() {
        return Err(AppError::BadRequest("missing `file` field".into()));
    }
    let single = files.len() == 1;
    let vault = db::active_vault(&state.db, &state.user_id).await?;

    let mut uploaded: Vec<Uploaded> = Vec::new();
    let mut rejected: Vec<Rejected> = Vec::new();
    for file in files {
        if file.bytes.is_empty() {
            rejected.push(Rejected {
                filename: file.filename.clone(),
                reason: "this file is empty".into(),
            });
            continue;
        }

        // kind comes from the bytes, not the name; unreadable types are refused before storage
        let (kind, media_type) = match analyzer::identify(
            &file.bytes,
            file.filename.as_deref(),
            file.media_type.as_deref().filter(|m| *m != "application/octet-stream"),
        ) {
            Ok(found) => found,
            Err(e) => {
                let reason = e.to_string();
                let reason = reason.strip_prefix("unsupported: ").unwrap_or(&reason).to_string();
                rejected.push(Rejected { filename: file.filename.clone(), reason });
                continue;
            }
        };

        let stored = state.storage.put(&file.bytes).await?;
        let stored_new_blob = stored.is_new;

        let source = db::insert_source(
            &state.db,
            NewSource {
                owner_id: state.user_id.clone(),
                vault_id: vault.id.clone(),
                folder_id: folder_id.clone(),
                title: title
                    .clone()
                    .filter(|_| single)
                    .or_else(|| file.filename.clone())
                    .unwrap_or_else(|| format!("untitled-{}", &stored.sha256[..8])),
                original_filename: file.filename,
                kind: kind.as_str().to_string(),
                media_type,
                byte_size: stored.byte_size as i64,
                sha256: stored.sha256,
                storage_path: stored.relative_path,
            },
        )
        .await?;

        uploaded.push(Uploaded { source, stored_new_blob });
    }

    // analysis is spawned so it outlives the request
    analyzer::spawn_analysis(&state, uploaded.iter().map(|u| u.source.clone()).collect());

    Ok(Json(json!({ "uploaded": uploaded, "rejected": rejected })))
}

pub async fn list(State(state): State<AppState>) -> AppResult<Json<Value>> {
    let vault = db::active_vault(&state.db, &state.user_id).await?;
    let sources = db::list_sources(&state.db, &vault.id).await?;
    let folders = db::list_folders(&state.db, &vault.id).await?;
    Ok(Json(json!({ "vault_id": vault.id, "sources": sources, "folders": folders })))
}

pub async fn get(State(state): State<AppState>, Path(id): Path<String>) -> AppResult<Json<Source>> {
    Ok(Json(load(&state, &id).await?))
}

// absent vs present-null separates "leave the folder" from "move to root"
fn double_option<'de, D>(de: D) -> Result<Option<Option<String>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    serde::Deserialize::deserialize(de).map(Some)
}

#[derive(Deserialize)]
pub struct SourcePatch {
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default, deserialize_with = "double_option")]
    pub folder_id: Option<Option<String>>,
    #[serde(default)]
    pub vault_id: Option<String>,
}

pub async fn patch(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<SourcePatch>,
) -> AppResult<Json<Source>> {
    let source = load(&state, &id).await?;
    let title = body.title.map(|t| t.trim().to_string()).filter(|t| !t.is_empty());
    db::update_source(&state.db, &source.id, title.as_deref(), body.folder_id).await?;
    if let Some(vault) = &body.vault_id {
        let vault = owned_vault(&state, vault).await?;
        db::move_source_to_vault(&state.db, &source.id, &vault.id).await?;
    }
    Ok(Json(load(&state, &id).await?))
}

async fn owned_vault(state: &AppState, id: &str) -> AppResult<db::Vault> {
    db::get_vault(&state.db, &state.user_id, id)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("vault {id}")))
}

pub async fn document(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> AppResult<Json<Value>> {
    let source = load(&state, &id).await?;
    let doc = db::get_document(&state.db, &source.id)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("rendition for source {id}")))?;
    let chunks = db::chunks_for_source(&state.db, &source.id).await?;
    Ok(Json(json!({ "source": source, "document": doc, "chunks": chunks })))
}

pub async fn raw(State(state): State<AppState>, Path(id): Path<String>) -> AppResult<Response> {
    let source = load(&state, &id).await?;
    let bytes = state.storage.read(&source.storage_path).await?;
    let filename = source.original_filename.unwrap_or(source.title).replace('"', "");

    Ok((
        [
            (header::CONTENT_TYPE, source.media_type),
            (header::CONTENT_DISPOSITION, format!("inline; filename=\"{filename}\"")),
        ],
        Body::from(bytes),
    )
        .into_response())
}

#[derive(Deserialize, Default)]
pub struct ReingestBody {
    // set when retrying after a rate limit on another model
    #[serde(default)]
    pub model_id: Option<String>,
}

pub async fn reingest(
    State(state): State<AppState>,
    Path(id): Path<String>,
    body: Option<Json<ReingestBody>>,
) -> AppResult<Json<Source>> {
    let source = load(&state, &id).await?;
    let model_id = body.and_then(|Json(b)| b.model_id).filter(|m| !m.trim().is_empty());

    db::set_source_result(&state.db, &source.id, "pending", None, None).await?;
    match model_id {
        // hand picked model runs alone, not through the queue that resolves the role
        Some(model) => {
            let state = state.clone();
            let source = source.clone();
            tokio::spawn(async move {
                let _slot = state.analysis.clone().acquire_owned().await;
                if let Err(e) = analyzer::ingest_with(&state, &source, Some(&model)).await {
                    tracing::warn!(source = %source.id, error = %e, "analysis failed");
                }
            });
        }
        None => analyzer::spawn_analysis(&state, vec![source.clone()]),
    }

    Ok(Json(load(&state, &id).await?))
}

#[derive(Deserialize)]
pub struct AskBody {
    pub question: String,
}

// same call the researcher makes as a tool
pub async fn ask(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<AskBody>,
) -> AppResult<Json<Value>> {
    let source = load(&state, &id).await?;
    let answer = analyzer::ask_source(&state, &source, &body.question).await?;
    Ok(Json(json!({ "source_id": source.id, "answer": answer })))
}

pub async fn delete(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> AppResult<Json<Value>> {
    let source = load(&state, &id).await?;
    db::delete_source(&state.db, &source.id).await?;

    let remaining = db::count_sources_with_sha(&state.db, &source.sha256).await?;
    state.storage.remove_if_unreferenced(&source.storage_path, remaining).await?;

    Ok(Json(json!({ "deleted": source.id })))
}

#[derive(Deserialize)]
pub struct SearchParams {
    pub q: String,
    #[serde(default)]
    pub source_id: Option<String>,
    #[serde(default)]
    pub limit: Option<i64>,
}

pub async fn search(
    State(state): State<AppState>,
    Query(params): Query<SearchParams>,
) -> AppResult<Json<Value>> {
    let limit = params.limit.unwrap_or(10).clamp(1, 50);
    let vault = db::active_vault(&state.db, &state.user_id).await?;
    let hits =
        db::search_chunks(&state.db, &vault.id, &params.q, params.source_id.as_deref(), limit)
            .await?;
    Ok(Json(json!({ "query": params.q, "hits": hits })))
}

#[derive(Deserialize)]
pub struct NewFolder {
    pub name: String,
    #[serde(default)]
    pub parent_id: Option<String>,
}

pub async fn create_folder(
    State(state): State<AppState>,
    Json(body): Json<NewFolder>,
) -> AppResult<Json<db::Folder>> {
    let name = body.name.trim();
    if name.is_empty() {
        return Err(AppError::BadRequest("folder name is empty".into()));
    }
    let vault = db::active_vault(&state.db, &state.user_id).await?;
    Ok(Json(
        db::create_folder(&state.db, &state.user_id, &vault.id, name, body.parent_id.as_deref())
            .await?,
    ))
}

#[derive(Deserialize)]
pub struct FolderPatch {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default, deserialize_with = "double_option")]
    pub parent_id: Option<Option<String>>,
    #[serde(default)]
    pub vault_id: Option<String>,
}

pub async fn patch_folder(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<FolderPatch>,
) -> AppResult<Json<Value>> {
    // a folder can't be its own parent; only cycle one move can make
    if let Some(Some(parent)) = &body.parent_id {
        if *parent == id {
            return Err(AppError::BadRequest("a folder cannot contain itself".into()));
        }
    }
    let name = body.name.map(|n| n.trim().to_string()).filter(|n| !n.is_empty());
    db::update_folder(&state.db, &state.user_id, &id, name.as_deref(), body.parent_id).await?;
    if let Some(vault) = &body.vault_id {
        let vault = owned_vault(&state, vault).await?;
        db::move_folder_to_vault(&state.db, &state.user_id, &id, &vault.id).await?;
    }
    Ok(Json(json!({ "updated": id })))
}

pub async fn delete_folder(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> AppResult<Json<Value>> {
    db::delete_folder(&state.db, &state.user_id, &id).await?;
    Ok(Json(json!({ "deleted": id })))
}

async fn load(state: &AppState, id: &str) -> AppResult<Source> {
    db::get_source(&state.db, id)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("source {id}")))
}
