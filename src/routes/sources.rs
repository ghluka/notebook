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
    /// False when these exact bytes were already in the store.
    stored_new_blob: bool,
}

/// A file that was refused, and why. One bad file in a drop of twenty should
/// not sink the other nineteen, so these travel back beside the successes.
#[derive(Serialize)]
pub struct Rejected {
    filename: Option<String>,
    reason: String,
}

/// `POST /api/sources` takes any number of `file` parts, so one drop of a dozen
/// files is one request. An optional `title` renames a single upload, and
/// `folder_id` puts them all in a folder.
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

        /* What it is comes from the bytes, not from the name. A file whose
           extension lies is analyzed for what it actually is, and one that is
           nothing the analyzer reads is turned away here, before it takes up
           disk and a slot in the queue. */
        let (kind, media_type) = match analyzer::identify(
            &file.bytes,
            file.filename.as_deref(),
            file.media_type.as_deref().filter(|m| *m != "application/octet-stream"),
        ) {
            Ok(found) => found,
            Err(e) => {
                // The bare sentence, since the file name is already beside it.
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

    // Storing is done; analyzing starts now and outlives this request, so the
    // response returns as fast as the disk writes.
    analyzer::spawn_analysis(&state, uploaded.iter().map(|u| u.source.clone()).collect());

    // Even a drop where nothing survived answers the same way, one line per
    // file: the client shows those reasons next to the names they belong to,
    // which reads better than one status code standing for twenty files.
    Ok(Json(json!({ "uploaded": uploaded, "rejected": rejected })))
}

/// `GET /api/sources` returns everything the explorer draws: folders and
/// sources together, each carrying its parent.
pub async fn list(State(state): State<AppState>) -> AppResult<Json<Value>> {
    let sources = db::list_sources(&state.db, &state.user_id).await?;
    let folders = db::list_folders(&state.db, &state.user_id).await?;
    Ok(Json(json!({ "sources": sources, "folders": folders })))
}

pub async fn get(State(state): State<AppState>, Path(id): Path<String>) -> AppResult<Json<Source>> {
    Ok(Json(load(&state, &id).await?))
}

/// Distinguishes "field absent" from "field present and null", which is what
/// separates "leave the folder alone" from "move to the root".
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
}

/// `PATCH /api/sources/{id}` renames a source or moves it between folders.
pub async fn patch(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<SourcePatch>,
) -> AppResult<Json<Source>> {
    let source = load(&state, &id).await?;
    let title = body.title.map(|t| t.trim().to_string()).filter(|t| !t.is_empty());
    db::update_source(&state.db, &source.id, title.as_deref(), body.folder_id).await?;
    Ok(Json(load(&state, &id).await?))
}

/// `GET /api/sources/{id}/document` returns the analyzer's markdown rendition.
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

/// `GET /api/sources/{id}/raw` serves the original bytes inline, so the viewer
/// can show a PDF or an image without a second copy of the file.
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
    /// Run on this model instead of the one holding the analyzer role. Set by
    /// the client when retrying after a rate limit on a different model.
    #[serde(default)]
    pub model_id: Option<String>,
}

/// `POST /api/sources/{id}/reingest` queues the analyzer over stored bytes and
/// returns at once. Watch the source's status for the outcome.
pub async fn reingest(
    State(state): State<AppState>,
    Path(id): Path<String>,
    body: Option<Json<ReingestBody>>,
) -> AppResult<Json<Source>> {
    let source = load(&state, &id).await?;
    let model_id = body.and_then(|Json(b)| b.model_id).filter(|m| !m.trim().is_empty());

    db::set_source_result(&state.db, &source.id, "pending", None, None).await?;
    match model_id {
        // A hand picked model is a one off, so it runs on its own rather than
        // through the shared queue path that always resolves the role.
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

/// `POST /api/sources/{id}/ask` puts a question to the analyzer about the
/// ORIGINAL file. This is the same call the researcher makes as a tool.
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

/// `GET /api/search?q=` is the retrieval surface the researcher uses.
pub async fn search(
    State(state): State<AppState>,
    Query(params): Query<SearchParams>,
) -> AppResult<Json<Value>> {
    let limit = params.limit.unwrap_or(10).clamp(1, 50);
    let hits = db::search_chunks(&state.db, &params.q, params.source_id.as_deref(), limit).await?;
    Ok(Json(json!({ "query": params.q, "hits": hits })))
}

// ---------------------------------------------------------------- folders

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
    Ok(Json(db::create_folder(&state.db, &state.user_id, name, body.parent_id.as_deref()).await?))
}

#[derive(Deserialize)]
pub struct FolderPatch {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default, deserialize_with = "double_option")]
    pub parent_id: Option<Option<String>>,
}

pub async fn patch_folder(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<FolderPatch>,
) -> AppResult<Json<Value>> {
    // A folder cannot be its own parent, which is the only cycle one move can
    // create from the explorer.
    if let Some(Some(parent)) = &body.parent_id {
        if *parent == id {
            return Err(AppError::BadRequest("a folder cannot contain itself".into()));
        }
    }
    let name = body.name.map(|n| n.trim().to_string()).filter(|n| !n.is_empty());
    db::update_folder(&state.db, &state.user_id, &id, name.as_deref(), body.parent_id).await?;
    Ok(Json(json!({ "updated": id })))
}

/// Deleting a folder keeps the files; they fall back to the root.
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
