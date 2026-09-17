//! separate libraries, one open at a time

use axum::Json;
use axum::extract::{Path, State};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::db;
use crate::error::{AppError, AppResult};
use crate::state::AppState;

pub async fn list(State(state): State<AppState>) -> AppResult<Json<Value>> {
    Ok(Json(listing(&state).await?))
}

async fn listing(state: &AppState) -> AppResult<Value> {
    let active = db::active_vault(&state.db, &state.user_id).await?;
    let vaults = db::list_vaults(&state.db, &state.user_id).await?;
    Ok(json!({ "vaults": vaults, "active": active.id }))
}

#[derive(Deserialize)]
pub struct VaultBody {
    pub name: String,
}

pub async fn create(
    State(state): State<AppState>,
    Json(body): Json<VaultBody>,
) -> AppResult<Json<db::Vault>> {
    let name = usable_name(&state, &body.name, None).await?;
    Ok(Json(db::create_vault(&state.db, &state.user_id, &name).await?))
}

pub async fn rename(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<VaultBody>,
) -> AppResult<Json<Value>> {
    let vault = load(&state, &id).await?;
    let name = usable_name(&state, &body.name, Some(&vault.id)).await?;
    db::rename_vault(&state.db, &state.user_id, &vault.id, &name).await?;
    Ok(Json(json!({ "id": vault.id, "name": name })))
}

pub async fn open(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> AppResult<Json<Value>> {
    let vault = load(&state, &id).await?;
    db::open_vault(&state.db, &state.user_id, &vault.id).await?;
    Ok(Json(listing(&state).await?))
}

// deletes its sources, folders and conversations; blobs go only when no other vault shares them
pub async fn delete(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> AppResult<Json<Value>> {
    let vault = load(&state, &id).await?;
    if db::list_vaults(&state.db, &state.user_id).await?.len() <= 1 {
        return Err(AppError::BadRequest(
            "this is the only vault; make another one before deleting it".into(),
        ));
    }

    let sources = db::list_sources(&state.db, &vault.id).await?;
    db::delete_vault(&state.db, &state.user_id, &vault.id).await?;

    let mut blobs: Vec<(&str, &str)> =
        sources.iter().map(|s| (s.sha256.as_str(), s.storage_path.as_str())).collect();
    blobs.sort();
    blobs.dedup();
    for (sha256, path) in blobs {
        let remaining = db::count_sources_with_sha(&state.db, sha256).await?;
        state.storage.remove_if_unreferenced(path, remaining).await?;
    }

    Ok(Json(json!({ "deleted": vault.id, "sources": sources.len() })))
}

async fn usable_name(state: &AppState, name: &str, except: Option<&str>) -> AppResult<String> {
    let name = name.trim();
    if name.is_empty() {
        return Err(AppError::BadRequest("a vault needs a name".into()));
    }
    if name.chars().count() > 80 {
        return Err(AppError::BadRequest("keep the name under 80 characters".into()));
    }
    // the unique index is ASCII case-insensitive, so match that
    let taken = db::list_vaults(&state.db, &state.user_id)
        .await?
        .iter()
        .any(|v| v.name.eq_ignore_ascii_case(name) && Some(v.id.as_str()) != except);
    if taken {
        return Err(AppError::BadRequest(format!("there is already a vault called \"{name}\"")));
    }
    Ok(name.to_string())
}

async fn load(state: &AppState, id: &str) -> AppResult<db::Vault> {
    db::get_vault(&state.db, &state.user_id, id)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("vault {id}")))
}
