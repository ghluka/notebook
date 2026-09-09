//! Everything the prompt bar and the Configure panel talk to.
//!
//! API keys go in and are never handed back: a provider is returned with a
//! `key_hint` ("••••abcd") and nothing else. Every query is scoped to the
//! current user (`state.user_id`), which is the local one until auth exists.

use axum::Json;
use axum::extract::{Path, Query, State};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::error::{AppError, AppResult};
use crate::models::{self, Model, ModelPatch, NewModel, PRESETS, Provider};
use crate::state::AppState;

#[derive(Serialize)]
pub struct ProviderView {
    #[serde(flatten)]
    provider: Provider,
    key_hint: Option<String>,
    has_key: bool,
    model_count: usize,
}

fn view(p: Provider, model_count: usize) -> ProviderView {
    ProviderView { key_hint: p.key_hint(), has_key: !p.api_key.is_empty(), provider: p, model_count }
}

/// `GET /api/me` says who the server thinks you are. One local user for now.
pub async fn me(State(state): State<AppState>) -> AppResult<Json<Value>> {
    let user = models::get_user(&state.db, &state.user_id)
        .await?
        .ok_or_else(|| AppError::NotFound("user".into()))?;
    Ok(Json(json!({ "user": user, "auth": "local" })))
}

/// `GET /api/providers/presets` fills the "Add provider" menu.
pub async fn presets() -> Json<Value> {
    Json(json!({ "presets": PRESETS }))
}

pub async fn list_providers(State(state): State<AppState>) -> AppResult<Json<Value>> {
    let models = models::list_models(&state.db, &state.user_id).await?;
    let providers: Vec<ProviderView> = models::list_providers(&state.db, &state.user_id)
        .await?
        .into_iter()
        .map(|p| {
            let n = models.iter().filter(|m| m.provider_id == p.id).count();
            view(p, n)
        })
        .collect();
    Ok(Json(json!({ "providers": providers })))
}

#[derive(Deserialize)]
pub struct NewProvider {
    pub name: String,
    pub api_style: String,
    pub base_url: String,
    #[serde(default)]
    pub api_key: String,
}

/// Creating a provider immediately imports its catalogue, so the model list the
/// user sees is the one that endpoint actually serves. A provider whose list
/// call fails is still created, and the import error is reported alongside it.
pub async fn create_provider(
    State(state): State<AppState>,
    Json(body): Json<NewProvider>,
) -> AppResult<Json<Value>> {
    if body.api_style != "openai" && body.api_style != "anthropic" {
        return Err(AppError::BadRequest("api_style must be openai or anthropic".into()));
    }
    if body.base_url.trim().is_empty() {
        return Err(AppError::BadRequest("base_url is required".into()));
    }

    let provider = models::create_provider(
        &state.db,
        &state.user_id,
        body.name.trim(),
        &body.api_style,
        body.base_url.trim(),
        body.api_key.trim(),
    )
    .await?;

    let (imported, import_error) =
        match models::import_models(&state.db, &state.http, &provider).await {
            Ok(r) => (r.total, None),
            Err(e) => (0, Some(e.to_string())),
        };

    Ok(Json(json!({
        "provider": view(provider, imported),
        "imported": imported,
        "import_error": import_error,
    })))
}

#[derive(Deserialize)]
pub struct ProviderPatch {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
    /// Absent keeps the stored key; empty string clears it.
    #[serde(default)]
    pub api_key: Option<String>,
}

pub async fn update_provider(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<ProviderPatch>,
) -> AppResult<Json<ProviderView>> {
    let p = models::update_provider(
        &state.db,
        &state.user_id,
        &id,
        body.name.as_deref(),
        body.base_url.as_deref(),
        body.api_key.as_deref(),
    )
    .await?
    .ok_or_else(|| AppError::NotFound(format!("provider {id}")))?;
    Ok(Json(view(p, 0)))
}

pub async fn delete_provider(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> AppResult<Json<Value>> {
    models::delete_provider(&state.db, &state.user_id, &id).await?;
    Ok(Json(json!({ "deleted": id })))
}

/// `POST /api/providers/{id}/refresh` re-reads the endpoint's catalogue.
/// Hidden and pinned choices survive; new ids appear, missing ones stay put.
pub async fn refresh_provider(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> AppResult<Json<Value>> {
    let provider = models::get_provider(&state.db, &state.user_id, &id)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("provider {id}")))?;
    let report = models::import_models(&state.db, &state.http, &provider).await?;
    Ok(Json(json!({
        "total": report.total,
        "added": report.added,
        "non_chat": report.non_chat,
    })))
}

// ----------------------------------------------------------------- models

#[derive(Serialize)]
pub struct ModelView {
    #[serde(flatten)]
    model: Model,
    provider_name: String,
    api_style: String,
}

#[derive(Deserialize)]
pub struct ModelQuery {
    /// Hidden models are excluded unless asked for.
    #[serde(default)]
    pub include_hidden: bool,
    #[serde(default)]
    pub provider_id: Option<String>,
}

pub async fn list_models(
    State(state): State<AppState>,
    Query(q): Query<ModelQuery>,
) -> AppResult<Json<Value>> {
    let providers = models::list_providers(&state.db, &state.user_id).await?;
    let all = models::list_models(&state.db, &state.user_id).await?;
    let hidden_count = all.iter().filter(|m| m.hidden).count();

    let views: Vec<ModelView> = all
        .into_iter()
        .filter(|m| q.include_hidden || !m.hidden)
        .filter(|m| q.provider_id.as_ref().is_none_or(|p| *p == m.provider_id))
        .map(|m| {
            let p = providers.iter().find(|p| p.id == m.provider_id);
            ModelView {
                provider_name: p.map(|p| p.name.clone()).unwrap_or_default(),
                api_style: p.map(|p| p.api_style.clone()).unwrap_or_default(),
                model: m,
            }
        })
        .collect();

    Ok(Json(json!({
        "models": views,
        "hidden_count": hidden_count,
        "researcher_model": models::get_setting(&state.db, &state.user_id, "researcher_model").await?,
        "analyzer_model": models::get_setting(&state.db, &state.user_id, "analyzer_model").await?,
        "thinking_effort": models::effort(&state.db, &state.user_id).await?.as_str(),
    })))
}

/// Add a model the endpoint does not advertise (a fine-tune, a private
/// deployment). The normal path is importing the provider's own list.
pub async fn add_model(
    State(state): State<AppState>,
    Json(body): Json<NewModel>,
) -> AppResult<Json<Model>> {
    if models::get_provider(&state.db, &state.user_id, &body.provider_id).await?.is_none() {
        return Err(AppError::NotFound(format!("provider {}", body.provider_id)));
    }
    Ok(Json(models::add_model(&state.db, body, "manual").await?))
}

pub async fn update_model(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<ModelPatch>,
) -> AppResult<Json<Model>> {
    let m = models::update_model(&state.db, &state.user_id, &id, body)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("model {id}")))?;
    Ok(Json(m))
}

pub async fn delete_model(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> AppResult<Json<Value>> {
    models::delete_model(&state.db, &state.user_id, &id).await?;
    Ok(Json(json!({ "deleted": id })))
}

// --------------------------------------------------------------- settings

#[derive(Deserialize)]
pub struct SettingsPatch {
    #[serde(default)]
    pub researcher_model: Option<String>,
    #[serde(default)]
    pub analyzer_model: Option<String>,
    #[serde(default)]
    pub thinking_effort: Option<String>,
}

pub async fn get_settings(State(state): State<AppState>) -> AppResult<Json<Value>> {
    Ok(Json(settings_json(&state).await?))
}

pub async fn patch_settings(
    State(state): State<AppState>,
    Json(body): Json<SettingsPatch>,
) -> AppResult<Json<Value>> {
    for (key, value) in
        [("researcher_model", &body.researcher_model), ("analyzer_model", &body.analyzer_model)]
    {
        if let Some(model_id) = value {
            if models::get_model(&state.db, &state.user_id, model_id).await?.is_none() {
                return Err(AppError::NotFound(format!("model {model_id}")));
            }
            models::set_setting(&state.db, &state.user_id, key, model_id).await?;
        }
    }
    if let Some(effort) = &body.thinking_effort {
        let parsed = crate::llm::Effort::parse(effort)
            .ok_or_else(|| AppError::BadRequest("effort must be off|low|medium|high".into()))?;
        models::set_setting(&state.db, &state.user_id, "thinking_effort", parsed.as_str()).await?;
    }
    Ok(Json(settings_json(&state).await?))
}

/// What the prompt bar needs in one request: the models to offer, the current
/// role assignments and effort, and each model's context window.
async fn settings_json(state: &AppState) -> AppResult<Value> {
    let providers = models::list_providers(&state.db, &state.user_id).await?;
    let all = models::list_models(&state.db, &state.user_id).await?;

    // Pinned models are the picker. Before anything is pinned, offer everything
    // visible rather than an empty menu.
    let any_pinned = all.iter().any(|m| m.pinned && !m.hidden);
    let offered: Vec<Value> = all
        .iter()
        .filter(|m| !m.hidden && (!any_pinned || m.pinned))
        .map(|m| {
            json!({
                "id": m.id,
                "display_name": m.display_name,
                "model_id": m.model_id,
                "context_window": m.context_window,
                "supports_vision": m.supports_vision,
                "supports_thinking": m.supports_thinking,
                "pinned": m.pinned,
                "provider_name": providers.iter().find(|p| p.id == m.provider_id)
                                          .map(|p| p.name.clone()).unwrap_or_default(),
            })
        })
        .collect();

    let researcher = models::resolve_role(&state.db, &state.user_id, "researcher").await?;
    let analyzer = models::resolve_role(&state.db, &state.user_id, "analyzer").await?;

    Ok(json!({
        "pinned": offered,
        "any_pinned": any_pinned,
        "thinking_effort": models::effort(&state.db, &state.user_id).await?.as_str(),
        "researcher": researcher.map(|r| json!({
            "id": r.model.id,
            "display_name": r.model.display_name,
            "context_window": r.model.context_window,
            "supports_thinking": r.model.supports_thinking,
            "provider_name": r.provider.name,
        })),
        "analyzer": analyzer.map(|r| json!({
            "id": r.model.id,
            "display_name": r.model.display_name,
            "supports_vision": r.model.supports_vision,
            "provider_name": r.provider.name,
        })),
        "provider_count": providers.len(),
        "model_count": all.len(),
    }))
}
