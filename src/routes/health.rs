use axum::Json;
use axum::extract::State;
use serde_json::{Value, json};
use sqlx::Row;

use crate::error::AppResult;
use crate::models;
use crate::state::AppState;

// says whether a key is present, never what it is
pub async fn health(State(state): State<AppState>) -> AppResult<Json<Value>> {
    let row = sqlx::query("SELECT count(*) AS n FROM sources WHERE owner_id = ?1")
        .bind(&state.user_id)
        .fetch_one(&state.db)
        .await?;
    let sources: i64 = row.get("n");

    let researcher = models::resolve_role(&state.db, &state.user_id, "researcher").await?;
    let analyzer = models::resolve_role(&state.db, &state.user_id, "analyzer").await?;
    let describe = |r: Option<models::Resolved>| {
        r.map(|r| {
            json!({
                "provider": r.provider.name,
                "api_style": r.provider.api_style,
                "model": r.model.model_id,
                "has_key": !r.provider.api_key.is_empty(),
            })
        })
    };

    Ok(Json(json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "sources": sources,
        "user": state.user_id,
        "providers": models::list_providers(&state.db, &state.user_id).await?.len(),
        "models": models::list_models(&state.db, &state.user_id).await?.len(),
        "thinking_effort": models::effort(&state.db, &state.user_id).await?.as_str(),
        "researcher": describe(researcher),
        "analyzer": describe(analyzer),
    })))
}
