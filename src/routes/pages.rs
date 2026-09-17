//! served by hand so the client's base path is burned in before its first fetch

use axum::extract::State;
use axum::http::{HeaderMap, header};
use axum::response::{IntoResponse, Redirect, Response};

use crate::base;
use crate::error::{AppError, AppResult};
use crate::state::AppState;

const TOKEN: &str = "__BASE_PATH__";

async fn page(file: &str, state: &AppState, headers: &HeaderMap) -> AppResult<Response> {
    let html = tokio::fs::read_to_string(file)
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("reading {file}: {e}")))?;
    // the browser must use the same prefix this request came in under
    let base = base::effective(&state.config.base_path, headers);
    Ok((
        [
            (header::CONTENT_TYPE, "text/html; charset=utf-8"),
            // body depends on the request, so no shared cache
            (header::CACHE_CONTROL, "no-store"),
        ],
        html.replace(TOKEN, &base),
    )
        .into_response())
}

pub async fn index(State(state): State<AppState>, headers: HeaderMap) -> AppResult<Response> {
    page("static/index.html", &state, &headers).await
}

pub async fn login(State(state): State<AppState>, headers: HeaderMap) -> AppResult<Response> {
    page("static/login.html", &state, &headers).await
}

// bare mount point goes to the trailing-slash spelling instead of 401ing through the gate
pub async fn mount_point(State(state): State<AppState>) -> Redirect {
    Redirect::permanent(&format!("{}/", state.config.base_path))
}
