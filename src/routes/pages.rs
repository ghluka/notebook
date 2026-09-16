//! The two HTML entry points.
//!
//! They are served rather than handed to `ServeDir` for one reason: the client
//! has to know which prefix it is behind before it makes its first fetch, and
//! the server is the only one that knows. Both pages carry a `__BASE_PATH__`
//! placeholder in `<head>`, burned in here, and every URL the client builds is
//! prefixed with it.

use axum::extract::State;
use axum::http::{HeaderMap, header};
use axum::response::{IntoResponse, Redirect, Response};

use crate::base;
use crate::error::{AppError, AppResult};
use crate::state::AppState;

/// The placeholder both pages carry in `<head>`.
const TOKEN: &str = "__BASE_PATH__";

async fn page(file: &str, state: &AppState, headers: &HeaderMap) -> AppResult<Response> {
    let html = tokio::fs::read_to_string(file)
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("reading {file}: {e}")))?;
    // Whatever prefix this request came in under is the one the browser has to
    // use, since it is looking at the app through that same proxy.
    let base = base::effective(&state.config.base_path, headers);
    Ok((
        [
            (header::CONTENT_TYPE, "text/html; charset=utf-8"),
            // The body depends on the request, so a shared cache must not hold
            // one deployment's paths and hand them to another.
            (header::CACHE_CONTROL, "no-store"),
        ],
        html.replace(TOKEN, &base),
    )
        .into_response())
}

/// The app itself, at `/`, at `/index.html`, and at every conversation
/// permalink.
pub async fn index(State(state): State<AppState>, headers: HeaderMap) -> AppResult<Response> {
    page("static/index.html", &state, &headers).await
}

/// The one page an anonymous browser may see.
pub async fn login(State(state): State<AppState>, headers: HeaderMap) -> AppResult<Response> {
    page("static/login.html", &state, &headers).await
}

/// The mount point written without its trailing slash. Inside a mounted app
/// that path is the root, so it goes to the spelling the rest of the app uses
/// rather than answering 401 through the gate, which is what sent people
/// looking for a login page that was already there.
pub async fn mount_point(State(state): State<AppState>) -> Redirect {
    Redirect::permanent(&format!("{}/", state.config.base_path))
}
