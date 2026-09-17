pub mod auth;
pub mod chat;
pub mod conversations;
pub mod health;
pub mod models;
pub mod pages;
pub mod sources;
pub mod vaults;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::middleware;
use axum::routing::{get, patch, post};
use tower_http::services::ServeDir;
use tower_http::trace::TraceLayer;

use crate::state::AppState;

pub fn router(state: AppState) -> Router {
    let max_upload = state.config.max_upload_bytes;
    let base = state.config.base_path.clone();

    let api = Router::new()
        .route("/health", get(health::health))
        .route("/vaults", get(vaults::list).post(vaults::create))
        .route("/vaults/{id}", patch(vaults::rename).delete(vaults::delete))
        .route("/vaults/{id}/open", post(vaults::open))
        .route("/sources", get(sources::list).post(sources::upload))
        .route(
            "/sources/{id}",
            get(sources::get).patch(sources::patch).delete(sources::delete),
        )
        .route("/sources/{id}/document", get(sources::document))
        .route("/sources/{id}/raw", get(sources::raw))
        .route("/folders", post(sources::create_folder))
        .route(
            "/folders/{id}",
            patch(sources::patch_folder).delete(sources::delete_folder),
        )
        .route("/sources/{id}/reingest", post(sources::reingest))
        .route("/sources/{id}/ask", post(sources::ask))
        .route("/search", get(sources::search))
        .route("/chat", post(chat::chat))
        .route("/chat/stream", post(chat::stream))
        .route("/conversations", get(conversations::list))
        .route(
            "/conversations/{id}",
            get(conversations::get).patch(conversations::rename).delete(conversations::delete),
        )
        .route("/conversations/{id}/compact", post(conversations::compact))
        .route("/conversations/{id}/rewind", post(conversations::rewind))
        .route("/me", get(models::me))
        .route("/providers/presets", get(models::presets))
        .route("/providers", get(models::list_providers).post(models::create_provider))
        .route(
            "/providers/{id}",
            patch(models::update_provider).delete(models::delete_provider),
        )
        .route("/providers/{id}/refresh", post(models::refresh_provider))
        .route("/models", get(models::list_models).post(models::add_model))
        .route("/models/{id}", patch(models::update_model).delete(models::delete_model))
        .route("/settings", get(models::get_settings).patch(models::patch_settings))
        .route("/auth/status", get(auth::status))
        .route("/auth/challenge", post(auth::challenge))
        .route("/auth/login", post(auth::login))
        .route("/auth/setup", post(auth::setup))
        .route("/auth/logout", post(auth::logout))
        .route("/auth/logout_all", post(auth::logout_all))
        .route("/auth/password", patch(auth::change_password))
        .route("/auth/sessions", get(auth::sessions))
        .layer(DefaultBodyLimit::max(max_upload));

    let app = Router::new()
        .nest("/api", api)
        // served by a handler, not ServeDir, so the base path is burned into the html
        .route("/login", get(pages::login))
        .route("/login.html", get(pages::login))
        .route("/", get(pages::index))
        .route("/index.html", get(pages::index))
        .route("/health", get(health::health))
        .route("/c/{id}", get(pages::index))
        .layer(middleware::from_fn_with_state(state.clone(), crate::auth::require_auth))
        .layer(TraceLayer::new_for_http())
        .with_state(state.clone());

    // static fallback sits behind the same gate, so unknown paths 401 instead of 404
    let files = Router::new()
        .fallback_service(ServeDir::new("static").append_index_html_on_directories(true))
        .layer(middleware::from_fn_with_state(state.clone(), crate::auth::require_auth))
        .with_state(state.clone());

    if base.is_empty() {
        return app.fallback_service(files);
    }

    // mounted under a prefix, the table answers both at root and at the mount point,
    // since a proxy may strip the prefix or pass it through; bare mount point redirects
    app.clone()
        .nest(&format!("{base}/"), app)
        .route(&base, get(pages::mount_point).with_state(state.clone()))
        .fallback_service(files)
}
