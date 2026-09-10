pub mod chat;
pub mod conversations;
pub mod health;
pub mod models;
pub mod sources;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::routing::{get, get_service, patch, post};
use tower_http::services::{ServeDir, ServeFile};
use tower_http::trace::TraceLayer;

use crate::state::AppState;

pub fn router(state: AppState) -> Router {
    let max_upload = state.config.max_upload_bytes;

    let api = Router::new()
        .route("/health", get(health::health))
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
        .route("/conversations/{id}/retry", post(conversations::retry))
        // Model configuration: what the prompt bar and Configure panel use.
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
        .layer(DefaultBodyLimit::max(max_upload));

    Router::new()
        .nest("/api", api)
        .route("/health", get(health::health))
        // Conversation permalinks. The client reads the id out of the path, so
        // every one of these serves the same page.
        .route("/c/{id}", get_service(ServeFile::new("static/index.html")))
        .fallback_service(ServeDir::new("static").append_index_html_on_directories(true))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
