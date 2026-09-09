//! notebook: an agent harness over your own sources. See AGENTS.md.

mod analyzer;
mod config;
mod db;
mod error;
mod llm;
mod models;
mod routes;
mod state;
mod storage;

use config::Config;
use state::AppState;
use storage::Storage;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _ = dotenvy::dotenv();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "notebook=info,tower_http=info".into()),
        )
        .init();

    let config = Config::from_env();
    let db = db::connect(&config.database_url).await?;
    let storage = Storage::new(&config.upload_dir).await?;
    // Model configuration lives in the database; the environment only seeds it.
    let user_id = models::bootstrap(&db, &config).await?;

    let bind_addr = config.bind_addr.clone();
    let state = AppState::new(db, storage, config, user_id);
    let app = routes::router(state);

    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    tracing::info!("listening on http://{bind_addr}");

    axum::serve(listener, app).with_graceful_shutdown(shutdown()).await?;
    Ok(())
}

async fn shutdown() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("shutting down");
}
