use std::sync::Arc;
use std::time::Duration;

use crate::auth::AuthState;
use crate::config::Config;
use crate::db::Db;
use crate::error::{AppError, AppResult};
use crate::models::{self, Resolved};
use crate::storage::Storage;

#[derive(Clone)]
pub struct AppState(Arc<Inner>);

pub struct Inner {
    pub db: Db,
    pub storage: Storage,
    pub http: reqwest::Client,
    pub config: Config,
    /// no registration yet, always the seeded local user; session middleware fills this in
    pub user_id: String,
    pub auth: AuthState,
    /// one at a time, process wide, parallel analysis gets you rate limited
    pub analysis: Arc<tokio::sync::Semaphore>,
}

impl AppState {
    pub fn new(db: Db, storage: Storage, config: Config, user_id: String) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(600))
            .build()
            .expect("http client");
        AppState(Arc::new(Inner {
            db,
            storage,
            http,
            config,
            user_id,
            auth: AuthState::default(),
            analysis: Arc::new(tokio::sync::Semaphore::new(1)),
        }))
    }

    pub async fn role_model(&self, role: &str) -> AppResult<Resolved> {
        models::resolve_role(&self.db, &self.user_id, role).await?.ok_or_else(|| {
            AppError::Unsupported(format!(
                "no {role} model configured; add a provider and pin a model in Configure"
            ))
        })
    }
}

impl std::ops::Deref for AppState {
    type Target = Inner;

    fn deref(&self) -> &Inner {
        &self.0
    }
}
