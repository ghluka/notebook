use std::sync::Arc;
use std::time::Duration;

use crate::config::Config;
use crate::db::Db;
use crate::error::{AppError, AppResult};
use crate::models::{self, Resolved};
use crate::storage::Storage;

/// Shared, cheap to clone. Every handler gets one of these.
#[derive(Clone)]
pub struct AppState(Arc<Inner>);

pub struct Inner {
    pub db: Db,
    pub storage: Storage,
    /// One connection pool for every provider call.
    pub http: reqwest::Client,
    pub config: Config,
    /// Whose providers, keys and settings this request acts on. There is no
    /// registration yet, so it is always the seeded local user; when auth
    /// lands, this is what a session middleware fills in per request.
    pub user_id: String,
}

impl AppState {
    pub fn new(db: Db, storage: Storage, config: Config, user_id: String) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(600))
            .build()
            .expect("http client");
        AppState(Arc::new(Inner { db, storage, http, config, user_id }))
    }

    /// The model assigned to a role, with a message that points at the fix
    /// when nothing is configured.
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
