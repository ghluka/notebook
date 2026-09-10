//! One error type for every handler.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

pub type AppResult<T> = Result<T, AppError>;

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("{0}")]
    BadRequest(String),
    #[error("{0} not found")]
    NotFound(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("database: {0}")]
    Db(#[from] sqlx::Error),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("model: {0}")]
    Llm(#[from] crate::llm::LlmError),
    /// The provider was still rate limited after the full retry schedule.
    /// Carries the model that failed, so the client can offer to switch.
    #[error("{message}")]
    RateLimited {
        message: String,
        model_id: Option<String>,
        model_name: Option<String>,
        waited: u64,
    },
    #[error("{0}")]
    Internal(#[from] anyhow::Error),
}

impl AppError {
    /// The JSON body for this error. `kind` lets the client tell a rate limit
    /// apart from an ordinary failure without parsing prose.
    pub fn payload(&self) -> serde_json::Value {
        match self {
            AppError::RateLimited { message, model_id, model_name, waited } => json!({
                "error": message,
                "kind": "rate_limited",
                "model_id": model_id,
                "model_name": model_name,
                "waited_seconds": waited,
            }),
            other => json!({ "error": other.to_string() }),
        }
    }

    pub fn is_rate_limited(&self) -> bool {
        matches!(self, AppError::RateLimited { .. })
    }

    /// Wrap a provider rate limit with the model it happened on.
    pub fn from_rate_limit(
        error: crate::llm::LlmError,
        model_id: &str,
        model_name: &str,
        context: &str,
    ) -> AppError {
        let waited = match &error {
            crate::llm::LlmError::RateLimited { waited, .. } => *waited,
            _ => 0,
        };
        AppError::RateLimited {
            message: format!(
                "{model_name} is rate limited: the provider was still refusing after \
                 {waited}s of waiting{context}. Switch to another model or try again later."
            ),
            model_id: Some(model_id.to_string()),
            model_name: Some(model_name.to_string()),
            waited,
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = match &self {
            AppError::BadRequest(_) => StatusCode::BAD_REQUEST,
            AppError::NotFound(_) => StatusCode::NOT_FOUND,
            AppError::Unsupported(_) => StatusCode::UNPROCESSABLE_ENTITY,
            AppError::Llm(crate::llm::LlmError::MissingKey(_)) => StatusCode::SERVICE_UNAVAILABLE,
            AppError::RateLimited { .. } => StatusCode::TOO_MANY_REQUESTS,
            AppError::Llm(crate::llm::LlmError::RateLimited { .. }) => {
                StatusCode::TOO_MANY_REQUESTS
            }
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        if status.is_server_error() {
            tracing::error!(error = %self, "request failed");
        }
        (status, Json(self.payload())).into_response()
    }
}
