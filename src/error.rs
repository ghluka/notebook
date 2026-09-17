use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

pub type AppResult<T> = Result<T, AppError>;

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("{0}")]
    BadRequest(String),
    #[error("unauthorized: {0}")]
    Unauthorized(String),
    #[error("{0} not found")]
    NotFound(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// model gave a description instead of a transcription; another path on the same model repeats it
    #[error("{0}")]
    Rendition(String),
    #[error("database: {0}")]
    Db(#[from] sqlx::Error),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("model: {0}")]
    Llm(#[from] crate::llm::LlmError),
    #[error("{message}")]
    RateLimited {
        message: String,
        model_id: Option<String>,
        model_name: Option<String>,
        waited: u64,
    },
    /// endpoint refused this kind of content; the client can offer another model
    #[error("{message}")]
    Modality {
        message: String,
        model_id: Option<String>,
        model_name: Option<String>,
        /// what it would not take: "page images", "audio files"
        media: String,
    },
    #[error("{0}")]
    Internal(#[from] anyhow::Error),
}

impl AppError {
    /// kind is a contract: the client switches on it
    pub fn payload(&self) -> serde_json::Value {
        match self {
            AppError::Unauthorized(message) => json!({
                "error": message,
                "kind": "unauthorized",
            }),
            AppError::RateLimited { message, model_id, model_name, waited } => json!({
                "error": message,
                "kind": "rate_limited",
                "model_id": model_id,
                "model_name": model_name,
                "waited_seconds": waited,
            }),
            AppError::Modality { message, model_id, model_name, media } => json!({
                "error": message,
                "kind": "modality",
                "model_id": model_id,
                "model_name": model_name,
                "media": media,
            }),
            other => json!({ "error": other.to_string() }),
        }
    }

    pub fn is_rate_limited(&self) -> bool {
        matches!(self, AppError::RateLimited { .. })
    }

    pub fn is_rendition_failure(&self) -> bool {
        matches!(self, AppError::Rendition(_))
    }

    pub fn is_modality_refusal(&self) -> bool {
        matches!(self, AppError::Llm(e) if e.rejects_modality())
    }

    pub fn provider_message(&self) -> String {
        match self {
            AppError::Llm(e) => e.provider_message(),
            other => other.to_string(),
        }
    }

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
            AppError::Unauthorized(_) => StatusCode::UNAUTHORIZED,
            AppError::NotFound(_) => StatusCode::NOT_FOUND,
            AppError::Unsupported(_)
            | AppError::Rendition(_)
            | AppError::Modality { .. } => StatusCode::UNPROCESSABLE_ENTITY,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_failure_the_client_can_act_on_says_so() {
        let modality = AppError::Modality {
            message: "the analyzer model Nemotron 3 Nano 4b does not accept page images".into(),
            model_id: Some("m-1".into()),
            model_name: Some("Nemotron 3 Nano 4b".into()),
            media: "page images".into(),
        };
        let payload = modality.payload();
        assert_eq!(payload["kind"], "modality");
        assert_eq!(payload["model_id"], "m-1");
        assert_eq!(payload["media"], "page images");
        assert!(payload["error"].as_str().unwrap().contains("does not accept page images"));
        assert!(!modality.is_modality_refusal(), "this is the refusal itself, not one carrying an LlmError");

        let plain = AppError::Unsupported("no".into());
        assert!(plain.payload().get("kind").is_none());
    }

    #[test]
    fn a_provider_refusal_is_recognised_through_the_app_error() {
        let refusal = AppError::Llm(crate::llm::LlmError::Api {
            status: 400,
            body: r#"{"error":{"message":"model does not support image inputs"}}"#.into(),
            retry_after: None,
        });
        assert!(refusal.is_modality_refusal());
        assert_eq!(refusal.provider_message(), "model does not support image inputs");

        let other = AppError::Llm(crate::llm::LlmError::Api {
            status: 401,
            body: r#"{"error":{"message":"invalid api key"}}"#.into(),
            retry_after: None,
        });
        assert!(!other.is_modality_refusal());
    }
}
