//! The LLM layer: one neutral request type, two wire formats.
//!
//! Which model a role uses is not decided here; that lives in the `models`
//! registry, backed by the database. This module only knows how to talk.
//!
//! The neutral types cover the whole provider surface, including the
//! tool-calling pieces that only the phase-2 researcher loop will call.
#![allow(dead_code)]

pub mod anthropic;
pub mod openai;
pub mod types;

use async_trait::async_trait;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as B64;

pub use types::*;

#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("{0}")]
    MissingKey(String),
    #[error("provider returned {status}: {body}")]
    Api { status: u16, body: String },
    #[error("bad request: {0}")]
    Request(String),
    #[error("transport: {0}")]
    Http(#[from] reqwest::Error),
    #[error("malformed provider response: {0}")]
    Decode(#[from] serde_json::Error),
}

#[async_trait]
pub trait LlmProvider: Send + Sync {
    fn name(&self) -> &'static str;
    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse, LlmError>;
}

/// Wrap raw file bytes as a content part the analyzer can look at.
pub fn part_for_file(media_type: &str, bytes: &[u8], filename: Option<&str>) -> ContentPart {
    let data = B64.encode(bytes);
    if media_type.starts_with("image/") {
        ContentPart::Image { media_type: media_type.to_string(), data }
    } else {
        ContentPart::Document {
            media_type: media_type.to_string(),
            data,
            filename: filename.map(|s| s.to_string()),
        }
    }
}
