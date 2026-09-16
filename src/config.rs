//! Environment-driven configuration. See AGENTS.md §6 for the full table.

use std::env;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderKind {
    Anthropic,
    OpenAi,
}

impl ProviderKind {
    fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "anthropic" | "claude" => Some(ProviderKind::Anthropic),
            "openai" | "oai" | "compatible" => Some(ProviderKind::OpenAi),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub bind_addr: String,
    pub database_url: String,
    pub upload_dir: PathBuf,
    pub max_upload_bytes: usize,
    /// The URL prefix this is mounted under when a reverse proxy serves it from
    /// a subdirectory, such as `/notebook`. Empty means the site root. See
    /// `crate::base`.
    pub base_path: String,

    pub anthropic_api_key: String,
    pub anthropic_base_url: String,
    pub openai_api_key: String,
    pub openai_base_url: String,

    pub researcher_provider: ProviderKind,
    pub researcher_model: String,
    pub analyzer_provider: ProviderKind,
    pub analyzer_model: String,
}

fn var(key: &str, default: &str) -> String {
    env::var(key).ok().filter(|v| !v.trim().is_empty()).unwrap_or_else(|| default.to_string())
}

impl Config {
    pub fn from_env() -> Self {
        Config {
            bind_addr: var("BIND_ADDR", "127.0.0.1:8080"),
            database_url: var("DATABASE_URL", "sqlite://data.db"),
            upload_dir: PathBuf::from(var("UPLOAD_DIR", "uploads")),
            max_upload_bytes: var("MAX_UPLOAD_BYTES", "268435456")
                .parse()
                .unwrap_or(256 * 1024 * 1024),
            base_path: crate::base::normalize(&var("BASE_PATH", "")),

            anthropic_api_key: var("ANTHROPIC_API_KEY", ""),
            anthropic_base_url: var("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
            openai_api_key: var("OPENAI_API_KEY", ""),
            openai_base_url: var("OPENAI_BASE_URL", "https://api.openai.com/v1"),

            researcher_provider: ProviderKind::parse(&var("RESEARCHER_PROVIDER", "anthropic"))
                .unwrap_or(ProviderKind::Anthropic),
            researcher_model: var("RESEARCHER_MODEL", "claude-sonnet-5"),
            analyzer_provider: ProviderKind::parse(&var("ANALYZER_PROVIDER", "anthropic"))
                .unwrap_or(ProviderKind::Anthropic),
            analyzer_model: var("ANALYZER_MODEL", "claude-sonnet-5"),
        }
    }
}
