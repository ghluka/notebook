//! The model registry: users, their providers, and the models those providers
//! report.
//!
//! This is the source of truth at runtime; the environment only seeds it on an
//! empty database (see `bootstrap`). Everything the prompt bar and the Configure
//! panel show comes from these tables.
//!
//! Ownership: every provider belongs to a user, and models and settings hang off
//! providers. There is no registration yet, so `bootstrap` creates one local
//! user and the server treats it as the current one; the queries already take an
//! `owner` so adding real auth means changing where that string comes from, and
//! nothing else.

use serde::{Deserialize, Serialize};
use sqlx::Row;

use crate::config::Config;
use crate::db::{Db, new_id, now};
use crate::llm::{Effort, LlmError, LlmProvider, anthropic::AnthropicProvider, openai::OpenAiProvider};

pub const LOCAL_USER: &str = "local";

// ------------------------------------------------------------------- rows

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct User {
    pub id: String,
    pub name: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Provider {
    pub id: String,
    pub owner_id: String,
    pub name: String,
    pub api_style: String,
    pub base_url: String,
    /// Never serialized. The API exposes `key_hint` instead.
    #[serde(skip_serializing)]
    pub api_key: String,
    pub created_at: String,
    pub updated_at: String,
}

impl Provider {
    /// What the UI is allowed to see of a key: enough to recognise it, never
    /// enough to use it.
    pub fn key_hint(&self) -> Option<String> {
        let k = self.api_key.trim();
        if k.is_empty() {
            return None;
        }
        let tail: String = k.chars().rev().take(4).collect::<Vec<_>>().into_iter().rev().collect();
        Some(format!("••••{tail}"))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Model {
    pub id: String,
    pub provider_id: String,
    pub model_id: String,
    pub display_name: String,
    pub context_window: i64,
    pub max_output_tokens: i64,
    pub supports_vision: bool,
    pub supports_tools: bool,
    pub supports_thinking: bool,
    pub input_cost: Option<f64>,
    pub output_cost: Option<f64>,
    pub hidden: bool,
    pub pinned: bool,
    pub source: String,
    pub last_seen_at: Option<String>,
    pub sort_order: i64,
    pub created_at: String,
}

/// A model together with everything needed to call it.
pub struct Resolved {
    pub model: Model,
    pub provider: Provider,
}

impl Resolved {
    /// A remote endpoint with no stored key can only fail, and the provider's
    /// own 401 is a poor explanation, so say what to fix instead. Loopback
    /// endpoints are left alone: they legitimately take no key.
    pub fn client(&self, http: &reqwest::Client) -> Result<Box<dyn LlmProvider>, LlmError> {
        if self.provider.api_key.is_empty() && !is_loopback(&self.provider.base_url) {
            return Err(LlmError::MissingKey(format!(
                "no API key stored for {}; add one in Configure",
                self.provider.name
            )));
        }
        Ok(client_for(http, &self.provider))
    }
}

pub fn client_for(http: &reqwest::Client, p: &Provider) -> Box<dyn LlmProvider> {
    match p.api_style.as_str() {
        "anthropic" => {
            Box::new(AnthropicProvider::new(http.clone(), p.base_url.clone(), p.api_key.clone()))
        }
        _ => Box::new(OpenAiProvider::new(http.clone(), p.base_url.clone(), p.api_key.clone())),
    }
}

/// Whether a base URL points at this machine.
pub fn is_loopback(base_url: &str) -> bool {
    let after_scheme = base_url.split("://").last().unwrap_or(base_url);
    let authority = after_scheme.split('/').next().unwrap_or_default();
    let host = match authority.rsplit_once(':') {
        Some((h, port)) if port.chars().all(|c| c.is_ascii_digit()) => h,
        _ => authority,
    };
    matches!(host, "localhost" | "127.0.0.1" | "[::1]" | "::1" | "0.0.0.0")
        || host.ends_with(".local")
}

#[cfg(test)]
mod tests {
    use super::is_loopback;

    #[test]
    fn loopback_detection() {
        assert!(is_loopback("http://127.0.0.1:1234/v1"));
        assert!(is_loopback("http://localhost:11434/v1"));
        assert!(is_loopback("http://box.local:8000"));
        assert!(!is_loopback("https://api.anthropic.com"));
        assert!(!is_loopback("https://openrouter.ai/api/v1"));
    }
}

// ------------------------------------------------------------------ users

pub async fn get_user(db: &Db, id: &str) -> sqlx::Result<Option<User>> {
    sqlx::query_as::<_, User>("SELECT * FROM users WHERE id = ?1")
        .bind(id)
        .fetch_optional(db)
        .await
}

pub async fn ensure_user(db: &Db, id: &str, name: &str) -> sqlx::Result<User> {
    sqlx::query("INSERT OR IGNORE INTO users (id, name, created_at) VALUES (?1, ?2, ?3)")
        .bind(id)
        .bind(name)
        .bind(now())
        .execute(db)
        .await?;
    get_user(db, id).await?.ok_or(sqlx::Error::RowNotFound)
}

// -------------------------------------------------------------- providers

pub async fn list_providers(db: &Db, owner: &str) -> sqlx::Result<Vec<Provider>> {
    sqlx::query_as::<_, Provider>(
        "SELECT * FROM providers WHERE owner_id = ?1 ORDER BY created_at",
    )
    .bind(owner)
    .fetch_all(db)
    .await
}

/// Scoped by owner on purpose: a provider id from another user must read as
/// missing, not as forbidden.
pub async fn get_provider(db: &Db, owner: &str, id: &str) -> sqlx::Result<Option<Provider>> {
    sqlx::query_as::<_, Provider>("SELECT * FROM providers WHERE id = ?1 AND owner_id = ?2")
        .bind(id)
        .bind(owner)
        .fetch_optional(db)
        .await
}

pub async fn create_provider(
    db: &Db,
    owner: &str,
    name: &str,
    api_style: &str,
    base_url: &str,
    api_key: &str,
) -> sqlx::Result<Provider> {
    let id = new_id();
    let ts = now();
    sqlx::query(
        "INSERT INTO providers (id, owner_id, name, api_style, base_url, api_key,
                                created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7)",
    )
    .bind(&id)
    .bind(owner)
    .bind(name)
    .bind(api_style)
    .bind(base_url.trim_end_matches('/'))
    .bind(api_key)
    .bind(&ts)
    .execute(db)
    .await?;
    get_provider(db, owner, &id).await?.ok_or(sqlx::Error::RowNotFound)
}

/// An absent `api_key` keeps the stored one, so the UI never has to round-trip
/// a secret it cannot see.
pub async fn update_provider(
    db: &Db,
    owner: &str,
    id: &str,
    name: Option<&str>,
    base_url: Option<&str>,
    api_key: Option<&str>,
) -> sqlx::Result<Option<Provider>> {
    sqlx::query(
        "UPDATE providers
            SET name       = coalesce(?3, name),
                base_url   = coalesce(?4, base_url),
                api_key    = coalesce(?5, api_key),
                updated_at = ?6
          WHERE id = ?1 AND owner_id = ?2",
    )
    .bind(id)
    .bind(owner)
    .bind(name)
    .bind(base_url.map(|u| u.trim_end_matches('/')))
    .bind(api_key)
    .bind(now())
    .execute(db)
    .await?;
    get_provider(db, owner, id).await
}

pub async fn delete_provider(db: &Db, owner: &str, id: &str) -> sqlx::Result<()> {
    sqlx::query("DELETE FROM providers WHERE id = ?1 AND owner_id = ?2")
        .bind(id)
        .bind(owner)
        .execute(db)
        .await?;
    Ok(())
}

// ----------------------------------------------------------------- models

pub async fn list_models(db: &Db, owner: &str) -> sqlx::Result<Vec<Model>> {
    sqlx::query_as::<_, Model>(
        "SELECT m.* FROM models m
           JOIN providers p ON p.id = m.provider_id
          WHERE p.owner_id = ?1
          ORDER BY m.sort_order, m.display_name",
    )
    .bind(owner)
    .fetch_all(db)
    .await
}

pub async fn get_model(db: &Db, owner: &str, id: &str) -> sqlx::Result<Option<Model>> {
    sqlx::query_as::<_, Model>(
        "SELECT m.* FROM models m
           JOIN providers p ON p.id = m.provider_id
          WHERE m.id = ?1 AND p.owner_id = ?2",
    )
    .bind(id)
    .bind(owner)
    .fetch_optional(db)
    .await
}

#[derive(Debug, Clone, Deserialize)]
pub struct NewModel {
    pub provider_id: String,
    pub model_id: String,
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub context_window: Option<i64>,
    #[serde(default)]
    pub max_output_tokens: Option<i64>,
    #[serde(default)]
    pub supports_vision: Option<bool>,
    #[serde(default)]
    pub supports_tools: Option<bool>,
    #[serde(default)]
    pub supports_thinking: Option<bool>,
    #[serde(default)]
    pub pinned: Option<bool>,
}

/// Add a model the provider does not list (a fine-tune, a private deployment).
/// Re-adding an existing id just refreshes its metadata.
pub async fn add_model(db: &Db, m: NewModel, source: &str) -> sqlx::Result<Model> {
    let guess = Capabilities::guess(&m.model_id);
    upsert_model(
        db,
        &m.provider_id,
        &UpsertModel {
            model_id: m.model_id.clone(),
            display_name: m.display_name.clone().unwrap_or_else(|| m.model_id.clone()),
            context_window: m.context_window.unwrap_or(guess.context_window),
            max_output_tokens: m.max_output_tokens.unwrap_or(guess.max_output_tokens),
            supports_vision: m.supports_vision.unwrap_or(guess.vision),
            supports_tools: m.supports_tools.unwrap_or(guess.tools),
            supports_thinking: m.supports_thinking.unwrap_or(guess.thinking),
            pinned: m.pinned.unwrap_or(false),
            hidden: false,
            source: source.to_string(),
        },
    )
    .await?;

    sqlx::query_as::<_, Model>("SELECT * FROM models WHERE provider_id = ?1 AND model_id = ?2")
        .bind(&m.provider_id)
        .bind(&m.model_id)
        .fetch_one(db)
        .await
}

struct UpsertModel {
    model_id: String,
    display_name: String,
    context_window: i64,
    max_output_tokens: i64,
    supports_vision: bool,
    supports_tools: bool,
    supports_thinking: bool,
    pinned: bool,
    /// Applied on insert only. A re-import never re-hides what you unhid.
    hidden: bool,
    source: String,
}

/// Insert or refresh one model row. A re-import updates the provider's facts
/// but never touches the user's `hidden` / `pinned` choices.
async fn upsert_model(db: &Db, provider_id: &str, m: &UpsertModel) -> sqlx::Result<()> {
    sqlx::query(
        "INSERT INTO models (id, provider_id, model_id, display_name, context_window,
                             max_output_tokens, supports_vision, supports_tools,
                             supports_thinking, hidden, pinned, source, last_seen_at,
                             sort_order, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13,
                 (SELECT coalesce(max(sort_order), 0) + 1 FROM models), ?13)
         ON CONFLICT (provider_id, model_id) DO UPDATE SET
             display_name      = excluded.display_name,
             context_window    = excluded.context_window,
             max_output_tokens = excluded.max_output_tokens,
             supports_vision   = excluded.supports_vision,
             supports_tools    = excluded.supports_tools,
             supports_thinking = excluded.supports_thinking,
             last_seen_at      = excluded.last_seen_at,
             pinned            = models.pinned OR excluded.pinned",
    )
    .bind(new_id())
    .bind(provider_id)
    .bind(&m.model_id)
    .bind(&m.display_name)
    .bind(m.context_window)
    .bind(m.max_output_tokens)
    .bind(m.supports_vision)
    .bind(m.supports_tools)
    .bind(m.supports_thinking)
    .bind(m.hidden)
    .bind(m.pinned)
    .bind(&m.source)
    .bind(now())
    .execute(db)
    .await?;
    Ok(())
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ModelPatch {
    /// The id sent on the wire. Editable because endpoints that list nothing
    /// (DeepSeek's Anthropic-style API among them) are typed in by hand, and a
    /// typo should be fixable without losing the row's role and pin.
    pub model_id: Option<String>,
    pub display_name: Option<String>,
    pub context_window: Option<i64>,
    pub max_output_tokens: Option<i64>,
    pub supports_vision: Option<bool>,
    pub supports_tools: Option<bool>,
    pub supports_thinking: Option<bool>,
    pub hidden: Option<bool>,
    pub pinned: Option<bool>,
    pub sort_order: Option<i64>,
}

pub async fn update_model(
    db: &Db,
    owner: &str,
    id: &str,
    p: ModelPatch,
) -> sqlx::Result<Option<Model>> {
    sqlx::query(
        "UPDATE models
            SET model_id          = coalesce(?12, model_id),
                display_name      = coalesce(?2, display_name),
                context_window    = coalesce(?3, context_window),
                max_output_tokens = coalesce(?4, max_output_tokens),
                supports_vision   = coalesce(?5, supports_vision),
                supports_tools    = coalesce(?6, supports_tools),
                supports_thinking = coalesce(?7, supports_thinking),
                hidden            = coalesce(?8, hidden),
                pinned            = coalesce(?9, pinned),
                sort_order        = coalesce(?10, sort_order)
          WHERE id = ?1
            AND provider_id IN (SELECT id FROM providers WHERE owner_id = ?11)",
    )
    .bind(id)
    .bind(p.display_name)
    .bind(p.context_window)
    .bind(p.max_output_tokens)
    .bind(p.supports_vision)
    .bind(p.supports_tools)
    .bind(p.supports_thinking)
    .bind(p.hidden)
    .bind(p.pinned)
    .bind(p.sort_order)
    .bind(owner)
    .bind(p.model_id)
    .execute(db)
    .await?;
    get_model(db, owner, id).await
}

pub async fn delete_model(db: &Db, owner: &str, id: &str) -> sqlx::Result<()> {
    sqlx::query(
        "DELETE FROM models
          WHERE id = ?1 AND provider_id IN (SELECT id FROM providers WHERE owner_id = ?2)",
    )
    .bind(id)
    .bind(owner)
    .execute(db)
    .await?;
    // A deleted model must not stay assigned to a role.
    sqlx::query(
        "DELETE FROM settings
          WHERE owner_id = ?2 AND value = ?1
            AND key IN ('researcher_model', 'analyzer_model')",
    )
    .bind(id)
    .bind(owner)
    .execute(db)
    .await?;
    Ok(())
}

/// Pull the provider's catalogue into the models table. Existing rows keep the
/// user's hidden/pinned choices; ids the provider no longer reports are left in
/// place with a stale `last_seen_at` rather than deleted, since a key without
/// list access should never silently wipe a working configuration.
pub struct ImportReport {
    pub total: usize,
    pub added: usize,
    /// Non-chat models (embeddings, rerankers) imported hidden.
    pub non_chat: usize,
}

pub async fn import_models(
    db: &Db,
    http: &reqwest::Client,
    provider: &Provider,
) -> Result<ImportReport, LlmError> {
    let remote = fetch_remote_models(http, provider).await?;

    let before: Vec<String> = sqlx::query("SELECT model_id FROM models WHERE provider_id = ?1")
        .bind(&provider.id)
        .fetch_all(db)
        .await
        .map_err(|e| LlmError::Request(e.to_string()))?
        .into_iter()
        .map(|r| r.get::<String, _>("model_id"))
        .collect();

    let mut added = 0;
    for m in &remote {
        if !before.contains(&m.model_id) {
            added += 1;
        }
        let caps = Capabilities::guess(&m.model_id);
        upsert_model(
            db,
            &provider.id,
            &UpsertModel {
                model_id: m.model_id.clone(),
                display_name: m.display_name.clone(),
                context_window: m.context_window,
                max_output_tokens: caps.max_output_tokens.max(1024),
                supports_vision: m.supports_vision,
                supports_tools: m.supports_tools,
                supports_thinking: m.supports_thinking,
                pinned: false,
                hidden: !m.is_chat,
                source: "remote".into(),
            },
        )
        .await
        .map_err(|e| LlmError::Request(e.to_string()))?;
    }

    let non_chat = remote.iter().filter(|m| !m.is_chat).count();
    Ok(ImportReport { total: remote.len(), added, non_chat })
}

// --------------------------------------------------------------- settings

pub async fn get_setting(db: &Db, owner: &str, key: &str) -> sqlx::Result<Option<String>> {
    Ok(sqlx::query("SELECT value FROM settings WHERE owner_id = ?1 AND key = ?2")
        .bind(owner)
        .bind(key)
        .fetch_optional(db)
        .await?
        .map(|r| r.get::<String, _>("value")))
}

pub async fn set_setting(db: &Db, owner: &str, key: &str, value: &str) -> sqlx::Result<()> {
    sqlx::query(
        "INSERT INTO settings (owner_id, key, value, updated_at) VALUES (?1, ?2, ?3, ?4)
         ON CONFLICT (owner_id, key) DO UPDATE SET
             value = excluded.value, updated_at = excluded.updated_at",
    )
    .bind(owner)
    .bind(key)
    .bind(value)
    .bind(now())
    .execute(db)
    .await?;
    Ok(())
}

pub async fn effort(db: &Db, owner: &str) -> sqlx::Result<Effort> {
    Ok(get_setting(db, owner, "thinking_effort")
        .await?
        .and_then(|v| Effort::parse(&v))
        .unwrap_or(Effort::Off))
}

/// The model assigned to a role. Falls back to a pinned model, then to any
/// visible one, so a fresh install with one provider just works.
pub async fn resolve_role(db: &Db, owner: &str, role: &str) -> sqlx::Result<Option<Resolved>> {
    let assigned = get_setting(db, owner, &format!("{role}_model")).await?;

    let model = match assigned {
        Some(id) => match get_model(db, owner, &id).await? {
            Some(m) => Some(m),
            None => fallback_model(db, owner, role).await?,
        },
        None => fallback_model(db, owner, role).await?,
    };

    let Some(model) = model else { return Ok(None) };
    let Some(provider) = get_provider(db, owner, &model.provider_id).await? else {
        return Ok(None);
    };
    Ok(Some(Resolved { model, provider }))
}

/// The analyzer needs to be able to look at images; the researcher does not.
/// Pinned models win, then whatever is visible.
async fn fallback_model(db: &Db, owner: &str, role: &str) -> sqlx::Result<Option<Model>> {
    const ANY: &str = "SELECT m.* FROM models m
           JOIN providers p ON p.id = m.provider_id
          WHERE p.owner_id = ?1 AND m.hidden = 0 AND m.context_window > 0
          ORDER BY m.pinned DESC, m.sort_order
          LIMIT 1";
    const VISION: &str = "SELECT m.* FROM models m
           JOIN providers p ON p.id = m.provider_id
          WHERE p.owner_id = ?1 AND m.hidden = 0 AND m.supports_vision = 1
          ORDER BY m.pinned DESC, m.sort_order
          LIMIT 1";

    let sql = if role == "analyzer" { VISION } else { ANY };
    sqlx::query_as::<_, Model>(sql).bind(owner).fetch_optional(db).await
}

pub async fn resolve_model_id(db: &Db, owner: &str, id: &str) -> sqlx::Result<Option<Resolved>> {
    let Some(model) = get_model(db, owner, id).await? else { return Ok(None) };
    let Some(provider) = get_provider(db, owner, &model.provider_id).await? else {
        return Ok(None);
    };
    Ok(Some(Resolved { model, provider }))
}

// -------------------------------------------------------------- catalogue

/// Presets behind the "Add provider" menu. `api_style` decides the wire format;
/// anything not Anthropic speaks OpenAI's.
#[derive(Debug, Clone, Serialize)]
pub struct Preset {
    pub key: &'static str,
    pub name: &'static str,
    pub api_style: &'static str,
    pub base_url: &'static str,
    pub needs_key: bool,
}

pub const PRESETS: &[Preset] = &[
    Preset { key: "anthropic", name: "Anthropic", api_style: "anthropic",
             base_url: "https://api.anthropic.com", needs_key: true },
    Preset { key: "openai", name: "OpenAI", api_style: "openai",
             base_url: "https://api.openai.com/v1", needs_key: true },
    Preset { key: "azure", name: "Azure", api_style: "openai",
             base_url: "https://YOUR-RESOURCE.openai.azure.com/openai/v1", needs_key: true },
    Preset { key: "google", name: "Google", api_style: "openai",
             base_url: "https://generativelanguage.googleapis.com/v1beta/openai",
             needs_key: true },
    Preset { key: "openrouter", name: "OpenRouter", api_style: "openai",
             base_url: "https://openrouter.ai/api/v1", needs_key: true },
    Preset { key: "xai", name: "xAI", api_style: "openai",
             base_url: "https://api.x.ai/v1", needs_key: true },
    Preset { key: "ollama", name: "Ollama", api_style: "openai",
             base_url: "http://localhost:11434/v1", needs_key: false },
    Preset { key: "custom", name: "Custom Endpoint", api_style: "openai",
             base_url: "", needs_key: false },
];

/// Defaults for an imported model, from its id. Wrong guesses are editable in
/// the Configure panel; they only affect the context meter and role gating.
pub struct Capabilities {
    pub context_window: i64,
    pub max_output_tokens: i64,
    pub vision: bool,
    pub tools: bool,
    pub thinking: bool,
}

/// Substrings that mark a model as taking images. Used only when the endpoint
/// says nothing itself, which is the case for bare OpenAI-style listings
/// (NVIDIA, most proxies). It will be wrong sometimes; the Configure panel lets
/// you toggle the flag by hand.
const VISION_MARKERS: &[&str] = &[
    "vision", "-vl", "vl-", "vlm", "multimodal", "omni", "llava", "pixtral",
    "internvl", "minicpm-v", "idefics", "moondream", "smolvlm", "kosmos",
    "fuyu", "neva", "vila", "deplot", "paligemma", "nemotron", "cosmos",
    "gemma-3", "gemma-4", "mistral-small-3", "ministral-3", "phi-3-vision",
    "phi-4-multimodal", "llama-3.2-11b", "llama-3.2-90b", "llama-guard-4",
];

/// Ids that are not chat models at all.
const NON_CHAT_MARKERS: &[&str] = &[
    "embed", "rerank", "whisper", "tts", "dall-e", "moderation", "clip",
    "stable-diffusion", "guard", "safety", "reward", "detector",
];

fn is_chat_model(model_id: &str) -> bool {
    let id = model_id.to_ascii_lowercase();
    !NON_CHAT_MARKERS.iter().any(|m| id.contains(m))
}

impl Capabilities {
    pub fn guess(model_id: &str) -> Self {
        let id = model_id.to_ascii_lowercase();
        let has = |p: &str| id.contains(p);
        let vision_marker = VISION_MARKERS.iter().any(|m| id.contains(m));

        if has("claude") {
            return Capabilities {
                context_window: 200_000,
                max_output_tokens: 8192,
                vision: true,
                tools: true,
                thinking: !has("haiku-3") && !has("claude-2"),
            };
        }
        if has("gpt-5") || has("gpt-4.1") || has("o3") || has("o4") {
            return Capabilities {
                context_window: 200_000,
                max_output_tokens: 16384,
                vision: true,
                tools: true,
                thinking: has("o3") || has("o4") || has("gpt-5"),
            };
        }
        if has("gpt-4o") || has("gpt-4-turbo") {
            return Capabilities {
                context_window: 128_000,
                max_output_tokens: 8192,
                vision: true,
                tools: true,
                thinking: false,
            };
        }
        if has("gemini") {
            return Capabilities {
                context_window: 1_000_000,
                max_output_tokens: 8192,
                vision: true,
                tools: true,
                thinking: has("2.5") || has("3"),
            };
        }
        if has("grok") {
            return Capabilities {
                context_window: 131_072,
                max_output_tokens: 8192,
                vision: vision_marker || has("grok-4"),
                tools: true,
                thinking: true,
            };
        }
        if !is_chat_model(&id) {
            return Capabilities {
                context_window: 0,
                max_output_tokens: 0,
                vision: false,
                tools: false,
                thinking: false,
            };
        }
        Capabilities {
            context_window: 32_768,
            max_output_tokens: 4096,
            vision: vision_marker,
            tools: true,
            thinking: false,
        }
    }
}

// ------------------------------------------- listing a provider's models

#[derive(Debug, Clone, Serialize)]
pub struct RemoteModel {
    pub model_id: String,
    pub display_name: String,
    pub context_window: i64,
    pub supports_vision: bool,
    pub supports_tools: bool,
    pub supports_thinking: bool,
    /// False for embedding, reranking and other non-chat models. They are still
    /// imported, but hidden, so they do not clutter the picker.
    pub is_chat: bool,
}

/// Ask the provider what it can serve.
///
/// The OpenAI-compatible `/models` listing is the lowest common denominator and
/// often carries nothing but ids (LM Studio and NVIDIA both do this). When it
/// tells us nothing about capabilities, try the server's own richer listing
/// before falling back to guessing from the id.
pub async fn fetch_remote_models(
    http: &reqwest::Client,
    provider: &Provider,
) -> Result<Vec<RemoteModel>, LlmError> {
    let entries = match provider.api_style.as_str() {
        "anthropic" => {
            let url = format!("{}/v1/models?limit=1000", provider.base_url);
            let req = http
                .get(url)
                .header("x-api-key", &provider.api_key)
                .header("anthropic-version", "2023-06-01");
            data_array(req).await?
        }
        _ => {
            let req = get_with_key(http, &format!("{}/models", provider.base_url), provider);
            let compat = data_array(req).await?;

            if compat.iter().any(describes_capabilities) {
                compat
            } else {
                // LM Studio serves this alongside the compat API, with the real
                // context length and whether the model takes images.
                let native = format!("{}/api/v0/models", origin_of(&provider.base_url));
                match data_array(get_with_key(http, &native, provider)).await {
                    Ok(rich) if rich.iter().any(describes_capabilities) => rich,
                    _ => compat,
                }
            }
        }
    };

    let mut out: Vec<RemoteModel> = entries.iter().filter_map(parse_remote).collect();
    out.sort_by(|a, b| a.model_id.cmp(&b.model_id));
    Ok(out)
}

fn get_with_key(
    http: &reqwest::Client,
    url: &str,
    provider: &Provider,
) -> reqwest::RequestBuilder {
    let req = http.get(url);
    if provider.api_key.is_empty() { req } else { req.bearer_auth(&provider.api_key) }
}

async fn data_array(req: reqwest::RequestBuilder) -> Result<Vec<serde_json::Value>, LlmError> {
    let resp = req.send().await?;
    let status = resp.status();
    let body = resp.text().await?;
    if !status.is_success() {
        return Err(LlmError::Api { status: status.as_u16(), body, retry_after: None });
    }
    let v: serde_json::Value = serde_json::from_str(&body)?;
    Ok(v["data"].as_array().cloned().unwrap_or_default())
}

/// Strip the API path off a base URL so a server's non-OpenAI routes can be
/// reached: `http://host:1234/v1` becomes `http://host:1234`.
fn origin_of(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    match trimmed.rfind("/v1") {
        Some(i) if trimmed.len() == i + 3 => trimmed[..i].to_string(),
        _ => trimmed.to_string(),
    }
}

/// Whether a listing entry says anything about the model beyond its name.
fn describes_capabilities(m: &serde_json::Value) -> bool {
    !m["type"].is_null()
        || !m["capabilities"].is_null()
        || !m["max_context_length"].is_null()
        || !m["context_length"].is_null()
        || !m["architecture"].is_null()
}

fn strings_at(m: &serde_json::Value, key: &str) -> Vec<String> {
    m[key]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str()).map(str::to_lowercase).collect())
        .unwrap_or_default()
}

/// Read one listing entry, preferring what the endpoint states over what the id
/// suggests. Known shapes: LM Studio (`type`, `capabilities`,
/// `max_context_length`), OpenRouter (`architecture.input_modalities`,
/// `supported_parameters`, `context_length`), and bare OpenAI-style listings
/// that carry nothing at all.
fn parse_remote(m: &serde_json::Value) -> Option<RemoteModel> {
    let id = m["id"].as_str()?.to_string();
    let guess = Capabilities::guess(&id);

    let kind = m["type"].as_str().unwrap_or_default().to_lowercase();
    let capabilities = strings_at(m, "capabilities");
    let parameters = strings_at(m, "supported_parameters");
    let modalities = m["architecture"]["input_modalities"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str()).map(str::to_lowercase).collect::<Vec<_>>())
        .unwrap_or_default();

    let vision = kind == "vlm"
        || modalities.iter().any(|x| x == "image")
        || capabilities.iter().any(|c| c.contains("vision") || c.contains("image"))
        || (kind.is_empty() && modalities.is_empty() && guess.vision);

    // An explicit capability list is exhaustive: absence means "no".
    let tools = if !capabilities.is_empty() {
        capabilities.iter().any(|c| c.contains("tool") || c.contains("function"))
    } else if !parameters.is_empty() {
        parameters.iter().any(|p| p == "tools" || p == "tool_choice")
    } else {
        guess.tools
    };

    let thinking = capabilities.iter().any(|c| c.contains("reason") || c.contains("think"))
        || parameters.iter().any(|p| p.contains("reasoning") || p.contains("thinking"))
        || (capabilities.is_empty() && parameters.is_empty() && guess.thinking);

    let context_window = m["max_context_length"]
        .as_i64()
        .or_else(|| m["context_length"].as_i64())
        .or_else(|| m["top_provider"]["context_length"].as_i64())
        .filter(|n| *n > 0)
        .unwrap_or(guess.context_window);

    let is_chat = match kind.as_str() {
        "" => is_chat_model(&id),
        "llm" | "vlm" | "chat" => true,
        _ => false,
    };

    Some(RemoteModel {
        display_name: m["display_name"]
            .as_str()
            .or_else(|| m["name"].as_str())
            .unwrap_or(&id)
            .to_string(),
        model_id: id,
        context_window,
        supports_vision: vision,
        supports_tools: tools,
        supports_thinking: thinking,
        is_chat,
    })
}

// -------------------------------------------------------------- bootstrap

/// Create the local user, and seed providers from the environment the first
/// time the server runs against an empty database so an existing `.env` keeps
/// working. Once a provider exists, the environment is ignored entirely.
pub async fn bootstrap(db: &Db, cfg: &Config) -> anyhow::Result<String> {
    let user = ensure_user(db, LOCAL_USER, "Local").await?;
    // Every upload lands in a vault, so a fresh database gets one, and anything
    // the vaults migration could not place goes into it.
    let vault = crate::db::active_vault(db, &user.id).await?;
    crate::db::adopt_orphans(db, &user.id, &vault.id).await?;

    if !list_providers(db, &user.id).await?.is_empty() {
        return Ok(user.id);
    }

    let mut seeded: Vec<(crate::config::ProviderKind, Provider)> = Vec::new();
    if !cfg.anthropic_api_key.is_empty() {
        seeded.push((
            crate::config::ProviderKind::Anthropic,
            create_provider(
                db,
                &user.id,
                "Anthropic",
                "anthropic",
                &cfg.anthropic_base_url,
                &cfg.anthropic_api_key,
            )
            .await?,
        ));
    }
    if !cfg.openai_api_key.is_empty() {
        seeded.push((
            crate::config::ProviderKind::OpenAi,
            create_provider(
                db,
                &user.id,
                "OpenAI",
                "openai",
                &cfg.openai_base_url,
                &cfg.openai_api_key,
            )
            .await?,
        ));
    }
    if seeded.is_empty() {
        return Ok(user.id);
    }

    // The catalogue can only be imported once the server can reach the network,
    // so seed the env-named models directly and let the user refresh later.
    for (role, kind, model_id) in [
        ("researcher", cfg.researcher_provider, &cfg.researcher_model),
        ("analyzer", cfg.analyzer_provider, &cfg.analyzer_model),
    ] {
        let Some((_, provider)) = seeded.iter().find(|(k, _)| *k == kind) else { continue };
        let model = add_model(
            db,
            NewModel {
                provider_id: provider.id.clone(),
                model_id: model_id.clone(),
                display_name: None,
                context_window: None,
                max_output_tokens: None,
                supports_vision: None,
                supports_tools: None,
                supports_thinking: None,
                pinned: Some(true),
            },
            "manual",
        )
        .await?;
        set_setting(db, &user.id, &format!("{role}_model"), &model.id).await?;
    }

    tracing::info!("seeded provider configuration from environment");
    Ok(user.id)
}
