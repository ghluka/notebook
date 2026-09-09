-- Model configuration moves out of the environment and into the database, so
-- providers, keys and model visibility can be edited from the UI at runtime.
--
-- Everything here hangs off a user. There is no registration yet: the server
-- seeds a single local user and treats it as the current one. When auth lands,
-- the only change is where `owner_id` comes from; the queries already scope by
-- it.

CREATE TABLE users (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE providers (
    id         TEXT PRIMARY KEY,
    owner_id   TEXT NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    name       TEXT NOT NULL,
    -- Which wire format to speak. Everything OpenAI-compatible (OpenRouter,
    -- xAI, Gemini's compat endpoint, vLLM, Ollama) uses 'openai'.
    api_style  TEXT NOT NULL CHECK (api_style IN ('openai', 'anthropic')),
    base_url   TEXT NOT NULL,
    -- The user's key for this endpoint. Never leaves the server: the API
    -- returns a masked hint only.
    api_key    TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX providers_owner_idx ON providers (owner_id);

-- One row per model the provider reports, imported wholesale when a provider is
-- added or refreshed. `hidden` is the user's "I never want to see this one";
-- `pinned` is "put it in the prompt-bar picker".
CREATE TABLE models (
    id                TEXT PRIMARY KEY,
    provider_id       TEXT NOT NULL REFERENCES providers (id) ON DELETE CASCADE,
    -- The id sent on the wire, e.g. 'claude-sonnet-5'.
    model_id          TEXT NOT NULL,
    display_name      TEXT NOT NULL,
    context_window    INTEGER NOT NULL DEFAULT 0,
    max_output_tokens INTEGER NOT NULL DEFAULT 4096,
    supports_vision   INTEGER NOT NULL DEFAULT 0,
    supports_tools    INTEGER NOT NULL DEFAULT 1,
    supports_thinking INTEGER NOT NULL DEFAULT 0,
    -- Credits per million tokens, informational only.
    input_cost        REAL,
    output_cost       REAL,
    hidden            INTEGER NOT NULL DEFAULT 0,
    pinned            INTEGER NOT NULL DEFAULT 0,
    -- 'remote' came from the provider's model list, 'manual' was typed in.
    source            TEXT NOT NULL DEFAULT 'remote',
    -- Last time the provider still reported this id.
    last_seen_at      TEXT,
    sort_order        INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT NOT NULL
);
CREATE UNIQUE INDEX models_provider_model_idx ON models (provider_id, model_id);
CREATE INDEX models_visible_idx ON models (hidden, pinned);

-- Per-user role assignments (researcher_model, analyzer_model) and prompt-bar
-- state (thinking_effort).
CREATE TABLE settings (
    owner_id   TEXT NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (owner_id, key)
);
