-- Sources: one row per uploaded file / URL. Original bytes live on disk.
CREATE TABLE sources (
    id                TEXT PRIMARY KEY,
    title             TEXT NOT NULL,
    original_filename TEXT,
    kind              TEXT NOT NULL,              -- pdf | image | video | audio | text | url
    media_type        TEXT NOT NULL,
    byte_size         INTEGER NOT NULL,
    sha256            TEXT NOT NULL,
    storage_path      TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'pending', -- pending | analyzing | ready | failed
    error             TEXT,
    metadata          TEXT NOT NULL DEFAULT '{}', -- analyzer-extracted structured facts (json)
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);
CREATE INDEX sources_sha256_idx ON sources (sha256);
CREATE INDEX sources_status_idx ON sources (status);

-- The analyzer's rendition of a source: markdown with embedded LaTeX.
CREATE TABLE documents (
    id             TEXT PRIMARY KEY,
    source_id      TEXT NOT NULL REFERENCES sources (id) ON DELETE CASCADE,
    markdown       TEXT NOT NULL,
    summary        TEXT,
    analyzer_model TEXT,
    created_at     TEXT NOT NULL
);
CREATE UNIQUE INDEX documents_source_idx ON documents (source_id);

-- Retrieval surface. `locator` is how a human finds this in the ORIGINAL:
-- "p. 12", "00:14:03", "slide 4".
CREATE TABLE chunks (
    id             TEXT PRIMARY KEY,
    source_id      TEXT NOT NULL REFERENCES sources (id) ON DELETE CASCADE,
    document_id    TEXT NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    ordinal        INTEGER NOT NULL,
    heading        TEXT,
    locator        TEXT,
    content        TEXT NOT NULL,
    token_estimate INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX chunks_source_idx ON chunks (source_id, ordinal);

CREATE VIRTUAL TABLE chunks_fts USING fts5 (
    content,
    heading,
    chunk_id  UNINDEXED,
    source_id UNINDEXED,
    tokenize = 'porter unicode61'
);

CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts (content, heading, chunk_id, source_id)
    VALUES (new.content, coalesce(new.heading, ''), new.id, new.source_id);
END;

CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
    DELETE FROM chunks_fts WHERE chunk_id = old.id;
END;

CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
    DELETE FROM chunks_fts WHERE chunk_id = old.id;
    INSERT INTO chunks_fts (content, heading, chunk_id, source_id)
    VALUES (new.content, coalesce(new.heading, ''), new.id, new.source_id);
END;

CREATE TABLE conversations (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL DEFAULT 'Untitled',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    role            TEXT NOT NULL,              -- system | user | assistant | tool
    content         TEXT NOT NULL,
    tool_calls      TEXT,                       -- json array, assistant turns only
    citations       TEXT,                       -- json array of {source_id, chunk_id, locator}
    created_at      TEXT NOT NULL
);
CREATE INDEX messages_conversation_idx ON messages (conversation_id, created_at);
