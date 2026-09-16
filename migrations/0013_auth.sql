-- Password gate: one local user protected by a password, with sessions.
--
-- The app is single user (see 0002_models.sql). This adds the password hash to
-- that user and a table of login sessions. Proof of work challenges and rate
-- limit buckets live in memory in the server, since they expire in minutes.

ALTER TABLE users ADD COLUMN password_hash TEXT NOT NULL DEFAULT '';

CREATE TABLE sessions (
    id           TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    token_hash   TEXT NOT NULL UNIQUE,
    created_at   TEXT NOT NULL,
    expires_at   TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    ip           TEXT NOT NULL DEFAULT '',
    user_agent   TEXT NOT NULL DEFAULT ''
);
CREATE INDEX sessions_token_idx ON sessions (token_hash);
CREATE INDEX sessions_user_idx ON sessions (user_id);
