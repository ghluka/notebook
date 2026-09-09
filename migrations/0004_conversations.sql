-- Conversations belong to a user and survive a reload, and a conversation can
-- be compacted: the transcript so far is replaced, for context purposes, by a
-- summary. Compacted messages stay in the table so the history still reads in
-- full; they are simply skipped when the next prompt is assembled.

ALTER TABLE conversations ADD COLUMN owner_id TEXT NOT NULL DEFAULT 'local';
ALTER TABLE conversations ADD COLUMN summary TEXT;
ALTER TABLE conversations ADD COLUMN summarized_at TEXT;

ALTER TABLE messages ADD COLUMN compacted INTEGER NOT NULL DEFAULT 0;
-- Which model produced an assistant turn, so restored history can say so.
ALTER TABLE messages ADD COLUMN model TEXT;

CREATE INDEX conversations_owner_idx ON conversations (owner_id, updated_at);
CREATE INDEX messages_active_idx ON messages (conversation_id, compacted);
