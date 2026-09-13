-- Vaults are separate libraries: each holds its own sources, folders and
-- conversations, and the researcher only ever searches the one a conversation
-- belongs to. A user always has at least one, and one is open at a time (the
-- `vault` setting).

CREATE TABLE vaults (
    id         TEXT PRIMARY KEY,
    owner_id   TEXT NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    name       TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE UNIQUE INDEX vaults_owner_name_idx ON vaults (owner_id, name COLLATE NOCASE);

-- Everything that already exists goes into one vault per user, so upgrading
-- changes nothing until a second vault is made.
INSERT INTO vaults (id, owner_id, name, created_at)
SELECT lower(hex(randomblob(16))), id, 'Library', strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
  FROM users;

ALTER TABLE sources ADD COLUMN vault_id TEXT REFERENCES vaults (id) ON DELETE CASCADE;
ALTER TABLE folders ADD COLUMN vault_id TEXT REFERENCES vaults (id) ON DELETE CASCADE;
ALTER TABLE conversations ADD COLUMN vault_id TEXT REFERENCES vaults (id) ON DELETE CASCADE;

UPDATE sources SET vault_id = (SELECT v.id FROM vaults v WHERE v.owner_id = sources.owner_id);
UPDATE folders SET vault_id = (SELECT v.id FROM vaults v WHERE v.owner_id = folders.owner_id);
UPDATE conversations
   SET vault_id = (SELECT v.id FROM vaults v WHERE v.owner_id = conversations.owner_id);

CREATE INDEX sources_vault_idx ON sources (vault_id, folder_id);
CREATE INDEX folders_vault_idx ON folders (vault_id, parent_id);
CREATE INDEX conversations_vault_idx ON conversations (vault_id, updated_at);
