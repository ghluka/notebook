-- Folders are organisation only: a source's rendition, chunks and retrieval do
-- not care where it sits. Nesting is by parent_id, and deleting a folder takes
-- its subfolders with it while leaving the sources themselves at the root.

CREATE TABLE folders (
    id         TEXT PRIMARY KEY,
    owner_id   TEXT NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    parent_id  TEXT REFERENCES folders (id) ON DELETE CASCADE,
    name       TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX folders_owner_idx ON folders (owner_id, parent_id);

-- Sources predate users; everything already in the table belongs to the local
-- user that `bootstrap` seeds.
ALTER TABLE sources ADD COLUMN owner_id TEXT NOT NULL DEFAULT 'local';
ALTER TABLE sources ADD COLUMN folder_id TEXT REFERENCES folders (id) ON DELETE SET NULL;

CREATE INDEX sources_owner_idx ON sources (owner_id, folder_id);
