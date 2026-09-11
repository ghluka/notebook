-- The steps a turn took before its answer, in order: each stretch of reasoning
-- with how long it took, and each tool call with what it found. A reopened
-- conversation shows the same trace the live one did.
ALTER TABLE messages ADD COLUMN trace TEXT;
