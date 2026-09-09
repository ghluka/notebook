-- Token usage per assistant turn, so the context meter can be restored when a
-- conversation is reopened instead of resetting to zero.

ALTER TABLE messages ADD COLUMN input_tokens INTEGER;
ALTER TABLE messages ADD COLUMN output_tokens INTEGER;
