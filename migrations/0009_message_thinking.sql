-- What the model reasoned before answering, and for how long, so a reopened
-- conversation shows the same "Thought for N seconds" the live one did. Kept
-- apart from the answer: shown, never sent back to the model, never searched.
ALTER TABLE messages ADD COLUMN thinking TEXT;
ALTER TABLE messages ADD COLUMN thinking_ms INTEGER;
