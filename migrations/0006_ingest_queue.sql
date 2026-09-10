-- Storing a file and analyzing it are now separate steps. Uploads land as
-- `pending` and a background worker takes them from there, so a refresh or a
-- closed tab no longer abandons the work.

-- The failure as JSON (the same payload the API returns), so the client can
-- tell a rate limit from an unreadable file and offer to switch models.
ALTER TABLE sources ADD COLUMN error_detail TEXT;
