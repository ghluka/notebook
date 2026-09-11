-- The files a question was asked about, as JSON: each one's id, title and
-- kind, kept as they were when asked so a file deleted later still has a name.
-- They show under the question, and a retry or an edit asks with them again.
ALTER TABLE messages ADD COLUMN attachments TEXT;
