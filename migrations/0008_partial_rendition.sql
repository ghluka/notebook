-- A long PDF is many model calls, and losing all of them to a restart is not
-- acceptable. Finished batches are kept here, so analysis resumes where it
-- stopped instead of starting the book again.
ALTER TABLE sources ADD COLUMN partial TEXT;
