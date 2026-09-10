-- Long PDFs take many model calls, so the explorer shows how far along the
-- analyzer is rather than an anonymous spinner.
ALTER TABLE sources ADD COLUMN progress TEXT;
