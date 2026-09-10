//! Ingestion: original bytes → markdown-latex rendition → chunks.
//!
//! Text and markdown need no model at all. PDFs take one of two paths: when
//! the file carries a usable text layer (FlateDecode and ASCII85 streams are
//! decompressed, image streams are skipped), the whole layer goes to the
//! analyzer model as text for structuring, and any model will do. Scanned
//! PDFs with no text layer go to a vision-capable model as a native document.
//! Images, audio and video always go to the analyzer model directly: images
//! as image parts, audio and video as `audio_url` / `video_url` parts (the
//! OpenAI-style omni syntax). When the endpoint cannot read the file and no
//! text layer exists, ingestion fails with an honest error instead of storing
//! a guess.

pub mod text;
mod render;

use crate::db::{self, Source};
use crate::error::{AppError, AppResult};
use crate::state::AppState;

/// What the analyzer needs to know to pick a strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Text,
    Pdf,
    Image,
    Audio,
    Video,
    /// Constructed by URL ingestion (phase 4).
    #[allow(dead_code)]
    Url,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Text => "text",
            Kind::Pdf => "pdf",
            Kind::Image => "image",
            Kind::Audio => "audio",
            Kind::Video => "video",
            Kind::Url => "url",
        }
    }

    pub fn from_media_type(media_type: &str, filename: Option<&str>) -> Kind {
        // The extension wins when present, since uploads often arrive as
        // `application/octet-stream`.
        if let Some(name) = filename {
            let ext = name.rsplit('.').next().unwrap_or_default().to_ascii_lowercase();
            match ext.as_str() {
                "pdf" => return Kind::Pdf,
                "png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp" | "svg" | "heic"
                | "heif" | "avif" | "tif" | "tiff" => return Kind::Image,
                "mp3" | "wav" | "ogg" | "oga" | "flac" | "m4a" | "aac" | "opus"
                | "weba" | "mid" | "midi" => return Kind::Audio,
                "mp4" | "webm" | "mov" | "mkv" | "avi" | "m4v" | "ogv" => {
                    return Kind::Video;
                }
                _ => {}
            }
            let text_ext = ["md", "markdown", "txt", "rst", "org", "tex", "csv", "json",
                            "toml", "yaml", "yml", "rs", "py", "js", "ts", "go", "c",
                            "h", "cpp", "java"];
            if text_ext.contains(&ext.as_str()) {
                return Kind::Text;
            }
        }
        match media_type {
            m if m.starts_with("text/") => Kind::Text,
            "application/pdf" => Kind::Pdf,
            m if m.starts_with("image/") => Kind::Image,
            m if m.starts_with("audio/") => Kind::Audio,
            m if m.starts_with("video/") => Kind::Video,
            "application/json" | "application/x-yaml" | "application/toml" => Kind::Text,
            _ => Kind::Text,
        }
    }
}

/// One provider call with the retry schedule applied.
///
/// A rate limit that outlives the schedule becomes `AppError::RateLimited`,
/// carrying the model it happened on, so the client can offer to switch models
/// instead of being told the file is unreadable. Everything else passes through
/// unchanged.
async fn analyzer_chat(
    state: &AppState,
    resolved: &crate::models::Resolved,
    req: &crate::llm::ChatRequest,
    context: &str,
) -> AppResult<String> {
    let client = resolved.client(&state.http)?;
    match crate::llm::chat_with_retry(client.as_ref(), req).await {
        Ok(response) => Ok(response.text),
        Err(e) if e.is_rate_limited() => Err(AppError::from_rate_limit(
            e,
            &resolved.model.id,
            &resolved.model.display_name,
            context,
        )),
        Err(e) => Err(AppError::Llm(e)),
    }
}

/// Which model does the analyzing. `override_id` comes from a client retrying
/// after a rate limit with a different model picked by hand.
async fn analyzer_model(
    state: &AppState,
    override_id: Option<&str>,
) -> AppResult<Option<crate::models::Resolved>> {
    match override_id {
        Some(id) => Ok(Some(
            crate::models::resolve_model_id(&state.db, &state.user_id, id)
                .await?
                .ok_or_else(|| AppError::NotFound(format!("model {id}")))?,
        )),
        None => Ok(crate::models::resolve_role(&state.db, &state.user_id, "analyzer").await?),
    }
}

/// Analyze one source and persist its rendition. Sets the source status on the
/// way in and out, so a failure is visible in `GET /api/sources`.
pub async fn ingest(state: &AppState, source: &Source) -> AppResult<()> {
    ingest_with(state, source, None).await
}

/// Ingest with an explicit analyzer model, which is how a retry after a rate
/// limit runs on a different one.
pub async fn ingest_with(
    state: &AppState,
    source: &Source,
    model_override: Option<&str>,
) -> AppResult<()> {
    // A hand picked model means a deliberate fresh start, so nothing carries
    // over from the run that failed.
    if model_override.is_some() {
        db::clear_partial(&state.db, &source.id).await?;
    }
    db::set_source_status(&state.db, &source.id, "analyzing", None).await?;

    let result = run(state, source, model_override).await;

    match &result {
        Ok(()) => db::set_source_status(&state.db, &source.id, "ready", None).await?,
        Err(e) => {
            db::set_source_result(
                &state.db,
                &source.id,
                "failed",
                Some(&e.to_string()),
                Some(&e.payload().to_string()),
            )
            .await?
        }
    }
    result
}

/// Analyze these sources in the background, one at a time.
///
/// Uploading and analyzing are separate on purpose: a local file is stored in
/// milliseconds, while looking at it can take minutes, and tying the two
/// together meant a page refresh abandoned everything still queued. The work
/// now outlives the request that started it.
pub fn spawn_analysis(state: &AppState, sources: Vec<Source>) {
    if sources.is_empty() {
        return;
    }
    let state = state.clone();
    tokio::spawn(async move {
        for source in sources {
            // One permit, so a dozen files queue up instead of stampeding the
            // provider. Held across the whole of one source's analysis.
            let _slot = match state.analysis.clone().acquire_owned().await {
                Ok(slot) => slot,
                Err(_) => return,
            };
            // The row may have been deleted while it sat in the queue.
            match db::get_source(&state.db, &source.id).await {
                Ok(Some(current)) if current.status != "ready" => {
                    if let Err(e) = ingest(&state, &current).await {
                        tracing::warn!(source = %current.id, error = %e, "analysis failed");
                    }
                }
                Ok(_) => {}
                Err(e) => tracing::warn!(error = %e, "could not reload a queued source"),
            }
        }
    });
}

/// Pick up anything left `pending` or `analyzing` by a previous run.
pub async fn resume_pending(state: &AppState) {
    match db::pending_sources(&state.db).await {
        Ok(sources) if !sources.is_empty() => {
            tracing::info!(count = sources.len(), "resuming interrupted analysis");
            spawn_analysis(state, sources);
        }
        Ok(_) => {}
        Err(e) => tracing::warn!(error = %e, "could not list pending sources"),
    }
}

async fn run(
    state: &AppState,
    source: &Source,
    model_override: Option<&str>,
) -> AppResult<()> {
    let kind = Kind::from_media_type(&source.media_type, source.original_filename.as_deref());

    let (markdown, model) = match kind {
        Kind::Text => {
            let bytes = state.storage.read(&source.storage_path).await?;
            (String::from_utf8_lossy(&bytes).into_owned(), None::<String>)
        }
        Kind::Image | Kind::Audio | Kind::Video => {
            analyze_media(state, source, kind, model_override).await?
        }
        Kind::Pdf => analyze_pdf(state, source, model_override).await?,
        // Constructed by URL ingestion (phase 4).
        Kind::Url => {
            return Err(AppError::Unsupported(
                "URL ingestion is not implemented yet; upload the file directly".into(),
            ));
        }
    };

    let chunks = text::chunk_markdown(&markdown);
    let summary = text::naive_summary(&markdown);
    db::replace_document(
        &state.db,
        &source.id,
        &markdown,
        summary.as_deref(),
        model.as_deref(),
        &chunks,
    )
    .await?;

    tracing::info!(source = %source.id, chunks = chunks.len(), "ingested");
    Ok(())
}

/// Soft per-kind caps, so a huge upload fails with guidance instead of
/// blowing up the base64 body or the model's context. The hard cap stays in
/// `MAX_UPLOAD_BYTES`.
const IMAGE_MAX_BYTES: usize = 20 * 1024 * 1024;
const PDF_MAX_BYTES: usize = 64 * 1024 * 1024;
const AUDIO_MAX_BYTES: usize = 100 * 1024 * 1024;
const VIDEO_MAX_BYTES: usize = 200 * 1024 * 1024;

/// How much extracted PDF text counts as a real text layer.
const PDF_TEXT_SUBSTANTIAL: usize = 500;
/// Cap on extracted text sent to the model for structuring.
const PDF_TEXT_INPUT_MAX: usize = 100_000;

fn prompt_for(kind: Kind) -> &'static str {
    match kind {
        Kind::Text | Kind::Url => "",
        Kind::Image => {
            "Describe this image in detail for a searchable research notebook. \
             Cover: the main subject and every legible piece of text (OCR, quoted \
             exactly, never invented), each figure, table, diagram or chart with its \
             numbers and axis units, and any mathematics transcribed as LaTeX. \
             Write markdown with `#` headings. End with a one paragraph summary."
        }
        Kind::Pdf => {
            "Transcribe this document into markdown for a searchable research \
             notebook, page by page. Start each page with `## p. N`, then \
             transcribe its content faithfully in order: headings, paragraphs, \
             lists, tables with their numbers, and mathematics as LaTeX. This is \
             transcription, not summary: keep the document's own words, quote \
             exactly, and never condense it into notes about some other work. \
             The file name tells you nothing about the contents: never use it as \
             a source of facts, and never substitute a different document with a \
             similar title. If a page cannot be read, write its `## p. N` header \
             followed by one line saying it is unreadable, and continue. If no \
             page can be read, answer with that one line alone."
        }
        Kind::Audio => {
            "Transcribe this audio file for a searchable research notebook. \
             Write the full transcript with `HH:MM:SS` timestamps at least every \
             30 seconds and whenever the speaker or topic changes, labelled \
             `Speaker 1`, `Speaker 2`, and so on when you can tell voices apart. \
             Transcribe words exactly; mark uncertain passages `[inaudible]`. \
             End with a one paragraph summary and a bullet list of the key claims."
        }
        Kind::Video => {
            "Analyze this video for a searchable research notebook. Describe what is \
             shown scene by scene with `HH:MM:SS` timestamps, transcribe all speech \
             exactly (mark uncertain passages `[inaudible]`), and note on-screen \
             text, figures and tables with their numbers. End with a one paragraph \
             summary and a bullet list of the key claims."
        }
    }
}

/// Images, audio and video: the analyzer model looks at the original bytes.
/// Audio and video need an omni-style model on an OpenAI-style endpoint.
async fn analyze_media(
    state: &AppState,
    source: &Source,
    kind: Kind,
    model_override: Option<&str>,
) -> AppResult<(String, Option<String>)> {
    use crate::llm::{ChatRequest, ContentPart, Message, Role, part_for_file};
    use crate::models;

    let bytes = state.storage.read(&source.storage_path).await?;
    let limit = match kind {
        Kind::Image => IMAGE_MAX_BYTES,
        Kind::Audio => AUDIO_MAX_BYTES,
        Kind::Video => VIDEO_MAX_BYTES,
        _ => IMAGE_MAX_BYTES,
    };
    if bytes.len() > limit {
        return Err(AppError::Unsupported(format!(
            "this {} file is {} MiB, over the {} MiB the analyzer accepts; \
             compress or trim it and upload again",
            kind.as_str(),
            bytes.len() / (1024 * 1024),
            limit / (1024 * 1024)
        )));
    }

    let resolved = analyzer_model(state, model_override).await?.ok_or_else(|| {
        AppError::Unsupported(
            "no analyzer model configured; add a provider and pin a model in Configure"
                .into(),
        )
    })?;
    if !resolved.model.supports_vision {
        return Err(AppError::Unsupported(format!(
            "the analyzer model {} is not marked vision-capable, and {} files need \
             a multi-modal model; pin one (for example an omni or vlm model) in Configure",
            resolved.model.display_name,
            kind.as_str()
        )));
    }

    let filename = source.original_filename.as_deref().unwrap_or(&source.title);
    let content = vec![
        part_for_file(&source.media_type, &bytes, source.original_filename.as_deref()),
        ContentPart::text(format!(
            "File name: {filename}\n\n{}",
            prompt_for(kind)
        )),
    ];
    let req = ChatRequest::new(&resolved.model.model_id, vec![Message {
        role: Role::User,
        content,
    }])
    .system(
        "You are the analyzer. You turn one source file into a markdown-latex \
         rendition for a research notebook. Be thorough and precise, and only \
         describe what the file actually contains.",
    )
    .max_tokens(resolved.model.max_output_tokens.max(1024) as u32)
    .effort(models::effort(&state.db, &state.user_id).await?);

    let text = analyzer_chat(state, &resolved, &req, &format!(" on this {} file", kind.as_str()))
        .await?;
    if text.trim().is_empty() {
        return Err(AppError::Unsupported(format!(
            "the analyzer model returned no text for this {} file; \
             try a different analyzer model",
            kind.as_str()
        )));
    }
    Ok((text, Some(resolved.model.model_id.clone())))
}

/// PDFs take up to three paths, in order. Page images first: they capture text
/// and figures uniformly and are accepted by every endpoint, including ones
/// that reject native documents. Then the native document, for render failures.
/// Then the text layer alone. Junk extraction is never fed to a model, so a
/// failure stays a visible error instead of becoming a fluent fiction.
///
/// Page images per request. One page per request is faithful but far too slow
/// on local models; four keeps context bounded while staying ordered.
const PAGES_PER_REQUEST: usize = 4;

/// Transcribe a whole PDF from page images, a batch at a time.
///
/// Every page is covered, however long the file. The context limit is a limit
/// per request, not per document, so the document is split across requests and
/// the pieces are joined back into one rendition. Page numbers in the prompt
/// are the file's real ones, so a skipped page cannot shift the rest.
async fn analyze_page_images(
    state: &AppState,
    resolved: &crate::models::Resolved,
    source: &Source,
    filename: &str,
    bytes: &[u8],
    total: usize,
) -> AppResult<(String, Option<String>)> {
    use crate::llm::{ChatRequest, ContentPart, Message, Role};
    use crate::models;

    let effort = models::effort(&state.db, &state.user_id).await?;
    let max_tokens = resolved.model.max_output_tokens.max(4096) as u32;
    let mut sections = Vec::new();
    let mut unreadable = Vec::new();

    // Pick up where a previous run stopped. The saved progress counts pages of
    // this same file, so a resume only makes sense when the totals agree.
    let mut start = 0;
    if let (Some(partial), Some(progress)) = (&source.partial, &source.progress) {
        if let Some((done, saved_total)) = parse_progress(progress) {
            if saved_total == total && done > 0 && done < total {
                tracing::info!(source = %source.id, page = done, "resuming analysis");
                sections.push(partial.clone());
                start = done;
            }
        }
    }
    while start < total {
        // Rendered just before it is needed, so a long book never holds more
        // than one batch of page images in memory.
        let batch = {
            let bytes = bytes.to_vec();
            tokio::task::spawn_blocking(move || {
                render::render_pdf_range(&bytes, start, PAGES_PER_REQUEST)
            })
            .await
            .map_err(|e| AppError::Internal(anyhow::anyhow!("page rendering failed: {e}")))?
        };
        let batch_end = (start + PAGES_PER_REQUEST).min(total);
        db::set_source_progress(&state.db, &source.id, batch_end as i64, total as i64).await?;

        if batch.is_empty() {
            // Nothing in this slice could be rasterized. If that is true of the
            // very first slice the file is not renderable at all, and the
            // caller should try another path.
            if start == 0 {
                return Err(AppError::Unsupported(
                    "no page of this PDF could be rendered to an image".into(),
                ));
            }
            unreadable.push(format!("{} to {}", start + 1, batch_end));
            start = batch_end;
            continue;
        }

        let batch = &batch[..];
        let first = batch.first().map(|p| p.index + 1).unwrap_or(start + 1);
        let last = batch.last().map(|p| p.index + 1).unwrap_or(first);
        let mut content = vec![ContentPart::text(format!(
            "These are page images from the PDF file named {filename}: pages \
             {first} to {last} of {} in total, in order. Transcribe each page \
             faithfully under its own `## p. N` header using the actual page \
             number: the page's own words, lists, tables with their numbers, \
             mathematics as LaTeX. This is transcription, not summary: quote \
             exactly, never invent, never substitute a different document. \
             Briefly note figures, charts and diagrams with their numbers and \
             axis units. The file name tells you nothing about the contents: \
             never use it as a source of facts. If a page cannot be read, give \
             its header plus one line saying so.",
            total
        ))];
        for page in batch {
            content.push(crate::llm::part_for_file("image/png", &page.png, None));
        }
        let req = ChatRequest::new(&resolved.model.model_id, vec![Message {
            role: Role::User,
            content,
        }])
        .system(
            "You are the analyzer. You turn PDF page images into a \
             markdown-latex rendition for a research notebook. Transcribe what \
             the pages show and nothing else.",
        )
        .max_tokens(max_tokens)
        .effort(effort);
        let text = analyzer_chat(
            state,
            resolved,
            &req,
            &format!(" while transcribing pages {first} to {last}"),
        )
        .await
        .map_err(|e| match e {
            // A rate limit keeps its own shape; anything else is a page failure.
            rate @ AppError::RateLimited { .. } => rate,
            other => AppError::Unsupported(format!(
                "page-image analysis failed on pages {first} to {last}: {other}"
            )),
        })?;
        if text.trim().is_empty() {
            return Err(AppError::Unsupported(format!(
                "the analyzer model returned no text for pages {first} to {last}; \
                 try a different analyzer model"
            )));
        }
        sections.push(text);
        start = batch_end;
        // Saved after every batch, so a restart costs one batch, not the book.
        db::save_partial(
            &state.db,
            &source.id,
            &sections.join("\n\n"),
            batch_end as i64,
            total as i64,
        )
        .await?;
    }

    if !unreadable.is_empty() {
        sections.push(format!(
            "[Pages {} could not be rendered to images and are missing from this \
             rendition.]",
            unreadable.join(", ")
        ));
    }
    db::clear_partial(&state.db, &source.id).await?;
    Ok((sections.join("\n\n"), Some(resolved.model.model_id.clone())))
}

/// Structure a usable text layer into markdown. Works with any model, no
/// vision needed, and covers the whole file no matter how long.
async fn structure_text_layer(
    state: &AppState,
    resolved: &crate::models::Resolved,
    filename: &str,
    extracted: &str,
) -> AppResult<(String, Option<String>)> {
    use crate::llm::{ChatRequest, Message};
    use crate::models;

    let (input, truncated) = truncate_chars(extracted, PDF_TEXT_INPUT_MAX);
    let req = ChatRequest::new(
        &resolved.model.model_id,
        vec![Message::user(format!(
            "The text below is the complete text layer extracted from the PDF \
             file named {filename}. Turn it into a markdown rendition for a \
             research notebook: keep every section, in order, as faithful \
             transcription, not a summary. Use `#` headings for the \
             document's own headings, keep its lists and tables with their \
             numbers, and write mathematics as LaTeX. Page breaks are not \
             visible in the extraction, so do not invent `p. N` markers, \
             just transcribe in order.{}\n\n{}",
            if truncated {
                " The text was truncated to fit; transcribe what is here and \
                 end with one line saying the tail is missing."
            } else {
                ""
            },
            input
        ))],
    )
    .system(
        "You are the analyzer. You turn extracted document text into a \
         markdown-latex rendition for a research notebook. Transcribe what \
         is given and nothing else. The file name tells you nothing about \
         the contents: never use it as a source of facts, and never \
         substitute a different document with a similar title. If the text \
         is garbled or unreadable, say exactly that in one line instead of \
         guessing.",
    )
    .max_tokens(resolved.model.max_output_tokens.max(1024) as u32)
    .effort(models::effort(&state.db, &state.user_id).await?);

    let text = analyzer_chat(state, resolved, &req, " while structuring the text layer").await?;
    if text.trim().is_empty() {
        return Err(AppError::Unsupported(
            "the analyzer model returned no text for this PDF; try a \
             different analyzer model"
                .into(),
        ));
    }
    Ok((text, Some(resolved.model.model_id.clone())))
}

/// Send the document natively. Some endpoints take this; others reject the
/// part outright, and the caller decides what that means.
async fn analyze_native_document(
    state: &AppState,
    resolved: &crate::models::Resolved,
    source: &Source,
    filename: &str,
    bytes: &[u8],
) -> AppResult<(String, Option<String>)> {
    use crate::llm::{ChatRequest, ContentPart, Message, Role, part_for_file};
    use crate::models;

    let req = ChatRequest::new(
        &resolved.model.model_id,
        vec![Message {
            role: Role::User,
            content: vec![
                part_for_file(
                    &source.media_type,
                    bytes,
                    source.original_filename.as_deref(),
                ),
                ContentPart::text(format!(
                    "File name: {filename}\n\n{}",
                    prompt_for(Kind::Pdf)
                )),
            ],
        }],
    )
    .system(
        "You are the analyzer. You turn one source file into a markdown-latex \
         rendition for a research notebook. Transcribe what the file shows and \
         nothing else.",
    )
    .max_tokens(resolved.model.max_output_tokens.max(1024) as u32)
    .effort(models::effort(&state.db, &state.user_id).await?);

    let text = analyzer_chat(state, resolved, &req, " while reading the PDF document").await?;
    if text.trim().is_empty() {
        return Err(AppError::Unsupported(
            "the analyzer model returned no text for this PDF; try a different \
             analyzer model"
                .into(),
        ));
    }
    Ok((text, Some(resolved.model.model_id.clone())))
}

async fn analyze_pdf(
    state: &AppState,
    source: &Source,
    model_override: Option<&str>,
) -> AppResult<(String, Option<String>)> {

    let bytes = state.storage.read(&source.storage_path).await?;
    if bytes.len() > PDF_MAX_BYTES {
        return Err(AppError::Unsupported(format!(
            "this PDF is {} MiB, over the {} MiB the analyzer accepts; \
             split it and upload the parts",
            bytes.len() / (1024 * 1024),
            PDF_MAX_BYTES / (1024 * 1024)
        )));
    }

    let filename = source.original_filename.as_deref().unwrap_or(&source.title);
    let extracted = extract_pdf_text(&bytes);
    let has_text = usable_text_layer(&extracted);

    let resolved = match analyzer_model(state, model_override).await? {
        Some(r) => r,
        // No model configured, but the text layer is already the rendition.
        None if has_text => return Ok((extracted, None)),
        None => {
            return Err(AppError::Unsupported(
                "no analyzer model configured; add a provider and pin a model in \
                 Configure, or upload a PDF with an embedded text layer"
                    .into(),
            ));
        }
    };
    if !resolved.model.supports_vision && !has_text {
        return Err(AppError::Unsupported(format!(
            "this PDF has no readable text layer (it may be scanned), and the \
             analyzer model {} is not marked vision-capable; pin a \
             vision-capable model in Configure",
            resolved.model.display_name
        )));
    }

    if resolved.model.supports_vision {
        // Page images first: text and figures together, accepted everywhere.
        let total_pages = tokio::task::spawn_blocking({
            let bytes = bytes.clone();
            move || render::page_count(&bytes)
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("page rendering failed: {e}")))?;
        if total_pages > 0 {
            match analyze_page_images(state, &resolved, source, filename, &bytes, total_pages)
                .await
            {
                Ok(done) => return Ok(done),
                // Every remaining path calls the same endpoint, so a provider
                // that is out of capacity would only fail again, more slowly,
                // and end up reported as an unreadable file.
                Err(pages_err) if pages_err.is_rate_limited() => return Err(pages_err),
                Err(pages_err) => {
                    tracing::warn!(error = %pages_err, "page images failed, trying native document");
                    match analyze_native_document(state, &resolved, source, filename, &bytes)
                        .await
                    {
                        Ok(done) => return Ok(done),
                        Err(doc_err) if doc_err.is_rate_limited() => return Err(doc_err),
                        Err(_) if has_text => {
                            return structure_text_layer(
                                state,
                                &resolved,
                                filename,
                                &extracted,
                            )
                            .await;
                        }
                        Err(_) => return Err(honest_pdf_error(&pages_err)),
                    }
                }
            }
        }
        // The renderer could not parse the file: try the native document.
        match analyze_native_document(state, &resolved, source, filename, &bytes).await {
            Ok(done) => return Ok(done),
            Err(doc_err) if doc_err.is_rate_limited() => return Err(doc_err),
            Err(_) if has_text => {
                return structure_text_layer(state, &resolved, filename, &extracted).await;
            }
            Err(doc_err) => return Err(honest_pdf_error(&doc_err)),
        }
    }

    // Only a non-vision model with a usable text layer reaches here.
    structure_text_layer(state, &resolved, filename, &extracted).await
}

/// "12/177" back into numbers.
fn parse_progress(progress: &str) -> Option<(usize, usize)> {
    let (done, total) = progress.split_once('/')?;
    Some((done.trim().parse().ok()?, total.trim().parse().ok()?))
}

/// What the user sees when every PDF path failed: the provider's own words,
/// plus what to do about it. Never a guess presented as a rendition.
fn honest_pdf_error(cause: &AppError) -> AppError {
    AppError::Unsupported(format!(
        "the analyzer endpoint could not read this PDF ({cause}). The file has \
         no usable text layer and the endpoint rejected the document itself; \
         pin a model whose endpoint takes page images or PDF documents"
    ))
}

/// PDF text-layer extraction: FlateDecode and ASCII85/ASCIIHex streams are
/// decoded, image streams (DCT, CCITT, JBIG2, JPX) are skipped, and literal
/// `( ... )` plus hex `< ... >` strings are collected from everything
/// textual. Raw compressed bytes are never scanned, so binary junk cannot
/// pose as document text.
pub fn extract_pdf_text(bytes: &[u8]) -> String {
    let spans = stream_spans(bytes);

    // Everything textual: the gaps between streams (document structure and
    // metadata) plus each successfully decoded stream.
    let mut textual = Vec::with_capacity(bytes.len() / 2);
    let mut cursor = 0;
    for &(start, end) in &spans {
        textual.extend_from_slice(&bytes[cursor..start.min(bytes.len())]);
        cursor = end.min(bytes.len());
        let dict = &bytes[start.saturating_sub(1500)..start];
        if let Some(decoded) = decode_stream(parse_filters(dict), &bytes[start..cursor])
            && looks_textual(&decoded)
        {
            textual.extend_from_slice(&decoded);
            textual.push(b' ');
        }
    }
    textual.extend_from_slice(&bytes[cursor..]);

    scan_pdf_strings(&textual)
}

/// Whether bytes read like document structure or content rather than binary
/// residue. Unfiltered font, image and xref streams fail this and are skipped
/// instead of diluting the real text with junk tokens.
fn looks_textual(data: &[u8]) -> bool {
    let sample = &data[..data.len().min(4096)];
    if sample.is_empty() {
        return false;
    }
    let ok = sample.iter().filter(|b| matches!(b, 0x09 | 0x0A | 0x0D | 0x20..=0x7E)).count();
    ok * 10 >= sample.len() * 7
}

/// Byte spans of `stream ... endstream` data sections.
fn stream_spans(bytes: &[u8]) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut i = 0;
    while i + 6 < bytes.len() {
        if &bytes[i..i + 6] == b"stream"
            && matches!(bytes.get(i + 6), Some(b'\r' | b'\n'))
        {
            let mut data = i + 6;
            if bytes[data] == b'\r' && bytes.get(data + 1) == Some(&b'\n') {
                data += 2;
            } else {
                data += 1;
            }
            let rest = &bytes[data..];
            if let Some(off) = find_bytes(rest, b"endstream") {
                spans.push((data, data + off));
                i = data + off + 9;
                continue;
            }
            break;
        }
        i += 1;
    }
    spans
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Filters named by the stream's dictionary. `None` means the dictionary
/// could not be read (for example an indirect `/Filter 12 0 R`), in which
/// case the stream is skipped rather than scanned raw.
fn parse_filters(dict: &[u8]) -> Option<Vec<String>> {
    let at = match find_bytes(dict, b"/Filter") {
        Some(at) => at,
        // No filter key at all: the stream is raw.
        None => return Some(Vec::new()),
    };
    let mut rest = &dict[at + 7..];
    while rest.first().is_some_and(|b| b.is_ascii_whitespace()) {
        rest = &rest[1..];
    }
    let mut names = Vec::new();
    if rest.first() == Some(&b'[') {
        rest = &rest[1..];
        while !rest.is_empty() && rest[0] != b']' {
            while rest.first().is_some_and(|b| b.is_ascii_whitespace()) {
                rest = &rest[1..];
            }
            if rest.first() == Some(&b'/') {
                let len = rest[1..].iter().take_while(|b| is_name_char(**b)).count();
                names.push(String::from_utf8_lossy(&rest[1..1 + len]).into_owned());
                rest = &rest[1 + len..];
            } else if !rest.is_empty() {
                rest = &rest[1..];
            }
        }
        Some(names)
    } else if rest.first() == Some(&b'/') {
        let len = rest[1..].iter().take_while(|b| is_name_char(**b)).count();
        names.push(String::from_utf8_lossy(&rest[1..1 + len]).into_owned());
        Some(names)
    } else {
        // Indirect reference or unexpected shape: do not guess.
        None
    }
}

fn is_name_char(b: u8) -> bool {
    !matches!(b, b'\0'..=b' ' | b'/' | b'[' | b']' | b'<' | b'>' | b'(' | b')' | b'%')
}

/// Run a stream's filter chain. Anything image-shaped, encrypted-shaped or
/// unknown yields `None`: the stream is skipped.
fn decode_stream(filters: Option<Vec<String>>, data: &[u8]) -> Option<Vec<u8>> {
    let filters = filters?;
    if filters.is_empty() {
        return Some(data.to_vec());
    }
    let mut buf = data.to_vec();
    for f in &filters {
        match f.as_str() {
            "FlateDecode" | "Fl" => {
                buf = miniz_oxide::inflate::decompress_to_vec(&buf).ok()?;
            }
            "ASCII85Decode" | "A85" => buf = ascii85_decode(&buf)?,
            "ASCIIHexDecode" | "AHx" => buf = ascii_hex_decode(&buf)?,
            // Decryption without a password never yields text; image codecs
            // never do either.
            "Crypt" => {}
            "DCTDecode" | "DCT" | "CCITTFaxDecode" | "CCF" | "JBIG2Decode" | "JPXDecode" => {
                return None;
            }
            _ => return None,
        }
    }
    Some(buf)
}

/// Adobe ASCII85, with `z` shorthand, whitespace skipping and optional
/// `<~` `~>` wrappers.
fn ascii85_decode(data: &[u8]) -> Option<Vec<u8>> {
    let mut chars: Vec<u8> = data
        .iter()
        .filter(|b| !b.is_ascii_whitespace())
        .copied()
        .collect();
    if chars.len() >= 2 && chars[0] == b'<' && chars[1] == b'~' {
        chars.drain(..2);
    }
    if chars.len() >= 2 && chars[chars.len() - 2] == b'~' && chars[chars.len() - 1] == b'>' {
        chars.truncate(chars.len() - 2);
    }
    let mut out = Vec::with_capacity(chars.len() * 4 / 5);
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == b'z' {
            out.extend_from_slice(&[0, 0, 0, 0]);
            i += 1;
            continue;
        }
        let mut group = [84u32; 5];
        let mut n = 0;
        while n < 5 && i < chars.len() && chars[i] != b'z' {
            let c = chars[i];
            if !(b'!'..=b'u').contains(&c) {
                return None;
            }
            group[n] = (c - b'!') as u32;
            n += 1;
            i += 1;
        }
        if n == 0 {
            break;
        }
        if n == 1 {
            return None;
        }
        // A short final group pads with `u`, and loses its tail bytes.
        let mut value = 0u32;
        for g in group.iter() {
            value = value * 85 + g;
        }
        let bytes = value.to_be_bytes();
        out.extend_from_slice(&bytes[..n - 1]);
    }
    Some(out)
}

fn ascii_hex_decode(data: &[u8]) -> Option<Vec<u8>> {
    let hex: Vec<u8> =
        data.iter().filter(|b| !b.is_ascii_whitespace() && **b != b'>').copied().collect();
    if hex.is_empty() || !hex.iter().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let mut out = Vec::with_capacity(hex.len() / 2 + 1);
    let mut it = hex.chunks(2);
    for pair in &mut it {
        let hi = (pair[0] as char).to_digit(16)?;
        let lo = if pair.len() == 2 { (pair[1] as char).to_digit(16)? } else { 0 };
        out.push((hi * 16 + lo) as u8);
    }
    Some(out)
}

/// Collect literal `( ... )` strings (with escapes and nesting) and hex
/// `< ... >` strings from textual bytes.
fn scan_pdf_strings(buf: &[u8]) -> String {
    let mut out = Vec::<String>::new();
    let mut i = 0;
    while i < buf.len() {
        match buf[i] {
            b'(' => {
                let mut depth = 1usize;
                let mut str_buf = Vec::new();
                i += 1;
                while i < buf.len() && depth > 0 {
                    match buf[i] {
                        b'\\' if i + 1 < buf.len() => {
                            let n = buf[i + 1];
                            match n {
                                b'n' => str_buf.push(b'\n'),
                                b'r' => str_buf.push(b'\r'),
                                b't' => str_buf.push(b'\t'),
                                b'(' | b')' | b'\\' => str_buf.push(n),
                                b'\r' | b'\n' => {}
                                b'0'..=b'7' => {
                                    let mut val = 0u8;
                                    let mut j = i + 1;
                                    for _ in 0..3 {
                                        if j < buf.len() && buf[j].is_ascii_digit() && buf[j] < b'8' {
                                            val = val * 8 + (buf[j] - b'0');
                                            j += 1;
                                        } else {
                                            break;
                                        }
                                    }
                                    str_buf.push(val);
                                    i = j - 1;
                                }
                                _ => str_buf.push(n),
                            }
                            i += 2;
                        }
                        b'(' => {
                            depth += 1;
                            str_buf.push(b'(');
                            i += 1;
                        }
                        b')' => {
                            depth -= 1;
                            if depth > 0 {
                                str_buf.push(b')');
                            }
                            i += 1;
                        }
                        b => {
                            str_buf.push(b);
                            i += 1;
                        }
                    }
                }
                let s = String::from_utf8_lossy(&str_buf);
                if s.chars().filter(|c| c.is_alphanumeric()).count() >= 3 {
                    out.push(s.into_owned());
                }
            }
            b'<' if i + 1 < buf.len() && buf[i + 1] != b'<' => {
                if let Some(end) = buf[i + 1..].iter().position(|&b| b == b'>') {
                    let hex: Vec<u8> =
                        buf[i + 1..i + 1 + end].iter().filter(|b| !b.is_ascii_whitespace()).copied().collect();
                    if !hex.is_empty()
                        && hex.len().is_multiple_of(2)
                        && hex.iter().all(|b| b.is_ascii_hexdigit())
                    {
                        let decoded: Vec<u8> = hex
                            .chunks(2)
                            .map(|c| {
                                let hi = (c[0] as char).to_digit(16).unwrap_or(0);
                                let lo = (c[1] as char).to_digit(16).unwrap_or(0);
                                (hi * 16 + lo) as u8
                            })
                            .collect();
                        let s = String::from_utf8_lossy(&decoded);
                        if s.chars().filter(|c| c.is_alphanumeric()).count() >= 3 {
                            out.push(s.into_owned());
                        }
                    }
                    i += end + 2;
                } else {
                    i += 1;
                }
            }
            _ => i += 1,
        }
    }
    out.join(" ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Whether the extraction reads like a document rather than binary residue.
/// Junk passes a pure length check easily, so most tokens must also be
/// word-like and of sane length.
pub fn usable_text_layer(text: &str) -> bool {
    if text.chars().count() < PDF_TEXT_SUBSTANTIAL {
        return false;
    }
    let mut total = 0usize;
    let mut wordy = 0usize;
    for tok in text.split_whitespace() {
        total += 1;
        let len = tok.chars().count();
        let letters = tok.chars().filter(|c| c.is_alphabetic()).count();
        if len <= 40 && letters >= 2 && letters * 2 >= len {
            wordy += 1;
        }
    }
    // Fewer than 60 tokens, absurdly long tokens on average, or under 40%
    // word-like: not a text layer.
    total >= 60 && text.chars().count() / total <= 25 && wordy * 5 >= total * 2
}

/// First `max` characters, reporting whether anything was cut.
fn truncate_chars(s: &str, max: usize) -> (String, bool) {
    if s.chars().count() <= max {
        return (s.to_string(), false);
    }
    (s.chars().take(max).collect(), true)
}

/// Ask the analyzer a targeted question about one source's ORIGINAL bytes.
/// This is what the researcher reaches for when the rendition isn't enough
/// ("what are the axis units in figure 3?"). Wired into the agent loop in
/// phase 2; usable directly today.
pub async fn ask_source(state: &AppState, source: &Source, question: &str) -> AppResult<String> {
    use crate::llm::{ChatRequest, ContentPart, Message, Role, part_for_file};
    use crate::models;

    let resolved = state.role_model("analyzer").await?;
    if !resolved.model.supports_vision && !matches!(kind_of(source), Kind::Text) {
        return Err(AppError::Unsupported(format!(
            "the analyzer model {} is not marked vision-capable, and {} files need \
             a multi-modal model; pin one in Configure",
            resolved.model.display_name,
            kind_of(source).as_str()
        )));
    }

    let bytes = state.storage.read(&source.storage_path).await?;
    let kind = kind_of(source);
    let content = vec![
        part_for_file(&source.media_type, &bytes, source.original_filename.as_deref()),
        ContentPart::text(question),
    ];

    let req = ChatRequest::new(&resolved.model.model_id, vec![Message {
        role: Role::User,
        content,
    }])
    .system(
            "You are the analyzer. You are looking at the original source file. \
             Answer the question precisely and only from what the file actually shows. \
             Use markdown with LaTeX for any mathematics. If the file does not answer \
             the question, say so plainly.",
    )
    .max_tokens(resolved.model.max_output_tokens.max(1024) as u32)
    .effort(models::effort(&state.db, &state.user_id).await?);

    let client = resolved.client(&state.http)?;
    match crate::llm::chat_with_retry(client.as_ref(), &req).await {
        Ok(resp) => Ok(resp.text),
        Err(e) if e.is_rate_limited() => Err(AppError::from_rate_limit(
            e,
            &resolved.model.id,
            &resolved.model.display_name,
            " while reading this source",
        )),
        // Some endpoints reject the document part itself (notably ones that
        // take images but not files). For a PDF with a usable text layer,
        // answer from that instead of failing.
        Err(_) if matches!(kind, Kind::Pdf) => {
            let extracted = extract_pdf_text(&bytes);
            if !usable_text_layer(&extracted) {
                return Err(AppError::Unsupported(
                    "the analyzer endpoint rejected this PDF and it has no \
                     usable text layer to fall back on"
                        .into(),
                ));
            }
            let filename = source.original_filename.as_deref().unwrap_or(&source.title);
            let (input, _) = truncate_chars(&extracted, PDF_TEXT_INPUT_MAX);
            let fallback = ChatRequest::new(
                &resolved.model.model_id,
                vec![Message::user(format!(
                    "Answer this question only from the extracted text of the PDF \
                     file named {filename}. If the text does not answer it, say \
                     so plainly.\n\nQuestion: {question}\n\nText:\n{input}"
                ))],
            )
            .system(
                "You are the analyzer. Answer precisely and only from the given \
                 text, in markdown with LaTeX for any mathematics.",
            )
            .max_tokens(resolved.model.max_output_tokens.max(1024) as u32)
            .effort(models::effort(&state.db, &state.user_id).await?);
            analyzer_chat(state, &resolved, &fallback, " while reading this source").await
        }
        Err(e) => Err(e.into()),
    }
}

fn kind_of(source: &Source) -> Kind {
    Kind::from_media_type(&source.media_type, source.original_filename.as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_prefers_extension_over_octet_stream() {
        let pdf = Kind::from_media_type("application/octet-stream", Some("scan.pdf"));
        assert_eq!(pdf, Kind::Pdf);
        let img = Kind::from_media_type("application/octet-stream", Some("photo.jpg"));
        assert_eq!(img, Kind::Image);
        let audio = Kind::from_media_type("application/octet-stream", Some("talk.mp3"));
        assert_eq!(audio, Kind::Audio);
        let video = Kind::from_media_type("application/octet-stream", Some("demo.mp4"));
        assert_eq!(video, Kind::Video);
        let text = Kind::from_media_type("application/octet-stream", Some("notes.md"));
        assert_eq!(text, Kind::Text);
    }

    #[test]
    fn kind_from_media_type_without_filename() {
        assert_eq!(Kind::from_media_type("application/pdf", None), Kind::Pdf);
        assert_eq!(Kind::from_media_type("image/png", None), Kind::Image);
        assert_eq!(Kind::from_media_type("audio/wav", None), Kind::Audio);
        assert_eq!(Kind::from_media_type("video/mp4", None), Kind::Video);
        assert_eq!(Kind::from_media_type("text/plain", None), Kind::Text);
    }

    #[test]
    fn extracts_literal_strings_from_uncompressed_pdf() {
        let pdf = b"%PDF-1.4\n1 0 obj\n<< /Type /Page >>\nstream\nBT /F1 12 Tf \
            (Hello world, second page) Tj ET\nendstream\nendobj\ntrailer\n<<>>";
        let text = extract_pdf_text(pdf);
        assert!(text.contains("Hello world"), "got: {text}");
    }

    #[test]
    fn extracts_hex_strings_and_skips_dicts() {
        let pdf = b"<< /Title (Doc) >>\n[<48656c6c6f> 120 <20776f726c64>] TJ";
        let text = extract_pdf_text(pdf);
        assert!(text.contains("Hello"), "got: {text}");
        assert!(text.contains("world"), "got: {text}");
    }

    #[test]
    fn compressed_pdf_yields_nothing_useful() {
        let pdf = b"%PDF-1.4\n1 0 obj\n<< /Filter /FlateDecode /Length 20 >>\nstream\n\
            \x78\x9c\x2b\x49\x2d\x2e\x01\x00\x04\x5d\x01\xc1\nendstream\nendobj";
        assert!(extract_pdf_text(pdf).chars().count() < PDF_TEXT_SUBSTANTIAL);
    }

    #[test]
    fn decodes_flate_content_streams() {
        let content = b"BT /F1 12 Tf (Flate decoded sentence one.) Tj ET";
        let compressed = miniz_oxide::deflate::compress_to_vec(content, 6);
        let mut pdf = b"%PDF-1.4\n1 0 obj\n<< /Length 0 /Filter /FlateDecode >>\nstream\n".to_vec();
        pdf.extend_from_slice(&compressed);
        pdf.extend_from_slice(b"\nendstream\nendobj\ntrailer\n<<>>");
        let text = extract_pdf_text(&pdf);
        assert!(text.contains("Flate decoded sentence one"), "got: {text}");
    }

    #[test]
    fn skips_image_streams_even_with_text_lookalikes() {
        let pdf = b"%PDF-1.4\n1 0 obj\n<< /Filter /DCTDecode /Length 44 >>\nstream\n\
            \xff\xd8(SecretMarker Xqz should never appear)\xff\xd9\nendstream\nendobj";
        let text = extract_pdf_text(pdf);
        assert!(!text.contains("SecretMarker"), "got: {text}");
    }

    #[test]
    fn ascii85_decodes_a_known_vector() {
        // "87cUR" is the textbook encoding of "Hell".
        assert_eq!(ascii85_decode(b"87cUR"), Some(b"Hell".to_vec()));
        assert_eq!(ascii85_decode(b"z"), Some(vec![0, 0, 0, 0]));
        // `~` is outside the `!`..=`u` alphabet.
        assert_eq!(ascii85_decode(b"87cUR~"), None);
    }

    #[test]
    fn rejects_binary_residue_as_a_text_layer() {
        let mut junk = b"%PDF-1.4\n".to_vec();
        for i in 0..3000u32 {
            junk.extend_from_slice(
                format!("({i:08x}\x01\x02~|{} )", i.wrapping_mul(2654435761)).as_bytes(),
            );
        }
        let text = extract_pdf_text(&junk);
        assert!(!usable_text_layer(&text), "got: {text}");
    }

    #[test]
    fn accepts_a_real_text_layer() {
        let sentence = "The definition states that an integer n is even when another \
            integer m exists with n equal to twice m. ";
        let text = sentence.repeat(20);
        assert!(usable_text_layer(&text));
        assert!(!usable_text_layer("short"));
    }

    #[test]
    fn textual_gate_keeps_binary_streams_out() {
        assert!(looks_textual(b"BT /F1 12 Tf (Hello world) Tj ET"));
        assert!(!looks_textual(&[0xff, 0xd8, 0x00, 0x10, 0x4a, 0x46]));
        assert!(!looks_textual(b""));
    }

    #[test]
    fn binary_streams_do_not_poison_a_real_text_layer() {
        // A text content stream plus a large unfiltered binary stream, as in a
        // PDF with embedded font binaries. The binary must not drag the layer
        // below usable.
        let sentence = "The definition states that an integer n is even when another \
            integer m exists with n equal to twice m. ";
        let mut pdf = b"%PDF-1.4\n1 0 obj\n<< /Length 0 >>\nstream\n".to_vec();
        for _ in 0..20 {
            pdf.extend_from_slice(
                format!("BT /F1 12 Tf ({sentence}) Tj ET\n").as_bytes(),
            );
        }
        pdf.extend_from_slice(b"endstream\nendobj\n2 0 obj\n<< /Length 30000 >>\nstream\n");
        let mut x = 0x12345678u64;
        for i in 0..30000 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let b = if i % 7 == 0 {
                b' '
            } else if i % 11 == 0 {
                b'('
            } else {
                (x >> 33) as u8
            };
            pdf.push(b);
        }
        pdf.extend_from_slice(b"\nendstream\nendobj\ntrailer\n<<>>");
        let text = extract_pdf_text(&pdf);
        assert!(text.contains("integer n is even"), "got: {text}");
        assert!(usable_text_layer(&text), "got: {text}");
    }

    #[test]
    fn part_routing_covers_new_modalities() {
        use crate::llm::{ContentPart, part_for_file};
        assert!(matches!(
            part_for_file("audio/wav", b"data", None),
            ContentPart::Audio { .. }
        ));
        assert!(matches!(
            part_for_file("video/mp4", b"data", None),
            ContentPart::Video { .. }
        ));
        assert!(matches!(
            part_for_file("image/png", b"data", None),
            ContentPart::Image { .. }
        ));
        assert!(matches!(
            part_for_file("application/pdf", b"data", Some("a.pdf")),
            ContentPart::Document { .. }
        ));
    }
}
