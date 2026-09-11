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

pub mod sniff;
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

/// Text as a person would read it, whatever it was saved as.
///
/// UTF-8 is nearly everything, but a file that carries a UTF-16 byte order mark
/// would otherwise be indexed as every second character being a NUL, and one
/// written in an older single byte encoding would lose its accents. Both are
/// still someone's notes.
pub fn decode_text(bytes: &[u8]) -> String {
    match sniff::text_bom(bytes) {
        Some("text/plain") => String::from_utf8_lossy(&bytes[3..]).into_owned(),
        Some(mark) => {
            let little = mark.ends_with("le");
            let units: Vec<u16> = bytes[2..]
                .chunks_exact(2)
                .map(|p| if little {
                    u16::from_le_bytes([p[0], p[1]])
                } else {
                    u16::from_be_bytes([p[0], p[1]])
                })
                .collect();
            String::from_utf16_lossy(&units)
        }
        None => match std::str::from_utf8(bytes) {
            Ok(text) => text.to_string(),
            // Not UTF-8, and `sniff` already said it reads as text, so it is an
            // older single byte encoding: Latin-1 maps straight onto codepoints.
            Err(_) => bytes.iter().map(|&b| b as char).collect(),
        },
    }
}

/// What a file is, decided by its bytes rather than by its name.
///
/// The declared media type and the extension are claims made by whoever
/// uploaded the file, and they are wrong often enough to matter: an mp3 renamed
/// to `.pdf` would otherwise be rasterized as a document, fail, and cost a model
/// call to say so. The bytes decide; the name only fills in the finer label
/// (`text/markdown` rather than `text/plain`) once the bytes agree it is text.
pub fn identify(
    bytes: &[u8],
    filename: Option<&str>,
    declared: Option<&str>,
) -> AppResult<(Kind, String)> {
    let found = sniff::sniff(bytes);

    let Some(kind) = found.kind else {
        return Err(AppError::Unsupported(format!(
            "this file is {}, which the analyzer cannot read; it takes PDFs, \
             images, audio, video and text files",
            found.label
        )));
    };

    // For text, a name that says `text/markdown` or `text/csv` is more precise
    // than the bytes can be, so it wins. For everything else the bytes win.
    let media_type = if kind == Kind::Text {
        let named = declared
            .filter(|m| m.starts_with("text/") || *m == "application/json")
            .map(str::to_string)
            .or_else(|| {
                filename.map(|f| mime_guess::from_path(f).first_or_octet_stream().to_string())
            })
            .filter(|m| m.starts_with("text/") || m == "application/json");
        named.unwrap_or_else(|| found.media_type.to_string())
    } else {
        found.media_type.to_string()
    };

    if let Some(name) = filename {
        let claimed = Kind::from_media_type(
            declared.unwrap_or("application/octet-stream"),
            Some(name),
        );
        if claimed != kind {
            tracing::info!(
                file = %name, claimed = %claimed.as_str(), actual = %kind.as_str(),
                "the name and the bytes disagree; going with the bytes"
            );
        }
    }

    Ok((kind, media_type))
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
            /* A stored rendition that is itself a description of the file is
               exactly what this run set out to replace, and leaving it behind
               would keep the notebook answering questions about this file with
               something a model invented, under a source the explorer already
               marks failed. A rendition that reads as a real transcription
               stays: a failed rerun is no reason to throw away good work. */
            if e.is_rendition_failure()
                && let Ok(Some(stored)) = db::get_document(&state.db, &source.id).await
                && page_rendition_is_a_summary(&stored.markdown, 1)
            {
                tracing::warn!(source = %source.id, "dropping a stale description");
                db::delete_document(&state.db, &source.id).await?;
            }
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
    // Rows stored before the bytes were ever checked, and files whose name
    // lies, are both settled here: a header is cheap to read and never wrong.
    let head = state.storage.read_head(&source.storage_path, HEAD_BYTES).await?;
    let kind = match sniff::sniff(&head).kind {
        Some(kind) => kind,
        None => {
            return Err(AppError::Unsupported(format!(
                "this file is {}, which the analyzer cannot read; it takes PDFs, \
                 images, audio, video and text files",
                sniff::sniff(&head).label
            )));
        }
    };

    let (markdown, model) = match kind {
        Kind::Text => {
            let bytes = state.storage.read(&source.storage_path).await?;
            if bytes.len() > TEXT_MAX_BYTES {
                return Err(AppError::Unsupported(format!(
                    "this text file is {} MiB, over the {} MiB the analyzer reads \
                     in one piece; split it and upload the parts",
                    bytes.len() / (1024 * 1024),
                    TEXT_MAX_BYTES / (1024 * 1024)
                )));
            }
            (decode_text(&bytes), None::<String>)
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

/// Text is read whole and chunked, so it has a ceiling of its own.
const TEXT_MAX_BYTES: usize = 32 * 1024 * 1024;
/// A PDF claiming more pages than any real document has is broken or hostile.
/// Refusing loudly beats spending a day of model calls on a generated file.
const PDF_MAX_PAGES: usize = 5_000;
/// A decompression bomb is a small file that claims to be enormous: a few
/// hundred kilobytes of PNG can unpack to a gigabyte of pixels. The header
/// says how big it intends to be, which is enough to turn it away.
const IMAGE_MAX_PIXELS: u64 = 120_000_000;
/// How many bytes of a file are enough to say what it is.
const HEAD_BYTES: usize = 8 * 1024;

/// What a faithful page transcription runs to, per page, at the very least.
/// A caption of a whole batch of pages comes in far under this.
const MIN_TRANSCRIPT_CHARS_PER_PAGE: usize = 250;
/// How much of an extracted text layer a structured rendition should still
/// carry. Losing more than this means it was summarised, not transcribed.
const MIN_TEXT_LAYER_KEPT_PERCENT: usize = 40;

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

/// A page count no real document has. Generated files can claim millions of
/// pages, and every page is a model call, so this refuses rather than settling
/// in for a week of work. It is a refusal, not a silent truncation: a document
/// that is analyzed at all is analyzed in full.
fn check_page_budget(total_pages: usize) -> AppResult<()> {
    if total_pages > PDF_MAX_PAGES {
        return Err(AppError::Unsupported(format!(
            "this PDF claims {total_pages} pages, past the {PDF_MAX_PAGES} pages the              analyzer will work through; if that is real, split it and upload the parts"
        )));
    }
    Ok(())
}

/// Whether a rendition transcribes the pages or merely talks about them.
///
/// A small model handed page images will often answer with a caption: "The PDF
/// contains the 2026 Fall timetable for a university, listing various courses".
/// Fluent, wrong in the details, and useless as a rendition, since the notebook
/// then indexes a description of the document instead of the document. Storing
/// that is worse than failing, because the researcher will quote it.
///
/// Three things separate the two. A transcription follows the `## p. N` shape it
/// was asked for; it does not open by naming the artefact it came from; and it
/// runs to roughly the length of the pages it covers.
fn page_rendition_is_a_summary(rendition: &str, pages: usize) -> bool {
    let body = rendition.trim();
    if body.is_empty() {
        return true;
    }
    if has_page_headers(body) {
        return false;
    }
    if opens_by_describing(body) {
        return true;
    }
    body.chars().count() < MIN_TRANSCRIPT_CHARS_PER_PAGE * pages.max(1)
}

/// The same question for a rendition built from an extracted text layer, where
/// the input is known: a transcription keeps most of what it was given.
fn text_rendition_is_a_summary(rendition: &str, extracted: &str) -> bool {
    let kept = rendition.trim().chars().count();
    if kept == 0 {
        return true;
    }
    if opens_by_describing(rendition.trim()) {
        return true;
    }
    let given = extracted.trim().chars().count().min(PDF_TEXT_INPUT_MAX);
    given > 0 && kept * 100 < given * MIN_TEXT_LAYER_KEPT_PERCENT
}

/// `## p. 4`, `### Page 4`, and the other shapes a model reaches for when it is
/// transcribing page by page as asked.
fn has_page_headers(body: &str) -> bool {
    body.lines().any(|line| {
        let line = line.trim_start();
        let Some(rest) = line.strip_prefix('#') else { return false };
        let rest = rest.trim_start_matches('#').trim_start().to_ascii_lowercase();
        let rest = rest
            .strip_prefix("page")
            .or_else(|| rest.strip_prefix("p."))
            .or_else(|| rest.strip_prefix('p'))
            .unwrap_or("");
        rest.trim_start_matches([' ', '.', ':']).starts_with(|c: char| c.is_ascii_digit())
    })
}

/// An answer that begins by naming the thing it was given is describing it.
/// Nobody's page begins "This PDF contains".
fn opens_by_describing(body: &str) -> bool {
    let opening: String = body
        .chars()
        .filter(|c| !matches!(c, '#' | '*' | '>' | '`' | '_'))
        .take(60)
        .collect::<String>()
        .to_ascii_lowercase();
    let opening = opening.trim_start();
    let Some(rest) = opening
        .strip_prefix("the ")
        .or_else(|| opening.strip_prefix("this "))
        .or_else(|| opening.strip_prefix("here is the "))
        .or_else(|| opening.strip_prefix("here is a "))
    else {
        return false;
    };
    [
        "pdf", "document", "file", "image", "images", "page", "pages", "screenshot",
        "slide", "slides", "attachment", "text",
    ]
    .iter()
    .any(|subject| {
        rest.strip_prefix(subject)
            .is_some_and(|tail| tail.starts_with(' ') || tail.starts_with(','))
    })
}

/// What to say when the endpoint refuses the kind of content outright.
///
/// This is a configuration problem, not a file problem, and the fix is exact:
/// the model is marked as taking images and does not. Say which model, quote
/// what the endpoint said, and name both ways out.
fn modality_error(resolved: &crate::models::Resolved, media: &str, cause: &AppError) -> AppError {
    let model = &resolved.model.display_name;
    let said = cause.provider_message();
    AppError::Modality {
        message: format!(
            "the analyzer model {model} does not accept {media}: the endpoint said \
             \"{said}\". This file cannot be read without them, so run it on a model \
             that takes them, and clear the vision flag on {model} so it is not tried \
             again"
        ),
        model_id: Some(resolved.model.id.clone()),
        model_name: Some(model.clone()),
        media: media.to_string(),
    }
}

/// What to say when the model will not transcribe. Every other path asks the
/// same model, so the answer is a different model, not a different prompt.
fn summarised_error(model: &str, what: &str) -> AppError {
    AppError::Rendition(format!(
        "the analyzer model {model} answered with a description of {what} rather than \
         a transcription, so there is no rendition to store; a rendition has to be \
         the document's own words. Pin a stronger analyzer model in Configure and \
         reingest this file"
    ))
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

    // A small file is not a small image. Ask the header what it unpacks to.
    if kind == Kind::Image
        && let Some((w, h)) = sniff::image_dimensions(&bytes)
        && w.saturating_mul(h) > IMAGE_MAX_PIXELS
    {
        return Err(AppError::Unsupported(format!(
            "this image says it is {w} by {h} pixels, {} megapixels, which is \
             far past the {} megapixels the analyzer will decode; downscale it \
             and upload again",
            w.saturating_mul(h) / 1_000_000,
            IMAGE_MAX_PIXELS / 1_000_000
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
        .await
        .map_err(|e| {
            // An omni model is not the default anywhere, and "unknown variant
            // `audio_url`" tells a person nothing about what to do next.
            if e.is_modality_refusal() {
                modality_error(&resolved, &format!("{} files", kind.as_str()), &e)
            } else {
                e
            }
        })?;
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
            // A rate limit keeps its own shape, and so does a refusal of images
            // as such: both mean something to the caller, and wrapping them in
            // prose about page numbers would throw that away and read worse.
            keep @ AppError::RateLimited { .. } => keep,
            keep if keep.is_modality_refusal() => keep,
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
        // A model that answers with a description of the pages has not
        // transcribed them, and a description stored as a rendition is worse
        // than no rendition: it reads as the document and is not.
        if page_rendition_is_a_summary(&text, batch.len()) {
            return Err(summarised_error(&resolved.model.display_name, "these pages"));
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
    // The input is known here, so a rendition that dropped most of it was a
    // summary however confidently it reads.
    if text_rendition_is_a_summary(&text, extracted) {
        return Err(summarised_error(&resolved.model.display_name, "this text"));
    }
    Ok((text, Some(resolved.model.model_id.clone())))
}

/// Structure the text layer, and if the model will not do that either, store
/// the text layer as it came out of the file.
///
/// A plain rendition that is true beats a fluent one that is not: the notebook
/// can still search it, cite it and show it, and nothing in it was invented.
/// The note at the top says what happened, so the file is not silently second
/// rate.
async fn text_layer_fallback(
    state: &AppState,
    resolved: &crate::models::Resolved,
    filename: &str,
    extracted: &str,
) -> AppResult<(String, Option<String>)> {
    match structure_text_layer(state, resolved, filename, extracted).await {
        Ok(done) => Ok(done),
        Err(e) if e.is_rate_limited() => Err(e),
        Err(e) => {
            tracing::warn!(error = %e, "storing the raw text layer instead");
            Ok((
                format!(
                    "> The analyzer model would not transcribe this file, so this \
                     rendition is the PDF's own text layer, unedited. Reingest with a \
                     stronger analyzer model for a structured one.\n\n{extracted}"
                ),
                None,
            ))
        }
    }
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

    let text = analyzer_chat(state, resolved, &req, " while reading the PDF document")
        .await
        .map_err(|e| {
            if e.is_modality_refusal() {
                modality_error(resolved, "PDF documents", &e)
            } else {
                e
            }
        })?;
    if page_rendition_is_a_summary(&text, 1) {
        return Err(summarised_error(&resolved.model.display_name, "this document"));
    }
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
        check_page_budget(total_pages)?;
        if total_pages > 0 {
            match analyze_page_images(state, &resolved, source, filename, &bytes, total_pages)
                .await
            {
                Ok(done) => return Ok(done),
                // Every remaining path calls the same endpoint, so a provider
                // that is out of capacity would only fail again, more slowly,
                // and end up reported as an unreadable file.
                Err(pages_err) if pages_err.is_rate_limited() => return Err(pages_err),
                // Nor is there any point asking the same model to describe the
                // same file another way. Only the file's own text is left.
                Err(pages_err) if pages_err.is_rendition_failure() => {
                    return if has_text {
                        text_layer_fallback(state, &resolved, filename, &extracted).await
                    } else {
                        Err(pages_err)
                    };
                }
                /* The endpoint has said this model does not take images. The
                   native document part is the same kind of refusal one step
                   later, and on endpoints that do not implement it at all the
                   second error is worse than the first, so it is not tried. */
                Err(pages_err) if pages_err.is_modality_refusal() => {
                    if has_text {
                        return text_layer_fallback(state, &resolved, filename, &extracted).await;
                    }
                    return Err(modality_error(&resolved, "page images", &pages_err));
                }
                Err(pages_err) => {
                    tracing::warn!(error = %pages_err, "page images failed, trying native document");
                    match analyze_native_document(state, &resolved, source, filename, &bytes)
                        .await
                    {
                        Ok(done) => return Ok(done),
                        Err(doc_err) if doc_err.is_rate_limited() => return Err(doc_err),
                        Err(_) if has_text => {
                            return text_layer_fallback(state, &resolved, filename, &extracted)
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
                return text_layer_fallback(state, &resolved, filename, &extracted).await;
            }
            Err(doc_err) => return Err(honest_pdf_error(&doc_err)),
        }
    }

    // Only a non-vision model with a usable text layer reaches here.
    text_layer_fallback(state, &resolved, filename, &extracted).await
}

/// "12/177" back into numbers.
fn parse_progress(progress: &str) -> Option<(usize, usize)> {
    let (done, total) = progress.split_once('/')?;
    Some((done.trim().parse().ok()?, total.trim().parse().ok()?))
}

/// What the user sees when every PDF path failed: the provider's own words,
/// plus what to do about it. Never a guess presented as a rendition.
fn honest_pdf_error(cause: &AppError) -> AppError {
    // One sentence a person can act on, with the provider's own words in it
    // rather than the JSON they arrived in.
    AppError::Unsupported(format!(
        "the analyzer endpoint could not read this PDF: {}. The file has no usable \
         text layer and the endpoint took neither page images nor the document \
         itself; pin a model whose endpoint takes one of them",
        cause.provider_message()
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
        if textual.len() < EXTRACT_MAX_BYTES
            && let Some(decoded) = decode_stream(parse_filters(dict), &bytes[start..cursor])
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
/// The most one PDF stream may decompress to, and the most all of them
/// together may add up to. Both exist to bound a decompression bomb.
const STREAM_MAX_BYTES: usize = 16 * 1024 * 1024;
const EXTRACT_MAX_BYTES: usize = 64 * 1024 * 1024;

fn decode_stream(filters: Option<Vec<String>>, data: &[u8]) -> Option<Vec<u8>> {
    let filters = filters?;
    if filters.is_empty() {
        return Some(data.to_vec());
    }
    let mut buf = data.to_vec();
    for f in &filters {
        match f.as_str() {
            "FlateDecode" | "Fl" => {
                // Unbounded inflate is how a kilobyte of PDF turns into a
                // gigabyte of memory. A stream past this ceiling is not text
                // anyone wrote, so the whole stream is dropped.
                buf = miniz_oxide::inflate::decompress_to_vec_with_limit(&buf, STREAM_MAX_BYTES)
                    .ok()?;
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

    /// Verbatim, from smolvlm2-2.2b-instruct handed a one page timetable. It
    /// reads well and is wrong: "PHL" is philosophy, not "Person".
    const CAPTION: &str = "The PDF contains the 2026 Fall timetable for a \
university, listing various courses and their corresponding times on weekdays. \
The schedule is organized by days of the week from Monday to Friday, with each \
day having a list of courses starting at different times. The courses are listed \
in columns under \"PHL\" (Person), indicating that they are taught by a specific \
person or group. The timetable also includes information about the location and \
time duration for each course.";

    #[test]
    fn a_caption_is_not_a_rendition() {
        assert!(page_rendition_is_a_summary(CAPTION, 1));
        // Long enough to pass a length test on its own, so the opening is what
        // has to catch it.
        assert!(CAPTION.len() > MIN_TRANSCRIPT_CHARS_PER_PAGE);

        for opening in [
            "This document is a syllabus for PHL245.",
            "The image shows a weekly timetable.",
            "Here is the PDF, summarised for you:",
            "**The file contains** three sections.",
        ] {
            assert!(page_rendition_is_a_summary(opening, 1), "{opening}");
        }
    }

    #[test]
    fn a_transcription_is_left_alone() {
        let transcribed = "## p. 1\n\nPHL245H5F LEC0101 Mon 10:00 to 12:00 IB 345\n";
        assert!(!page_rendition_is_a_summary(transcribed, 1));
        // Page headers carry it even for a nearly empty page, which is exactly
        // what a title page or a blank one looks like.
        assert!(!page_rendition_is_a_summary("### Page 7\n\n(blank)", 1));
        assert!(!page_rendition_is_a_summary("# p.12\n\nnothing here", 1));

        // No headers, but plainly a transcription: it is long, and it does not
        // start by naming the file.
        let long = "PHL245H5F LEC0101 Monday 10:00 to 12:00 room IB 345. "
            .repeat(12);
        assert!(!page_rendition_is_a_summary(&long, 1));

        // A sentence that merely mentions a document is not an opening about one.
        assert!(!page_rendition_is_a_summary(
            &("Theorem 4. The document of a divisor is defined as follows. ".repeat(8)),
            1
        ));
    }

    #[test]
    fn one_paragraph_for_a_whole_batch_is_a_summary() {
        // Eight pages of lecture notes do not fit in three sentences.
        let thin = "Slides about groups, rings and fields, with examples. ".repeat(6);
        assert!(page_rendition_is_a_summary(&thin, 8));
        assert!(!page_rendition_is_a_summary(&thin, 1));
    }

    #[test]
    fn a_structured_text_layer_has_to_keep_the_text() {
        let extracted = "Course PHL245H5F meets Mondays at ten. ".repeat(60);
        let faithful = "Course PHL245H5F meets Mondays at ten. ".repeat(50);
        let boiled_down = "The document lists courses and times.";

        assert!(!text_rendition_is_a_summary(&faithful, &extracted));
        assert!(text_rendition_is_a_summary(boiled_down, &extracted));
        assert!(text_rendition_is_a_summary("", &extracted));
        // Nothing to compare against means the length test cannot fire.
        assert!(!text_rendition_is_a_summary("a short note", ""));
    }

    #[test]
    fn an_impossible_page_count_is_refused_not_truncated() {
        assert!(check_page_budget(PDF_MAX_PAGES).is_ok());
        let err = check_page_budget(PDF_MAX_PAGES + 1).unwrap_err();
        assert!(err.to_string().contains("split it"), "{err}");
        // The number it claims is in the message, so the refusal is checkable.
        assert!(check_page_budget(2_000_000).unwrap_err().to_string().contains("2000000"));
    }

    #[test]
    fn text_survives_the_encoding_it_was_saved_in() {
        assert_eq!(decode_text(b"plain ascii"), "plain ascii");
        assert_eq!(decode_text("\u{feff}marked utf-8".as_bytes()), "marked utf-8");

        let mut utf16 = b"\xff\xfe".to_vec();
        for c in "wide text".encode_utf16() {
            utf16.extend_from_slice(&c.to_le_bytes());
        }
        assert_eq!(decode_text(&utf16), "wide text");

        let mut utf16be = b"\xfe\xff".to_vec();
        for c in "wide text".encode_utf16() {
            utf16be.extend_from_slice(&c.to_be_bytes());
        }
        assert_eq!(decode_text(&utf16be), "wide text");

        // Latin-1 keeps its accents rather than becoming replacement marks.
        assert_eq!(decode_text(b"caf\xe9"), "caf\u{e9}");
    }

    #[test]
    fn a_lying_extension_loses_to_the_bytes() {
        // An mp3 someone renamed to .pdf, declared as a PDF for good measure.
        let mp3 = b"ID3\x04\x00\x00\x00\x00\x00\x00audio payload here";
        let (kind, media_type) =
            identify(mp3, Some("lecture.pdf"), Some("application/pdf")).expect("identified");
        assert_eq!(kind, Kind::Audio);
        assert_eq!(media_type, "audio/mpeg");

        // And the other way round.
        let pdf = b"%PDF-1.7\n1 0 obj\n<< >>\nendobj\n";
        let (kind, media_type) =
            identify(pdf, Some("song.mp3"), Some("audio/mpeg")).expect("identified");
        assert_eq!(kind, Kind::Pdf);
        assert_eq!(media_type, "application/pdf");
    }

    #[test]
    fn a_text_file_keeps_the_finer_name_its_extension_gives_it() {
        let (kind, media_type) =
            identify(b"# Notes\n\nprose\n", Some("notes.md"), None).expect("identified");
        assert_eq!(kind, Kind::Text);
        assert_eq!(media_type, "text/markdown");

        let (kind, media_type) =
            identify(b"a,b\n1,2\n", Some("table.csv"), Some("text/csv")).expect("identified");
        assert_eq!(kind, Kind::Text);
        assert_eq!(media_type, "text/csv");
    }

    #[test]
    fn a_file_the_analyzer_cannot_read_is_refused_by_name() {
        let zip = b"PK\x03\x04\x14\x00\x00\x00\x08\x00word/document.xml";
        let err = identify(zip, Some("paper.pdf"), Some("application/pdf")).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("a Word document"), "{message}");

        let exe = b"MZ\x90\x00\x03\x00\x00\x00";
        assert!(identify(exe, Some("notes.txt"), None).unwrap_err().to_string().contains(
            "a Windows executable"
        ));

        // Binary residue with no recognisable header is refused too.
        let noise: Vec<u8> = (0u8..=255).cycle().take(4096).collect();
        assert!(identify(&noise, Some("data.txt"), None).is_err());
    }

    /// A PDF stream that inflates far past the ceiling is dropped whole rather
    /// than allocated. Without the limit this is how a small file eats memory.
    #[test]
    fn a_decompression_bomb_is_dropped_not_inflated() {
        let payload = vec![b' '; STREAM_MAX_BYTES + 1024];
        let squashed = miniz_oxide::deflate::compress_to_vec(&payload, 9);
        assert!(squashed.len() < 100_000, "the bomb should be small on disk");
        assert!(decode_stream(Some(vec!["FlateDecode".into()]), &squashed).is_none());

        // A stream inside the ceiling still decodes, so the guard only bites
        // on the absurd.
        let ordinary = miniz_oxide::deflate::compress_to_vec(b"BT (real text) Tj ET", 6);
        let out = decode_stream(Some(vec!["FlateDecode".into()]), &ordinary).expect("decoded");
        assert!(out.starts_with(b"BT ("));
    }

    /// A PDF whose bytes are damaged past parsing yields no pages and no text,
    /// which is what sends the analyzer to its fallbacks instead of panicking.
    #[test]
    fn a_corrupt_pdf_yields_nothing_without_panicking() {
        let mut broken = b"%PDF-1.5\n".to_vec();
        broken.extend((0u8..=255).cycle().take(3000));
        assert_eq!(render::page_count(&broken), 0);
        assert!(render::render_pdf_range(&broken, 0, 8).is_empty());
        // It is still recognisably a PDF, so it is not turned away at upload:
        // the analyzer gets its chance to read whatever survived.
        assert_eq!(sniff::sniff(&broken).kind, Some(Kind::Pdf));
        let _ = extract_pdf_text(&broken);
    }

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
