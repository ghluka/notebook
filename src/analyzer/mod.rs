//! Ingestion: original bytes → markdown-latex rendition → chunks.
//!
//! Phase 0 implements the text/markdown path, which needs no model at all.
//! Phase 1 fills in `analyze_with_model` for pdf / image / audio / video: the
//! same function, except the rendition comes from the multi-modal analyzer
//! looking at the original bytes. The rest of the system never learns which
//! path produced a document.

pub mod text;

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
        let text_ext = ["md", "markdown", "txt", "rst", "org", "tex", "csv", "json", "toml",
                        "yaml", "yml", "rs", "py", "js", "ts", "go", "c", "h", "cpp", "java"];
        if let Some(name) = filename {
            if let Some(ext) = name.rsplit('.').next() {
                if text_ext.contains(&ext.to_ascii_lowercase().as_str()) {
                    return Kind::Text;
                }
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

/// Analyze one source and persist its rendition. Sets the source status on the
/// way in and out, so a failure is visible in `GET /api/sources`.
pub async fn ingest(state: &AppState, source: &Source) -> AppResult<()> {
    db::set_source_status(&state.db, &source.id, "analyzing", None).await?;

    let result = run(state, source).await;

    match &result {
        Ok(()) => db::set_source_status(&state.db, &source.id, "ready", None).await?,
        Err(e) => {
            db::set_source_status(&state.db, &source.id, "failed", Some(&e.to_string())).await?
        }
    }
    result
}

async fn run(state: &AppState, source: &Source) -> AppResult<()> {
    let kind = Kind::from_media_type(&source.media_type, source.original_filename.as_deref());

    let (markdown, model) = match kind {
        Kind::Text => {
            let bytes = state.storage.read(&source.storage_path).await?;
            (String::from_utf8_lossy(&bytes).into_owned(), None)
        }
        // Phase 1: hand the original bytes to the multi-modal analyzer.
        Kind::Pdf | Kind::Image | Kind::Audio | Kind::Video | Kind::Url => {
            return Err(AppError::Unsupported(format!(
                "the {} analyzer is not implemented yet (see AGENTS.md, phase 1); \
                 the original file is stored and can be re-ingested later",
                kind.as_str()
            )));
        }
    };

    let chunks = text::chunk_markdown(&markdown);
    let summary = text::naive_summary(&markdown);
    db::replace_document(
        &state.db,
        &source.id,
        &markdown,
        summary.as_deref(),
        model,
        &chunks,
    )
    .await?;

    tracing::info!(source = %source.id, chunks = chunks.len(), "ingested");
    Ok(())
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
            "the analyzer model {} cannot read {} files; pin a vision-capable model",
            resolved.model.display_name,
            source.kind
        )));
    }

    let bytes = state.storage.read(&source.storage_path).await?;
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
    .max_tokens(resolved.model.max_output_tokens as u32)
    .effort(models::effort(&state.db, &state.user_id).await?);

    Ok(resolved.client(&state.http)?.chat(&req).await?.text)
}

fn kind_of(source: &Source) -> Kind {
    Kind::from_media_type(&source.media_type, source.original_filename.as_deref())
}
