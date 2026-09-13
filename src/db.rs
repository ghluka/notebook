//! SQLite access. Runtime queries only, so the build never needs a live DB.

use std::str::FromStr;

use serde::{Deserialize, Serialize};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::{Row, SqlitePool};

pub type Db = SqlitePool;

pub async fn connect(database_url: &str) -> anyhow::Result<Db> {
    let opts = SqliteConnectOptions::from_str(database_url)?
        .create_if_missing(true)
        .foreign_keys(true)
        .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal)
        .busy_timeout(std::time::Duration::from_secs(10));

    let pool = SqlitePoolOptions::new().max_connections(8).connect_with(opts).await?;
    sqlx::migrate!("./migrations").run(&pool).await?;
    Ok(pool)
}

pub fn now() -> String {
    chrono::Utc::now().to_rfc3339()
}

pub fn new_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

// ---------------------------------------------------------------- sources

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Source {
    pub id: String,
    pub owner_id: String,
    pub vault_id: Option<String>,
    pub folder_id: Option<String>,
    pub title: String,
    pub original_filename: Option<String>,
    pub kind: String,
    pub media_type: String,
    pub byte_size: i64,
    pub sha256: String,
    pub storage_path: String,
    pub status: String,
    pub error: Option<String>,
    /// The failure as JSON, when there was one. Same shape the API returns.
    pub error_detail: Option<String>,
    /// How far the analyzer has got, as "done/total", while it is working.
    pub progress: Option<String>,
    /// Finished sections of a long rendition, kept so a restart resumes.
    #[serde(skip_serializing)]
    pub partial: Option<String>,
    pub metadata: String,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone)]
pub struct NewSource {
    pub owner_id: String,
    pub vault_id: String,
    pub folder_id: Option<String>,
    pub title: String,
    pub original_filename: Option<String>,
    pub kind: String,
    pub media_type: String,
    pub byte_size: i64,
    pub sha256: String,
    pub storage_path: String,
}

pub async fn insert_source(db: &Db, s: NewSource) -> sqlx::Result<Source> {
    let id = new_id();
    let ts = now();
    sqlx::query(
        "INSERT INTO sources (id, owner_id, vault_id, folder_id, title, original_filename,
                              kind, media_type, byte_size, sha256, storage_path, status,
                              metadata, created_at, updated_at)
         VALUES (?1, ?10, ?12, ?11, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'pending', '{}', ?9, ?9)",
    )
    .bind(&id)
    .bind(&s.title)
    .bind(&s.original_filename)
    .bind(&s.kind)
    .bind(&s.media_type)
    .bind(s.byte_size)
    .bind(&s.sha256)
    .bind(&s.storage_path)
    .bind(&ts)
    .bind(&s.owner_id)
    .bind(&s.folder_id)
    .bind(&s.vault_id)
    .execute(db)
    .await?;

    get_source(db, &id).await?.ok_or(sqlx::Error::RowNotFound)
}

pub async fn get_source(db: &Db, id: &str) -> sqlx::Result<Option<Source>> {
    sqlx::query_as::<_, Source>("SELECT * FROM sources WHERE id = ?1")
        .bind(id)
        .fetch_optional(db)
        .await
}

pub async fn list_sources(db: &Db, vault: &str) -> sqlx::Result<Vec<Source>> {
    sqlx::query_as::<_, Source>(
        "SELECT * FROM sources WHERE vault_id = ?1 ORDER BY title COLLATE NOCASE",
    )
    .bind(vault)
    .fetch_all(db)
    .await
}

/// Rename a source or move it between folders. `None` leaves a field alone;
/// moving to the root is `folder_id: Some(None)`.
pub async fn update_source(
    db: &Db,
    id: &str,
    title: Option<&str>,
    folder_id: Option<Option<String>>,
) -> sqlx::Result<()> {
    if let Some(title) = title {
        sqlx::query("UPDATE sources SET title = ?2, updated_at = ?3 WHERE id = ?1")
            .bind(id)
            .bind(title)
            .bind(now())
            .execute(db)
            .await?;
    }
    if let Some(folder_id) = folder_id {
        sqlx::query("UPDATE sources SET folder_id = ?2, updated_at = ?3 WHERE id = ?1")
            .bind(id)
            .bind(folder_id)
            .bind(now())
            .execute(db)
            .await?;
    }
    Ok(())
}

// ---------------------------------------------------------------- folders

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Folder {
    pub id: String,
    pub owner_id: String,
    pub vault_id: Option<String>,
    pub parent_id: Option<String>,
    pub name: String,
    pub created_at: String,
}

pub async fn list_folders(db: &Db, vault: &str) -> sqlx::Result<Vec<Folder>> {
    sqlx::query_as::<_, Folder>(
        "SELECT * FROM folders WHERE vault_id = ?1 ORDER BY name COLLATE NOCASE",
    )
    .bind(vault)
    .fetch_all(db)
    .await
}

pub async fn create_folder(
    db: &Db,
    owner: &str,
    vault: &str,
    name: &str,
    parent_id: Option<&str>,
) -> sqlx::Result<Folder> {
    let id = new_id();
    sqlx::query(
        "INSERT INTO folders (id, owner_id, vault_id, parent_id, name, created_at)
         VALUES (?1, ?2, ?6, ?3, ?4, ?5)",
    )
    .bind(&id)
    .bind(owner)
    .bind(parent_id)
    .bind(name)
    .bind(now())
    .bind(vault)
    .execute(db)
    .await?;

    sqlx::query_as::<_, Folder>("SELECT * FROM folders WHERE id = ?1")
        .bind(&id)
        .fetch_one(db)
        .await
}

pub async fn update_folder(
    db: &Db,
    owner: &str,
    id: &str,
    name: Option<&str>,
    parent_id: Option<Option<String>>,
) -> sqlx::Result<()> {
    if let Some(name) = name {
        sqlx::query("UPDATE folders SET name = ?3 WHERE id = ?1 AND owner_id = ?2")
            .bind(id)
            .bind(owner)
            .bind(name)
            .execute(db)
            .await?;
    }
    if let Some(parent_id) = parent_id {
        sqlx::query("UPDATE folders SET parent_id = ?3 WHERE id = ?1 AND owner_id = ?2")
            .bind(id)
            .bind(owner)
            .bind(parent_id)
            .execute(db)
            .await?;
    }
    Ok(())
}

/// Deleting a folder keeps the files: they fall back to the root.
pub async fn delete_folder(db: &Db, owner: &str, id: &str) -> sqlx::Result<()> {
    sqlx::query("DELETE FROM folders WHERE id = ?1 AND owner_id = ?2")
        .bind(id)
        .bind(owner)
        .execute(db)
        .await?;
    Ok(())
}

// ----------------------------------------------------------------- vaults

/// A separate library: its own sources, folders and conversations. One is
/// open at a time, and that is the one every list and every search sees.
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Vault {
    pub id: String,
    pub owner_id: String,
    pub name: String,
    pub created_at: String,
}

/// A vault as the switcher lists it, with what it holds.
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct VaultSummary {
    pub id: String,
    pub name: String,
    pub created_at: String,
    pub source_count: i64,
    pub conversation_count: i64,
}

/// The setting that says which vault is open.
const ACTIVE_VAULT: &str = "vault";

pub async fn list_vaults(db: &Db, owner: &str) -> sqlx::Result<Vec<VaultSummary>> {
    sqlx::query_as::<_, VaultSummary>(
        "SELECT v.id, v.name, v.created_at,
                (SELECT count(*) FROM sources s WHERE s.vault_id = v.id) AS source_count,
                (SELECT count(*) FROM conversations c
                  WHERE c.vault_id = v.id
                    AND EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = c.id))
                    AS conversation_count
           FROM vaults v
          WHERE v.owner_id = ?1
          ORDER BY v.name COLLATE NOCASE",
    )
    .bind(owner)
    .fetch_all(db)
    .await
}

pub async fn get_vault(db: &Db, owner: &str, id: &str) -> sqlx::Result<Option<Vault>> {
    sqlx::query_as::<_, Vault>("SELECT * FROM vaults WHERE id = ?1 AND owner_id = ?2")
        .bind(id)
        .bind(owner)
        .fetch_optional(db)
        .await
}

pub async fn create_vault(db: &Db, owner: &str, name: &str) -> sqlx::Result<Vault> {
    let id = new_id();
    sqlx::query("INSERT INTO vaults (id, owner_id, name, created_at) VALUES (?1, ?2, ?3, ?4)")
        .bind(&id)
        .bind(owner)
        .bind(name)
        .bind(now())
        .execute(db)
        .await?;
    get_vault(db, owner, &id).await?.ok_or(sqlx::Error::RowNotFound)
}

pub async fn rename_vault(db: &Db, owner: &str, id: &str, name: &str) -> sqlx::Result<()> {
    sqlx::query("UPDATE vaults SET name = ?3 WHERE id = ?1 AND owner_id = ?2")
        .bind(id)
        .bind(owner)
        .bind(name)
        .execute(db)
        .await?;
    Ok(())
}

/// Everything in the vault goes with it: sources and their renditions,
/// folders, conversations. The caller deals with the files on disk.
pub async fn delete_vault(db: &Db, owner: &str, id: &str) -> sqlx::Result<()> {
    sqlx::query("DELETE FROM vaults WHERE id = ?1 AND owner_id = ?2")
        .bind(id)
        .bind(owner)
        .execute(db)
        .await?;
    Ok(())
}

/// The vault that is open. A setting that points at a deleted vault falls back
/// to the oldest one, and a user with none gets one, so there is always
/// somewhere for a file to land.
pub async fn active_vault(db: &Db, owner: &str) -> sqlx::Result<Vault> {
    if let Some(id) = crate::models::get_setting(db, owner, ACTIVE_VAULT).await?
        && let Some(vault) = get_vault(db, owner, &id).await?
    {
        return Ok(vault);
    }
    let oldest = sqlx::query_as::<_, Vault>(
        "SELECT * FROM vaults WHERE owner_id = ?1 ORDER BY created_at, name LIMIT 1",
    )
    .bind(owner)
    .fetch_optional(db)
    .await?;
    let vault = match oldest {
        Some(vault) => vault,
        None => create_vault(db, owner, "Library").await?,
    };
    open_vault(db, owner, &vault.id).await?;
    Ok(vault)
}

pub async fn open_vault(db: &Db, owner: &str, id: &str) -> sqlx::Result<()> {
    crate::models::set_setting(db, owner, ACTIVE_VAULT, id).await
}

/// Rows with no vault, which the migration leaves behind only when their owner
/// had no user row, go into this one rather than vanishing from every list.
pub async fn adopt_orphans(db: &Db, owner: &str, vault: &str) -> sqlx::Result<()> {
    for sql in [
        "UPDATE sources SET vault_id = ?2 WHERE vault_id IS NULL AND owner_id = ?1",
        "UPDATE folders SET vault_id = ?2 WHERE vault_id IS NULL AND owner_id = ?1",
        "UPDATE conversations SET vault_id = ?2 WHERE vault_id IS NULL AND owner_id = ?1",
    ] {
        sqlx::query(sql).bind(owner).bind(vault).execute(db).await?;
    }
    Ok(())
}

/// Move a source into another vault. Its folder stays behind, so it lands at
/// the root.
pub async fn move_source_to_vault(db: &Db, id: &str, vault: &str) -> sqlx::Result<()> {
    sqlx::query(
        "UPDATE sources SET vault_id = ?2, folder_id = NULL, updated_at = ?3 WHERE id = ?1",
    )
    .bind(id)
    .bind(vault)
    .bind(now())
    .execute(db)
    .await?;
    Ok(())
}

/// Move a folder into another vault with everything under it: subfolders keep
/// their shape and files stay where they were inside them. The folder itself
/// lands at the root. `UNION` rather than `UNION ALL`, so a cycle in the
/// parent links ends the walk instead of running it forever.
pub async fn move_folder_to_vault(db: &Db, owner: &str, id: &str, vault: &str) -> sqlx::Result<()> {
    let mut tx = db.begin().await?;
    sqlx::query(
        "WITH RECURSIVE sub(id) AS (
             SELECT id FROM folders WHERE id = ?1 AND owner_id = ?2
             UNION SELECT f.id FROM folders f JOIN sub ON f.parent_id = sub.id)
         UPDATE sources SET vault_id = ?3 WHERE folder_id IN (SELECT id FROM sub)",
    )
    .bind(id)
    .bind(owner)
    .bind(vault)
    .execute(&mut *tx)
    .await?;
    sqlx::query(
        "WITH RECURSIVE sub(id) AS (
             SELECT id FROM folders WHERE id = ?1 AND owner_id = ?2
             UNION SELECT f.id FROM folders f JOIN sub ON f.parent_id = sub.id)
         UPDATE folders SET vault_id = ?3 WHERE id IN (SELECT id FROM sub)",
    )
    .bind(id)
    .bind(owner)
    .bind(vault)
    .execute(&mut *tx)
    .await?;
    sqlx::query("UPDATE folders SET parent_id = NULL WHERE id = ?1 AND owner_id = ?2")
        .bind(id)
        .bind(owner)
        .execute(&mut *tx)
        .await?;
    tx.commit().await
}

pub async fn set_source_status(
    db: &Db,
    id: &str,
    status: &str,
    error: Option<&str>,
) -> sqlx::Result<()> {
    set_source_result(db, id, status, error, None).await
}

/// Status plus the structured failure, which the client reads to tell a rate
/// limit apart from a file it will never be able to read.
pub async fn set_source_result(
    db: &Db,
    id: &str,
    status: &str,
    error: Option<&str>,
    detail: Option<&str>,
) -> sqlx::Result<()> {
    sqlx::query(
        "UPDATE sources SET status = ?2, error = ?3, error_detail = ?4, updated_at = ?5,
                progress = CASE WHEN ?2 IN ('pending', 'analyzing') THEN progress END
          WHERE id = ?1",
    )
    .bind(id)
    .bind(status)
    .bind(error)
    .bind(detail)
    .bind(now())
    .execute(db)
    .await?;
    Ok(())
}

/// How far through a long file the analyzer is. Cleared when it finishes, so
/// a stale count never sits under a ready source.
pub async fn set_source_progress(
    db: &Db,
    id: &str,
    done: i64,
    total: i64,
) -> sqlx::Result<()> {
    sqlx::query("UPDATE sources SET progress = ?2 WHERE id = ?1")
        .bind(id)
        .bind(format!("{done}/{total}"))
        .execute(db)
        .await?;
    Ok(())
}

/// Save the work done so far on a long file, with the page it reached.
pub async fn save_partial(
    db: &Db,
    id: &str,
    markdown: &str,
    done: i64,
    total: i64,
) -> sqlx::Result<()> {
    sqlx::query("UPDATE sources SET partial = ?2, progress = ?3 WHERE id = ?1")
        .bind(id)
        .bind(markdown)
        .bind(format!("{done}/{total}"))
        .execute(db)
        .await?;
    Ok(())
}

pub async fn clear_partial(db: &Db, id: &str) -> sqlx::Result<()> {
    sqlx::query("UPDATE sources SET partial = NULL WHERE id = ?1")
        .bind(id)
        .execute(db)
        .await?;
    Ok(())
}

/// Sources waiting for the analyzer, oldest first. Used to pick work back up
/// after a restart, so nothing is stranded mid-queue.
pub async fn pending_sources(db: &Db) -> sqlx::Result<Vec<Source>> {
    sqlx::query_as::<_, Source>(
        "SELECT * FROM sources WHERE status IN ('pending', 'analyzing')
          ORDER BY created_at",
    )
    .fetch_all(db)
    .await
}

pub async fn delete_source(db: &Db, id: &str) -> sqlx::Result<()> {
    sqlx::query("DELETE FROM sources WHERE id = ?1").bind(id).execute(db).await?;
    Ok(())
}

/// How many sources still point at this blob, which decides whether the file
/// on disk can go.
pub async fn count_sources_with_sha(db: &Db, sha256: &str) -> sqlx::Result<i64> {
    let row = sqlx::query("SELECT count(*) AS n FROM sources WHERE sha256 = ?1")
        .bind(sha256)
        .fetch_one(db)
        .await?;
    Ok(row.get::<i64, _>("n"))
}

// -------------------------------------------------------------- documents

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Document {
    pub id: String,
    pub source_id: String,
    pub markdown: String,
    pub summary: Option<String>,
    pub analyzer_model: Option<String>,
    pub created_at: String,
}

pub async fn get_document(db: &Db, source_id: &str) -> sqlx::Result<Option<Document>> {
    sqlx::query_as::<_, Document>("SELECT * FROM documents WHERE source_id = ?1")
        .bind(source_id)
        .fetch_optional(db)
        .await
}

/// Take the rendition away, chunks and index entries with it, leaving the
/// source itself in place.
pub async fn delete_document(db: &Db, source_id: &str) -> sqlx::Result<()> {
    sqlx::query("DELETE FROM documents WHERE source_id = ?1")
        .bind(source_id)
        .execute(db)
        .await?;
    Ok(())
}

/// One rendition per source: replacing it drops the old chunks with it.
pub async fn replace_document(
    db: &Db,
    source_id: &str,
    markdown: &str,
    summary: Option<&str>,
    analyzer_model: Option<&str>,
    chunks: &[NewChunk],
) -> sqlx::Result<String> {
    let mut tx = db.begin().await?;
    sqlx::query("DELETE FROM documents WHERE source_id = ?1")
        .bind(source_id)
        .execute(&mut *tx)
        .await?;

    let doc_id = new_id();
    sqlx::query(
        "INSERT INTO documents (id, source_id, markdown, summary, analyzer_model, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
    )
    .bind(&doc_id)
    .bind(source_id)
    .bind(markdown)
    .bind(summary)
    .bind(analyzer_model)
    .bind(now())
    .execute(&mut *tx)
    .await?;

    for (i, c) in chunks.iter().enumerate() {
        sqlx::query(
            "INSERT INTO chunks (id, source_id, document_id, ordinal, heading, locator,
                                 content, token_estimate)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        )
        .bind(new_id())
        .bind(source_id)
        .bind(&doc_id)
        .bind(i as i64)
        .bind(&c.heading)
        .bind(&c.locator)
        .bind(&c.content)
        .bind((c.content.len() / 4) as i64)
        .execute(&mut *tx)
        .await?;
    }

    tx.commit().await?;
    Ok(doc_id)
}

// ----------------------------------------------------------------- chunks

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewChunk {
    pub heading: Option<String>,
    pub locator: Option<String>,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct SearchHit {
    pub chunk_id: String,
    pub source_id: String,
    pub source_title: String,
    pub heading: Option<String>,
    pub locator: Option<String>,
    pub ordinal: i64,
    /// The matched text with markers, for the search UI.
    pub snippet: String,
    /// The whole chunk. What a model is given: a worked example does not fit
    /// in a 32 token snippet, and an answer cannot be built from ellipses.
    pub content: String,
    pub score: f64,
}

const SEARCH_SQL: &str = "
    SELECT c.id           AS chunk_id,
           c.source_id    AS source_id,
           s.title        AS source_title,
           c.heading      AS heading,
           c.locator      AS locator,
           c.ordinal      AS ordinal,
           snippet(chunks_fts, 0, '**', '**', ' ... ', 32) AS snippet,
           c.content      AS content,
           bm25(chunks_fts) AS score
    FROM chunks_fts
    JOIN chunks  c ON c.id = chunks_fts.chunk_id
    JOIN sources s ON s.id = c.source_id
    WHERE chunks_fts MATCH ?1
      AND (?2 IS NULL OR c.source_id = ?2)
      AND s.vault_id = ?4
    ORDER BY score
    LIMIT ?3";

/// Free-text search over chunk renditions. All-terms matches come first, then
/// any-term matches fill up to the limit. Requiring every term in one chunk
/// fails exactly when the terms live in different files: "second" in some
/// proof and "reading" in the readings index, so the index never surfaces and
/// the model declares the file absent after one search. Only the given vault
/// is searched.
pub async fn search_chunks(
    db: &Db,
    vault: &str,
    query: &str,
    source_id: Option<&str>,
    limit: i64,
) -> sqlx::Result<Vec<SearchHit>> {
    let terms = fts_terms(query);
    if terms.is_empty() {
        return Ok(Vec::new());
    }

    let mut hits: Vec<SearchHit> = Vec::new();
    // One term is one query; running it twice only wastes time.
    let joiners: &[&str] = if terms.len() == 1 { &[" OR "] } else { &[" AND ", " OR "] };
    for joiner in joiners {
        let expr = terms.join(joiner);
        let batch = sqlx::query_as::<_, SearchHit>(SEARCH_SQL)
            .bind(&expr)
            .bind(source_id)
            .bind(limit)
            .bind(vault)
            .fetch_all(db)
            .await?;
        for hit in batch {
            if hits.len() >= limit as usize {
                break;
            }
            if !hits.iter().any(|h: &SearchHit| h.chunk_id == hit.chunk_id) {
                hits.push(hit);
            }
        }
        if hits.len() >= limit as usize {
            break;
        }
    }
    Ok(hits)
}

pub async fn chunks_for_source(db: &Db, source_id: &str) -> sqlx::Result<Vec<SearchHit>> {
    sqlx::query_as::<_, SearchHit>(
        "SELECT c.id AS chunk_id, c.source_id AS source_id, s.title AS source_title,
                c.heading AS heading, c.locator AS locator, c.ordinal AS ordinal,
                c.content AS snippet, c.content AS content, 0.0 AS score
         FROM chunks c JOIN sources s ON s.id = c.source_id
         WHERE c.source_id = ?1 ORDER BY c.ordinal",
    )
    .bind(source_id)
    .fetch_all(db)
    .await
}

/// Words that say how to answer rather than what to look for. A question like
/// "whats an lde, give an example and explain it in depth" otherwise buries the
/// one rare word that matters under chunks matching "give", "explain" and
/// "depth", which is exactly the ranking bm25 cannot save you from.
const STOPWORDS: &[&str] = &[
    "a", "about", "all", "am", "an", "and", "any", "are", "as", "at", "be", "been", "being",
    "but", "by", "can", "could", "describe", "did", "do", "does", "explain", "for", "from",
    "get", "give", "had", "has", "have", "he", "her", "him", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "just", "know", "let", "like", "make", "me", "mean", "more",
    "much", "my", "need", "of", "on", "one", "or", "our", "out", "please", "say", "she",
    "should", "show", "so", "some", "such", "tell", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "those", "to", "up", "us", "was", "we", "were",
    "what", "whats", "when", "where", "which", "while", "who", "why", "will", "with", "would",
    "you", "your", "depth", "detail", "please", "thanks",
];

/// The chunks either side of a hit.
///
/// A match often lands on the sentence that names a thing while the worked
/// example runs on into the next chunk, so a hit is read with its neighbours
/// rather than alone.
pub async fn chunks_around(
    db: &Db,
    source_id: &str,
    ordinal: i64,
    radius: i64,
) -> sqlx::Result<Vec<SearchHit>> {
    sqlx::query_as::<_, SearchHit>(
        "SELECT c.id AS chunk_id, c.source_id AS source_id, s.title AS source_title,
                c.heading AS heading, c.locator AS locator, c.ordinal AS ordinal,
                c.content AS snippet, c.content AS content, 0.0 AS score
           FROM chunks c JOIN sources s ON s.id = c.source_id
          WHERE c.source_id = ?1 AND c.ordinal BETWEEN ?2 AND ?3
          ORDER BY c.ordinal",
    )
    .bind(source_id)
    .bind(ordinal - radius)
    .bind(ordinal + radius)
    .fetch_all(db)
    .await
}

/// A slice of a rendition, by line, with line numbers kept.
///
/// Locators are lines of this same text, so "read around line 249" is how a
/// model follows a citation to the passage it came from.
pub struct DocumentSlice {
    pub title: String,
    pub from_line: usize,
    pub to_line: usize,
    pub total_lines: usize,
    pub text: String,
}

pub async fn read_document_lines(
    db: &Db,
    owner: &str,
    source_id: &str,
    from_line: usize,
    lines: usize,
    max_chars: usize,
) -> sqlx::Result<Option<DocumentSlice>> {
    let Some(source) = get_source(db, source_id).await? else { return Ok(None) };
    if source.owner_id != owner {
        return Ok(None);
    }
    let Some(doc) = get_document(db, source_id).await? else { return Ok(None) };

    let all: Vec<&str> = doc.markdown.lines().collect();
    let total_lines = all.len();
    let start = from_line.saturating_sub(1).min(total_lines);
    let end = (start + lines.max(1)).min(total_lines);

    let mut text = String::new();
    for (offset, line) in all[start..end].iter().enumerate() {
        let numbered = format!("{:>5}  {}\n", start + offset + 1, line);
        if text.len() + numbered.len() > max_chars {
            break;
        }
        text.push_str(&numbered);
    }

    Ok(Some(DocumentSlice {
        title: source.title,
        from_line: start + 1,
        to_line: end,
        total_lines,
        text,
    }))
}

/// Titles and summaries, for a model deciding where to look.
pub struct SourceBrief {
    pub id: String,
    pub title: String,
    pub kind: String,
    pub status: String,
    pub summary: Option<String>,
    pub total_lines: usize,
}

pub async fn source_briefs(db: &Db, vault: &str) -> sqlx::Result<Vec<SourceBrief>> {
    let sources = list_sources(db, vault).await?;
    let mut briefs = Vec::new();
    for source in sources {
        let doc = get_document(db, &source.id).await?;
        briefs.push(SourceBrief {
            id: source.id,
            title: source.title,
            kind: source.kind,
            status: source.status,
            summary: doc.as_ref().and_then(|d| d.summary.clone()),
            total_lines: doc.map(|d| d.markdown.lines().count()).unwrap_or(0),
        });
    }
    Ok(briefs)
}

/// Turn arbitrary user text into FTS5 terms. Everything is quoted, so operator
/// characters in a question can never produce a malformed MATCH expression.
///
/// Function words are dropped, unless that would leave nothing to search for.
fn fts_terms(query: &str) -> Vec<String> {
    let words: Vec<String> = query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.chars().count() > 1)
        .map(str::to_lowercase)
        .take(32)
        .collect();

    let content: Vec<&String> =
        words.iter().filter(|w| !STOPWORDS.contains(&w.as_str())).collect();

    let chosen: Vec<&String> = if content.is_empty() { words.iter().collect() } else { content };
    chosen.into_iter().take(24).map(|t| format!("\"{t}\"")).collect()
}

/// The words of a query worth searching for: what `fts_terms` keeps, without
/// its fallback to every word. How many there are tells a question that names
/// something from one that only points back at the conversation.
pub fn search_words(query: &str) -> Vec<String> {
    query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.chars().count() > 1)
        .map(str::to_lowercase)
        .filter(|w| !STOPWORDS.contains(&w.as_str()))
        .collect()
}

#[cfg(test)]
mod query_tests {
    use super::fts_terms;

    #[test]
    fn drops_the_words_that_bury_the_rare_one() {
        let terms = fts_terms("whats an lde, give an example and explain it in depth");
        assert_eq!(terms, vec!["\"lde\"", "\"example\""]);
    }

    #[test]
    fn a_question_of_only_function_words_still_searches() {
        let terms = fts_terms("what is it about");
        assert!(!terms.is_empty(), "dropping everything would return nothing at all");
    }

    #[test]
    fn quotes_everything_so_operators_cannot_leak() {
        let terms = fts_terms("NOT (a OR b) AND \"c\"");
        assert!(terms.iter().all(|t| t.starts_with('"') && t.ends_with('"')));
    }
}

#[cfg(test)]
mod search_tests {
    use super::*;

    async fn scratch() -> Db {
        let db = connect("sqlite::memory:").await.expect("in memory db");
        sqlx::query("INSERT INTO users (id, name, created_at) VALUES ('u', 'test', ?1)")
            .bind(now())
            .execute(&db)
            .await
            .expect("user");
        sqlx::query("INSERT INTO vaults (id, owner_id, name, created_at) VALUES ('v', 'u', 'test', ?1)")
            .bind(now())
            .execute(&db)
            .await
            .expect("vault");
        db
    }

    async fn add_source(db: &Db, title: &str, chunks: &[&str]) -> String {
        add_source_in(db, "v", None, title, chunks).await
    }

    async fn add_source_in(
        db: &Db,
        vault: &str,
        folder: Option<&str>,
        title: &str,
        chunks: &[&str],
    ) -> String {
        let source = insert_source(
            db,
            NewSource {
                owner_id: "u".into(),
                vault_id: vault.into(),
                folder_id: folder.map(str::to_string),
                title: title.into(),
                original_filename: Some(title.into()),
                kind: "text".into(),
                media_type: "text/markdown".into(),
                byte_size: 100,
                sha256: format!("sha-{title}"),
                storage_path: format!("ab/sha-{title}"),
            },
        )
        .await
        .expect("source");
        replace_document(
            db,
            &source.id,
            &chunks.join("\n\n"),
            None,
            None,
            &chunks
                .iter()
                .enumerate()
                .map(|(i, c)| NewChunk {
                    heading: None,
                    locator: Some(format!("line {}", i + 1)),
                    content: c.to_string(),
                })
                .collect::<Vec<_>>(),
        )
        .await
        .expect("document");
        source.id
    }

    /// The "second reading" failure: one term lives in a proof, the other in
    /// the readings index. All-terms alone returns only the proof; the merged
    /// search must also surface the index behind it.
    #[tokio::test]
    async fn any_term_hits_fill_in_behind_all_term_hits() {
        let db = scratch().await;
        add_source(&db, "readings.md", &["Here is the weekly reading list.", "Week 02 is Sets and Propositions."])
            .await;
        add_source(&db, "notes.pdf", &["Check the second item on the list."]).await;
        add_source(&db, "both.pdf", &["The second reading group meets Friday."]).await;

        let hits = search_chunks(&db, "v", "second reading", None, 10).await.expect("search");
        let titles: Vec<&str> = hits.iter().map(|h| h.source_title.as_str()).collect();

        assert_eq!(titles[0], "both.pdf", "the chunk with every term still ranks first");
        assert!(titles.contains(&"readings.md"), "the index is present: {titles:?}");
        assert!(titles.contains(&"notes.pdf"), "the single-term file is present: {titles:?}");
    }

    #[tokio::test]
    async fn single_term_queries_run_once() {
        let db = scratch().await;
        add_source(&db, "a.md", &["Something about induction."]).await;

        let hits = search_chunks(&db, "v", "induction", None, 10).await.expect("search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].source_title, "a.md");
    }

    /// A course's notes never answer a question asked in another course.
    #[tokio::test]
    async fn a_search_stays_in_its_vault() {
        let db = scratch().await;
        let other = create_vault(&db, "u", "other").await.expect("vault");
        add_source(&db, "here.md", &["Induction on the naturals."]).await;
        add_source_in(&db, &other.id, None, "there.md", &["Induction on trees."]).await;

        let hits = search_chunks(&db, "v", "induction", None, 10).await.expect("search");
        let titles: Vec<&str> = hits.iter().map(|h| h.source_title.as_str()).collect();
        assert_eq!(titles, vec!["here.md"]);
        assert_eq!(source_briefs(&db, &other.id).await.expect("briefs").len(), 1);
    }

    /// Moving a folder takes its subfolders and their files, and leaves the
    /// rest of the vault alone.
    #[tokio::test]
    async fn moving_a_folder_takes_everything_under_it() {
        let db = scratch().await;
        let other = create_vault(&db, "u", "other").await.expect("vault");
        let top = create_folder(&db, "u", "v", "MAT102", None).await.expect("folder");
        let inner = create_folder(&db, "u", "v", "week 1", Some(&top.id)).await.expect("folder");
        add_source_in(&db, "v", Some(&top.id), "syllabus.md", &["x"]).await;
        add_source_in(&db, "v", Some(&inner.id), "sets.md", &["y"]).await;
        add_source(&db, "unrelated.md", &["z"]).await;

        move_folder_to_vault(&db, "u", &top.id, &other.id).await.expect("move");

        let moved: Vec<String> = list_sources(&db, &other.id)
            .await
            .expect("list")
            .into_iter()
            .map(|s| s.title)
            .collect();
        assert_eq!(moved, vec!["sets.md", "syllabus.md"]);
        assert_eq!(list_folders(&db, &other.id).await.expect("folders").len(), 2);
        assert_eq!(list_sources(&db, "v").await.expect("list").len(), 1);
    }
}

// ---------------------------------------------------------- conversations

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Conversation {
    pub id: String,
    pub owner_id: String,
    /// The vault whose sources it searches. Opening it opens that vault.
    pub vault_id: Option<String>,
    pub title: String,
    /// Set once the conversation has been compacted. Stands in for every
    /// message marked `compacted` when the next prompt is assembled.
    pub summary: Option<String>,
    pub summarized_at: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

pub async fn create_conversation(
    db: &Db,
    owner: &str,
    vault: &str,
    title: &str,
) -> sqlx::Result<Conversation> {
    let id = new_id();
    let ts = now();
    sqlx::query(
        "INSERT INTO conversations (id, owner_id, vault_id, title, created_at, updated_at)
         VALUES (?1, ?2, ?5, ?3, ?4, ?4)",
    )
    .bind(&id)
    .bind(owner)
    .bind(title)
    .bind(&ts)
    .bind(vault)
    .execute(db)
    .await?;
    get_conversation(db, owner, &id).await?.ok_or(sqlx::Error::RowNotFound)
}

pub async fn get_conversation(
    db: &Db,
    owner: &str,
    id: &str,
) -> sqlx::Result<Option<Conversation>> {
    sqlx::query_as::<_, Conversation>(
        "SELECT * FROM conversations WHERE id = ?1 AND owner_id = ?2",
    )
    .bind(id)
    .bind(owner)
    .fetch_optional(db)
    .await
}

/// One row per conversation for the sidebar, newest first.
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct ConversationSummary {
    pub id: String,
    pub title: String,
    pub created_at: String,
    pub updated_at: String,
    pub message_count: i64,
    pub compacted: bool,
}

pub async fn list_conversations(
    db: &Db,
    owner: &str,
    vault: &str,
) -> sqlx::Result<Vec<ConversationSummary>> {
    sqlx::query_as::<_, ConversationSummary>(
        "SELECT c.id, c.title, c.created_at, c.updated_at,
                (SELECT count(*) FROM messages m WHERE m.conversation_id = c.id)
                    AS message_count,
                (c.summary IS NOT NULL) AS compacted
           FROM conversations c
          WHERE c.owner_id = ?1 AND c.vault_id = ?2
            AND EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = c.id)
          ORDER BY c.updated_at DESC",
    )
    .bind(owner)
    .bind(vault)
    .fetch_all(db)
    .await
}

pub async fn rename_conversation(db: &Db, owner: &str, id: &str, title: &str) -> sqlx::Result<()> {
    sqlx::query("UPDATE conversations SET title = ?3 WHERE id = ?1 AND owner_id = ?2")
        .bind(id)
        .bind(owner)
        .bind(title)
        .execute(db)
        .await?;
    Ok(())
}

pub async fn delete_conversation(db: &Db, owner: &str, id: &str) -> sqlx::Result<()> {
    sqlx::query("DELETE FROM conversations WHERE id = ?1 AND owner_id = ?2")
        .bind(id)
        .bind(owner)
        .execute(db)
        .await?;
    Ok(())
}

/// Fold everything said so far into `summary` and mark it compacted. The rows
/// stay so the transcript still reads in full.
pub async fn compact_conversation(
    db: &Db,
    owner: &str,
    id: &str,
    summary: &str,
) -> sqlx::Result<i64> {
    let mut tx = db.begin().await?;
    let marked = sqlx::query(
        "UPDATE messages SET compacted = 1
          WHERE compacted = 0
            AND conversation_id IN (SELECT id FROM conversations
                                     WHERE id = ?1 AND owner_id = ?2)",
    )
    .bind(id)
    .bind(owner)
    .execute(&mut *tx)
    .await?
    .rows_affected() as i64;

    sqlx::query(
        "UPDATE conversations SET summary = ?3, summarized_at = ?4, updated_at = ?4
          WHERE id = ?1 AND owner_id = ?2",
    )
    .bind(id)
    .bind(owner)
    .bind(summary)
    .bind(now())
    .execute(&mut *tx)
    .await?;

    tx.commit().await?;
    Ok(marked)
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct StoredMessage {
    pub id: String,
    pub conversation_id: String,
    pub role: String,
    pub content: String,
    pub tool_calls: Option<String>,
    pub citations: Option<String>,
    pub compacted: bool,
    pub model: Option<String>,
    /// Present on assistant turns, so reopening a conversation can restore the
    /// context meter rather than showing zero.
    pub input_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
    /// The model's reasoning before this answer, and how long it took, when it
    /// exposed any. Shown on reopening, never sent back to the model.
    pub thinking: Option<String>,
    pub thinking_ms: Option<i64>,
    /// The steps the turn took before this answer, as JSON: stretches of
    /// reasoning and tool calls, in order. See `routes::chat::TraceStep`.
    pub trace: Option<String>,
    /// The files a question was asked about, as JSON: id, title and kind,
    /// kept as they were when it was asked.
    pub attachments: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

pub async fn append_message(
    db: &Db,
    conversation_id: &str,
    role: &str,
    content: &str,
    citations: Option<&str>,
    model: Option<&str>,
    usage: Option<Usage>,
) -> sqlx::Result<String> {
    let id = new_id();
    sqlx::query(
        "INSERT INTO messages (id, conversation_id, role, content, citations, model,
                               input_tokens, output_tokens, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
    )
    .bind(&id)
    .bind(conversation_id)
    .bind(role)
    .bind(content)
    .bind(citations)
    .bind(model)
    .bind(usage.map(|u| u.input_tokens as i64))
    .bind(usage.map(|u| u.output_tokens as i64))
    .bind(now())
    .execute(db)
    .await?;
    sqlx::query("UPDATE conversations SET updated_at = ?2 WHERE id = ?1")
        .bind(conversation_id)
        .bind(now())
        .execute(db)
        .await?;
    Ok(id)
}

/// The reasoning behind an assistant turn, kept beside the answer rather than
/// in it: shown when the conversation is reopened, never sent back to the
/// model, never searched.
pub async fn set_message_thinking(
    db: &Db,
    id: &str,
    thinking: &str,
    thinking_ms: u64,
) -> sqlx::Result<()> {
    sqlx::query("UPDATE messages SET thinking = ?2, thinking_ms = ?3 WHERE id = ?1")
        .bind(id)
        .bind(thinking)
        .bind(thinking_ms as i64)
        .execute(db)
        .await?;
    Ok(())
}

/// The steps behind an assistant turn, kept for display like its reasoning.
pub async fn set_message_trace(db: &Db, id: &str, trace: &str) -> sqlx::Result<()> {
    sqlx::query("UPDATE messages SET trace = ?2 WHERE id = ?1")
        .bind(id)
        .bind(trace)
        .execute(db)
        .await?;
    Ok(())
}

/// The files a question was asked about, kept with it.
pub async fn set_message_attachments(db: &Db, id: &str, files: &str) -> sqlx::Result<()> {
    sqlx::query("UPDATE messages SET attachments = ?2 WHERE id = ?1")
        .bind(id)
        .bind(files)
        .execute(db)
        .await?;
    Ok(())
}

/// Everything ever said, for display.
pub async fn conversation_messages(db: &Db, id: &str) -> sqlx::Result<Vec<StoredMessage>> {
    sqlx::query_as::<_, StoredMessage>(
        "SELECT * FROM messages WHERE conversation_id = ?1 ORDER BY created_at, id",
    )
    .bind(id)
    .fetch_all(db)
    .await
}

/// Unasking a question: the question, whatever it produced and everything said
/// after it all go, and the question comes back so the caller can decide what
/// to do with it. Point at an answer and the question behind it is the one that
/// goes; point at a question and it is that one. Returns None when the message
/// is not there, or when nothing at or before it was a question.
#[cfg(test)]
pub async fn rewind_to_question(
    db: &Db,
    conversation_id: &str,
    message_id: &str,
) -> sqlx::Result<Option<(String, usize)>> {
    Ok(rewind_question(db, conversation_id, message_id).await?.map(|(m, n)| (m.content, n)))
}

/// Unasking a question, handing back the whole of it: its text and the files
/// it was asked about, so asking it again asks it about the same files.
pub async fn rewind_question(
    db: &Db,
    conversation_id: &str,
    message_id: &str,
) -> sqlx::Result<Option<(StoredMessage, usize)>> {
    let messages = conversation_messages(db, conversation_id).await?;
    let Some(at) = messages.iter().position(|m| m.id == message_id) else {
        return Ok(None);
    };
    let Some(question) = messages[..=at].iter().rposition(|m| m.role == "user") else {
        return Ok(None);
    };

    let doomed = &messages[question..];
    for m in doomed {
        sqlx::query("DELETE FROM messages WHERE id = ?1").bind(&m.id).execute(db).await?;
    }
    sqlx::query("UPDATE conversations SET updated_at = ?2 WHERE id = ?1")
        .bind(conversation_id)
        .bind(now())
        .execute(db)
        .await?;

    Ok(Some((messages[question].clone(), doomed.len())))
}

/// What the next prompt should carry: only what has not been compacted away.
pub async fn active_messages(db: &Db, id: &str) -> sqlx::Result<Vec<StoredMessage>> {
    sqlx::query_as::<_, StoredMessage>(
        "SELECT * FROM messages
          WHERE conversation_id = ?1 AND compacted = 0
          ORDER BY created_at, id",
    )
    .bind(id)
    .fetch_all(db)
    .await
}

#[cfg(test)]
mod rewind_tests {
    use super::*;

    async fn scratch() -> (Db, String) {
        let db = connect("sqlite::memory:").await.expect("in memory db");
        sqlx::query("INSERT INTO users (id, name, created_at) VALUES ('u', 'test', ?1)")
            .bind(now())
            .execute(&db)
            .await
            .expect("user");
        let vault = create_vault(&db, "u", "test").await.expect("vault");
        let c = create_conversation(&db, "u", &vault.id, "test").await.expect("conversation");
        (db, c.id)
    }

    async fn say(db: &Db, id: &str, role: &str, text: &str) -> String {
        append_message(db, id, role, text, None, None, None).await.expect("message")
    }

    /// A retried or edited question asks about the same files it did before.
    #[tokio::test]
    async fn a_rewound_question_keeps_its_files() {
        let (db, c) = scratch().await;
        let asked = say(&db, &c, "user", "what is a set?").await;
        say(&db, &c, "assistant", "a collection").await;
        let files = r#"[{"id":"s1","title":"symbols.md","kind":"text"}]"#;
        set_message_attachments(&db, &asked, files).await.expect("files");

        let (question, removed) =
            rewind_question(&db, &c, &asked).await.expect("query").expect("rewound");
        assert_eq!(question.content, "what is a set?");
        assert_eq!(question.attachments.as_deref(), Some(files));
        assert_eq!(removed, 2);
    }

    #[tokio::test]
    async fn rewind_takes_the_answer_its_question_and_what_followed() {
        let (db, c) = scratch().await;
        say(&db, &c, "user", "first question").await;
        say(&db, &c, "assistant", "first answer").await;
        say(&db, &c, "user", "second question").await;
        let target = say(&db, &c, "assistant", "second answer").await;
        say(&db, &c, "user", "third question").await;
        say(&db, &c, "assistant", "third answer").await;

        let (question, removed) =
            rewind_to_question(&db, &c, &target).await.expect("query").expect("rewound");
        assert_eq!(question, "second question");
        assert_eq!(removed, 4);

        let left = conversation_messages(&db, &c).await.expect("messages");
        let texts: Vec<&str> = left.iter().map(|m| m.content.as_str()).collect();
        assert_eq!(texts, vec!["first question", "first answer"]);
    }

    #[tokio::test]
    async fn rewinding_at_a_question_takes_that_question() {
        let (db, c) = scratch().await;
        say(&db, &c, "user", "first question").await;
        say(&db, &c, "assistant", "first answer").await;
        let target = say(&db, &c, "user", "second question").await;
        say(&db, &c, "assistant", "second answer").await;

        let (question, removed) =
            rewind_to_question(&db, &c, &target).await.expect("query").expect("rewound");
        assert_eq!(question, "second question");
        assert_eq!(removed, 2);

        let left = conversation_messages(&db, &c).await.expect("messages");
        assert_eq!(left.len(), 2);
    }

    #[tokio::test]
    async fn an_answer_with_no_question_behind_it_cannot_be_retried() {
        let (db, c) = scratch().await;
        let orphan = say(&db, &c, "assistant", "an answer to nothing").await;
        assert!(rewind_to_question(&db, &c, &orphan).await.expect("query").is_none());
        assert!(rewind_to_question(&db, &c, "no such message").await.expect("query").is_none());
        assert_eq!(conversation_messages(&db, &c).await.expect("messages").len(), 1);
    }
}
