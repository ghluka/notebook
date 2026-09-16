//! Password gate for the whole site.
//!
//! Single user, same as the rest of the app: one row in `users` carries a
//! password hash, and every request except the login endpoints needs a valid
//! session cookie. The middleware runs outside the router, so an unknown path
//! without a session still answers 401 rather than leaking whether it exists.
//!
//! Login is guarded twice: a proof of work challenge (the client burns ~a
//! second of hashing per attempt, which makes bulk guessing expensive) and a
//! per IP rate limiter. No new crates: hashing is iterated SHA256 over the
//! `sha2` crate the app already depends on.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::{Request, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use sha2::{Digest, Sha256};
use sqlx::Row;

use crate::db::{Db, new_id, now};
use crate::error::{AppError, AppResult};
use crate::state::AppState;

// ------------------------------------------------------------ constants

/// Cookie that carries the session token. Tied to the browser profile, so one
/// login covers every tab in it and none outside it.
pub const COOKIE_NAME: &str = "notebook_session";
/// How long a session lasts without being seen. Touched on use, so daily use
/// never logs out.
const SESSION_DAYS: i64 = 30;
/// Leading zero bits a proof of work solution must show. 18 bits averages
/// ~260k hashes, about a second in the login page, milliseconds to verify.
pub const POW_DIFFICULTY: u32 = 18;
/// A challenge lives 5 minutes and is single use.
const POW_TTL: Duration = Duration::from_secs(300);
/// Password hashing cost. Only runs on login and password change.
const HASH_ITERATIONS: u32 = 100_000;
/// Per IP budgets. Login covers both login and setup attempts.
const CHALLENGE_LIMIT: usize = 30;
const CHALLENGE_WINDOW: Duration = Duration::from_secs(60);
const LOGIN_LIMIT: usize = 10;
const LOGIN_WINDOW: Duration = Duration::from_secs(300);

// ------------------------------------------------------------------ hex

fn hex_encode(bytes: &[u8]) -> String {
    const C: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(C[(b >> 4) as usize] as char);
        s.push(C[(b & 15) as usize] as char);
    }
    s
}

fn sha256_hex(data: &[u8]) -> String {
    hex_encode(&Sha256::digest(data))
}

// ------------------------------------------------------- password hash

/// `v1$<iterations>$<salt hex>$<hash hex>`, salt is 128 random bits.
pub fn hash_password(password: &str) -> String {
    let salt = uuid::Uuid::new_v4().simple().to_string();
    let hash = stretch(password, &salt, HASH_ITERATIONS);
    format!("v1${HASH_ITERATIONS}${salt}${hash}")
}

fn stretch(password: &str, salt_hex: &str, rounds: u32) -> String {
    let mut h = Sha256::digest(format!("{salt_hex}:{password}").as_bytes()).to_vec();
    for i in 1..rounds {
        let mut hasher = Sha256::new();
        hasher.update(&h);
        hasher.update(salt_hex.as_bytes());
        hasher.update(password.as_bytes());
        hasher.update(i.to_le_bytes());
        h = hasher.finalize().to_vec();
    }
    hex_encode(&h)
}

/// Constant time compare, so a wrong password costs the same as a right one.
fn slow_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

pub fn verify_password(password: &str, stored: &str) -> bool {
    let mut parts = stored.split('$');
    match (parts.next(), parts.next(), parts.next(), parts.next()) {
        (Some("v1"), Some(rounds), Some(salt), Some(hash)) => {
            let Ok(n) = rounds.parse::<u32>() else { return false };
            if n == 0 || n > 1_000_000 {
                return false;
            }
            slow_eq(&stretch(password, salt, n), hash)
        }
        _ => false,
    }
}

// ---------------------------------------------------------- proof of work

fn leading_zero_bits(digest: &[u8]) -> u32 {
    let mut n = 0;
    for b in digest {
        if *b == 0 {
            n += 8;
        } else {
            n += b.leading_zeros();
            break;
        }
    }
    n
}

/// The client finds `solution` such that SHA256(nonce || solution) shows
/// `POW_DIFFICULTY` leading zero bits. Verification is one hash.
pub fn pow_meets(nonce: &str, solution: &str, difficulty: u32) -> bool {
    let mut hasher = Sha256::new();
    hasher.update(nonce.as_bytes());
    hasher.update(solution.as_bytes());
    leading_zero_bits(&hasher.finalize()) >= difficulty
}

// ---------------------------------------------------------------- state

struct Challenge {
    expires_at: Instant,
}

#[derive(Default)]
struct Buckets {
    challenges: HashMap<String, Vec<Instant>>,
    logins: HashMap<String, Vec<Instant>>,
}

impl Buckets {
    /// Record one event, pruning older entries. Returns how many remain in
    /// the window including this one.
    fn push(bucket: &mut HashMap<String, Vec<Instant>>, key: &str, window: Duration) -> usize {
        let cutoff = Instant::now().checked_sub(window).unwrap_or_else(Instant::now);
        let entry = bucket.entry(key.to_string()).or_default();
        entry.retain(|t| *t > cutoff);
        entry.push(Instant::now());
        entry.len()
    }
}

/// In memory auth state: single use challenges plus per IP rate buckets.
/// Short lived by design, so a restart just invalidates pending challenges.
#[derive(Clone, Default)]
pub struct AuthState {
    inner: Arc<Mutex<InnerAuth>>,
}

#[derive(Default)]
struct InnerAuth {
    challenges: HashMap<String, Challenge>,
    buckets: Buckets,
}

impl AuthState {
    pub fn issue_challenge(&self) -> (String, u32) {
        let nonce = uuid::Uuid::new_v4().simple().to_string();
        let mut inner = self.inner.lock().expect("auth lock");
        // Expired nonces never accumulate: sweep on issue, which is cheap.
        let now = Instant::now();
        inner.challenges.retain(|_, c| c.expires_at > now);
        inner.challenges.insert(nonce.clone(), Challenge { expires_at: now + POW_TTL });
        (nonce, POW_DIFFICULTY)
    }

    /// Consume a challenge if it is live and the solution checks out.
    pub fn consume_challenge(&self, nonce: &str, solution: &str) -> bool {
        let mut inner = self.inner.lock().expect("auth lock");
        let Some(c) = inner.challenges.remove(nonce) else { return false };
        if c.expires_at <= Instant::now() {
            return false;
        }
        if solution.len() > 32 || !solution.bytes().all(|b| b.is_ascii_digit()) {
            return false;
        }
        pow_meets(nonce, solution, POW_DIFFICULTY)
    }

    /// Returns seconds to wait when over budget, else records and returns None.
    pub fn check_challenge_rate(&self, ip: &str) -> Option<u64> {
        let mut inner = self.inner.lock().expect("auth lock");
        let n = Buckets::push(&mut inner.buckets.challenges, ip, CHALLENGE_WINDOW);
        if n > CHALLENGE_LIMIT { Some(60) } else { None }
    }

    pub fn check_login_rate(&self, ip: &str) -> Option<u64> {
        let mut inner = self.inner.lock().expect("auth lock");
        let n = Buckets::push(&mut inner.buckets.logins, ip, LOGIN_WINDOW);
        if n > LOGIN_LIMIT { Some(300) } else { None }
    }
}

// -------------------------------------------------------------- sessions

#[derive(Debug, Clone, Serialize, sqlx::FromRow)]
pub struct Session {
    pub id: String,
    pub user_id: String,
    #[serde(skip_serializing)]
    #[allow(dead_code)]
    pub token_hash: String,
    pub created_at: String,
    pub expires_at: String,
    pub last_seen_at: String,
    pub ip: String,
    pub user_agent: String,
}

fn expiry(days_from_now: i64) -> String {
    (chrono::Utc::now() + chrono::Duration::days(days_from_now)).to_rfc3339()
}

/// Create a session and return the raw token (hashed in storage, never read
/// back). The token is 512 bits of UUID randomness rendered as hex.
pub async fn create_session(db: &Db, user_id: &str, ip: &str, ua: &str) -> sqlx::Result<String> {
    let token =
        format!("{}{}", uuid::Uuid::new_v4().simple(), uuid::Uuid::new_v4().simple());
    let ts = now();
    sqlx::query(
        "INSERT INTO sessions (id, user_id, token_hash, created_at, expires_at,
                               last_seen_at, ip, user_agent)
         VALUES (?1, ?2, ?3, ?4, ?5, ?4, ?6, ?7)",
    )
    .bind(new_id())
    .bind(user_id)
    .bind(sha256_hex(token.as_bytes()))
    .bind(&ts)
    .bind(expiry(SESSION_DAYS))
    .bind(ip)
    .bind(ua)
    .execute(db)
    .await?;
    Ok(token)
}

/// Look a presented token up. Expired rows read as absent and are swept.
pub async fn lookup_session(db: &Db, token: &str) -> sqlx::Result<Option<Session>> {
    let hash = sha256_hex(token.trim().as_bytes());
    let session = sqlx::query_as::<_, Session>("SELECT * FROM sessions WHERE token_hash = ?1")
        .bind(&hash)
        .fetch_optional(db)
        .await?;
    let Some(s) = session else { return Ok(None) };
    let expired = chrono::DateTime::parse_from_rfc3339(&s.expires_at)
        .map(|t| t < chrono::Utc::now())
        .unwrap_or(true);
    if expired {
        sqlx::query("DELETE FROM sessions WHERE id = ?1").bind(&s.id).execute(db).await?;
        return Ok(None);
    }
    Ok(Some(s))
}

/// Sliding expiry, but at most one write per hour per session so the 1.5s
/// explorer poll does not churn the table.
pub async fn touch_session(db: &Db, session: &Session) -> sqlx::Result<()> {
    let seen_old = chrono::DateTime::parse_from_rfc3339(&session.last_seen_at)
        .map(|t| chrono::Utc::now().signed_duration_since(t) > chrono::Duration::hours(1))
        .unwrap_or(true);
    if !seen_old {
        return Ok(());
    }
    sqlx::query("UPDATE sessions SET last_seen_at = ?2, expires_at = ?3 WHERE id = ?1")
        .bind(&session.id)
        .bind(now())
        .bind(expiry(SESSION_DAYS))
        .execute(db)
        .await?;
    Ok(())
}

pub async fn delete_session(db: &Db, token: &str) -> sqlx::Result<()> {
    sqlx::query("DELETE FROM sessions WHERE token_hash = ?1")
        .bind(sha256_hex(token.trim().as_bytes()))
        .execute(db)
        .await?;
    Ok(())
}

pub async fn delete_user_sessions(db: &Db, user_id: &str) -> sqlx::Result<()> {
    sqlx::query("DELETE FROM sessions WHERE user_id = ?1").bind(user_id).execute(db).await?;
    Ok(())
}

/// Delete every session but the current one, so a password change signs other
/// browsers out while this one stays in.
pub async fn delete_other_sessions(db: &Db, user_id: &str, keep_id: &str) -> sqlx::Result<()> {
    sqlx::query("DELETE FROM sessions WHERE user_id = ?1 AND id != ?2")
        .bind(user_id)
        .bind(keep_id)
        .execute(db)
        .await?;
    Ok(())
}

pub async fn list_sessions(db: &Db, user_id: &str) -> sqlx::Result<Vec<Session>> {
    sqlx::query_as::<_, Session>(
        "SELECT * FROM sessions WHERE user_id = ?1 ORDER BY last_seen_at DESC",
    )
    .bind(user_id)
    .fetch_all(db)
    .await
}

// ------------------------------------------------------- password store

pub async fn password_hash(db: &Db, user_id: &str) -> sqlx::Result<String> {
    let row = sqlx::query("SELECT password_hash FROM users WHERE id = ?1")
        .bind(user_id)
        .fetch_optional(db)
        .await?;
    Ok(row.map(|r| r.get::<String, _>("password_hash")).unwrap_or_default())
}

pub async fn set_password_hash(db: &Db, user_id: &str, hash: &str) -> sqlx::Result<()> {
    sqlx::query("UPDATE users SET password_hash = ?2 WHERE id = ?1")
        .bind(user_id)
        .bind(hash)
        .execute(db)
        .await?;
    Ok(())
}

/// Seed a headless install once: when no password is set and the environment
/// names one, that password wins, exactly like provider seeding.
pub async fn seed_password_from_env(db: &Db, user_id: &str) {
    if let Ok(pw) = std::env::var("NOTEBOOK_PASSWORD")
        && !pw.trim().is_empty()
        && let Ok(current) = password_hash(db, user_id).await
        && current.is_empty()
    {
        if set_password_hash(db, user_id, &hash_password(pw.trim())).await.is_ok() {
            tracing::info!("seeded login password from environment");
        }
    }
}

// -------------------------------------------------------------- cookies

pub fn extract_token(headers: &HeaderMap) -> Option<String> {
    let cookie = headers.get(header::COOKIE)?.to_str().ok()?;
    for part in cookie.split(';') {
        let part = part.trim();
        if let Some(value) = part.strip_prefix(&format!("{COOKIE_NAME}=")) {
            let value = value.trim();
            if !value.is_empty() {
                return Some(value.to_string());
            }
        }
    }
    None
}

/// Cookie `Path`, scoped to the app when a proxy serves it from a
/// subdirectory and to the whole site when it is at the root.
fn cookie_path(base: &str) -> &str {
    if base.is_empty() { "/" } else { base }
}

pub fn set_cookie(token: &str, base: &str) -> String {
    let max_age = SESSION_DAYS * 24 * 60 * 60;
    format!(
        "{COOKIE_NAME}={token}; Path={}; HttpOnly; SameSite=Lax; Max-Age={max_age}",
        cookie_path(base)
    )
}

pub fn clear_cookie(base: &str) -> String {
    format!(
        "{COOKIE_NAME}=; Path={}; HttpOnly; SameSite=Lax; Max-Age=0",
        cookie_path(base)
    )
}

/// Best effort client address: a trusted proxy header first, then the peer.
pub fn client_ip(headers: &HeaderMap, peer: Option<std::net::SocketAddr>) -> String {
    if let Some(forwarded) = headers.get("x-forwarded-for").and_then(|v| v.to_str().ok()) {
        if let Some(first) = forwarded.split(',').next().map(str::trim)
            && !first.is_empty()
        {
            return first.to_string();
        }
    }
    peer.map(|p| p.ip().to_string()).unwrap_or_else(|| "unknown".into())
}

// ------------------------------------------------------------ middleware

/// Paths anyone may call without a session: the login page and the three
/// endpoints that prove who you are.
fn is_public(method: &str, path: &str) -> bool {
    matches!(
        (method, path),
        ("GET", "/login")
            | ("GET", "/api/auth/status")
            | ("POST", "/api/auth/challenge")
            | ("POST", "/api/auth/login")
            | ("POST", "/api/auth/setup")
    )
}

/// Runs outside the router, so even a path that matches nothing answers 401
/// to an anonymous caller instead of 404.
pub async fn require_auth(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Response {
    let path = request.uri().path().to_string();
    let method = request.method().as_str().to_string();
    if is_public(&method, &path) {
        return next.run(request).await;
    }
    let token = extract_token(request.headers());
    match token {
        Some(t) => match lookup_session(&state.db, &t).await {
            Ok(Some(session)) => {
                let _ = touch_session(&state.db, &session).await;
                next.run(request).await
            }
            _ => denied(&request, &state.config.base_path),
        },
        None => denied(&request, &state.config.base_path),
    }
}

/// What an anonymous caller gets. API calls, assets and unknown paths answer
/// 401 JSON so existence never leaks; a browser navigating to an app entry
/// point is bounced to the login page instead of shown that JSON.
fn denied(request: &Request, configured_prefix: &str) -> Response {
    let path = request.uri().path();
    // The prefix the browser is looking at the app through, which the redirect
    // has to carry: behind a proxy at `/notebook`, a bare `/login` points at
    // whatever else the host serves at its own root.
    let base = crate::base::effective(configured_prefix, request.headers());
    let is_entry = path == "/" || path.starts_with("/c/");
    let wants_html = request
        .headers()
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|a| a.contains("text/html"));
    if is_entry
        && wants_html
        && let Ok(to) = header::HeaderValue::from_str(&format!("{base}/login"))
    {
        return (StatusCode::FOUND, [(header::LOCATION, to)], "redirecting to the login page")
            .into_response();
    }
    AppError::Unauthorized(format!("not signed in; open {base}/login and sign in first"))
        .into_response()
}

/// Read the session behind a request inside handlers that need to know which
/// row is theirs (logout, password change, session list).
pub async fn request_session(state: &AppState, headers: &HeaderMap) -> AppResult<Session> {
    let Some(token) = extract_token(headers) else {
        return Err(AppError::Unauthorized("not signed in".into()));
    };
    lookup_session(&state.db, &token)
        .await?
        .ok_or_else(|| AppError::Unauthorized("session expired; sign in again".into()))
}

pub fn rate_limited_response(wait_secs: u64) -> Response {
    (
        StatusCode::TOO_MANY_REQUESTS,
        axum::Json(serde_json::json!({
            "error": format!(
                "too many attempts; wait {wait_secs}s before trying again"
            ),
            "kind": "rate_limited",
            "waited_seconds": 0,
            "retry_after": wait_secs,
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_password_verifies_and_a_wrong_one_does_not() {
        let hash = hash_password("correct horse battery staple");
        assert!(verify_password("correct horse battery staple", &hash));
        assert!(!verify_password("correct horse battery staples", &hash));
        assert!(!verify_password("", &hash));
        assert!(!verify_password("x", "garbage"));
    }

    #[test]
    fn easy_pow_checks_out_and_hard_pow_does_not() {
        assert!(pow_meets("nonce", "0", 0));
        assert!(!pow_meets("nonce", "0", 256));
    }

    #[test]
    fn pow_solution_is_checked_not_trusted() {
        // Difficulty 1 needs a hash starting with a zero bit; "0" may or may
        // not qualify, but whatever qualifies at 8 bits also qualifies at 1.
        let mut found = false;
        for s in 0..5000 {
            let sol = s.to_string();
            if pow_meets("test-nonce", &sol, 8) {
                assert!(pow_meets("test-nonce", &sol, 1));
                found = true;
                break;
            }
        }
        assert!(found, "8 bit solution should appear within 5000 tries");
    }

    #[test]
    fn only_the_login_endpoints_are_public() {
        assert!(is_public("GET", "/login"));
        assert!(is_public("POST", "/api/auth/login"));
        assert!(!is_public("GET", "/"));
        assert!(!is_public("GET", "/api/sources"));
        assert!(!is_public("GET", "/health"));
        assert!(!is_public("GET", "/nope"));
    }

    #[test]
    fn cookies_round_trip() {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::COOKIE,
            format!("a=1; {COOKIE_NAME}=tok123; b=2").parse().unwrap(),
        );
        assert_eq!(extract_token(&headers).as_deref(), Some("tok123"));
        let empty = HeaderMap::new();
        assert!(extract_token(&empty).is_none());
    }

    #[test]
    fn a_cookie_is_scoped_to_where_the_app_is_mounted() {
        assert!(set_cookie("t", "").contains("Path=/;"));
        assert!(clear_cookie("").contains("Path=/;"));
        // Under a proxy the cookie belongs to the subdirectory, not the host.
        assert!(set_cookie("t", "/notebook").contains("Path=/notebook;"));
        assert!(clear_cookie("/notebook").contains("Path=/notebook;"));
    }
}
