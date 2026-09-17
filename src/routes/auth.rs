use axum::Json;
use axum::extract::State;
use axum::http::{HeaderMap, header};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::auth::{
    self, clear_cookie, client_ip, create_session, delete_other_sessions, delete_session,
    delete_user_sessions, hash_password, list_sessions, password_hash, rate_limited_response,
    request_session, set_cookie, set_password_hash, verify_password,
};
use crate::error::{AppError, AppResult};
use crate::models;
use crate::state::AppState;

fn peer(headers: &HeaderMap) -> String {
    client_ip(headers, None)
}

fn base(state: &AppState, headers: &HeaderMap) -> String {
    crate::base::effective(&state.config.base_path, headers)
}

fn user_agent(headers: &HeaderMap) -> String {
    headers
        .get(header::USER_AGENT)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .chars()
        .take(300)
        .collect()
}

pub async fn status(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> AppResult<Json<serde_json::Value>> {
    let setup_required = password_hash(&state.db, &state.user_id).await?.is_empty();
    let authenticated = match auth::extract_token(&headers) {
        Some(t) => auth::lookup_session(&state.db, &t).await?.is_some(),
        None => false,
    };
    Ok(Json(json!({
        "authenticated": authenticated,
        "setup_required": setup_required,
        "auth": if authenticated { "session" } else { "local" },
    })))
}

#[derive(Serialize)]
pub struct Challenge {
    nonce: String,
    difficulty: u32,
    expires_in: u64,
}

pub async fn challenge(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> AppResult<Response> {
    let ip = peer(&headers);
    if let Some(wait) = state.auth.check_challenge_rate(&ip) {
        return Ok(rate_limited_response(wait));
    }
    let (nonce, difficulty) = state.auth.issue_challenge();
    Ok(Json(Challenge { nonce, difficulty, expires_in: 300 }).into_response())
}

#[derive(Deserialize)]
pub struct LoginBody {
    #[serde(default)]
    nonce: String,
    #[serde(default)]
    solution: String,
    #[serde(default)]
    password: String,
}

fn check_pow(state: &AppState, body: &LoginBody) -> AppResult<()> {
    if body.nonce.is_empty() || body.solution.is_empty() {
        return Err(AppError::BadRequest(
            "a fresh proof of work is required with every attempt".into(),
        ));
    }
    if !state.auth.consume_challenge(&body.nonce, &body.solution) {
        return Err(AppError::BadRequest(
            "that proof of work is missing, expired or wrong; fetch a new challenge".into(),
        ));
    }
    Ok(())
}

fn cookie_response(token: &str, base: &str, body: serde_json::Value) -> Response {
    let mut headers = HeaderMap::new();
    headers.insert(
        header::SET_COOKIE,
        set_cookie(token, base).parse().expect("cookie value"),
    );
    (headers, Json(body)).into_response()
}

pub async fn login(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<LoginBody>,
) -> AppResult<Response> {
    let ip = peer(&headers);
    if let Some(wait) = state.auth.check_login_rate(&ip) {
        return Ok(rate_limited_response(wait));
    }
    check_pow(&state, &body)?;

    let stored = password_hash(&state.db, &state.user_id).await?;
    if stored.is_empty() {
        return Err(AppError::BadRequest(
            "no password is set yet; use the setup form instead".into(),
        ));
    }
    if body.password.is_empty() || !verify_password(&body.password, &stored) {
        return Err(AppError::Unauthorized("wrong password".into()));
    }
    let token =
        create_session(&state.db, &state.user_id, &ip, &user_agent(&headers)).await?;
    Ok(cookie_response(&token, &base(&state, &headers), json!({ "ok": true })))
}

// only while no password exists, the first browser claims the install
pub async fn setup(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<LoginBody>,
) -> AppResult<Response> {
    let ip = peer(&headers);
    if let Some(wait) = state.auth.check_login_rate(&ip) {
        return Ok(rate_limited_response(wait));
    }
    check_pow(&state, &body)?;

    if !password_hash(&state.db, &state.user_id).await?.is_empty() {
        return Err(AppError::BadRequest(
            "a password already exists; sign in instead".into(),
        ));
    }
    validate_new_password(&body.password)?;
    set_password_hash(&state.db, &state.user_id, &hash_password(&body.password)).await?;
    let token =
        create_session(&state.db, &state.user_id, &ip, &user_agent(&headers)).await?;
    Ok(cookie_response(&token, &base(&state, &headers), json!({ "ok": true })))
}

pub async fn logout(State(state): State<AppState>, headers: HeaderMap) -> AppResult<Response> {
    if let Some(token) = auth::extract_token(&headers) {
        delete_session(&state.db, &token).await?;
    }
    let mut out = HeaderMap::new();
    out.insert(
        header::SET_COOKIE,
        clear_cookie(&base(&state, &headers)).parse().expect("cookie value"),
    );
    Ok((out, Json(json!({ "ok": true }))).into_response())
}

pub async fn logout_all(State(state): State<AppState>, headers: HeaderMap) -> AppResult<Response> {
    request_session(&state, &headers).await?;
    delete_user_sessions(&state.db, &state.user_id).await?;
    let mut out = HeaderMap::new();
    out.insert(
        header::SET_COOKIE,
        clear_cookie(&base(&state, &headers)).parse().expect("cookie value"),
    );
    Ok((out, Json(json!({ "ok": true }))).into_response())
}

#[derive(Deserialize)]
pub struct PasswordBody {
    #[serde(default)]
    current: String,
    #[serde(default)]
    new: String,
}

fn validate_new_password(pw: &str) -> AppResult<()> {
    if pw.chars().count() < 8 {
        return Err(AppError::BadRequest("the new password needs at least 8 characters".into()));
    }
    if pw.chars().count() > 256 {
        return Err(AppError::BadRequest("the new password must fit in 256 characters".into()));
    }
    Ok(())
}

// other browsers get signed out
pub async fn change_password(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<PasswordBody>,
) -> AppResult<Json<serde_json::Value>> {
    let session = request_session(&state, &headers).await?;
    let stored = password_hash(&state.db, &state.user_id).await?;
    if stored.is_empty() || !verify_password(&body.current, &stored) {
        return Err(AppError::Unauthorized("the current password is wrong".into()));
    }
    validate_new_password(&body.new)?;
    set_password_hash(&state.db, &state.user_id, &hash_password(&body.new)).await?;
    delete_other_sessions(&state.db, &state.user_id, &session.id).await?;
    Ok(Json(json!({ "ok": true })))
}

pub async fn sessions(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> AppResult<Json<serde_json::Value>> {    let current = request_session(&state, &headers).await?;
    let all = list_sessions(&state.db, &state.user_id).await?;
    let user = models::get_user(&state.db, &state.user_id)
        .await?
        .ok_or_else(|| AppError::NotFound("user".into()))?;
    Ok(Json(json!({
        "user": user,
        "sessions": all.iter().map(|s| json!({
            "id": s.id,
            "current": s.id == current.id,
            "ip": s.ip,
            "user_agent": s.user_agent,
            "created_at": s.created_at,
            "last_seen_at": s.last_seen_at,
            "expires_at": s.expires_at,
        })).collect::<Vec<_>>(),
    })))
}

#[cfg(test)]
mod gate_tests {
    use super::*;

    fn solve(nonce: &str, difficulty: u32) -> String {
        for s in 0..10_000_000u64 {
            let sol = s.to_string();
            if crate::auth::pow_meets(nonce, &sol, difficulty) {
                return sol;
            }
        }
        panic!("no solution found");
    }

    async fn live_server() -> (String, tokio::task::JoinHandle<()>) {
        live_server_at("").await
    }

    async fn live_server_at(base: &str) -> (String, tokio::task::JoinHandle<()>) {
        let db = crate::db::connect("sqlite::memory:").await.expect("db");
        let dir = std::env::temp_dir().join(format!("notebook-gate-{}", crate::db::new_id()));
        let storage = crate::storage::Storage::new(&dir).await.expect("storage");
        let mut config = crate::config::Config::from_env();
        // The developer's own environment must not decide what the tests see.
        config.base_path = crate::base::normalize(base);
        let user_id = crate::models::bootstrap(&db, &config).await.expect("bootstrap");
        let state = AppState::new(db, storage, config, user_id);
        let app = super::super::router(state);
        let listener =
            tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let base = format!("http://{}", listener.local_addr().expect("addr"));
        let handle = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        (base, handle)
    }

    async fn claim(base_url: &str, prefix: &str) -> String {
        let anon = reqwest::Client::new();
        let chal: serde_json::Value = anon
            .post(format!("{base_url}{prefix}/api/auth/challenge"))
            .send().await.expect("challenge").json().await.expect("json");
        let nonce = chal["nonce"].as_str().expect("nonce");
        let difficulty = chal["difficulty"].as_u64().expect("difficulty") as u32;
        let r = anon
            .post(format!("{base_url}{prefix}/api/auth/setup"))
            .json(&json!({
                "nonce": nonce,
                "solution": solve(nonce, difficulty),
                "password": "correct horse battery staple",
            }))
            .send().await.expect("setup");
        assert!(r.status().is_success(), "setup claims the install");
        r.headers()
            .get("set-cookie")
            .expect("cookie")
            .to_str()
            .expect("str")
            .to_string()
}

    #[tokio::test]
    async fn a_deployment_under_a_subdirectory_answers_under_its_prefix() {
        let (base, server) = live_server_at("/notebook").await;
        let prefix = "/notebook";
        let anon = reqwest::Client::new();

        let r = anon.get(format!("{base}{prefix}/login")).send().await.expect("login page");
        assert!(r.status().is_success(), "the prefixed login page is served");
        let html = r.text().await.expect("html");
        assert!(
            html.contains(r#"window.__BASE__ = "/notebook""#),
            "the page is told its prefix"
        );
        assert!(!html.contains("__BASE_PATH__"), "the placeholder is filled in");

        let status: serde_json::Value = anon
            .get(format!("{base}{prefix}/api/auth/status"))
            .send().await.expect("status").json().await.expect("json");
        assert_eq!(status["setup_required"], true);

        let plain = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("client");
        let r = plain
            .get(format!("{base}{prefix}/"))
            .header("accept", "text/html,application/xhtml+xml")
            .send().await.expect("root");
        assert_eq!(r.status(), 302, "the app root redirects a browser");
        assert_eq!(
            r.headers().get("location").expect("location").to_str().expect("str"),
            "/notebook/login",
            "the redirect lands inside the subdirectory"
        );

        let r = plain
            .get(format!("{base}{prefix}"))
            .header("accept", "text/html")
            .send().await.expect("mount point");
        assert_eq!(r.status(), 308, "the bare mount point is canonicalised");
        assert_eq!(
            r.headers().get("location").expect("location").to_str().expect("str"),
            "/notebook/"
        );

        let r = plain
            .get(format!("{base}{prefix}/c/some-chat"))
            .header("accept", "text/html")
            .send().await.expect("permalink");
        assert_eq!(r.status(), 302);
        let r = anon.get(format!("{base}{prefix}/api/sources")).send().await.expect("sources");
        assert_eq!(r.status(), 401, "the API is gated under the prefix too");
        let body: serde_json::Value = r.json().await.expect("json");
        assert!(
            body["error"].as_str().expect("error").contains("/notebook/login"),
            "the refusal names the prefixed login page: {body}"
        );

        let r = anon.get(format!("{base}/api/sources")).send().await.expect("direct");
        assert_eq!(r.status(), 401, "the gate holds direct as well");
        let cookie = claim(&base, prefix).await;
        assert!(cookie.contains("Path=/notebook;"), "cookie is scoped: {cookie}");
        let session = cookie.split(';').next().expect("pair").to_string();
        let authed = anon
            .get(format!("{base}{prefix}/api/sources"))
            .header("cookie", session)
            .send().await.expect("sources");
        assert!(authed.status().is_success(), "the cookie opens the prefixed API");

        server.abort();
    }

    #[tokio::test]
    async fn a_forwarded_prefix_is_honoured_when_base_path_is_unset() {
        let (base, server) = live_server().await;
        let anon = reqwest::Client::new();
        let r = anon
            .get(format!("{base}/login"))
            .header("x-forwarded-prefix", "/notebook/")
            .send().await.expect("login page");
        assert!(r.status().is_success(), "the login page is served at the app's own root");
        let html = r.text().await.expect("html");
        assert!(html.contains(r#"window.__BASE__ = "/notebook""#), "the page is told its prefix");

        let plain = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("client");
        let r = plain
            .get(format!("{base}/"))
            .header("accept", "text/html")
            .header("x-forwarded-prefix", "/notebook/")
            .send().await.expect("root");
        assert_eq!(r.status(), 302);
        assert_eq!(
            r.headers().get("location").expect("location").to_str().expect("str"),
            "/notebook/login"
        );
        server.abort();
    }

    #[tokio::test]
    async fn the_gate_holds_and_the_cookie_opens_it() {
        let (base, server) = live_server().await;
        let anon = reqwest::Client::new();

        for path in ["/api/sources", "/api/settings", "/health", "/", "/does-not-exist"] {
            let r = anon.get(format!("{base}{path}")).send().await.expect("get");
            assert_eq!(r.status(), 401, "anonymous {path} must be 401");
            let body: serde_json::Value = r.json().await.expect("json");
            assert_eq!(body["kind"], "unauthorized");
        }

        let r = anon.get(format!("{base}/login")).send().await.expect("login page");
        assert!(r.status().is_success(), "the login page stays public");

        let plain = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("client");
        let r = plain
            .get(format!("{base}/"))
            .header("accept", "text/html,application/xhtml+xml")
            .send().await.expect("root");
        assert_eq!(r.status(), 302, "browser navigation to / redirects");
        assert_eq!(r.headers().get("location").expect("location").to_str().expect("str"), "/login");
        let r = plain
            .get(format!("{base}/c/some-chat"))
            .header("accept", "text/html")
            .send().await.expect("permalink");
        assert_eq!(r.status(), 302, "browser navigation to a permalink redirects");

        let status: serde_json::Value =
            anon.get(format!("{base}/api/auth/status")).send().await.expect("status").json().await.expect("json");
        assert_eq!(status["setup_required"], true);
        assert_eq!(status["authenticated"], false);

        let chal: serde_json::Value = anon
            .post(format!("{base}/api/auth/challenge"))
            .send().await.expect("challenge").json().await.expect("json");
        let nonce = chal["nonce"].as_str().expect("nonce");
        let difficulty = chal["difficulty"].as_u64().expect("difficulty") as u32;
        let bad = anon
            .post(format!("{base}/api/auth/setup"))
            .json(&json!({
                "nonce": nonce,
                "solution": solve(nonce, difficulty),
                "password": "short",
            }))
            .send().await.expect("setup");
        assert_eq!(bad.status(), 400, "short passwords are refused");

        let chal: serde_json::Value = anon
            .post(format!("{base}/api/auth/challenge"))
            .send().await.expect("challenge").json().await.expect("json");
        let nonce = chal["nonce"].as_str().expect("nonce");
        let difficulty = chal["difficulty"].as_u64().expect("difficulty") as u32;
        let r = anon
            .post(format!("{base}/api/auth/setup"))
            .json(&json!({
                "nonce": nonce,
                "solution": solve(nonce, difficulty),
                "password": "correct horse battery staple",
            }))
            .send().await.expect("setup");
        assert!(r.status().is_success(), "setup claims the install");
        let cookie = r.headers().get("set-cookie").expect("cookie").to_str().expect("str").to_string();
        assert!(cookie.contains("notebook_session"), "a session cookie is set: {cookie}");
        let session = cookie.split(';').next().expect("pair").to_string();

        let authed = reqwest::Client::builder().build().expect("client");
        let get = |path: &str| {
            authed.get(format!("{base}{path}")).header("cookie", session.clone()).send()
        };
        let r = get("/api/sources").await.expect("sources");
        assert!(r.status().is_success(), "the cookie opens the API");

        let sessions: serde_json::Value =
            get("/api/auth/sessions").await.expect("sessions").json().await.expect("json");
        assert_eq!(sessions["sessions"].as_array().expect("array").len(), 1);

        let replay = anon
            .post(format!("{base}/api/auth/login"))
            .json(&json!({
                "nonce": nonce,
                "solution": solve(nonce, difficulty),
                "password": "correct horse battery staple",
            }))
            .send().await.expect("replay");
        assert_eq!(replay.status(), 400, "a spent challenge is dead");

        let chal: serde_json::Value = anon
            .post(format!("{base}/api/auth/challenge"))
            .send().await.expect("challenge").json().await.expect("json");
        let nonce = chal["nonce"].as_str().expect("nonce");
        let difficulty = chal["difficulty"].as_u64().expect("difficulty") as u32;
        let wrong = anon
            .post(format!("{base}/api/auth/login"))
            .json(&json!({
                "nonce": nonce,
                "solution": solve(nonce, difficulty),
                "password": "nope",
            }))
            .send().await.expect("login");
        assert_eq!(wrong.status(), 401, "a wrong password stays out");

        let changed = authed
            .patch(format!("{base}/api/auth/password"))
            .header("cookie", session.clone())
            .json(&json!({ "current": "nope", "new": "a brand new password" }))
            .send().await.expect("change");
        assert_eq!(changed.status(), 401, "the old password must match");
        let changed = authed
            .patch(format!("{base}/api/auth/password"))
            .header("cookie", session.clone())
            .json(&json!({ "current": "correct horse battery staple", "new": "short" }))
            .send().await.expect("change");
        assert_eq!(changed.status(), 400, "the new password must be long enough");
        let changed = authed
            .patch(format!("{base}/api/auth/password"))
            .header("cookie", session.clone())
            .json(&json!({ "current": "correct horse battery staple", "new": "a brand new password" }))
            .send().await.expect("change");
        assert!(changed.status().is_success(), "password change works");

        let out = authed
            .post(format!("{base}/api/auth/logout"))
            .header("cookie", session.clone())
            .send().await.expect("logout");
        assert!(out.status().is_success());
        let r = get("/api/sources").await.expect("sources");
        assert_eq!(r.status(), 401, "logout closes the gate again");

        server.abort();
    }
}
