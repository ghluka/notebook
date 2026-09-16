//! Serving the app under a path prefix, the way nginx mounts it one level down
//! (`https://host/notebook/`).
//!
//! Every route in this app is written against the site root: the API lives at
//! `/api/...`, the login page is `/login`, a conversation permalink is
//! `/c/{id}`. A reverse proxy that puts all of that under a subdirectory has to
//! be told about, or nothing lines up: the browser asks for `/notebook/login`,
//! the router only knows `/login`, and an anonymous caller is refused on the
//! login page itself.
//!
//! So the router mounts the same table twice, at the root and under the prefix,
//! and every URL the app *emits* carries the prefix: the redirect that sends an
//! anonymous browser to the login page, the cookie's `Path`, and the HTML that
//! tells the client where to send its fetches.
//!
//! `BASE_PATH` names the prefix. A proxy may instead send `X-Forwarded-Prefix`,
//! which is honoured when `BASE_PATH` is unset.

use axum::http::HeaderMap;

/// Normalise a prefix to `""` or `"/something"`: one leading slash, no
/// trailing one, no traversal, no query and no fragment. Anything that cannot
/// be a plain path prefix is refused rather than half honoured, since it ends
/// up in a `Location` header and a cookie `Path`.
pub fn normalize(raw: &str) -> String {
    let raw = raw.trim();
    if raw.is_empty() {
        return String::new();
    }
    // A proxy may pass a list; the first entry is the one in front of us.
    let raw = raw.split(',').next().unwrap_or("").trim();
    if raw.contains('?') || raw.contains('#') || raw.contains('\\') {
        return String::new();
    }
    let mut out = String::new();
    for segment in raw.split('/') {
        match segment {
            "" | "." => continue,
            ".." => return String::new(),
            s if s.bytes().all(|b| {
                b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~' | b'%')
            }) =>
            {
                out.push('/');
                out.push_str(s);
            }
            _ => return String::new(),
        }
    }
    if out.len() > 200 { String::new() } else { out }
}

/// The prefix for this request: the configured one wins, otherwise whatever
/// the proxy declared.
pub fn effective(configured: &str, headers: &HeaderMap) -> String {
    if !configured.is_empty() {
        return configured.to_string();
    }
    headers
        .get("x-forwarded-prefix")
        .and_then(|value| value.to_str().ok())
        .map(normalize)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_prefix_is_normalised_to_one_slash_and_no_trailing_one() {
        assert_eq!(normalize(""), "");
        assert_eq!(normalize("/"), "");
        assert_eq!(normalize("   "), "");
        assert_eq!(normalize("/notebook"), "/notebook");
        assert_eq!(normalize("/notebook/"), "/notebook");
        assert_eq!(normalize("/notebook//"), "/notebook");
        assert_eq!(normalize("notebook"), "/notebook");
        assert_eq!(normalize("/a/b/"), "/a/b");
        assert_eq!(normalize("/notebook, /other"), "/notebook");
    }

    #[test]
    fn a_prefix_that_is_not_a_plain_path_is_refused() {
        assert_eq!(normalize("/../etc"), "");
        assert_eq!(normalize("/note book"), "");
        assert_eq!(normalize("/notebook?x=1"), "");
        assert_eq!(normalize("/notebook#frag"), "");
        assert_eq!(normalize("https://host/notebook"), "");
        assert_eq!(normalize(&format!("/{}", "a".repeat(300))), "");
    }

    #[test]
    fn the_configured_prefix_beats_the_forwarded_one() {
        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-prefix", "/from-proxy/".parse().unwrap());
        assert_eq!(effective("/configured", &headers), "/configured");
        assert_eq!(effective("", &headers), "/from-proxy");
        assert_eq!(effective("", &HeaderMap::new()), "");
    }
}
