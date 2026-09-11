//! The LLM layer: one neutral request type, three wire formats.
//!
//! Anthropic-style `/v1/messages`, and the two OpenAI-style formats:
//! `/responses`, tried first, and `/chat/completions`, for endpoints without
//! the newer route and for audio and video (see `openai.rs`).
//!
//! Which model a role uses is not decided here; that lives in the `models`
//! registry, backed by the database. This module only knows how to talk.
//!
//! The neutral types cover the whole provider surface, including the
//! tool-calling pieces that only the phase-2 researcher loop will call.
#![allow(dead_code)]

pub mod anthropic;
pub mod openai;
pub mod responses;
pub mod types;

use async_trait::async_trait;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as B64;
use futures::Stream;
use std::pin::Pin;

pub use types::*;

#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("{0}")]
    MissingKey(String),
    #[error("provider returned {status}: {body}")]
    Api {
        status: u16,
        body: String,
        /// Seconds the provider asked us to wait, when it said so.
        retry_after: Option<u64>,
    },
    #[error("the provider stayed rate limited after {waited}s of waiting: {body}")]
    RateLimited { status: u16, body: String, waited: u64 },
    #[error("bad request: {0}")]
    Request(String),
    #[error("transport: {0}")]
    Http(#[from] reqwest::Error),
    #[error("malformed provider response: {0}")]
    Decode(#[from] serde_json::Error),
}

impl LlmError {
    /// Whether waiting could plausibly fix this. Rate limits and the various
    /// "busy right now" statuses qualify; a 400 or a 401 never will.
    pub fn is_retryable(&self) -> bool {
        match self {
            LlmError::Api { status, body, .. } => {
                matches!(status, 408 | 409 | 425 | 429 | 500 | 502 | 503 | 504)
                    || mentions_rate_limit(body)
            }
            // A dropped or timed out request is worth one more go.
            LlmError::Http(e) => e.is_timeout() || e.is_connect() || e.is_request(),
            _ => false,
        }
    }

    pub fn is_rate_limited(&self) -> bool {
        matches!(self, LlmError::RateLimited { .. })
    }

    /// Whether the endpoint said, in whatever words, that this model cannot
    /// take this kind of content at all.
    ///
    /// It is worth telling apart because no amount of retrying, and no other
    /// way of sending the same file, will change the answer: the fix is a
    /// different model.
    pub fn rejects_modality(&self) -> bool {
        match self {
            LlmError::Api { status, body, .. } => {
                (*status == 400 || *status == 415 || *status == 422)
                    && mentions_unsupported_modality(body)
            }
            _ => false,
        }
    }

    /// The sentence a person should read, dug out of whatever JSON the
    /// provider wrapped it in. Providers bury it in `error.message`,
    /// `message`, or `detail`; some just send prose.
    pub fn provider_message(&self) -> String {
        let body = match self {
            LlmError::Api { body, .. } | LlmError::RateLimited { body, .. } => body.as_str(),
            other => return other.to_string(),
        };
        human_message(body)
    }

    fn retry_after(&self) -> Option<u64> {
        match self {
            LlmError::Api { retry_after, .. } => *retry_after,
            _ => None,
        }
    }
}

/// Pull the human sentence out of a provider error body.
fn human_message(body: &str) -> String {
    let trimmed = body.trim();
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) {
        let candidates = [
            value.pointer("/error/message"),
            value.pointer("/message"),
            value.pointer("/detail"),
            value.pointer("/error"),
            value.pointer("/error/0/message"),
        ];
        for found in candidates.into_iter().flatten() {
            if let Some(text) = found.as_str()
                && !text.trim().is_empty()
            {
                return tidy(text);
            }
        }
    }
    tidy(trimmed)
}

/// One line, no runaway payloads.
fn tidy(text: &str) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let flat = flat.trim_end_matches(['.', ' ']).to_string();
    if flat.chars().count() > 300 {
        let cut: String = flat.chars().take(300).collect();
        format!("{cut}...")
    } else {
        flat
    }
}

/// The many ways an endpoint says "not that kind of content".
///
/// The first two are real answers this project has been given: NVIDIA's hosted
/// endpoint for a text-only model, and LM Studio faced with an OpenAI `file`
/// part it does not implement.
fn mentions_unsupported_modality(body: &str) -> bool {
    let body = body.to_ascii_lowercase();
    let refusal = [
        "does not support image",
        "do not support image",
        "does not support vision",
        "image input",
        "image_input",
        "does not support audio",
        "does not support video",
        "unsupported content",
        "unsupported media",
        "content type is not supported",
        "must have a 'type' field",
        "must have a \"type\" field",
        "unknown variant",
        "invalid content type",
        "multimodal",
        "not a vision",
        "no vision",
    ];
    refusal.iter().any(|m| body.contains(m))
}

/// Providers phrase exhaustion differently and not all of them use a 429.
/// Gemini says "high demand", NVIDIA says "ResourceExhausted", Anthropic says
/// "overloaded_error".
fn mentions_rate_limit(body: &str) -> bool {
    let body = body.to_ascii_lowercase();
    [
        "rate limit",
        "rate_limit",
        "ratelimit",
        "resourceexhausted",
        "resource_exhausted",
        "quota",
        "high demand",
        "overloaded",
        "unavailable",
        "too many requests",
        "capacity",
        "try again later",
    ]
    .iter()
    .any(|m| body.contains(m))
}

/// How long to wait between attempts. A short first wait rides out a burst;
/// the two long ones cover a provider that is genuinely saturated. After the
/// last one the caller is told, so a person can decide what to do.
pub const RETRY_WAITS: &[u64] = &[5, 60, 60];
/// Never wait longer than this, whatever a `Retry-After` header claims.
const MAX_WAIT: u64 = 120;

#[async_trait]
pub trait LlmProvider: Send + Sync {
    fn name(&self) -> &'static str;
    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse, LlmError>;
    /// The same turn as a token stream. The default refuses, which lets
    /// callers fall back to one-shot `chat` for providers that never learned
    /// to stream.
    async fn chat_stream(&self, req: &ChatRequest) -> Result<TokenStream, LlmError> {
        let _ = req;
        Err(LlmError::Request("this provider does not support streaming".into()))
    }
}

/// Tokens as they arrive. Only the establishment of the stream is retried
/// (see `chat_stream_with_retry`): once tokens flow, an error ends the turn
/// instead of restarting it halfway through an answer.
pub type TokenStream = Pin<Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send>>;

/// One SSE record: the optional `event:` name and its `data:` payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseRecord {
    pub event: Option<String>,
    pub data: String,
}

/// Pull complete records off the front of `buf`, leaving the partial tail in
/// place. Comments and keep-alives carry no data and are dropped.
pub fn split_sse_records(buf: &mut String) -> Vec<SseRecord> {
    let mut out = Vec::new();
    loop {
        let end = match (buf.find("\r\n\r\n"), buf.find("\n\n")) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        let Some(end) = end else { break };
        let sep = if buf.as_bytes()[end] == b'\r' { 4 } else { 2 };
        let raw: String = buf.drain(..end + sep).collect();
        let mut event = None;
        let mut data = Vec::new();
        for line in raw.lines() {
            if let Some(name) = line.strip_prefix("event:") {
                event = Some(name.trim().to_string());
            } else if let Some(payload) = line.strip_prefix("data:") {
                data.push(payload.strip_prefix(' ').unwrap_or(payload));
            }
        }
        if data.is_empty() {
            continue;
        }
        out.push(SseRecord { event, data: data.join("\n") });
    }
    out
}

/// Call a provider, waiting out rate limits on the schedule in `RETRY_WAITS`.
///
/// A provider that is still refusing after the last wait returns
/// `LlmError::RateLimited`, which callers surface as its own kind of failure
/// rather than pretending the request was impossible: waiting longer or moving
/// to another model would both have worked.
pub async fn chat_with_retry(
    client: &dyn LlmProvider,
    req: &ChatRequest,
) -> Result<ChatResponse, LlmError> {
    chat_with_schedule(client, req, RETRY_WAITS).await
}

/// The same, with the waits given explicitly. Tests pass zeroes.
pub async fn chat_with_schedule(
    client: &dyn LlmProvider,
    req: &ChatRequest,
    waits: &[u64],
) -> Result<ChatResponse, LlmError> {
    let mut waited = 0;

    for (attempt, base) in waits.iter().enumerate() {
        match client.chat(req).await {
            Ok(response) => return Ok(response),
            Err(e) if e.is_retryable() => {
                // Honour the provider's own advice when it gives any.
                let wait = e.retry_after().unwrap_or(*base).clamp(*base, MAX_WAIT.max(*base));
                tracing::warn!(
                    attempt = attempt + 1,
                    wait_seconds = wait,
                    error = %e,
                    "provider is busy, waiting before retrying"
                );
                tokio::time::sleep(std::time::Duration::from_secs(wait)).await;
                waited += wait;
            }
            Err(e) => return Err(e),
        }
    }

    match client.chat(req).await {
        Ok(response) => Ok(response),
        Err(LlmError::Api { status, body, .. }) => {
            Err(LlmError::RateLimited { status, body, waited })
        }
        Err(e) if e.is_retryable() => {
            Err(LlmError::RateLimited { status: 0, body: e.to_string(), waited })
        }
        Err(e) => Err(e),
    }
}

/// Open a token stream, waiting out rate limits on the same schedule. Only
/// the handshake is retried: a refusal to stream at all (a 400 naming the
/// `stream` parameter, a provider without support) comes back at once so the
/// caller can fall back to one-shot chat for that round.
pub async fn chat_stream_with_retry(
    client: &dyn LlmProvider,
    req: &ChatRequest,
) -> Result<TokenStream, LlmError> {
    chat_stream_with_schedule(client, req, RETRY_WAITS).await
}

/// The same, with the waits given explicitly. Tests pass zeroes.
pub async fn chat_stream_with_schedule(
    client: &dyn LlmProvider,
    req: &ChatRequest,
    waits: &[u64],
) -> Result<TokenStream, LlmError> {
    let mut waited = 0;

    for (attempt, base) in waits.iter().enumerate() {
        match client.chat_stream(req).await {
            Ok(stream) => return Ok(stream),
            Err(e) if e.is_retryable() => {
                let wait = e.retry_after().unwrap_or(*base).clamp(*base, MAX_WAIT.max(*base));
                tracing::warn!(
                    attempt = attempt + 1,
                    wait_seconds = wait,
                    error = %e,
                    "provider is busy, waiting before retrying the stream"
                );
                tokio::time::sleep(std::time::Duration::from_secs(wait)).await;
                waited += wait;
            }
            Err(e) => return Err(e),
        }
    }

    match client.chat_stream(req).await {
        Ok(stream) => Ok(stream),
        Err(LlmError::Api { status, body, .. }) => {
            Err(LlmError::RateLimited { status, body, waited })
        }
        Err(e) if e.is_retryable() => {
            Err(LlmError::RateLimited { status: 0, body: e.to_string(), waited })
        }
        Err(e) => Err(e),
    }
}

/// Wrap raw file bytes as a content part the analyzer can look at.
pub fn part_for_file(media_type: &str, bytes: &[u8], filename: Option<&str>) -> ContentPart {
    let data = B64.encode(bytes);
    if media_type.starts_with("image/") {
        ContentPart::Image { media_type: media_type.to_string(), data }
    } else if media_type.starts_with("audio/") {
        ContentPart::Audio { media_type: media_type.to_string(), data }
    } else if media_type.starts_with("video/") {
        ContentPart::Video { media_type: media_type.to_string(), data }
    } else {
        ContentPart::Document {
            media_type: media_type.to_string(),
            data,
            filename: filename.map(|s| s.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// NVIDIA's hosted endpoint, asked to look at a page image with a text
    /// only model.
    const NO_IMAGES: &str = r#"{ "error": { "message": "The provided messages contain images, but nvidia/nemotron-3-nano-4b does not support image inputs.", "type": "invalid_request_error", "param": "messages", "code": "invalid_value" } }"#;

    /// LM Studio, handed the OpenAI `file` part it does not implement.
    const NO_FILE_PART: &str =
        r#"{"error":"Invalid 'content': 'content' objects must have a 'type' field that is either 'text' or 'image_url'"}"#;

    /// vLLM style, handed an `audio_url` part by a model that has no ears.
    const NO_AUDIO: &str = r#"{"error":{"message":"Failed to deserialize the JSON body into the target type: messages[1]: unknown variant `audio_url`, expected one of `text`, `image_url`","type":"invalid_request_error"}}"#;

    fn api(status: u16, body: &str) -> LlmError {
        LlmError::Api { status, body: body.to_string(), retry_after: None }
    }

    #[test]
    fn a_refusal_of_the_content_kind_is_told_apart() {
        assert!(api(400, NO_IMAGES).rejects_modality());
        assert!(api(400, NO_FILE_PART).rejects_modality());
        assert!(api(400, NO_AUDIO).rejects_modality());

        // A refusal of the content kind is not a rate limit and not retryable:
        // the same request will be refused for ever.
        assert!(!api(400, NO_IMAGES).is_retryable());

        // Ordinary failures are not mistaken for it.
        assert!(!api(400, r#"{"error":{"message":"invalid api key"}}"#).rejects_modality());
        assert!(!api(429, r#"{"error":{"message":"rate limit exceeded"}}"#).rejects_modality());
        assert!(!api(500, NO_IMAGES).rejects_modality(), "a server fault is not a refusal");
    }

    #[test]
    fn the_providers_own_sentence_comes_out_of_the_json() {
        // The sentence, whole, with the JSON and the trailing full stop gone.
        let images = api(400, NO_IMAGES).provider_message();
        assert!(images.starts_with("The provided messages contain images"), "{images}");
        assert!(images.ends_with("does not support image inputs"), "{images}");
        assert!(!images.contains('{'), "{images}");

        let part = api(400, NO_FILE_PART).provider_message();
        assert!(part.starts_with("Invalid 'content'"), "{part}");
        assert!(part.ends_with("either 'text' or 'image_url'"), "{part}");
        // Prose, or a shape nobody recognises, comes back as it is.
        assert_eq!(api(502, "  upstream is down  ").provider_message(), "upstream is down");
        assert_eq!(api(400, "{}").provider_message(), "{}");

        // A wall of payload is cut to something readable.
        let huge = format!(r#"{{"error":{{"message":"{}"}}}}"#, "x".repeat(900));
        let message = api(400, &huge).provider_message();
        assert!(message.chars().count() <= 303, "{}", message.len());
        assert!(message.ends_with("..."));
    }
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Fails `fail_times` times with `status` and `body`, then succeeds.
    struct Flaky {
        calls: AtomicUsize,
        fail_times: usize,
        status: u16,
        body: &'static str,
    }

    impl Flaky {
        fn new(fail_times: usize, status: u16, body: &'static str) -> Self {
            Flaky { calls: AtomicUsize::new(0), fail_times, status, body }
        }
    }

    #[async_trait]
    impl LlmProvider for Flaky {
        fn name(&self) -> &'static str {
            "flaky"
        }

        async fn chat(&self, _req: &ChatRequest) -> Result<ChatResponse, LlmError> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            if n < self.fail_times {
                return Err(LlmError::Api {
                    status: self.status,
                    body: self.body.to_string(),
                    retry_after: None,
                });
            }
            Ok(ChatResponse {
                text: "ok".into(),
                thinking: None,
                tool_calls: Vec::new(),
                stop_reason: "end_turn".into(),
                usage: Usage::default(),
                model: "flaky".into(),
            })
        }
    }

    fn request() -> ChatRequest {
        ChatRequest::new("m", vec![Message::user("hi")])
    }

    /// The two bodies that started this: NVIDIA and Gemini, both saying busy
    /// without saying 429.
    #[test]
    fn recognises_real_rate_limit_bodies() {
        let nvidia = LlmError::Api {
            status: 503,
            body: r#"{"error":{"message":"ResourceExhausted: Worker local total request limit reached (16/16)","type":"Service Unavailable","code":503}}"#.into(),
            retry_after: None,
        };
        let gemini = LlmError::Api {
            status: 503,
            body: r#"[{ "error": { "code": 503, "message": "This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.", "status": "UNAVAILABLE" } } ]"#.into(),
            retry_after: None,
        };
        let too_many = LlmError::Api { status: 429, body: "slow down".into(), retry_after: None };

        assert!(nvidia.is_retryable());
        assert!(gemini.is_retryable());
        assert!(too_many.is_retryable());
    }

    #[test]
    fn leaves_real_failures_alone() {
        let bad_request = LlmError::Api {
            status: 400,
            body: "model does not accept image parts".into(),
            retry_after: None,
        };
        let unauthorised =
            LlmError::Api { status: 401, body: "invalid api key".into(), retry_after: None };

        assert!(!bad_request.is_retryable());
        assert!(!unauthorised.is_retryable());
    }

    #[tokio::test]
    async fn retries_until_the_provider_recovers() {
        let flaky = Flaky::new(2, 503, "high demand");
        let out = chat_with_schedule(&flaky, &request(), &[0, 0, 0]).await.unwrap();

        assert_eq!(out.text, "ok");
        assert_eq!(flaky.calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn gives_up_as_a_rate_limit_not_a_dead_end() {
        let flaky = Flaky::new(usize::MAX, 503, "ResourceExhausted");
        let err = chat_with_schedule(&flaky, &request(), &[0, 0, 0]).await.unwrap_err();

        assert!(err.is_rate_limited(), "got {err:?}");
        // One attempt per wait, plus the final one after the last wait.
        assert_eq!(flaky.calls.load(Ordering::SeqCst), 4);
    }

    #[tokio::test]
    async fn a_bad_request_fails_immediately() {
        let flaky = Flaky::new(usize::MAX, 400, "unsupported content part");
        let err = chat_with_schedule(&flaky, &request(), &[0, 0, 0]).await.unwrap_err();

        assert!(!err.is_rate_limited());
        assert_eq!(flaky.calls.load(Ordering::SeqCst), 1, "must not retry a 400");
    }

    #[test]
    fn splits_records_across_chunk_boundaries() {
        let mut buf = "event: message_start\ndata: {\"a\":1}\n\ndata: {\"b\"".to_string();
        let first = split_sse_records(&mut buf);
        assert_eq!(
            first,
            vec![SseRecord { event: Some("message_start".into()), data: "{\"a\":1}".into() }]
        );
        // The partial tail waits for the rest.
        assert!(buf.contains("{\"b\""));

        buf.push_str(":2}\n\n:keep-alive\n\n");
        let rest = split_sse_records(&mut buf);
        assert_eq!(rest, vec![SseRecord { event: None, data: "{\"b\":2}".into() }]);
        assert!(buf.is_empty());
    }

    #[test]
    fn understands_crlf_and_multi_line_data() {
        let mut buf = "event: ping\r\ndata: one\r\ndata: two\r\n\r\n".to_string();
        let out = split_sse_records(&mut buf);
        assert_eq!(out, vec![SseRecord { event: Some("ping".into()), data: "one\ntwo".into() }]);
    }

    /// A provider that streams canned events, failing `fail_times` handshakes
    /// first, so the retry schedule has something to chew on.
    struct Scripted {
        calls: AtomicUsize,
        fail_times: usize,
        events: Vec<StreamEvent>,
    }

    #[async_trait]
    impl LlmProvider for Scripted {
        fn name(&self) -> &'static str {
            "scripted"
        }

        async fn chat(&self, _req: &ChatRequest) -> Result<ChatResponse, LlmError> {
            unreachable!("streaming tests never call one-shot chat");
        }

        async fn chat_stream(&self, _req: &ChatRequest) -> Result<TokenStream, LlmError> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            if n < self.fail_times {
                return Err(LlmError::Api {
                    status: 503,
                    body: "high demand".into(),
                    retry_after: None,
                });
            }
            let events = self
                .events
                .iter()
                .cloned()
                .map(Ok)
                .collect::<Vec<Result<StreamEvent, LlmError>>>();
            Ok(Box::pin(futures::stream::iter(events)))
        }
    }

    #[tokio::test]
    async fn streams_recover_after_a_busy_handshake() {
        use futures::StreamExt;
        let scripted = Scripted {
            calls: AtomicUsize::new(0),
            fail_times: 1,
            events: vec![
                StreamEvent::Text("hel".into()),
                StreamEvent::Text("lo".into()),
                StreamEvent::Done(StreamDone {
                    tool_calls: Vec::new(),
                    input_tokens: 3,
                    output_tokens: 2,
                }),
            ],
        };
        let Ok(mut stream) = chat_stream_with_schedule(&scripted, &request(), &[0, 0, 0]).await
        else {
            panic!("expected a stream after the retry");
        };
        let mut text = String::new();
        let mut done = None;
        while let Some(ev) = stream.next().await {
            match ev.unwrap() {
                StreamEvent::Thinking(_) => {}
                StreamEvent::Text(t) => text.push_str(&t),
                StreamEvent::Done(d) => done = Some(d),
            }
        }
        assert_eq!(text, "hello");
        assert_eq!(done.unwrap().output_tokens, 2);
        assert_eq!(scripted.calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn providers_without_streaming_say_so_at_once() {
        let flaky = Flaky::new(0, 200, "");
        let Err(err) = chat_stream_with_schedule(&flaky, &request(), &[0, 0, 0]).await else {
            panic!("expected streaming to be refused");
        };
        assert!(!err.is_rate_limited());
        assert!(err.to_string().contains("streaming"));
    }
}
