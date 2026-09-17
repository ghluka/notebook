//! openai-style endpoints in two wire formats: /responses first, /chat/completions
//! as fallback. support is per model, not per endpoint, and remembered per process.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{Value, json};

use super::types::*;
use super::{LlmError, LlmProvider, SseRecord, TokenStream, responses, split_sse_records};

static SPEAKS_RESPONSES: LazyLock<Mutex<HashMap<(String, String), bool>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

pub struct OpenAiProvider {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
}

impl OpenAiProvider {
    pub fn new(http: reqwest::Client, base_url: String, api_key: String) -> Self {
        OpenAiProvider { http, base_url: base_url.trim_end_matches('/').to_string(), api_key }
    }

    fn key(&self, model: &str) -> (String, String) {
        (self.base_url.clone(), model.to_string())
    }

    fn remembered(&self, model: &str) -> Option<bool> {
        SPEAKS_RESPONSES.lock().ok()?.get(&self.key(model)).copied()
    }

    fn remember(&self, model: &str, speaks: bool) {
        if let Ok(mut known) = SPEAKS_RESPONSES.lock() {
            known.insert(self.key(model), speaks);
        }
    }

    fn try_responses(&self, req: &ChatRequest) -> bool {
        responses::can_carry(req) && self.remembered(&req.model) != Some(false)
    }

    // once a model is known served there, a 404 is something else; falling back would hide it
    fn should_fall_back(&self, model: &str, e: &LlmError) -> bool {
        self.remembered(model) != Some(true) && lacks_route(e)
    }

    async fn post(&self, path: &str, body: &Value) -> Result<reqwest::Response, LlmError> {
        let mut builder = self.http.post(format!("{}{path}", self.base_url));
        if !self.api_key.is_empty() {
            builder = builder.bearer_auth(&self.api_key);
        }
        let resp = builder.json(body).send().await?;

        let status = resp.status();
        if status.is_success() {
            return Ok(resp);
        }
        let retry_after = resp
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.trim().parse::<u64>().ok());
        let raw = resp.text().await.unwrap_or_default();
        Err(LlmError::Api { status: status.as_u16(), body: raw, retry_after })
    }

    fn push_message(out: &mut Vec<Value>, m: &Message) -> Result<(), LlmError> {
        let mut parts: Vec<Value> = Vec::new();
        let mut tool_calls: Vec<Value> = Vec::new();

        for part in &m.content {
            match part {
                ContentPart::Text { text } => parts.push(json!({ "type": "text", "text": text })),
                ContentPart::Image { media_type, data } => parts.push(json!({
                    "type": "image_url",
                    "image_url": { "url": format!("data:{media_type};base64,{data}") }
                })),
                ContentPart::Document { media_type, data, filename } => parts.push(json!({
                    "type": "file",
                    "file": {
                        "filename": filename.clone().unwrap_or_else(|| "document".into()),
                        "file_data": format!("data:{media_type};base64,{data}")
                    }
                })),
                // data URIs work without sharing a filesystem, unlike file:// uris
                ContentPart::Audio { media_type, data } => parts.push(json!({
                    "type": "audio_url",
                    "audio_url": { "url": format!("data:{media_type};base64,{data}") }
                })),
                ContentPart::Video { media_type, data } => parts.push(json!({
                    "type": "video_url",
                    "video_url": { "url": format!("data:{media_type};base64,{data}") }
                })),
                ContentPart::ToolUse { id, name, input, signature } => {
                    let mut call = json!({
                        "id": id,
                        "type": "function",
                        "function": { "name": name, "arguments": input.to_string() }
                    });
                    // gemini 400s ("missing a thought_signature") if a signed call goes back unsigned
                    if let Some(signature) = signature {
                        call["extra_content"] =
                            json!({ "google": { "thought_signature": signature } });
                    }
                    tool_calls.push(call);
                }
                ContentPart::ToolResult { tool_use_id, content, .. } => out.push(json!({
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": content
                })),
            }
        }

        if parts.is_empty() && tool_calls.is_empty() {
            return Ok(());
        }

        let role = match m.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            // bare text on a tool turn has nowhere to go; treated as user context
            Role::Tool => "user",
        };
        let mut msg = json!({ "role": role });
        if !parts.is_empty() {
            msg["content"] = json!(parts);
        }
        if !tool_calls.is_empty() {
            msg["tool_calls"] = json!(tool_calls);
        }
        out.push(msg);
        Ok(())
    }

    fn build_body(req: &ChatRequest) -> Result<Value, LlmError> {
        let mut messages = Vec::new();
        if let Some(system) = &req.system {
            messages.push(json!({ "role": "system", "content": system }));
        }
        for m in &req.messages {
            Self::push_message(&mut messages, m)?;
        }

        let mut body = json!({
            "model": req.model,
            "messages": messages,
            "max_completion_tokens": req.max_tokens,
        });
        if req.effort == Effort::Off {
            if let Some(t) = req.temperature {
                body["temperature"] = json!(t);
            }
        } else {
            body["reasoning_effort"] = json!(req.effort.as_str());
        }
        if !req.tools.is_empty() {
            body["tools"] = json!(
                req.tools
                    .iter()
                    .map(|t| json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters
                        }
                    }))
                    .collect::<Vec<_>>()
            );
        }
        Ok(body)
    }

    async fn chat_completions(&self, req: &ChatRequest) -> Result<ChatResponse, LlmError> {
        let body = Self::build_body(req)?;
        let raw = self.post("/chat/completions", &body).await?.text().await?;
        let v: Value = serde_json::from_str(&raw)?;
        Ok(parse_chat_completion(&v, &req.model))
    }

    async fn responses(&self, req: &ChatRequest) -> Result<ChatResponse, LlmError> {
        let body = responses::build_body(req)?;
        let raw = self.post("/responses", &body).await?.text().await?;
        let v: Value = serde_json::from_str(&raw)?;
        responses::parse_response(&v, &req.model)
    }

    async fn chat_completions_stream(&self, req: &ChatRequest) -> Result<TokenStream, LlmError> {
        let mut body = Self::build_body(req)?;
        body["stream"] = json!(true);
        // most servers omit usage from streamed chunks without this
        body["stream_options"] = json!({ "include_usage": true });
        let resp = self.post("/chat/completions", &body).await?;
        Ok(pump(resp, OpenAiStreamState::default(), feed_chat, "the stream ended before [DONE]"))
    }

    async fn responses_stream(&self, req: &ChatRequest) -> Result<TokenStream, LlmError> {
        let mut body = responses::build_body(req)?;
        body["stream"] = json!(true);
        let resp = self.post("/responses", &body).await?;

        // a json body where a stream was asked for; old LM Studio says no-such-route this way
        let is_json = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|t| t.contains("application/json"));
        if is_json {
            let v: Value = serde_json::from_str(&resp.text().await?)?;
            return Err(responses::failure_of(&v).unwrap_or_else(|| {
                LlmError::Request("asked for a stream, got a single JSON body".into())
            }));
        }

        Ok(pump(
            resp,
            responses::StreamState::default(),
            responses::feed,
            "the stream ended before the response completed",
        ))
    }
}

#[async_trait]
impl LlmProvider for OpenAiProvider {
    fn name(&self) -> &'static str {
        "openai"
    }

    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse, LlmError> {
        if !self.try_responses(req) {
            return self.chat_completions(req).await;
        }
        match self.responses(req).await {
            Ok(out) => {
                self.remember(&req.model, true);
                Ok(out)
            }
            Err(e) if self.should_fall_back(&req.model, &e) => {
                tracing::info!(
                    endpoint = %self.base_url, model = %req.model, reason = %e.provider_message(),
                    "no /responses for this model; using /chat/completions"
                );
                let out = self.chat_completions(req).await;
                if out.is_ok() {
                    self.remember(&req.model, false);
                }
                out
            }
            Err(e) => Err(e),
        }
    }

    async fn chat_stream(&self, req: &ChatRequest) -> Result<TokenStream, LlmError> {
        if !self.try_responses(req) {
            return self.chat_completions_stream(req).await;
        }
        match self.responses_stream(req).await {
            Ok(stream) => {
                self.remember(&req.model, true);
                Ok(stream)
            }
            Err(e) if self.should_fall_back(&req.model, &e) => {
                tracing::info!(
                    endpoint = %self.base_url, model = %req.model, reason = %e.provider_message(),
                    "no /responses for this model; using /chat/completions"
                );
                let out = self.chat_completions_stream(req).await;
                if out.is_ok() {
                    self.remember(&req.model, false);
                }
                out
            }
            Err(e) => Err(e),
        }
    }
}

fn lacks_route(e: &LlmError) -> bool {
    match e {
        LlmError::Api { status: 404 | 405 | 501, .. } => true,
        LlmError::Api { status: 400, body, .. } => {
            let body = body.to_ascii_lowercase();
            ["unexpected endpoint", "unrecognized request url", "no route", "unknown url"]
                .iter()
                .any(|m| body.contains(m))
        }
        _ => false,
    }
}

fn parse_chat_completion(v: &Value, requested_model: &str) -> ChatResponse {
    let choice = &v["choices"][0];
    let message = &choice["message"];

    let tool_calls = message["tool_calls"]
        .as_array()
        .map(|calls| {
            calls
                .iter()
                .map(|c| ToolCall {
                    id: c["id"].as_str().unwrap_or_default().to_string(),
                    name: c["function"]["name"].as_str().unwrap_or_default().to_string(),
                    // openai sends arguments as a JSON string
                    arguments: c["function"]["arguments"]
                        .as_str()
                        .and_then(|s| serde_json::from_str(s).ok())
                        .unwrap_or(Value::Null),
                    signature: thought_signature(c),
                })
                .collect()
        })
        .unwrap_or_default();

    ChatResponse {
        // content is a string on some servers, an array of text parts on others
        text: match &message["content"] {
            Value::String(s) => s.clone(),
            Value::Array(arr) => arr
                .iter()
                .filter_map(|p| p["text"].as_str().or_else(|| p.as_str()))
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        },
        thinking: message["reasoning_content"]
            .as_str()
            .or(message["reasoning"].as_str())
            .map(str::to_string),
        tool_calls,
        stop_reason: choice["finish_reason"].as_str().unwrap_or("stop").to_string(),
        usage: Usage {
            input_tokens: v["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32,
            output_tokens: v["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32,
        },
        model: v["model"].as_str().unwrap_or(requested_model).to_string(),
    }
}

type Sender = futures::channel::mpsc::UnboundedSender<Result<StreamEvent, LlmError>>;
type Feed<S> = fn(&mut S, &SseRecord) -> Result<Vec<StreamEvent>, LlmError>;

// handshake is done, so errors here end the turn; a dropped stream kills the provider call
fn pump<S: Send + 'static>(
    resp: reqwest::Response,
    mut state: S,
    feed: Feed<S>,
    unfinished: &'static str,
) -> TokenStream {
    let (tx, rx) = futures::channel::mpsc::unbounded();
    tokio::spawn(async move {
        let mut bytes = resp.bytes_stream();
        let mut buf = String::new();
        loop {
            match bytes.next().await {
                Some(Ok(chunk)) => buf.push_str(&String::from_utf8_lossy(&chunk)),
                Some(Err(e)) => {
                    let _ = tx.unbounded_send(Err(LlmError::Http(e)));
                    return;
                }
                None => break,
            }
            for rec in split_sse_records(&mut buf) {
                if forward(&tx, &mut state, feed, &rec) {
                    return;
                }
            }
        }
        // flush trailing bytes; a well formed stream closes itself, ending without that means data was lost
        buf.push_str("\n\n");
        for rec in split_sse_records(&mut buf) {
            if forward(&tx, &mut state, feed, &rec) {
                return;
            }
        }
        let _ = tx.unbounded_send(Err(LlmError::Request(unfinished.into())));
    });
    Box::pin(rx)
}

fn forward<S>(tx: &Sender, state: &mut S, feed: Feed<S>, rec: &SseRecord) -> bool {
    match feed(state, rec) {
        Ok(events) => {
            for ev in events {
                let done = matches!(ev, StreamEvent::Done(_));
                let _ = tx.unbounded_send(Ok(ev));
                if done {
                    return true;
                }
            }
            false
        }
        Err(e) => {
            let _ = tx.unbounded_send(Err(e));
            true
        }
    }
}

fn feed_chat(state: &mut OpenAiStreamState, rec: &SseRecord) -> Result<Vec<StreamEvent>, LlmError> {
    feed_openai_record(state, rec).map(|ev| ev.into_iter().collect())
}

#[derive(Debug, Default)]
struct OpenAiStreamState {
    tools: std::collections::BTreeMap<u64, OpenAiToolBuilder>,
    // gemini omits index and sends each call whole; without this parallel calls pile into slot 0
    next_slot: u64,
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Debug, Default)]
struct OpenAiToolBuilder {
    id: String,
    name: String,
    arguments: String,
    signature: Option<String>,
}

// gemini hangs the signature off the call; the nested under function spot has been seen too
fn thought_signature(call: &Value) -> Option<String> {
    for place in [&call["extra_content"], &call["function"]["extra_content"]] {
        if let Some(sig) = place["google"]["thought_signature"].as_str() {
            return Some(sig.to_string());
        }
    }
    None
}

// explicit index wins; without one a fragment naming a function starts a new call
fn tool_slot(state: &mut OpenAiStreamState, tc: &Value) -> u64 {
    if let Some(index) = tc["index"].as_u64() {
        state.next_slot = state.next_slot.max(index + 1);
        return index;
    }
    if tc["function"]["name"].is_string() || tc["id"].is_string() {
        let slot = state.next_slot;
        state.next_slot += 1;
        return slot;
    }
    state.next_slot.saturating_sub(1)
}

fn feed_openai_record(
    state: &mut OpenAiStreamState,
    rec: &SseRecord,
) -> Result<Option<StreamEvent>, LlmError> {
    if rec.data.trim() == "[DONE]" {
        let tool_calls = state
            .tools
            .values()
            .map(|b| ToolCall {
                id: b.id.clone(),
                name: b.name.clone(),
                arguments: serde_json::from_str(&b.arguments).unwrap_or(Value::Null),
                signature: b.signature.clone(),
            })
            .collect();
        return Ok(Some(StreamEvent::Done(StreamDone {
            tool_calls,
            input_tokens: state.input_tokens,
            output_tokens: state.output_tokens,
        })));
    }

    let v: Value = serde_json::from_str(&rec.data)?;
    if let Some(u) = v.get("usage") {
        state.input_tokens = u["prompt_tokens"].as_u64().unwrap_or(0) as u32;
        state.output_tokens = u["completion_tokens"].as_u64().unwrap_or(0) as u32;
    }
    let delta = &v["choices"][0]["delta"];
    for tc in delta["tool_calls"].as_array().cloned().unwrap_or_default() {
        let signature = thought_signature(&tc);
        let slot = tool_slot(state, &tc);
        let builder = state.tools.entry(slot).or_default();
        if let Some(id) = tc["id"].as_str() {
            builder.id = id.to_string();
        }
        if let Some(name) = tc["function"]["name"].as_str() {
            builder.name = name.to_string();
        }
        if let Some(args) = tc["function"]["arguments"].as_str() {
            builder.arguments.push_str(args);
        }
        if signature.is_some() {
            builder.signature = signature;
        }
    }
    if let Some(t) = delta["content"].as_str().filter(|t| !t.is_empty()) {
        return Ok(Some(StreamEvent::Text(t.to_string())));
    }
    // reasoning rides in reasoning_content (deepseek/vllm/lm studio) or reasoning (openrouter/ollama)
    match delta["reasoning_content"].as_str().or(delta["reasoning"].as_str()) {
        Some(t) if !t.is_empty() => Ok(Some(StreamEvent::Thinking(t.to_string()))),
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(data: &str) -> SseRecord {
        SseRecord { event: None, data: data.into() }
    }

    #[test]
    fn streams_text_and_usage() {
        let mut state = OpenAiStreamState::default();
        let first = feed_openai_record(
            &mut state,
            &record(r#"{"choices":[{"delta":{"content":"hel"},"index":0}]}"#),
        )
        .unwrap();
        assert!(matches!(first, Some(StreamEvent::Text(ref t)) if t == "hel"));

        let usage = feed_openai_record(
            &mut state,
            &record(r#"{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":3}}"#),
        )
        .unwrap();
        assert!(usage.is_none());

        let done = feed_openai_record(&mut state, &record("[DONE]")).unwrap();
        match done {
            Some(StreamEvent::Done(d)) => {
                assert!(d.tool_calls.is_empty());
                assert_eq!((d.input_tokens, d.output_tokens), (10, 3));
            }
            other => panic!("expected Done, got {other:?}"),
        }
    }

    #[test]
    fn assembles_fragmented_tool_calls() {
        let mut state = OpenAiStreamState::default();
        for data in [
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"search_sources","arguments":""}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"query\":"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"diffusion\"}"}}]}}]}"#,
        ] {
            assert!(feed_openai_record(&mut state, &record(data)).unwrap().is_none());
        }
        match feed_openai_record(&mut state, &record("[DONE]")).unwrap() {
            Some(StreamEvent::Done(d)) => {
                assert_eq!(d.tool_calls.len(), 1);
                assert_eq!(d.tool_calls[0].name, "search_sources");
                assert_eq!(d.tool_calls[0].arguments["query"], "diffusion");
            }
            other => panic!("expected Done, got {other:?}"),
        }
    }

    #[test]
    fn keeps_gemini_signatures_and_separates_unindexed_calls() {
        let mut state = OpenAiStreamState::default();
        for data in [
            r#"{"choices":[{"delta":{"tool_calls":[{"id":"c1","function":{"name":"search_sources","arguments":"{\"query\":\"a\"}"},"extra_content":{"google":{"thought_signature":"sig-one"}}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"id":"c2","function":{"name":"read_source","arguments":"{\"title\":\"b\"}"}}]}}]}"#,
        ] {
            assert!(feed_openai_record(&mut state, &record(data)).unwrap().is_none());
        }
        match feed_openai_record(&mut state, &record("[DONE]")).unwrap() {
            Some(StreamEvent::Done(d)) => {
                assert_eq!(d.tool_calls.len(), 2);
                assert_eq!(d.tool_calls[0].arguments["query"], "a");
                assert_eq!(d.tool_calls[0].signature.as_deref(), Some("sig-one"));
                assert_eq!(d.tool_calls[1].arguments["title"], "b");
                assert!(d.tool_calls[1].signature.is_none());
            }
            other => panic!("expected Done, got {other:?}"),
        }
    }

    #[test]
    fn signed_calls_go_back_signed() {
        let history = vec![Message {
            role: Role::Assistant,
            content: vec![
                ContentPart::ToolUse {
                    id: "c1".into(),
                    name: "search_sources".into(),
                    input: json!({ "query": "a" }),
                    signature: Some("sig-one".into()),
                },
                ContentPart::ToolUse {
                    id: "c2".into(),
                    name: "read_source".into(),
                    input: json!({ "title": "b" }),
                    signature: None,
                },
            ],
        }];
        let body = OpenAiProvider::build_body(&ChatRequest::new("gemini", history)).unwrap();
        let calls = &body["messages"][0]["tool_calls"];

        assert_eq!(calls[0]["extra_content"]["google"]["thought_signature"], "sig-one");
        assert!(calls[1].get("extra_content").is_none());
    }

    #[test]
    fn broken_arguments_fall_back_to_null_like_chat() {
        let mut state = OpenAiStreamState::default();
        feed_openai_record(
            &mut state,
            &record(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c","function":{"name":"f","arguments":"{oops"}}]}}]}"#),
        )
        .unwrap();
        match feed_openai_record(&mut state, &record("[DONE]")).unwrap() {
            Some(StreamEvent::Done(d)) => assert_eq!(d.tool_calls[0].arguments, Value::Null),
            other => panic!("expected Done, got {other:?}"),
        }
    }

    #[test]
    fn reasoning_streams_beside_the_answer() {
        let mut state = OpenAiStreamState::default();
        let thought = feed_openai_record(
            &mut state,
            &record(r#"{"choices":[{"delta":{"reasoning_content":"The user wants"}}]}"#),
        )
        .unwrap();
        assert!(matches!(thought, Some(StreamEvent::Thinking(ref t)) if t == "The user wants"));

        let thought = feed_openai_record(
            &mut state,
            &record(r#"{"choices":[{"delta":{"reasoning":" a summary."}}]}"#),
        )
        .unwrap();
        assert!(matches!(thought, Some(StreamEvent::Thinking(ref t)) if t == " a summary."));

        let answer = feed_openai_record(
            &mut state,
            &record(r#"{"choices":[{"delta":{"content":"Here","reasoning_content":null}}]}"#),
        )
        .unwrap();
        assert!(matches!(answer, Some(StreamEvent::Text(ref t)) if t == "Here"));
    }

    fn api(status: u16, body: &str) -> LlmError {
        LlmError::Api { status, body: body.into(), retry_after: None }
    }

    #[test]
    fn a_missing_route_is_told_apart_from_a_bad_request() {
        assert!(lacks_route(&api(404, "")));
        assert!(lacks_route(&api(405, "method not allowed")));
        assert!(lacks_route(&api(501, "")));
        assert!(lacks_route(&api(400, "Unrecognized request URL (POST: /v1/responses)")));

        assert!(!lacks_route(&api(400, r#"{"error":{"message":"model field is required"}}"#)));
        assert!(!lacks_route(&api(429, "rate limit exceeded")));
        assert!(!lacks_route(&api(503, "high demand")));
        assert!(!lacks_route(&api(401, "invalid api key")));
    }

    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const JSON: &str = "application/json";
    const SSE: &str = "text/event-stream";

    const CHAT_OK: &str = r#"{"model":"m","choices":[{"message":{"content":"from chat"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2}}"#;
    const RESPONSES_OK: &str = r#"{"status":"completed","model":"m","output":[{"type":"message","content":[{"type":"output_text","text":"from responses"}]}],"usage":{"input_tokens":1,"output_tokens":2}}"#;
    const CHAT_SSE: &str = "data: {\"choices\":[{\"delta\":{\"content\":\"from chat\"}}]}\n\ndata: [DONE]\n\n";
    const RESPONSES_SSE: &str = "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"from responses\"}\n\nevent: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":1,\"output_tokens\":2}}}\n\n";

    type Script = (u16, &'static str, &'static str);

    struct Mock {
        base: String,
        responses_hits: Arc<AtomicUsize>,
        chat_hits: Arc<AtomicUsize>,
    }

    impl Mock {
        async fn start(responses: Script, chat: Script) -> Mock {
            use axum::http::{StatusCode, header};
            use axum::routing::post;

            let responses_hits = Arc::new(AtomicUsize::new(0));
            let chat_hits = Arc::new(AtomicUsize::new(0));
            let route = |hits: Arc<AtomicUsize>, (status, kind, body): Script| {
                post(move || {
                    let hits = hits.clone();
                    async move {
                        hits.fetch_add(1, Ordering::SeqCst);
                        (StatusCode::from_u16(status).unwrap(), [(header::CONTENT_TYPE, kind)], body)
                    }
                })
            };
            let app = axum::Router::new()
                .route("/v1/responses", route(responses_hits.clone(), responses))
                .route("/v1/chat/completions", route(chat_hits.clone(), chat));

            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            Mock { base: format!("http://{addr}/v1"), responses_hits, chat_hits }
        }

        fn provider(&self) -> OpenAiProvider {
            OpenAiProvider::new(reqwest::Client::new(), self.base.clone(), String::new())
        }

        fn hits(&self) -> (usize, usize) {
            (self.responses_hits.load(Ordering::SeqCst), self.chat_hits.load(Ordering::SeqCst))
        }
    }

    fn ask() -> ChatRequest {
        ChatRequest::new("m", vec![Message::user("hi")])
    }

    async fn collect(stream: TokenStream) -> (String, Option<StreamDone>) {
        let mut text = String::new();
        let mut done = None;
        for event in stream.collect::<Vec<_>>().await {
            match event.unwrap() {
                StreamEvent::Thinking(_) => {}
                StreamEvent::Text(t) => text.push_str(&t),
                StreamEvent::Done(d) => done = Some(d),
            }
        }
        (text, done)
    }

    #[tokio::test]
    async fn responses_is_used_where_it_is_served() {
        let mock = Mock::start((200, JSON, RESPONSES_OK), (500, JSON, "{}")).await;
        let out = mock.provider().chat(&ask()).await.unwrap();
        assert_eq!(out.text, "from responses");
        assert_eq!(mock.hits(), (1, 0), "chat completions is never touched");
    }

    #[tokio::test]
    async fn a_missing_route_falls_back_once_and_is_remembered() {
        let mock = Mock::start((404, JSON, ""), (200, JSON, CHAT_OK)).await;
        let provider = mock.provider();

        assert_eq!(provider.chat(&ask()).await.unwrap().text, "from chat");
        assert_eq!(provider.chat(&ask()).await.unwrap().text, "from chat");
        assert_eq!(mock.hits(), (1, 2));

        let mock_sse = Mock::start((404, JSON, ""), (200, SSE, CHAT_SSE)).await;
        let (text, done) = collect(mock_sse.provider().chat_stream(&ask()).await.unwrap()).await;
        assert_eq!(text, "from chat");
        assert!(done.is_some());
        assert_eq!(mock_sse.hits(), (1, 1));
    }

    #[tokio::test]
    async fn support_is_learned_per_model_not_per_endpoint() {
        use axum::http::{StatusCode, header};
        use axum::routing::post;

        let app = axum::Router::new()
            .route("/v1/responses", post(|body: String| async move {
                let v: Value = serde_json::from_str(&body).unwrap_or_default();
                if v["model"] == "served" {
                    (StatusCode::OK, [(header::CONTENT_TYPE, JSON)], RESPONSES_OK)
                } else {
                    (StatusCode::NOT_FOUND, [(header::CONTENT_TYPE, JSON)], "")
                }
            }))
            .route("/v1/chat/completions", post(|| async {
                (StatusCode::OK, [(header::CONTENT_TYPE, JSON)], CHAT_OK)
            }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let provider =
            OpenAiProvider::new(reqwest::Client::new(), format!("http://{addr}/v1"), String::new());

        let served = ChatRequest::new("served", vec![Message::user("hi")]);
        let unserved = ChatRequest::new("unserved", vec![Message::user("hi")]);
        assert_eq!(provider.chat(&served).await.unwrap().text, "from responses");
        assert_eq!(provider.chat(&unserved).await.unwrap().text, "from chat");
        assert_eq!(provider.chat(&served).await.unwrap().text, "from responses");
    }

    #[tokio::test]
    async fn a_busy_route_is_not_mistaken_for_a_missing_one() {
        let mock = Mock::start(
            (429, JSON, r#"{"error":{"message":"rate limit exceeded"}}"#),
            (200, JSON, CHAT_OK),
        )
        .await;
        let err = mock.provider().chat(&ask()).await.unwrap_err();
        assert!(err.is_retryable(), "the retry schedule gets it, not the fallback");
        assert_eq!(mock.hits(), (1, 0));
    }

    #[tokio::test]
    async fn streams_run_over_responses() {
        let mock = Mock::start((200, SSE, RESPONSES_SSE), (500, JSON, "{}")).await;
        let (text, done) = collect(mock.provider().chat_stream(&ask()).await.unwrap()).await;
        assert_eq!(text, "from responses");
        let done = done.expect("the turn closes");
        assert_eq!((done.input_tokens, done.output_tokens), (1, 2));
        assert_eq!(mock.hits(), (1, 0));
    }

    #[tokio::test]
    async fn a_200_that_says_no_such_route_still_falls_back() {
        let old = r#"{"error":"Unexpected endpoint or method. (POST /v1/responses)"}"#;

        let mock = Mock::start((200, JSON, old), (200, SSE, CHAT_SSE)).await;
        let (text, _) = collect(mock.provider().chat_stream(&ask()).await.unwrap()).await;
        assert_eq!(text, "from chat");

        let mock = Mock::start((200, JSON, old), (200, JSON, CHAT_OK)).await;
        assert_eq!(mock.provider().chat(&ask()).await.unwrap().text, "from chat");
    }

    #[tokio::test]
    async fn audio_goes_straight_to_chat_completions() {
        let mock = Mock::start((200, JSON, RESPONSES_OK), (200, JSON, CHAT_OK)).await;
        let req = ChatRequest::new(
            "m",
            vec![Message {
                role: Role::User,
                content: vec![ContentPart::Audio { media_type: "audio/mpeg".into(), data: "x".into() }],
            }],
        );
        assert_eq!(mock.provider().chat(&req).await.unwrap().text, "from chat");
        assert_eq!(mock.hits(), (0, 1), "Responses has no audio part to send it in");
    }
}
