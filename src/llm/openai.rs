//! OpenAI-style `/chat/completions`. Works against any compatible endpoint
//! (vLLM, llama.cpp, OpenRouter, Azure-style proxies) by changing the base URL.

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{Value, json};

use super::types::*;
use super::{LlmError, LlmProvider, SseRecord, TokenStream, split_sse_records};

pub struct OpenAiProvider {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
}

impl OpenAiProvider {
    pub fn new(http: reqwest::Client, base_url: String, api_key: String) -> Self {
        OpenAiProvider { http, base_url: base_url.trim_end_matches('/').to_string(), api_key }
    }

    /// OpenAI splits one neutral message into possibly several wire messages:
    /// each tool result is its own `role: "tool"` entry.
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
                // vLLM / Nemotron Omni style: data URIs work without sharing a
                // filesystem with the server, unlike `file://` URIs.
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
                    // Gemini rejects a history whose function calls come back
                    // unsigned ("missing a thought_signature"), so whatever it
                    // signed the call with goes back exactly where it came from.
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
            // Bare text on a Tool turn has nowhere to go; treat it as user context.
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
}

#[async_trait]
impl LlmProvider for OpenAiProvider {
    fn name(&self) -> &'static str {
        "openai"
    }

    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse, LlmError> {
        let body = Self::build_body(req)?;

        // A local endpoint (LM Studio, Ollama, vLLM) takes no key at all.
        let mut req_builder = self.http.post(format!("{}/chat/completions", self.base_url));
        if !self.api_key.is_empty() {
            req_builder = req_builder.bearer_auth(&self.api_key);
        }
        let resp = req_builder.json(&body).send().await?;

        let status = resp.status();
        let retry_after = resp
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.trim().parse::<u64>().ok());
        let raw = resp.text().await?;
        if !status.is_success() {
            return Err(LlmError::Api { status: status.as_u16(), body: raw, retry_after });
        }
        let v: Value = serde_json::from_str(&raw)?;
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
                        // OpenAI sends arguments as a JSON *string*.
                        arguments: c["function"]["arguments"]
                            .as_str()
                            .and_then(|s| serde_json::from_str(s).ok())
                            .unwrap_or(Value::Null),
                        signature: thought_signature(c),
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(ChatResponse {
            // Some servers return content as a string, others as an array of
            // `{ "type": "text", "text": ... }` parts.
            text: match &message["content"] {
                Value::String(s) => s.clone(),
                Value::Array(arr) => arr
                    .iter()
                    .filter_map(|p| p["text"].as_str().or_else(|| p.as_str()))
                    .collect::<Vec<_>>()
                    .join("\n"),
                _ => String::new(),
            },
            thinking: message["reasoning_content"].as_str().map(str::to_string),
            tool_calls,
            stop_reason: choice["finish_reason"].as_str().unwrap_or("stop").to_string(),
            usage: Usage {
                input_tokens: v["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32,
                output_tokens: v["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32,
            },
            model: v["model"].as_str().unwrap_or(&req.model).to_string(),
        })
    }

    async fn chat_stream(&self, req: &ChatRequest) -> Result<TokenStream, LlmError> {
        let mut body = Self::build_body(req)?;
        body["stream"] = json!(true);
        // Without this most servers omit usage from streamed chunks.
        body["stream_options"] = json!({ "include_usage": true });

        // A local endpoint (LM Studio, Ollama, vLLM) takes no key at all.
        let mut req_builder = self.http.post(format!("{}/chat/completions", self.base_url));
        if !self.api_key.is_empty() {
            req_builder = req_builder.bearer_auth(&self.api_key);
        }
        let resp = req_builder.json(&body).send().await?;

        let status = resp.status();
        let retry_after = resp
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.trim().parse::<u64>().ok());
        if !status.is_success() {
            let raw = resp.text().await.unwrap_or_default();
            return Err(LlmError::Api { status: status.as_u16(), body: raw, retry_after });
        }

        // The handshake is done, so establishment errors above stay retryable
        // while everything below ends the turn instead. A pump task forwards
        // events: dropping the stream (a stopped turn) drops the receiver,
        // the sends fail, and the task and its provider connection go with it.
        let (tx, rx) = futures::channel::mpsc::unbounded();
        tokio::spawn(async move {
            let mut bytes = resp.bytes_stream();
            let mut buf = String::new();
            let mut state = OpenAiStreamState::default();
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
                    if pump_openai_record(&tx, &mut state, &rec) {
                        return;
                    }
                }
            }
            // Flush whatever the body ended with. A well-formed stream closes
            // with [DONE]; anything else lost data on the wire.
            buf.push_str("\n\n");
            for rec in split_sse_records(&mut buf) {
                if pump_openai_record(&tx, &mut state, &rec) {
                    return;
                }
            }
            let _ = tx.unbounded_send(Err(LlmError::Request(
                "the stream ended before [DONE]".into(),
            )));
        });
        Ok(Box::pin(rx))
    }
}

/// Forwards one record's event, if any. True when the turn is over.
fn pump_openai_record(
    tx: &futures::channel::mpsc::UnboundedSender<Result<StreamEvent, LlmError>>,
    state: &mut OpenAiStreamState,
    rec: &SseRecord,
) -> bool {
    match feed_openai_record(state, rec) {
        Ok(Some(ev)) => {
            let done = matches!(ev, StreamEvent::Done(_));
            let _ = tx.unbounded_send(Ok(ev));
            done
        }
        Ok(None) => false,
        Err(e) => {
            let _ = tx.unbounded_send(Err(e));
            true
        }
    }
}

/// Tool call fragments in flight, keyed by their `index`.
#[derive(Debug, Default)]
struct OpenAiStreamState {
    tools: std::collections::BTreeMap<u64, OpenAiToolBuilder>,
    /// Where the next call without an `index` goes. Gemini's compatibility
    /// endpoint omits the field and sends each call whole in one chunk, so
    /// without this every parallel call would pile into slot 0.
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

/// Gemini hangs its per-call thought signature off the tool call itself. The
/// nesting under `function` has been seen in the wild too, so look in both.
fn thought_signature(call: &Value) -> Option<String> {
    for place in [&call["extra_content"], &call["function"]["extra_content"]] {
        if let Some(sig) = place["google"]["thought_signature"].as_str() {
            return Some(sig.to_string());
        }
    }
    None
}

/// The slot a streamed fragment belongs to. An explicit `index` wins; without
/// one, a fragment that names a function starts a new call and anything else
/// continues the call in flight.
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

/// Fold one SSE record into text, tool fragments and usage. `[DONE]` closes
/// the turn with everything accumulated alongside it.
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
                // Arguments arrive as JSON *string* fragments, same as chat().
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
        // Usage-only chunks carry no choices at all.
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
    match delta["content"].as_str() {
        Some(t) if !t.is_empty() => Ok(Some(StreamEvent::Text(t.to_string()))),
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

        // A usage-only chunk carries no choices; the text keeps flowing after.
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

    /// Gemini streams each call complete in one chunk, with a signature and
    /// no `index`. Two of them are two calls, not one call twice.
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

    /// The signature has to leave the way it arrived, or the next request is a
    /// 400 naming the call it belongs to.
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
        // An unsigned call stays clean, so plain OpenAI endpoints see no change.
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
}
