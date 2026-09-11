//! Anthropic-style `/v1/messages`.

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{Value, json};

use super::types::*;
use super::{LlmError, LlmProvider, SseRecord, TokenStream, split_sse_records};

const API_VERSION: &str = "2023-06-01";

pub struct AnthropicProvider {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
}

impl AnthropicProvider {
    pub fn new(http: reqwest::Client, base_url: String, api_key: String) -> Self {
        AnthropicProvider { http, base_url: base_url.trim_end_matches('/').to_string(), api_key }
    }

    fn part_to_json(part: &ContentPart) -> Result<Value, LlmError> {
        Ok(match part {
            ContentPart::Text { text } => json!({ "type": "text", "text": text }),
            ContentPart::Image { media_type, data } => json!({
                "type": "image",
                "source": { "type": "base64", "media_type": media_type, "data": data }
            }),
            ContentPart::Document { media_type, data, .. } => json!({
                "type": "document",
                "source": { "type": "base64", "media_type": media_type, "data": data }
            }),
            ContentPart::Audio { .. } => {
                return Err(LlmError::Request(
                    "this Anthropic-style endpoint does not take audio input; \
                     assign an OpenAI-style omni model as the analyzer"
                        .into(),
                ));
            }
            ContentPart::Video { .. } => {
                return Err(LlmError::Request(
                    "this Anthropic-style endpoint does not take video input; \
                     assign an OpenAI-style omni model as the analyzer"
                        .into(),
                ));
            }
            ContentPart::ToolUse { id, name, input, .. } => json!({
                "type": "tool_use", "id": id, "name": name, "input": input
            }),
            ContentPart::ToolResult { tool_use_id, content, is_error } => json!({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": content,
                "is_error": is_error
            }),
        })
    }

    fn build_body(req: &ChatRequest) -> Result<Value, LlmError> {
        let mut messages = Vec::new();
        for m in &req.messages {
            // Anthropic has no `tool` role: tool results ride on a user turn.
            let role = match m.role {
                Role::Assistant => "assistant",
                Role::User | Role::Tool => "user",
                Role::System => {
                    return Err(LlmError::Request(
                        "system messages belong in ChatRequest::system".into(),
                    ));
                }
            };
            let content: Vec<Value> =
                m.content.iter().map(Self::part_to_json).collect::<Result<_, _>>()?;
            messages.push(json!({ "role": role, "content": content }));
        }

        // Thinking needs headroom above the budget, and forbids temperature.
        let budget = req.effort.budget_tokens();
        let max_tokens = match budget {
            Some(b) => req.max_tokens.max(b + 1024),
            None => req.max_tokens,
        };

        let mut body = json!({
            "model": req.model,
            "max_tokens": max_tokens,
            "messages": messages,
        });
        if let Some(system) = &req.system {
            body["system"] = json!(system);
        }
        match budget {
            Some(b) => {
                body["thinking"] = json!({ "type": "enabled", "budget_tokens": b });
            }
            None => {
                if let Some(t) = req.temperature {
                    body["temperature"] = json!(t);
                }
            }
        }
        if !req.tools.is_empty() {
            body["tools"] = json!(
                req.tools
                    .iter()
                    .map(|t| json!({
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.parameters
                    }))
                    .collect::<Vec<_>>()
            );
        }
        Ok(body)
    }
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    fn name(&self) -> &'static str {
        "anthropic"
    }

    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse, LlmError> {
        let body = Self::build_body(req)?;

        let resp = self
            .http
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
            .json(&body)
            .send()
            .await?;

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

        let mut text = String::new();
        let mut thinking = String::new();
        let mut tool_calls = Vec::new();
        for block in v["content"].as_array().unwrap_or(&Vec::new()) {
            match block["type"].as_str().unwrap_or_default() {
                "text" => {
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(block["text"].as_str().unwrap_or_default());
                }
                "thinking" => {
                    thinking.push_str(block["thinking"].as_str().unwrap_or_default());
                }
                "tool_use" => tool_calls.push(ToolCall {
                    id: block["id"].as_str().unwrap_or_default().to_string(),
                    name: block["name"].as_str().unwrap_or_default().to_string(),
                    arguments: block["input"].clone(),
                    signature: None,
                }),
                _ => {}
            }
        }

        Ok(ChatResponse {
            text,
            thinking: (!thinking.is_empty()).then_some(thinking),
            tool_calls,
            stop_reason: v["stop_reason"].as_str().unwrap_or("end_turn").to_string(),
            usage: Usage {
                input_tokens: v["usage"]["input_tokens"].as_u64().unwrap_or(0) as u32,
                output_tokens: v["usage"]["output_tokens"].as_u64().unwrap_or(0) as u32,
            },
            model: v["model"].as_str().unwrap_or(&req.model).to_string(),
        })
    }

    async fn chat_stream(&self, req: &ChatRequest) -> Result<TokenStream, LlmError> {
        let mut body = Self::build_body(req)?;
        body["stream"] = json!(true);

        let resp = self
            .http
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let raw = resp.text().await.unwrap_or_default();
            return Err(LlmError::Api { status: status.as_u16(), body: raw, retry_after: None });
        }

        // Same shape as the OpenAI pump: the handshake above stays retryable,
        // everything below ends the turn, and dropping the stream stops the
        // task with its provider connection.
        let (tx, rx) = futures::channel::mpsc::unbounded();
        tokio::spawn(async move {
            let mut bytes = resp.bytes_stream();
            let mut buf = String::new();
            let mut state = AnthropicStreamState::default();
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
                    if pump_anthropic_record(&tx, &mut state, &rec) {
                        return;
                    }
                }
            }
            // A well-formed stream closes with message_stop; anything else
            // lost data on the wire.
            buf.push_str("\n\n");
            for rec in split_sse_records(&mut buf) {
                if pump_anthropic_record(&tx, &mut state, &rec) {
                    return;
                }
            }
            let _ = tx.unbounded_send(Err(LlmError::Request(
                "the stream ended before message_stop".into(),
            )));
        });
        Ok(Box::pin(rx))
    }
}

/// Forwards one record's event, if any. True when the turn is over.
fn pump_anthropic_record(
    tx: &futures::channel::mpsc::UnboundedSender<Result<StreamEvent, LlmError>>,
    state: &mut AnthropicStreamState,
    rec: &SseRecord,
) -> bool {
    match feed_anthropic_record(state, rec) {
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

/// Tool input fragments in flight, keyed by content block index.
#[derive(Debug, Default)]
struct AnthropicStreamState {
    tools: std::collections::BTreeMap<u64, AnthropicToolBuilder>,
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Debug, Default)]
struct AnthropicToolBuilder {
    id: String,
    name: String,
    input: String,
}

/// Fold one SSE record into text, tool fragments and usage. `message_stop`
/// closes the turn with everything accumulated alongside it. Anything else
/// (pings, block opens and closes) yields nothing on its own.
fn feed_anthropic_record(
    state: &mut AnthropicStreamState,
    rec: &SseRecord,
) -> Result<Option<StreamEvent>, LlmError> {
    let v: Value = serde_json::from_str(&rec.data)?;
    match rec.event.as_deref() {
        Some("message_start") => {
            state.input_tokens = v["message"]["usage"]["input_tokens"].as_u64().unwrap_or(0) as u32;
            Ok(None)
        }
        Some("content_block_start") => {
            let block = &v["content_block"];
            if block["type"].as_str() == Some("tool_use") {
                let index = v["index"].as_u64().unwrap_or(0);
                state.tools.insert(index, AnthropicToolBuilder {
                    id: block["id"].as_str().unwrap_or_default().to_string(),
                    name: block["name"].as_str().unwrap_or_default().to_string(),
                    input: String::new(),
                });
            }
            Ok(None)
        }
        Some("content_block_delta") => {
            let delta = &v["delta"];
            match delta["type"].as_str() {
                Some("text_delta") => match delta["text"].as_str() {
                    Some(t) if !t.is_empty() => Ok(Some(StreamEvent::Text(t.to_string()))),
                    _ => Ok(None),
                },
                // Extended thinking streams as its own block, before the answer.
                Some("thinking_delta") => match delta["thinking"].as_str() {
                    Some(t) if !t.is_empty() => Ok(Some(StreamEvent::Thinking(t.to_string()))),
                    _ => Ok(None),
                },
                Some("input_json_delta") => {
                    let index = v["index"].as_u64().unwrap_or(0);
                    if let Some(part) = delta["partial_json"].as_str() {
                        state.tools.entry(index).or_default().input.push_str(part);
                    }
                    Ok(None)
                }
                _ => Ok(None),
            }
        }
        Some("message_delta") => {
            state.output_tokens =
                v["usage"]["output_tokens"].as_u64().unwrap_or(0) as u32;
            Ok(None)
        }
        Some("message_stop") => {
            let tool_calls = state
                .tools
                .values()
                .map(|b| ToolCall {
                    id: b.id.clone(),
                    name: b.name.clone(),
                    // Same leniency as chat(): broken JSON becomes Null.
                    arguments: serde_json::from_str(&b.input).unwrap_or(Value::Null),
                    signature: None,
                })
                .collect();
            Ok(Some(StreamEvent::Done(StreamDone {
                tool_calls,
                input_tokens: state.input_tokens,
                output_tokens: state.output_tokens,
            })))
        }
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(event: &str, data: &str) -> SseRecord {
        SseRecord { event: Some(event.into()), data: data.into() }
    }

    #[test]
    fn streams_text_and_usage() {
        let mut state = AnthropicStreamState::default();
        let start = feed_anthropic_record(
            &mut state,
            &record("message_start", r#"{"message":{"usage":{"input_tokens":12}}}"#),
        )
        .unwrap();
        assert!(start.is_none());

        let text = feed_anthropic_record(
            &mut state,
            &record(
                "content_block_delta",
                r#"{"index":0,"delta":{"type":"text_delta","text":"hi"}}"#,
            ),
        )
        .unwrap();
        assert!(matches!(text, Some(StreamEvent::Text(ref t)) if t == "hi"));

        // Reasoning streams on its own channel, never as answer text.
        let thought = feed_anthropic_record(
            &mut state,
            &record(
                "content_block_delta",
                r#"{"index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}"#,
            ),
        )
        .unwrap();
        assert!(matches!(thought, Some(StreamEvent::Thinking(ref t)) if t == "hmm"));

        let delta = feed_anthropic_record(
            &mut state,
            &record("message_delta", r#"{"usage":{"output_tokens":4}}"#),
        )
        .unwrap();
        assert!(delta.is_none());

        match feed_anthropic_record(&mut state, &record("message_stop", "{}")).unwrap() {
            Some(StreamEvent::Done(d)) => {
                assert!(d.tool_calls.is_empty());
                assert_eq!((d.input_tokens, d.output_tokens), (12, 4));
            }
            other => panic!("expected Done, got {other:?}"),
        }
    }

    #[test]
    fn assembles_fragmented_tool_input() {
        let mut state = AnthropicStreamState::default();
        for (event, data) in [
            (
                "content_block_start",
                r#"{"index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"read_source"}}"#,
            ),
            (
                "content_block_delta",
                r#"{"index":1,"delta":{"type":"input_json_delta","partial_json":"{\"title\":"}}"#,
            ),
            (
                "content_block_delta",
                r#"{"index":1,"delta":{"type":"input_json_delta","partial_json":"\"fuchs.pdf\"}"}}"#,
            ),
        ] {
            assert!(feed_anthropic_record(&mut state, &record(event, data)).unwrap().is_none());
        }
        match feed_anthropic_record(&mut state, &record("message_stop", "{}")).unwrap() {
            Some(StreamEvent::Done(d)) => {
                assert_eq!(d.tool_calls.len(), 1);
                assert_eq!(d.tool_calls[0].name, "read_source");
                assert_eq!(d.tool_calls[0].arguments["title"], "fuchs.pdf");
            }
            other => panic!("expected Done, got {other:?}"),
        }
    }

    #[test]
    fn ignores_pings_and_unknown_blocks() {
        let mut state = AnthropicStreamState::default();
        let ping = SseRecord { event: Some("ping".into()), data: r#"{"type":"ping"}"#.into() };
        assert!(feed_anthropic_record(&mut state, &ping).unwrap().is_none());
    }
}
