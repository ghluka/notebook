//! Anthropic-style `/v1/messages`.

use async_trait::async_trait;
use serde_json::{Value, json};

use super::types::*;
use super::{LlmError, LlmProvider};

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
            ContentPart::ToolUse { id, name, input } => json!({
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
}
