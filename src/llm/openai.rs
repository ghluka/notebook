//! OpenAI-style `/chat/completions`. Works against any compatible endpoint
//! (vLLM, llama.cpp, OpenRouter, Azure-style proxies) by changing the base URL.

use async_trait::async_trait;
use serde_json::{Value, json};

use super::types::*;
use super::{LlmError, LlmProvider};

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
                ContentPart::ToolUse { id, name, input } => tool_calls.push(json!({
                    "id": id,
                    "type": "function",
                    "function": { "name": name, "arguments": input.to_string() }
                })),
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
}
