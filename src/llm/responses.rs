//! OpenAI-style `/responses`, the newer of OpenAI's two wire formats.
//!
//! `openai.rs` decides which format a request goes out in and does the HTTP;
//! this file only translates. Responses carries the same conversation as chat
//! completions in a different shape. The system prompt is `instructions`. The
//! history is a flat list of `input` items, where a tool call and its result
//! are items in their own right rather than fields hanging off a message. And
//! reasoning comes back as an item beside the answer instead of a field inside
//! it, which is how LM Studio, NVIDIA and DeepSeek all return it.
//!
//! Audio and video have no part type in this format at all, so requests that
//! carry them stay on chat completions, where the `audio_url` and `video_url`
//! extension lives. `can_carry` is that test.

use std::collections::BTreeMap;

use serde_json::{Value, json};

use super::types::*;
use super::{LlmError, SseRecord, mentions_rate_limit};

/// Whether this request can be said in this format at all.
pub fn can_carry(req: &ChatRequest) -> bool {
    !req.messages
        .iter()
        .flat_map(|m| &m.content)
        .any(|p| matches!(p, ContentPart::Audio { .. } | ContentPart::Video { .. }))
}

pub fn build_body(req: &ChatRequest) -> Result<Value, LlmError> {
    let mut input = Vec::new();
    for m in &req.messages {
        push_items(&mut input, m)?;
    }

    let mut body = json!({
        "model": req.model,
        "input": input,
        "max_output_tokens": req.max_tokens,
        // Every request carries its whole history, so nothing needs keeping on
        // the provider's side, and a notebook's sources have no business sitting
        // in someone else's request log by default.
        "store": false,
    });
    if let Some(system) = &req.system {
        body["instructions"] = json!(system);
    }
    if req.effort == Effort::Off {
        if let Some(t) = req.temperature {
            body["temperature"] = json!(t);
        }
    } else {
        body["reasoning"] = json!({ "effort": req.effort.as_str() });
    }
    if !req.tools.is_empty() {
        // Flat, unlike chat completions, which nests these under `function`.
        body["tools"] = json!(
            req.tools
                .iter()
                .map(|t| json!({
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }))
                .collect::<Vec<_>>()
        );
    }
    Ok(body)
}

/// One neutral message becomes up to three kinds of item: the message itself,
/// then any calls it made, while results for earlier calls stand on their own.
fn push_items(out: &mut Vec<Value>, m: &Message) -> Result<(), LlmError> {
    let mut parts: Vec<Value> = Vec::new();
    let mut said: Vec<&str> = Vec::new();
    let mut calls: Vec<Value> = Vec::new();

    for part in &m.content {
        match part {
            ContentPart::Text { text } => {
                said.push(text);
                parts.push(json!({ "type": "input_text", "text": text }));
            }
            ContentPart::Image { media_type, data } => parts.push(json!({
                "type": "input_image",
                "image_url": format!("data:{media_type};base64,{data}"),
            })),
            ContentPart::Document { media_type, data, filename } => parts.push(json!({
                "type": "input_file",
                "filename": filename.clone().unwrap_or_else(|| "document".into()),
                "file_data": format!("data:{media_type};base64,{data}"),
            })),
            ContentPart::Audio { .. } | ContentPart::Video { .. } => {
                return Err(LlmError::Request(
                    "audio and video have no part in the Responses format; this request \
                     belongs on chat completions"
                        .into(),
                ));
            }
            ContentPart::ToolUse { id, name, input, .. } => calls.push(json!({
                "type": "function_call",
                "call_id": id,
                "name": name,
                "arguments": input.to_string(),
            })),
            ContentPart::ToolResult { tool_use_id, content, .. } => out.push(json!({
                "type": "function_call_output",
                "call_id": tool_use_id,
                "output": content,
            })),
        }
    }

    match m.role {
        // What the model said before goes back as a plain string: `input_text`
        // is refused on an assistant turn, and a string is the one shape every
        // implementation accepts there.
        Role::Assistant => {
            if parts.len() != said.len() {
                return Err(LlmError::Request(
                    "an assistant turn can only carry text and tool calls".into(),
                ));
            }
            if !said.is_empty() {
                out.push(json!({ "role": "assistant", "content": said.join("\n") }));
            }
        }
        role => {
            if !parts.is_empty() {
                // Bare text on a tool turn has nowhere else to go, same as in
                // chat completions: it is user context.
                let role = if role == Role::System { "system" } else { "user" };
                out.push(json!({ "role": role, "content": parts }));
            }
        }
    }
    out.extend(calls);
    Ok(())
}

/// A finished response, as the neutral type.
pub fn parse_response(v: &Value, requested_model: &str) -> Result<ChatResponse, LlmError> {
    if let Some(failure) = failure_of(v) {
        return Err(failure);
    }

    let mut text = Vec::new();
    let mut thinking = Vec::new();
    let mut tool_calls = Vec::new();
    for item in v["output"].as_array().into_iter().flatten() {
        match item["type"].as_str() {
            Some("message") => text.push(message_text(item)),
            Some("reasoning") => thinking.extend(reasoning_text(item)),
            Some("function_call") => tool_calls.push(call_of(item)),
            _ => {}
        }
    }
    let mut text = text.into_iter().filter(|t| !t.is_empty()).collect::<Vec<_>>().join("\n");
    // A convenience field some servers add. The items are the authority.
    if text.is_empty()
        && let Some(t) = v["output_text"].as_str()
    {
        text = t.to_string();
    }

    // Named the way chat completions names them, so nothing downstream has to
    // know which format the answer arrived in.
    let stop_reason = match v["status"].as_str() {
        Some("incomplete") => match v["incomplete_details"]["reason"].as_str() {
            Some("max_output_tokens") | None => "length".to_string(),
            Some(other) => other.to_string(),
        },
        _ if !tool_calls.is_empty() => "tool_calls".to_string(),
        _ => "stop".to_string(),
    };

    Ok(ChatResponse {
        text,
        thinking: (!thinking.is_empty()).then(|| thinking.join("\n")),
        tool_calls,
        stop_reason,
        usage: usage_of(&v["usage"]),
        model: v["model"].as_str().unwrap_or(requested_model).to_string(),
    })
}

/// A message item's text. Content is usually a list of parts but a bare
/// string has been seen too. A refusal is what the model said, so it counts.
fn message_text(item: &Value) -> String {
    match &item["content"] {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| match p["type"].as_str() {
                Some("output_text") | Some("text") => p["text"].as_str(),
                Some("refusal") => p["refusal"].as_str(),
                _ => None,
            })
            .collect::<String>(),
        _ => String::new(),
    }
}

/// OpenAI exposes reasoning as a `summary`; LM Studio, NVIDIA and DeepSeek
/// send the reasoning itself as `reasoning_text` content. Either will do.
fn reasoning_text(item: &Value) -> Vec<String> {
    ["summary", "content"]
        .iter()
        .flat_map(|field| item[*field].as_array().into_iter().flatten())
        .filter_map(|p| p["text"].as_str())
        .filter(|t| !t.trim().is_empty())
        .map(str::to_string)
        .collect()
}

fn call_of(item: &Value) -> ToolCall {
    ToolCall {
        // `call_id` is what the result has to answer to; `id` names the item.
        id: item["call_id"].as_str().or(item["id"].as_str()).unwrap_or_default().to_string(),
        name: item["name"].as_str().unwrap_or_default().to_string(),
        arguments: parse_arguments(&item["arguments"]),
        signature: None,
    }
}

/// Arguments arrive as a JSON string, as in chat completions, and a broken one
/// becomes null the same way. An object is taken as it is.
fn parse_arguments(raw: &Value) -> Value {
    match raw {
        Value::String(s) => serde_json::from_str(s).unwrap_or(Value::Null),
        Value::Object(_) => raw.clone(),
        _ => Value::Null,
    }
}

fn usage_of(u: &Value) -> Usage {
    Usage {
        input_tokens: u["input_tokens"].as_u64().unwrap_or(0) as u32,
        output_tokens: u["output_tokens"].as_u64().unwrap_or(0) as u32,
    }
}

/// A 200 is not always a success in this format: a response can report its
/// own failure, and an older LM Studio answers a route it does not have with a
/// 200 and an error string.
pub(super) fn failure_of(v: &Value) -> Option<LlmError> {
    let error = &v["error"];
    let failed = v["status"].as_str() == Some("failed");
    if error.is_null() && !failed {
        return None;
    }
    let body = if error.is_null() { v.to_string() } else { json!({ "error": error }).to_string() };
    Some(LlmError::Api { status: failure_status(error), body, retry_after: None })
}

/// The HTTP status a failure reported inside a body amounts to, so the retry
/// and fallback rules treat it as they would the real thing.
fn failure_status(error: &Value) -> u16 {
    let said = match error {
        Value::String(s) => s.to_ascii_lowercase(),
        other => other.to_string().to_ascii_lowercase(),
    };
    if said.contains("unexpected endpoint") || (said.contains("not found") && said.contains("endpoint"))
    {
        404
    } else if mentions_rate_limit(&said) {
        429
    } else if said.contains("server_error") || said.contains("internal") {
        500
    } else {
        400
    }
}

/// What a streamed response has accumulated so far.
#[derive(Debug, Default)]
pub struct StreamState {
    /// Calls in flight, keyed by the output slot they occupy, so they come out
    /// in the order the model made them.
    calls: BTreeMap<u64, CallBuilder>,
    /// Whether any answer text has gone out yet. A server that only ever sends
    /// the finished text still gets its answer through, exactly once.
    streamed_text: bool,
}

#[derive(Debug, Default)]
struct CallBuilder {
    id: String,
    name: String,
    arguments: String,
}

impl CallBuilder {
    /// Take what a `function_call` item says. The finished item is the
    /// authority on arguments; the opening one usually has none yet.
    fn fill(&mut self, item: &Value, finished: bool) {
        if let Some(id) = item["call_id"].as_str() {
            self.id = id.to_string();
        } else if self.id.is_empty()
            && let Some(id) = item["id"].as_str()
        {
            self.id = id.to_string();
        }
        if let Some(name) = item["name"].as_str() {
            self.name = name.to_string();
        }
        if let Some(arguments) = item["arguments"].as_str()
            && (finished || self.arguments.is_empty())
        {
            self.arguments = arguments.to_string();
        }
    }
}

/// Fold one SSE record into the events it amounts to. A record yields at most
/// a piece of text and the close of the turn, in that order.
///
/// Implementations differ in how much they stream. OpenAI streams argument
/// deltas; LM Studio sends a call's arguments only once they are complete. So
/// every event that can carry a call is read, and the latest complete value
/// wins over whatever the deltas built up.
pub fn feed(state: &mut StreamState, rec: &SseRecord) -> Result<Vec<StreamEvent>, LlmError> {
    if rec.data.trim() == "[DONE]" {
        return Ok(vec![finish(state, &Value::Null)]);
    }
    let v: Value = serde_json::from_str(&rec.data)?;
    // The event name rides on the `event:` line and again inside the payload;
    // the payload is the copy every implementation sends.
    let kind = v["type"].as_str().or(rec.event.as_deref()).unwrap_or_default();
    let slot = v["output_index"].as_u64().unwrap_or(0);

    match kind {
        "response.output_text.delta" => {
            let delta = v["delta"].as_str().unwrap_or_default();
            if delta.is_empty() {
                return Ok(Vec::new());
            }
            state.streamed_text = true;
            Ok(vec![StreamEvent::Text(delta.to_string())])
        }
        "response.output_text.done" if !state.streamed_text => match v["text"].as_str() {
            Some(text) if !text.is_empty() => {
                state.streamed_text = true;
                Ok(vec![StreamEvent::Text(text.to_string())])
            }
            _ => Ok(Vec::new()),
        },
        // Reasoning, as it is thought. LM Studio, NVIDIA and DeepSeek stream
        // the reasoning itself, OpenAI a summary of it; either way it is not
        // the answer, and it goes out on its own channel.
        "response.reasoning_text.delta" | "response.reasoning_summary_text.delta" => {
            match v["delta"].as_str() {
                Some(d) if !d.is_empty() => Ok(vec![StreamEvent::Thinking(d.to_string())]),
                _ => Ok(Vec::new()),
            }
        }
        "response.output_item.added" | "response.output_item.done" => {
            if v["item"]["type"] == "function_call" {
                let finished = kind.ends_with(".done");
                state.calls.entry(slot).or_default().fill(&v["item"], finished);
            }
            Ok(Vec::new())
        }
        "response.function_call_arguments.delta" => {
            if let Some(delta) = v["delta"].as_str() {
                state.calls.entry(slot).or_default().arguments.push_str(delta);
            }
            Ok(Vec::new())
        }
        "response.function_call_arguments.done" => {
            if let Some(arguments) = v["arguments"].as_str() {
                state.calls.entry(slot).or_default().arguments = arguments.to_string();
            }
            Ok(Vec::new())
        }
        "response.completed" | "response.incomplete" => {
            let response = &v["response"];
            let mut events = Vec::new();
            if !state.streamed_text {
                let text = response["output"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter(|item| item["type"] == "message")
                    .map(message_text)
                    .collect::<String>();
                if !text.is_empty() {
                    state.streamed_text = true;
                    events.push(StreamEvent::Text(text));
                }
            }
            events.push(finish(state, response));
            Ok(events)
        }
        "response.failed" => Err(failure_of(&v["response"]).unwrap_or(LlmError::Api {
            status: 500,
            body: v.to_string(),
            retry_after: None,
        })),
        // OpenAI's stream-level error carries its fields at the top.
        "error" => {
            let error = json!({ "message": v["message"], "code": v["code"] });
            Err(LlmError::Api {
                status: failure_status(&error),
                body: json!({ "error": error }).to_string(),
                retry_after: None,
            })
        }
        _ => Ok(Vec::new()),
    }
}

/// Close the turn. A server that never streamed its calls still lists them in
/// the final response, so that is read too when nothing else arrived.
fn finish(state: &mut StreamState, response: &Value) -> StreamEvent {
    if state.calls.is_empty() {
        for (i, item) in response["output"].as_array().into_iter().flatten().enumerate() {
            if item["type"] == "function_call" {
                state.calls.entry(i as u64).or_default().fill(item, true);
            }
        }
    }
    let usage = usage_of(&response["usage"]);
    let tool_calls = state
        .calls
        .values()
        .map(|b| ToolCall {
            id: b.id.clone(),
            name: b.name.clone(),
            arguments: parse_arguments(&Value::String(b.arguments.clone())),
            signature: None,
        })
        .collect();
    StreamEvent::Done(StreamDone {
        tool_calls,
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(data: &str) -> SseRecord {
        SseRecord { event: None, data: data.into() }
    }

    fn feed_all(state: &mut StreamState, records: &[&str]) -> Vec<StreamEvent> {
        records.iter().flat_map(|r| feed(state, &record(r)).unwrap()).collect()
    }

    fn done_of(events: &[StreamEvent]) -> &StreamDone {
        match events.last() {
            Some(StreamEvent::Done(d)) => d,
            other => panic!("expected the turn to close, got {other:?}"),
        }
    }

    #[test]
    fn a_conversation_becomes_items() {
        let history = vec![
            Message {
                role: Role::User,
                content: vec![
                    ContentPart::text("what is on this page?"),
                    ContentPart::Image { media_type: "image/png".into(), data: "AAAA".into() },
                    ContentPart::Document {
                        media_type: "application/pdf".into(),
                        data: "BBBB".into(),
                        filename: Some("notes.pdf".into()),
                    },
                ],
            },
            Message {
                role: Role::Assistant,
                content: vec![
                    ContentPart::text("Let me look."),
                    ContentPart::ToolUse {
                        id: "call_1".into(),
                        name: "search_sources".into(),
                        input: json!({ "query": "gcd" }),
                        signature: None,
                    },
                ],
            },
            Message::tool_results(vec![ContentPart::ToolResult {
                tool_use_id: "call_1".into(),
                content: "three excerpts".into(),
                is_error: false,
            }]),
        ];
        let req = ChatRequest::new("m", history)
            .system("You are the researcher.")
            .max_tokens(900)
            .effort(Effort::High)
            .tools(vec![Tool {
                name: "search_sources".into(),
                description: "Search.".into(),
                parameters: json!({ "type": "object" }),
            }]);
        let body = build_body(&req).unwrap();

        assert_eq!(body["instructions"], "You are the researcher.");
        assert_eq!(body["max_output_tokens"], 900);
        assert_eq!(body["store"], false);
        assert_eq!(body["reasoning"]["effort"], "high");
        assert!(body.get("temperature").is_none(), "reasoning models refuse a temperature");
        assert_eq!(body["tools"][0]["name"], "search_sources", "tools are flat here");
        assert!(body["tools"][0].get("function").is_none());

        let input = body["input"].as_array().unwrap();
        assert_eq!(input.len(), 4, "{input:#?}");
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[0]["content"][0]["type"], "input_text");
        assert_eq!(input[0]["content"][1]["type"], "input_image");
        assert_eq!(input[0]["content"][1]["image_url"], "data:image/png;base64,AAAA");
        assert_eq!(input[0]["content"][2]["type"], "input_file");
        assert_eq!(input[0]["content"][2]["filename"], "notes.pdf");
        // What it said, then the call it made, then the answer to that call.
        assert_eq!(input[1], json!({ "role": "assistant", "content": "Let me look." }));
        assert_eq!(input[2]["type"], "function_call");
        assert_eq!(input[2]["call_id"], "call_1");
        assert_eq!(input[2]["arguments"], r#"{"query":"gcd"}"#);
        assert_eq!(input[3]["type"], "function_call_output");
        assert_eq!(input[3]["call_id"], "call_1");
        assert_eq!(input[3]["output"], "three excerpts");
    }

    #[test]
    fn audio_and_video_stay_on_chat_completions() {
        let plain = ChatRequest::new("m", vec![Message::user("hi")]);
        assert!(can_carry(&plain));

        for part in [
            ContentPart::Audio { media_type: "audio/mpeg".into(), data: "x".into() },
            ContentPart::Video { media_type: "video/mp4".into(), data: "x".into() },
        ] {
            let req =
                ChatRequest::new("m", vec![Message { role: Role::User, content: vec![part] }]);
            assert!(!can_carry(&req));
            assert!(build_body(&req).is_err(), "refused, never silently dropped");
        }
    }

    /// LM Studio's answer to "say ok" with 16 tokens to spare: all of them went
    /// on reasoning, and that is the whole of the output.
    #[test]
    fn a_response_cut_short_while_reasoning_says_so() {
        let v = json!({
            "status": "incomplete",
            "incomplete_details": { "reason": "max_output_tokens" },
            "model": "nvidia/nemotron-3-nano-4b",
            "output": [{
                "type": "reasoning",
                "summary": [],
                "content": [{ "type": "reasoning_text", "text": "The user says: say ok." }]
            }],
            "error": null,
            "usage": { "input_tokens": 23, "output_tokens": 16 }
        });
        let out = parse_response(&v, "fallback").unwrap();
        assert_eq!(out.text, "");
        assert_eq!(out.stop_reason, "length");
        assert_eq!(out.thinking.as_deref(), Some("The user says: say ok."));
        assert_eq!((out.usage.input_tokens, out.usage.output_tokens), (23, 16));
        assert_eq!(out.model, "nvidia/nemotron-3-nano-4b");
    }

    #[test]
    fn answers_and_calls_come_out_of_the_items() {
        let v = json!({
            "status": "completed",
            "output": [
                { "type": "reasoning", "summary": [{ "type": "summary_text", "text": "Need a search." }] },
                { "type": "message", "role": "assistant",
                  "content": [{ "type": "output_text", "text": "Searching " },
                              { "type": "output_text", "text": "now." }] },
                { "type": "function_call", "id": "fc_1", "call_id": "call_9",
                  "name": "search_sources", "arguments": "{\"query\":\"bezout\"}" }
            ],
            "usage": { "input_tokens": 100, "output_tokens": 20 }
        });
        let out = parse_response(&v, "m").unwrap();
        assert_eq!(out.text, "Searching now.");
        assert_eq!(out.thinking.as_deref(), Some("Need a search."));
        assert_eq!(out.stop_reason, "tool_calls");
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].id, "call_9", "the result answers to call_id, not the item id");
        assert_eq!(out.tool_calls[0].arguments["query"], "bezout");
    }

    #[test]
    fn a_failure_inside_a_200_is_still_a_failure() {
        // An older LM Studio, asked for a route it does not have.
        let missing = json!({ "error": "Unexpected endpoint or method. (POST /v1/responses)" });
        match parse_response(&missing, "m").unwrap_err() {
            LlmError::Api { status, .. } => assert_eq!(status, 404, "so the fallback fires"),
            other => panic!("expected an API error, got {other:?}"),
        }

        let busy = json!({ "status": "failed",
                           "error": { "code": "rate_limit_exceeded", "message": "slow down" } });
        let err = parse_response(&busy, "m").unwrap_err();
        assert!(err.is_retryable(), "a rate limit reported in the body still waits");
        assert_eq!(err.provider_message(), "slow down");
    }

    /// The shape LM Studio actually streamed for a tool call: reasoning deltas,
    /// then the call announced empty, then its arguments whole, never in deltas.
    #[test]
    fn a_call_that_arrives_whole_is_assembled() {
        let mut state = StreamState::default();
        let events = feed_all(&mut state, &[
            r#"{"type":"response.created","response":{"status":"in_progress"}}"#,
            r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","summary":[],"content":[]}}"#,
            r#"{"type":"response.reasoning_text.delta","output_index":0,"delta":"I need"}"#,
            r#"{"type":"response.output_item.added","output_index":1,"item":{"id":"fc_c7","type":"function_call","status":"in_progress","arguments":"","call_id":"call_378","name":"get_weather"}}"#,
            r#"{"type":"response.function_call_arguments.done","item_id":"fc_c7","output_index":1,"arguments":"{\"city\":\"Paris\"}"}"#,
            r#"{"type":"response.output_item.done","output_index":1,"item":{"id":"fc_c7","type":"function_call","status":"completed","arguments":"{\"city\":\"Paris\"}","call_id":"call_378","name":"get_weather"}}"#,
            r#"{"type":"response.completed","response":{"status":"completed","output":[],"usage":{"input_tokens":283,"output_tokens":17}}}"#,
        ]);
        assert!(
            !events.iter().any(|e| matches!(e, StreamEvent::Text(_))),
            "reasoning is not answer text: {events:?}"
        );
        assert!(matches!(&events[0], StreamEvent::Thinking(t) if t == "I need"), "{events:?}");
        let done = done_of(&events);
        assert_eq!(done.tool_calls.len(), 1);
        assert_eq!(done.tool_calls[0].id, "call_378");
        assert_eq!(done.tool_calls[0].name, "get_weather");
        assert_eq!(done.tool_calls[0].arguments["city"], "Paris");
        assert_eq!((done.input_tokens, done.output_tokens), (283, 17));
    }

    /// OpenAI's shape: text deltas, and a call whose arguments stream in pieces.
    #[test]
    fn deltas_are_streamed_and_assembled() {
        let mut state = StreamState::default();
        let events = feed_all(&mut state, &[
            r#"{"type":"response.output_text.delta","output_index":0,"delta":"Look"}"#,
            r#"{"type":"response.output_text.delta","output_index":0,"delta":"ing."}"#,
            r#"{"type":"response.output_text.done","output_index":0,"text":"Looking."}"#,
            r#"{"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","call_id":"c1","name":"search_sources","arguments":""}}"#,
            r#"{"type":"response.function_call_arguments.delta","output_index":1,"delta":"{\"query\":"}"#,
            r#"{"type":"response.function_call_arguments.delta","output_index":1,"delta":"\"gcd\"}"}"#,
            r#"{"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":5,"output_tokens":6}}}"#,
        ]);
        let text: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, vec!["Look", "ing."], "the finished text is not sent twice");
        let done = done_of(&events);
        assert_eq!(done.tool_calls[0].arguments["query"], "gcd");
    }

    #[test]
    fn a_server_that_never_streams_still_answers_once() {
        let mut state = StreamState::default();
        let events = feed_all(&mut state, &[
            r#"{"type":"response.completed","response":{"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"whole answer"}]},{"type":"function_call","call_id":"c2","name":"list_sources","arguments":"{}"}],"usage":{"input_tokens":1,"output_tokens":2}}}"#,
        ]);
        assert!(matches!(&events[0], StreamEvent::Text(t) if t == "whole answer"));
        let done = done_of(&events);
        assert_eq!(done.tool_calls[0].id, "c2");
        assert_eq!(done.tool_calls[0].arguments, json!({}));
    }

    #[test]
    fn a_failed_stream_ends_the_turn_with_the_reason() {
        let mut state = StreamState::default();
        let err = feed(&mut state, &record(
            r#"{"type":"response.failed","response":{"status":"failed","error":{"code":"server_error","message":"boom"}}}"#,
        ))
        .unwrap_err();
        assert_eq!(err.provider_message(), "boom");

        let err = feed(&mut state, &record(
            r#"{"type":"error","code":"rate_limit_exceeded","message":"too many requests"}"#,
        ))
        .unwrap_err();
        assert!(err.is_retryable());
    }
}
