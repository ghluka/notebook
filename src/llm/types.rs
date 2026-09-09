//! Provider-neutral chat types.
//!
//! Handlers and agents only ever speak this vocabulary; `openai.rs` and
//! `anthropic.rs` translate it to and from the wire. If a provider needs
//! something that isn't modelled here, extend these types first.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// One piece of a message. A message is a sequence of these so that a single
/// user turn can carry "here is a page image, here is the question about it".
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
    },
    /// Base64 image bytes. `media_type` is e.g. `image/png`.
    Image {
        media_type: String,
        data: String,
    },
    /// Base64 document bytes (PDF today). Providers that can't take documents
    /// natively will error rather than silently drop it.
    Document {
        media_type: String,
        data: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
    },
    /// A model's request to call a tool.
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// Our answer to a `ToolUse`.
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default)]
        is_error: bool,
    },
}

impl ContentPart {
    pub fn text(s: impl Into<String>) -> Self {
        ContentPart::Text { text: s.into() }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentPart>,
}

impl Message {
    pub fn user(text: impl Into<String>) -> Self {
        Message { role: Role::User, content: vec![ContentPart::text(text)] }
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Message { role: Role::Assistant, content: vec![ContentPart::text(text)] }
    }

    /// A turn carrying results for tools the model just called.
    pub fn tool_results(results: Vec<ContentPart>) -> Self {
        Message { role: Role::Tool, content: results }
    }

    /// All text parts joined, which is what you want for a plain answer.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|p| match p {
                ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// A tool the model may call. `parameters` is a JSON Schema object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// How hard the model should think before answering. Mapped per provider:
/// Anthropic gets a thinking budget, OpenAI gets `reasoning_effort`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Effort {
    #[default]
    Off,
    Low,
    Medium,
    High,
}

impl Effort {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" | "none" => Some(Effort::Off),
            "low" => Some(Effort::Low),
            "medium" | "med" => Some(Effort::Medium),
            "high" => Some(Effort::High),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Effort::Off => "off",
            Effort::Low => "low",
            Effort::Medium => "medium",
            Effort::High => "high",
        }
    }

    /// Anthropic thinking budget in tokens.
    pub fn budget_tokens(self) -> Option<u32> {
        match self {
            Effort::Off => None,
            Effort::Low => Some(2_048),
            Effort::Medium => Some(8_192),
            Effort::High => Some(16_384),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    pub messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    pub max_tokens: u32,
    #[serde(default)]
    pub effort: Effort,
}

impl ChatRequest {
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> Self {
        ChatRequest {
            model: model.into(),
            system: None,
            messages,
            tools: Vec::new(),
            temperature: None,
            max_tokens: 4096,
            effort: Effort::Off,
        }
    }

    pub fn system(mut self, s: impl Into<String>) -> Self {
        self.system = Some(s.into());
        self
    }

    pub fn tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = tools;
        self
    }

    pub fn max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = n;
        self
    }

    pub fn effort(mut self, effort: Effort) -> Self {
        self.effort = effort;
        self
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    /// Concatenated text output. Empty when the model only called tools.
    pub text: String,
    /// Reasoning the model exposed, when thinking was enabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    pub stop_reason: String,
    pub usage: Usage,
    pub model: String,
}

impl ChatResponse {
    /// The assistant turn to append to history before running tools.
    pub fn as_message(&self) -> Message {
        let mut content = Vec::new();
        if !self.text.is_empty() {
            content.push(ContentPart::text(self.text.clone()));
        }
        for call in &self.tool_calls {
            content.push(ContentPart::ToolUse {
                id: call.id.clone(),
                name: call.name.clone(),
                input: call.arguments.clone(),
            });
        }
        Message { role: Role::Assistant, content }
    }
}
