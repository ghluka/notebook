//! provider-neutral chat types; extend here before touching provider json.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
    },
    Image {
        media_type: String,
        data: String,
    },
    Document {
        media_type: String,
        data: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
    },
    /// openai-style sends audio_url as a data URI (nemotron omni); anthropic errors
    Audio {
        media_type: String,
        data: String,
    },
    Video {
        media_type: String,
        data: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
        /// opaque provider state; gemini signs every call and rejects history without it
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
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

    pub fn tool_results(results: Vec<ContentPart>) -> Self {
        Message { role: Role::Tool, content: results }
    }

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

/// mapped per provider: anthropic gets a thinking budget, openai reasoning_effort
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
    /// empty when the model only called tools
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    pub stop_reason: String,
    pub usage: Usage,
    pub model: String,
}

impl ChatResponse {
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
                signature: call.signature.clone(),
            });
        }
        Message { role: Role::Assistant, content }
    }
}

/// thinking is shown while waiting, never part of answer text, never sent back to the model
#[derive(Debug, Clone)]
pub enum StreamEvent {
    Thinking(String),
    Text(String),
    Done(StreamDone),
}

#[derive(Debug, Clone, Default)]
pub struct StreamDone {
    pub tool_calls: Vec<ToolCall>,
    pub input_tokens: u32,
    pub output_tokens: u32,
}
