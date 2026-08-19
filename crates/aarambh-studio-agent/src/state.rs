use std::path::Path;

use aarambh_studio_inference::{ToolCall, ToolDefinition};
use serde::{Deserialize, Serialize};

use crate::{AgentError, AgentResult};

/// Maximum accepted UTF-8 bytes in a text or error tool result.
pub const MAX_RESULT_TEXT_BYTES: usize = 64 * 1024;
/// Maximum accepted UTF-8 bytes in a media description.
pub const MAX_MEDIA_DESCRIPTION_BYTES: usize = 4 * 1024;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Policy used when a chain transcript approaches the model context limit.
///
/// This is the agent-crate view of the canonical
/// [`aarambh_studio_inference::ContextTruncationPolicy`] (`ARCHITECTURE_V4.md`
/// §66). The two map one-to-one; the canonical policy lives in the inference
/// crate so every long-context feature references the same definition.
pub enum EvictionPolicy {
    /// Remove the oldest completed exchanges while retaining recent turns —
    /// maps to [`aarambh_studio_inference::ContextTruncationPolicy::SlidingWindow`].
    #[default]
    DropOldest,
    /// Replace evicted exchanges with a compact model-produced summary — maps
    /// to [`aarambh_studio_inference::ContextTruncationPolicy::Summarize`].
    Summarise,
    /// Refuse to proceed rather than silently drop context — maps to
    /// [`aarambh_studio_inference::ContextTruncationPolicy::Reject`]. The mandatory default for anything
    /// safety- or execution-sensitive (Phase 47/48 sessions), where silently
    /// losing a turn would change the meaning of the chain.
    Reject,
}

impl From<EvictionPolicy> for aarambh_studio_inference::ContextTruncationPolicy {
    /// Map an agent-crate [`EvictionPolicy`] onto the canonical
    /// [`aarambh_studio_inference::ContextTruncationPolicy`].
    fn from(policy: EvictionPolicy) -> Self {
        match policy {
            EvictionPolicy::DropOldest => {
                aarambh_studio_inference::ContextTruncationPolicy::SlidingWindow
            }
            EvictionPolicy::Summarise => {
                aarambh_studio_inference::ContextTruncationPolicy::Summarize
            }
            EvictionPolicy::Reject => aarambh_studio_inference::ContextTruncationPolicy::Reject,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Outcome reported by a caller after executing a tool.
pub enum ToolResultStatus {
    /// The external tool completed successfully.
    Ok,
    /// The external tool failed and returned a bounded error message.
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
/// Typed content supplied by a caller as a tool result.
pub enum ToolResultContent {
    /// UTF-8 text content.
    Text {
        /// Tool-produced text.
        text: String,
    },
    /// Image path and optional textual description.
    Image {
        /// Caller-visible path to the image.
        path: String,
        /// Optional grounded description retained after the visual turn.
        #[serde(default)]
        description: String,
    },
    /// Video path and optional textual description.
    Video {
        /// Caller-visible path to the video.
        path: String,
        /// Optional grounded description retained after the visual turn.
        #[serde(default)]
        description: String,
    },
    /// Document path, optional page selection, and textual description.
    Document {
        /// Caller-visible path to the document.
        path: String,
        /// One-based pages to project; an empty list means decoder defaults.
        #[serde(default)]
        pages: Vec<u32>,
        /// Optional grounded description retained after the visual turn.
        #[serde(default)]
        description: String,
    },
}

impl ToolResultContent {
    /// Return true when this result can provide native multimodal embeddings.
    pub fn is_media(&self) -> bool {
        !matches!(self, Self::Text { .. })
    }

    /// Return the external media path, if this is a media result.
    pub fn media_path(&self) -> Option<&Path> {
        match self {
            Self::Text { .. } => None,
            Self::Image { path, .. } | Self::Video { path, .. } | Self::Document { path, .. } => {
                Some(Path::new(path))
            }
        }
    }

    /// Render bounded metadata retained in later text-only turns.
    pub fn metadata_text(&self) -> String {
        match self {
            Self::Text { text } => text.clone(),
            Self::Image { path, description } => {
                format!("image path={path:?} description={description:?}")
            }
            Self::Video { path, description } => {
                format!("video path={path:?} description={description:?}")
            }
            Self::Document {
                path,
                pages,
                description,
            } => format!("document path={path:?} pages={pages:?} description={description:?}"),
        }
    }

    pub(crate) fn validate(&self) -> AgentResult<()> {
        match self {
            Self::Text { text } => validate_bytes("tool result text", text, MAX_RESULT_TEXT_BYTES),
            Self::Image { path, description } | Self::Video { path, description } => {
                validate_path(path)?;
                validate_bytes(
                    "media description",
                    description,
                    MAX_MEDIA_DESCRIPTION_BYTES,
                )
            }
            Self::Document {
                path,
                pages,
                description,
            } => {
                validate_path(path)?;
                validate_bytes(
                    "document description",
                    description,
                    MAX_MEDIA_DESCRIPTION_BYTES,
                )?;
                let mut sorted = pages.clone();
                sorted.sort_unstable();
                sorted.dedup();
                if sorted.len() != pages.len() || pages.contains(&0) {
                    return Err(AgentError::ResultProtocol(
                        "document pages must be unique one-based page numbers".into(),
                    ));
                }
                Ok(())
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Caller-provided result for one model-selected tool call.
pub struct ToolResult {
    /// Chain-assigned call identifier.
    pub call_id: String,
    /// External execution outcome.
    pub status: ToolResultStatus,
    /// Typed successful content; required when status is `ok`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<ToolResultContent>,
    /// Error text; required when status is `error`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl ToolResult {
    /// Validate status/content exclusivity and all configured size limits.
    pub fn validate_for(&self, expected_call_id: &str) -> AgentResult<()> {
        if self.call_id != expected_call_id {
            return Err(AgentError::ResultProtocol(format!(
                "expected result for call_id {expected_call_id:?}, got {:?}",
                self.call_id
            )));
        }
        match (&self.status, &self.content, &self.error) {
            (ToolResultStatus::Ok, Some(content), None) => content.validate(),
            (ToolResultStatus::Error, None, Some(error)) => {
                if error.trim().is_empty() {
                    return Err(AgentError::ResultProtocol(
                        "error tool result must contain a non-empty error".into(),
                    ));
                }
                validate_bytes("tool result error", error, MAX_RESULT_TEXT_BYTES)
            }
            _ => Err(AgentError::ResultProtocol(
                "status=ok requires content only; status=error requires error only".into(),
            )),
        }
    }

    /// Return media content while it is eligible for the immediate next turn.
    pub fn media_content(&self) -> Option<&ToolResultContent> {
        self.content.as_ref().filter(|content| content.is_media())
    }

    /// Render the deterministic text representation stored in the transcript.
    pub fn transcript_text(&self) -> String {
        match (&self.status, &self.content, &self.error) {
            (ToolResultStatus::Ok, Some(content), None) => content.metadata_text(),
            (ToolResultStatus::Error, None, Some(error)) => format!("error: {error}"),
            _ => "invalid tool result".into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Request emitted to the caller for external tool execution.
pub struct ToolResultRequest {
    /// Chain-assigned call identifier.
    pub call_id: String,
    /// Schema-valid model-selected function call.
    pub call: ToolCall,
}

#[derive(Debug, Clone)]
/// One completed call/result exchange and its exact transcript token ids.
pub struct ToolExchange {
    /// Request sent to the caller.
    pub request: ToolResultRequest,
    /// Validated caller-provided result.
    pub result: ToolResult,
    /// Exact model-generated ids for the tool call.
    pub call_token_ids: Vec<u32>,
    /// Encoded result-turn ids.
    pub result_token_ids: Vec<u32>,
}

impl ToolExchange {
    /// Number of transcript tokens occupied by this exchange.
    pub fn token_len(&self) -> usize {
        self.call_token_ids.len() + self.result_token_ids.len()
    }
}

#[derive(Debug, Clone)]
/// Inspectable exact-token state for a running or completed chain.
pub struct ChainState {
    prompt: String,
    tools: Vec<ToolDefinition>,
    prefix_token_ids: Vec<u32>,
    exchanges: Vec<ToolExchange>,
    summary: Option<String>,
    evicted_exchanges: usize,
}

impl ChainState {
    pub(crate) fn new(
        prompt: String,
        tools: Vec<ToolDefinition>,
        prefix_token_ids: Vec<u32>,
    ) -> Self {
        Self {
            prompt,
            tools,
            prefix_token_ids,
            exchanges: Vec::new(),
            summary: None,
            evicted_exchanges: 0,
        }
    }

    /// Initial user prompt.
    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    /// Tool definitions available throughout this chain.
    pub fn tools(&self) -> &[ToolDefinition] {
        &self.tools
    }

    /// Completed exchanges still retained in the active context.
    pub fn exchanges(&self) -> &[ToolExchange] {
        &self.exchanges
    }

    /// Current compact history summary, when summarising eviction is enabled.
    pub fn summary(&self) -> Option<&str> {
        self.summary.as_deref()
    }

    /// Number of completed exchanges removed from exact context.
    pub fn evicted_exchanges(&self) -> usize {
        self.evicted_exchanges
    }

    /// Materialize the exact active transcript ids.
    pub fn transcript_token_ids(&self) -> Vec<u32> {
        let capacity = self.prefix_token_ids.len()
            + self
                .exchanges
                .iter()
                .map(ToolExchange::token_len)
                .sum::<usize>();
        let mut ids = Vec::with_capacity(capacity);
        ids.extend_from_slice(&self.prefix_token_ids);
        for exchange in &self.exchanges {
            ids.extend_from_slice(&exchange.call_token_ids);
            ids.extend_from_slice(&exchange.result_token_ids);
        }
        ids
    }

    pub(crate) fn push(&mut self, exchange: ToolExchange) {
        self.exchanges.push(exchange);
    }

    pub(crate) fn last_exchange_mut(&mut self) -> Option<&mut ToolExchange> {
        self.exchanges.last_mut()
    }

    pub(crate) fn pop_oldest(&mut self) -> Option<ToolExchange> {
        if self.exchanges.is_empty() {
            None
        } else {
            self.evicted_exchanges += 1;
            Some(self.exchanges.remove(0))
        }
    }

    pub(crate) fn replace_prefix(&mut self, summary: String, prefix_token_ids: Vec<u32>) {
        self.summary = Some(summary);
        self.prefix_token_ids = prefix_token_ids;
    }
}

fn validate_path(path: &str) -> AgentResult<()> {
    if path.trim().is_empty() || path.len() > 4096 {
        return Err(AgentError::ResultProtocol(
            "media path must contain 1..=4096 UTF-8 bytes".into(),
        ));
    }
    Ok(())
}

fn validate_bytes(label: &str, value: &str, max: usize) -> AgentResult<()> {
    if value.len() > max {
        return Err(AgentError::ResultProtocol(format!(
            "{label} exceeds {max} UTF-8 bytes"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{ToolResult, ToolResultContent, ToolResultStatus};

    #[test]
    fn result_status_is_strict() {
        let result = ToolResult {
            call_id: "call_0001".into(),
            status: ToolResultStatus::Ok,
            content: Some(ToolResultContent::Text {
                text: "sunny".into(),
            }),
            error: None,
        };
        assert!(result.validate_for("call_0001").is_ok());
        assert!(result.validate_for("call_0002").is_err());
    }

    #[test]
    fn document_pages_are_unique_and_one_based() {
        let result = ToolResult {
            call_id: "call_0001".into(),
            status: ToolResultStatus::Ok,
            content: Some(ToolResultContent::Document {
                path: "report.pdf".into(),
                pages: vec![1, 1],
                description: String::new(),
            }),
            error: None,
        };
        assert!(result.validate_for("call_0001").is_err());
    }
}
