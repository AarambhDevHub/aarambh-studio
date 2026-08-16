//! Autoregressive inference engine, sampling, streaming, KV cache, and thinking controls.
#![deny(missing_docs)]

/// Best-of-N test-time compute scaling.
pub mod best_of_n;
/// Generation engine and output types.
pub mod engine;
/// Grammar-constrained JSON decoding.
pub mod grammar;
/// Inference-time key/value cache.
pub mod kvcache;
/// One-checkpoint speculative decoding with multi-token prediction heads.
pub mod mtp_speculative;
/// Optional process-reward scoring for test-time compute scaling.
pub mod process_reward;
/// Temperature, top-k, top-p, and greedy sampling.
pub mod sampler;
/// Self-consistency majority-vote selection for test-time compute scaling.
pub mod self_consistency;
/// Exact draft-model speculative decoding.
pub mod speculative;
/// Streaming callback event types.
pub mod stream;
/// Thinking budget and forced-token controls.
pub mod thinking;
/// Tool definitions, call protocol, and decoding controller.
pub mod tool_calling;

pub use best_of_n::{
    BestOfNConfig, BestOfNEngine, BestOfNOutput, CompletionVerifier, SelectionRationale,
    SelectionStrategy,
};
pub use engine::{
    FinishReason, GenerationConfig, GenerationOutput, GenerationPhase, GenerationSession,
    GenerationStep, GenerationUsage, InferenceEngine,
};
pub use grammar::{JsonSchema, JsonSchemaGrammar};
pub use kvcache::KvCache;
pub use mtp_speculative::MtpSpeculativeEngine;
pub use process_reward::{HeuristicProcessRewardScorer, ProcessRewardHead, ProcessRewardScorer};
pub use sampler::{Sampler, TokenCandidate};
pub use self_consistency::{
    extract_final_answer, extract_final_number, majority_vote, self_consistency_select,
};
pub use speculative::{
    SpeculativeConfig, SpeculativeEngine, SpeculativeProposalSource, SpeculativeStats,
};
pub use stream::StreamEvent;
pub use thinking::{
    ForceToken, MODE_HIGH, MODE_LOW, MODE_MAX, MODE_MEDIUM, MODE_NONE, ThinkingController,
    ThinkingMode, parse_thinking_mode,
};
pub use tool_calling::{
    FINAL_MARKER, TOOL_CALL_END, TOOL_CALL_START, ToolCall, ToolCallController, ToolCallingConfig,
    ToolChoice, ToolDefinition,
};
