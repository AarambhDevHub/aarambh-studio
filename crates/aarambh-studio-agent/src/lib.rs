//! Bounded, caller-executed, long-horizon tool-use chains.
//!
//! Phase 47 (`ARCHITECTURE_V4.md` §61) extends this crate with sandboxed
//! tool execution: the model's tool calls can now be actually executed by
//! aarambh-studio itself, inside a strict, closed-world sandbox. See the
//! [`sandbox`] and [`authorization`] modules for the execution envelope.
//!
//! Phase 48 (`ARCHITECTURE_V4.md` §62) extends this crate with multi-agent
//! orchestration: one top-level orchestrating reasoning process delegates
//! independent sub-tasks to multiple parallel sandboxed tool-execution
//! sub-chains, then merges their results back into its own context. See the
//! [`orchestrator`] module for the delegation and aggregation envelope.
#![deny(missing_docs)]

/// Operator-controlled per-tool authorization.
pub mod authorization;
/// Multi-turn tool-chain orchestration.
pub mod chain;
/// Multi-agent orchestration (Phase 48).
pub mod orchestrator;
/// Interactive and replay-based tool-result ingestion.
pub mod result_ingestion;
/// Sandboxed tool execution (Phase 47).
pub mod sandbox;
/// Exact-token chain state and result protocol types.
pub mod state;

pub use authorization::AuthorizationScope;
pub use chain::{
    AgentError, AgentResult, ChainDecoder, ChainEvent, ChainMetrics, ChainOutput, ToolChain,
    ToolChainConfig,
};
pub use orchestrator::{
    DEFAULT_MAX_SUB_AGENTS, DEFAULT_MAX_TOTAL_TIME_MS, DelegatedSubTask, DelegationPlan,
    MAX_MAX_SUB_AGENTS, MIN_MAX_SUB_AGENTS, OrchestrationLimits, Orchestrator, SubChainOutcome,
    SubChainStatus,
};
pub use result_ingestion::{
    ReplayEntry, ReplayResultProvider, StdinResultProvider, ToolResultProvider,
};
pub use sandbox::{
    DEFAULT_MAX_ARGS_BYTES, DEFAULT_MAX_OUTPUT_BYTES, DEFAULT_TIMEOUT_MS, ExecContext, ExecError,
    ReadFileInWorkdir, ResourceKind, SandboxLimits, SandboxedToolProvider, StaticLookup,
    ToolExecutor, ToolSandbox, ValidatedArgs,
};
pub use state::{
    ChainState, EvictionPolicy, ToolExchange, ToolResult, ToolResultContent, ToolResultRequest,
    ToolResultStatus,
};
