//! Bounded, caller-executed, long-horizon tool-use chains.
//!
//! Phase 47 (`ARCHITECTURE_V4.md` §61) extends this crate with sandboxed
//! tool execution: the model's tool calls can now be actually executed by
//! aarambh-studio itself, inside a strict, closed-world sandbox. See the
//! [`sandbox`] and [`authorization`] modules for the execution envelope.
#![deny(missing_docs)]

/// Operator-controlled per-tool authorization.
pub mod authorization;
/// Multi-turn tool-chain orchestration.
pub mod chain;
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
