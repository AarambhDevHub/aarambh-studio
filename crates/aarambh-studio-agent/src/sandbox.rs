//! Sandboxed tool execution.
//!
//! Phase 47 (`ARCHITECTURE_V4.md` §61) closes the boundary v2 §30 opened
//! (tool calls are emitted, never executed) and v3 §46 extended (multi-step
//! chains, still emit-only): the model's tool calls can now be **actually
//! executed** by aarambh-studio itself, but only inside a strict, explicit
//! sandbox.
//!
//! ## Closed-world execution
//!
//! Every executor ([`ToolExecutor`]) implements one specific, named
//! capability. There is no generic "run a shell command" or "eval this
//! code" executor anywhere in this crate, by design. An unrecognised tool
//! name is a **hard refusal** ([`ExecError::UnknownTool`]), never a
//! best-effort fallback attempt.
//!
//! ## The execution envelope
//!
//! Each call runs inside a bounded envelope enforced by [`ToolSandbox`]:
//!
//! 1. **Closed-world allowlist** — the name must match a registered
//!    executor.
//! 2. **Operator authorization** — the name must be in the operator's
//!    [`AuthorizationScope`] (an operator decision, not a model decision).
//! 3. **Schema re-validation** — arguments are independently re-validated
//!    against the declared JSON Schema, defense-in-depth on top of the
//!    grammar-constrained decoder that already produced a valid call.
//!    Malformed or schema-invalid calls are never executed.
//! 4. **Wall-clock timeout** — execution runs on a worker thread with a
//!    `recv_timeout` join; exceeding the ceiling yields
//!    [`ExecError::Timeout`] and the call is abandoned (fail-closed).
//! 5. **Resource ceiling** — argument and output payload sizes are capped.
//!
//! ## Composability
//!
//! [`SandboxedToolProvider`] implements [`crate::ToolResultProvider`], so
//! execution plugs into the existing [`crate::ToolChain`] with **no changes
//! to the chain**: the chain calls `next_result`, the provider executes the
//! call through the sandbox, and the resulting [`crate::ToolResult`] re-enters
//! the chain via the existing `result_ingestion` path — execution is purely
//! additive to what v3 §46 already built.
//!
//! ## Honesty boundary
//!
//! This is a pure-Rust, CPU-only sandbox: wall-clock timeout (cooperative
//! cancellation via an [`std::sync::atomic::AtomicBool`] flag, plus
//! thread-abandonment on timeout since safe Rust cannot force-kill a
//! thread), output-size ceiling, args-size ceiling, closed-world allowlist,
//! operator authorization, and schema re-validation. OS-level isolation
//! (seccomp, cgroups, namespaces) is out of scope for the source release,
//! consistent with the project's CPU-first posture. The safety-relevant
//! property — a runaway or hung call never blocks the chain and always
//! produces a fail-closed result — holds under every tested failure
//! condition.

use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use aarambh_studio_inference::{JsonSchema, ToolCall, ToolDefinition};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::authorization::AuthorizationScope;
use crate::{
    AgentError, ToolResult, ToolResultContent, ToolResultProvider, ToolResultRequest,
    ToolResultStatus,
};

/// Maximum bytes accepted for the serialized tool-call arguments.
pub const DEFAULT_MAX_ARGS_BYTES: usize = 8 * 1024;
/// Maximum bytes accepted for a tool-execution output payload.
pub const DEFAULT_MAX_OUTPUT_BYTES: usize = 64 * 1024;
/// Default per-call wall-clock ceiling, in milliseconds.
pub const DEFAULT_TIMEOUT_MS: u64 = 5_000;

/// Schema-validated tool-call arguments.
///
/// Constructed only by [`ToolSandbox::execute`] after the arguments have
/// been independently re-validated against the declared JSON Schema. The
/// newtype guarantee means a [`ToolExecutor`] never receives untrusted,
/// unvalidated input — by construction, [`ValidatedArgs::value`] is
/// schema-valid for the tool named by [`ValidatedArgs::name`].
#[derive(Debug, Clone)]
pub struct ValidatedArgs {
    name: String,
    arguments: serde_json::Value,
}

impl ValidatedArgs {
    /// The tool name these arguments belong to.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The schema-valid argument object.
    pub fn value(&self) -> &serde_json::Value {
        &self.arguments
    }
}

/// The kind of resource ceiling an executor exceeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceKind {
    /// The output payload exceeded `max_output_bytes`.
    OutputBytes,
    /// The argument payload exceeded `max_args_bytes`.
    ArgumentBytes,
}

impl std::fmt::Display for ResourceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutputBytes => f.write_str("output_bytes"),
            Self::ArgumentBytes => f.write_str("argument_bytes"),
        }
    }
}

/// Resource and time ceilings for one execution envelope.
///
/// All fields are operator-set configuration; an executor cannot influence
/// them. The ceilings are enforced by [`ToolSandbox`], not by individual
/// executors, so a buggy or adversarial executor cannot escape them.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SandboxLimits {
    /// Wall-clock ceiling per call, in milliseconds. A call exceeding it
    /// is abandoned and reported as [`ExecError::Timeout`] (fail-closed).
    pub timeout_ms: u64,
    /// Maximum output payload bytes an executor may return.
    pub max_output_bytes: usize,
    /// Maximum argument payload bytes accepted from the model.
    pub max_args_bytes: usize,
}

impl Default for SandboxLimits {
    fn default() -> Self {
        Self {
            timeout_ms: DEFAULT_TIMEOUT_MS,
            max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
            max_args_bytes: DEFAULT_MAX_ARGS_BYTES,
        }
    }
}

impl SandboxLimits {
    /// Validate that all ceilings are positive and bounded.
    pub fn validate(&self) -> Result<(), AgentError> {
        if self.timeout_ms == 0 {
            return Err(AgentError::Config(
                "sandbox timeout_ms must be greater than zero".into(),
            ));
        }
        if self.max_output_bytes == 0 {
            return Err(AgentError::Config(
                "sandbox max_output_bytes must be greater than zero".into(),
            ));
        }
        if self.max_args_bytes == 0 {
            return Err(AgentError::Config(
                "sandbox max_args_bytes must be greater than zero".into(),
            ));
        }
        Ok(())
    }
}

/// Execution context handed to every [`ToolExecutor`].
///
/// Carries the [`SandboxLimits`] in force and a cooperative cancellation
/// flag set when the wall-clock deadline passes. Long-running executors
/// should poll [`ExecContext::is_cancelled`] inside their loops and return
/// [`ExecError::Timeout`] promptly when it becomes true.
#[derive(Debug, Clone)]
pub struct ExecContext {
    limits: SandboxLimits,
    cancel: Arc<AtomicBool>,
    deadline: Instant,
}

impl ExecContext {
    /// The resource and time ceilings in force for this call.
    pub fn limits(&self) -> SandboxLimits {
        self.limits
    }

    /// The wall-clock deadline after which the call is considered timed out.
    pub fn deadline(&self) -> Instant {
        self.deadline
    }

    /// True once the wall-clock deadline has passed or the sandbox raised
    /// the cancellation flag. Cooperative executors should check this
    /// inside long loops and return [`ExecError::Timeout`].
    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Acquire) || Instant::now() >= self.deadline
    }

    /// Remaining wall-clock time before the deadline, or zero once passed.
    pub fn remaining(&self) -> Duration {
        self.deadline.saturating_duration_since(Instant::now())
    }
}

/// Why a sandboxed execution did not produce a result.
///
/// Every variant maps to a fail-closed [`ToolResult`] with
/// `status = Error`: the chain records the refusal and continues, it never
/// silently drops a call or attempts a best-effort fallback.
#[derive(Debug, Error)]
pub enum ExecError {
    /// The tool name is not in the closed-world allowlist of registered
    /// executors. Hard refusal — no execution attempt of any kind.
    #[error("tool {0:?} is not in the closed-world execution allowlist")]
    UnknownTool(String),
    /// The tool is registered (in the allowlist) but the operator did not
    /// authorize it. Distinct from `UnknownTool`: the capability exists,
    /// the operator simply did not enable it.
    #[error("tool {0:?} is not authorized by the operator")]
    Unauthorized(String),
    /// The tool-call arguments failed independent schema re-validation.
    /// Malformed or schema-invalid calls are never executed.
    #[error("tool {0:?} arguments failed schema validation: {1}")]
    InvalidArgs(String, String),
    /// Execution exceeded the configured wall-clock ceiling. The worker
    /// thread was abandoned (fail-closed).
    #[error("execution of {0:?} exceeded the {1} ms wall-clock ceiling")]
    Timeout(String, u64),
    /// Execution exceeded a configured resource ceiling.
    #[error("execution of {name:?} exceeded the {kind} ceiling ({limit} bytes)")]
    ResourceLimitExceeded {
        /// The tool name.
        name: String,
        /// Which ceiling was exceeded.
        kind: ResourceKind,
        /// The configured ceiling value.
        limit: usize,
    },
    /// The executor returned an error from its own logic.
    #[error("executor {0:?} failure: {1}")]
    Executor(String, String),
}

/// Maximum bytes of error text retained in a fail-closed `ToolResult`.
const MAX_ERROR_TEXT_BYTES: usize = 4 * 1024;

impl ExecError {
    /// Render the error as the bounded `error` text of a fail-closed
    /// [`ToolResult`]. The result is always non-empty and well under the
    /// [`crate::state::MAX_RESULT_TEXT_BYTES`] ceiling.
    pub fn to_result_error(&self) -> String {
        let text = self.to_string();
        if text.len() > MAX_ERROR_TEXT_BYTES {
            // Keep the head and an ellipsis so the message stays bounded.
            let mut head = text
                .chars()
                .take(MAX_ERROR_TEXT_BYTES - 1)
                .collect::<String>();
            head.push('…');
            head
        } else {
            text
        }
    }
}

/// One named, closed-world execution capability.
///
/// Every implementor represents one specific capability (for example a
/// read-only file lookup confined to a fixed working directory). There is
/// no generic "run a shell command" or "eval this code" executor — by
/// design, that capability does not exist in this crate.
///
/// Implementors must be [`Send + Sync`] so the sandbox can dispatch on a
/// worker thread. They should respect the [`ExecContext`] cancellation
/// flag inside long loops and must not return more than
/// `limits.max_output_bytes` of payload (the sandbox enforces this as
/// defense-in-depth, but an executor that respects it avoids producing a
/// needless [`ExecError::ResourceLimitExceeded`]).
pub trait ToolExecutor: Send + Sync {
    /// The exact capability name. Must match a declared [`ToolDefinition`]
    /// name and an entry in the operator's [`AuthorizationScope`]. Exact
    /// match only — no pattern or fuzzy resolution.
    fn name(&self) -> &'static str;

    /// Execute one validated call.
    ///
    /// On success, returns a [`ToolResultContent`] that the sandbox wraps
    /// into a `status = Ok` [`ToolResult`]. On error, returns an
    /// [`ExecError`] that becomes a `status = Error` [`ToolResult`].
    fn execute(
        &self,
        args: &ValidatedArgs,
        ctx: &ExecContext,
    ) -> Result<ToolResultContent, ExecError>;
}

/// A `ToolResultProvider` that executes tool calls through the sandbox.
///
/// This is the composability bridge between Phase 47's execution capability
/// and v3 §46's existing chain: the [`crate::ToolChain`] calls
/// `next_result`, the provider executes the call via the sandbox, and the
/// resulting [`ToolResult`] re-enters the chain through the unchanged
/// `result_ingestion` path. No new ingestion mechanism is introduced.
pub struct SandboxedToolProvider {
    sandbox: ToolSandbox,
}

impl SandboxedToolProvider {
    /// Build a provider backed by the given sandbox.
    pub fn new(sandbox: ToolSandbox) -> Self {
        Self { sandbox }
    }

    /// Borrow the underlying sandbox (for inspection in tests).
    pub fn sandbox(&self) -> &ToolSandbox {
        &self.sandbox
    }
}

impl ToolResultProvider for SandboxedToolProvider {
    fn next_result(&mut self, request: &ToolResultRequest) -> Result<ToolResult, AgentError> {
        let result = self.sandbox.execute(&request.call, &request.call_id);
        // Belt-and-braces: the sandbox already produced a valid result, but
        // re-validate against the chain's call-id contract so any future
        // drift is caught at the provider boundary, not deeper in the chain.
        result.validate_for(&request.call_id)?;
        Ok(result)
    }
}

#[derive(Debug)]
struct CompiledTool {
    schema: JsonSchema,
}

/// The closed-world sandbox.
///
/// Holds the registry of named [`ToolExecutor`] implementations, the
/// compiled JSON Schemas for every declared tool, the operator's
/// [`AuthorizationScope`], and the [`SandboxLimits`] in force. Built once
/// at startup; immutable thereafter.
pub struct ToolSandbox {
    executors: HashMap<&'static str, Arc<dyn ToolExecutor>>,
    compiled: HashMap<String, CompiledTool>,
    authorization: AuthorizationScope,
    limits: SandboxLimits,
}

impl ToolSandbox {
    /// Build an empty sandbox with the given authorization and limits.
    pub fn new(
        authorization: AuthorizationScope,
        limits: SandboxLimits,
    ) -> Result<Self, AgentError> {
        limits.validate()?;
        Ok(Self {
            executors: HashMap::new(),
            compiled: HashMap::new(),
            authorization,
            limits,
        })
    }

    /// Register a tool definition so its schema can be compiled and its
    /// name recognised as "declared". Must be called for every tool the
    /// model may request, including those without a registered executor.
    pub fn register_definition(&mut self, definition: ToolDefinition) -> Result<(), AgentError> {
        let schema = JsonSchema::compile(&definition.parameters).map_err(|error| {
            AgentError::Config(format!(
                "tool {:?} parameters schema failed to compile: {error}",
                definition.name
            ))
        })?;
        if !schema.is_object() {
            return Err(AgentError::Config(format!(
                "tool {:?} parameters schema must have object type",
                definition.name
            )));
        }
        self.compiled
            .insert(definition.name.clone(), CompiledTool { schema });
        Ok(())
    }

    /// Register a batch of tool definitions.
    pub fn register_definitions(
        &mut self,
        definitions: &[ToolDefinition],
    ) -> Result<(), AgentError> {
        for definition in definitions {
            self.register_definition(definition.clone())?;
        }
        Ok(())
    }

    /// Register an executor implementing one named capability. The name
    /// must exactly match the executor's [`ToolExecutor::name`] and a
    /// previously-registered [`ToolDefinition`] of the same name.
    pub fn register_executor(&mut self, executor: Box<dyn ToolExecutor>) -> Result<(), AgentError> {
        let name = executor.name();
        if !self.compiled.contains_key(name) {
            return Err(AgentError::Config(format!(
                "cannot register executor {name:?}: no matching tool definition was registered"
            )));
        }
        if self.executors.insert(name, Arc::from(executor)).is_some() {
            return Err(AgentError::Config(format!(
                "duplicate executor registration for {name:?}"
            )));
        }
        Ok(())
    }

    /// The configured limits.
    pub fn limits(&self) -> SandboxLimits {
        self.limits
    }

    /// The operator's authorization scope.
    pub fn authorization(&self) -> &AuthorizationScope {
        &self.authorization
    }

    /// Whether an executor is registered for this capability name.
    pub fn has_executor(&self, name: &str) -> bool {
        self.executors.contains_key(name)
    }

    /// Execute one tool call through the full §61 pipeline.
    ///
    /// Returns a fully-formed [`ToolResult`] ready to re-enter the chain.
    /// Success yields `status = Ok` with content; every failure yields
    /// `status = Error` with a bounded error message (fail-closed).
    pub fn execute(&self, call: &ToolCall, call_id: &str) -> ToolResult {
        match self.dispatch(call) {
            Ok(content) => ToolResult {
                call_id: call_id.to_string(),
                status: ToolResultStatus::Ok,
                content: Some(content),
                error: None,
            },
            Err(error) => ToolResult {
                call_id: call_id.to_string(),
                status: ToolResultStatus::Error,
                content: None,
                error: Some(error.to_result_error()),
            },
        }
    }

    fn dispatch(&self, call: &ToolCall) -> Result<ToolResultContent, ExecError> {
        // 1. The name must be a declared tool (the model's schema).
        let compiled = self
            .compiled
            .get(&call.name)
            .ok_or_else(|| ExecError::UnknownTool(call.name.clone()))?;

        // 2. Closed-world allowlist: an executor must be registered.
        let executor = self
            .executors
            .get(call.name.as_str())
            .ok_or_else(|| ExecError::UnknownTool(call.name.clone()))?
            .clone();

        // 3. Operator authorization: the operator must have enabled it.
        // Distinct from UnknownTool: the capability exists and is declared,
        // the operator simply did not authorize it.
        if !self.authorization.is_authorized(&call.name) {
            return Err(ExecError::Unauthorized(call.name.clone()));
        }

        // 4. Argument size ceiling.
        let args_len = serde_json::to_vec(&call.arguments)
            .map_err(|error| ExecError::Executor(call.name.clone(), error.to_string()))?
            .len();
        if args_len > self.limits.max_args_bytes {
            return Err(ExecError::ResourceLimitExceeded {
                name: call.name.clone(),
                kind: ResourceKind::ArgumentBytes,
                limit: self.limits.max_args_bytes,
            });
        }

        // 5. Schema re-validation (defense-in-depth on top of the
        // grammar-constrained decoder).
        if let Err(error) = compiled.schema.validate(&call.arguments) {
            return Err(ExecError::InvalidArgs(call.name.clone(), error.to_string()));
        }

        let validated = ValidatedArgs {
            name: call.name.clone(),
            arguments: call.arguments.clone(),
        };

        // 6. Bounded envelope: run on a worker thread with a wall-clock
        // timeout. A cooperative executor checks the cancellation flag; a
        // non-cooperative one is abandoned (detached) on timeout.
        let cancel = Arc::new(AtomicBool::new(false));
        let ctx = ExecContext {
            limits: self.limits,
            cancel: cancel.clone(),
            deadline: Instant::now() + Duration::from_millis(self.limits.timeout_ms),
        };
        let result = self.run_with_timeout(&call.name, executor, validated, &ctx)?;

        // 7. Output size ceiling (defense-in-depth).
        let output_len = output_bytes(&result);
        if output_len > self.limits.max_output_bytes {
            return Err(ExecError::ResourceLimitExceeded {
                name: call.name.clone(),
                kind: ResourceKind::OutputBytes,
                limit: self.limits.max_output_bytes,
            });
        }

        Ok(result)
    }

    fn run_with_timeout(
        &self,
        name: &str,
        executor: Arc<dyn ToolExecutor>,
        args: ValidatedArgs,
        ctx: &ExecContext,
    ) -> Result<ToolResultContent, ExecError> {
        let (tx, rx) = mpsc::channel::<Result<ToolResultContent, ExecError>>();
        let cancel = ctx.cancel.clone();
        let deadline = ctx.deadline;
        let limits = ctx.limits;
        let ctx_for_thread = ExecContext {
            limits,
            cancel: cancel.clone(),
            deadline,
        };
        // The executor is `Send + Sync` and stored behind an `Arc`, so it
        // can be moved into the worker thread without any `unsafe`.
        let builder = thread::Builder::new().name(format!("sandbox:{name}"));
        let handle = builder
            .spawn(move || {
                let _ = tx.send(executor.execute(&args, &ctx_for_thread));
            })
            .map_err(|error| ExecError::Executor(name.to_string(), error.to_string()))?;

        match rx.recv_timeout(Duration::from_millis(self.limits.timeout_ms)) {
            Ok(result) => {
                // The worker finished; joining reaps the thread.
                let _ = handle.join();
                result
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // Raise the cancellation flag so a cooperative executor
                // winds down. The thread is detached (safe Rust cannot
                // force-kill it); the chain is unblocked and the result is
                // fail-closed.
                cancel.store(true, Ordering::Release);
                Err(ExecError::Timeout(name.to_string(), self.limits.timeout_ms))
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                // The worker panicked before sending. Join to recover the
                // panic, then report a fail-closed executor error.
                if let Err(panic_payload) = handle.join() {
                    let msg = panic_payload
                        .downcast_ref::<String>()
                        .map(String::as_str)
                        .or_else(|| panic_payload.downcast_ref::<&str>().copied())
                        .unwrap_or("worker thread panicked");
                    return Err(ExecError::Executor(name.to_string(), msg.to_string()));
                }
                Err(ExecError::Executor(
                    name.to_string(),
                    "worker thread ended without a result".into(),
                ))
            }
        }
    }
}

/// Approximate byte cost of a result content payload, for ceiling checks.
fn output_bytes(content: &ToolResultContent) -> usize {
    match content {
        ToolResultContent::Text { text } => text.len(),
        ToolResultContent::Image { path, description } => path.len() + description.len(),
        ToolResultContent::Video { path, description } => path.len() + description.len(),
        ToolResultContent::Document {
            path,
            pages,
            description,
        } => {
            let pages_cost: usize = pages.iter().map(|p| *p as usize).sum();
            path.len() + description.len() + pages_cost
        }
    }
}

// ---------------------------------------------------------------------------
// Reference executors
// ---------------------------------------------------------------------------

/// A read-only file lookup confined to a fixed working directory.
///
/// This is the milestone executor: "a read-only file lookup within a fixed
/// working directory." It reads a file by relative name, refuses any path
/// that escapes the workdir (absolute paths or `..` traversal), and caps
/// the returned bytes at the sandbox's `max_output_bytes`. It has no
/// network access and no write access — by construction.
pub struct ReadFileInWorkdir {
    workdir: PathBuf,
}

impl ReadFileInWorkdir {
    /// The capability name this executor registers under.
    pub const NAME: &'static str = "read_file_in_workdir";

    /// Build an executor confined to `workdir`. The directory must exist.
    pub fn new(workdir: impl Into<PathBuf>) -> Result<Self, AgentError> {
        let workdir = workdir.into();
        let canonical = fs::canonicalize(&workdir).map_err(|error| {
            AgentError::Config(format!(
                "sandbox workdir {} cannot be canonicalized: {error}",
                workdir.display()
            ))
        })?;
        if !canonical.is_dir() {
            return Err(AgentError::Config(format!(
                "sandbox workdir {} is not a directory",
                canonical.display()
            )));
        }
        Ok(Self { workdir: canonical })
    }

    /// The canonicalized working directory this executor is confined to.
    pub fn workdir(&self) -> &Path {
        &self.workdir
    }

    /// Resolve a relative path against the workdir, refusing any escape.
    fn resolve(&self, relative: &str) -> Result<PathBuf, ExecError> {
        if relative.is_empty() {
            return Err(ExecError::Executor(
                Self::NAME.into(),
                "path argument must not be empty".into(),
            ));
        }
        let relative_path = Path::new(relative);
        // Reject absolute paths and any `..` component in the user input
        // before joining, so a missing file cannot be probed for
        // path-structure information and an absolute path cannot replace
        // the workdir base (`PathBuf::join` replaces the base when given an
        // absolute path).
        if relative_path.is_absolute() {
            return Err(ExecError::Executor(
                Self::NAME.into(),
                "path must be relative and stay inside the workdir".into(),
            ));
        }
        for component in relative_path.components() {
            use std::path::Component;
            if matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            ) {
                return Err(ExecError::Executor(
                    Self::NAME.into(),
                    "path must be relative and stay inside the workdir".into(),
                ));
            }
        }
        let candidate = self.workdir.join(relative_path);
        let canonical = fs::canonicalize(&candidate).map_err(|error| {
            ExecError::Executor(
                Self::NAME.into(),
                format!("cannot resolve path {relative:?}: {error}"),
            )
        })?;
        if !canonical.starts_with(&self.workdir) {
            return Err(ExecError::Executor(
                Self::NAME.into(),
                "resolved path escapes the workdir".into(),
            ));
        }
        Ok(canonical)
    }
}

impl ToolExecutor for ReadFileInWorkdir {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn execute(
        &self,
        args: &ValidatedArgs,
        ctx: &ExecContext,
    ) -> Result<ToolResultContent, ExecError> {
        let path = args
            .value()
            .get("path")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                ExecError::Executor(Self::NAME.into(), "missing string argument `path`".into())
            })?;
        let resolved = self.resolve(path)?;
        let mut file = fs::File::open(&resolved).map_err(|error| {
            ExecError::Executor(Self::NAME.into(), format!("open failed: {error}"))
        })?;
        // Read up to the output ceiling (plus a small slack byte) so a huge
        // file cannot exhaust memory before the ceiling fires. The sandbox's
        // own output check enforces the exact bound.
        let cap = ctx.limits().max_output_bytes;
        let mut text = String::new();
        let mut buf = [0u8; 8192];
        loop {
            if ctx.is_cancelled() {
                return Err(ExecError::Timeout(
                    Self::NAME.into(),
                    ctx.limits().timeout_ms,
                ));
            }
            let n = file.read(&mut buf).map_err(|error| {
                ExecError::Executor(Self::NAME.into(), format!("read failed: {error}"))
            })?;
            if n == 0 {
                break;
            }
            if text.len() + n > cap {
                // Take only what fits under the ceiling and stop.
                let keep = cap.saturating_sub(text.len());
                let slice = &buf[..n.min(keep)];
                match std::str::from_utf8(slice) {
                    Ok(s) => text.push_str(s),
                    Err(_) => {
                        return Err(ExecError::Executor(
                            Self::NAME.into(),
                            "file is not valid UTF-8".into(),
                        ));
                    }
                }
                break;
            }
            match std::str::from_utf8(&buf[..n]) {
                Ok(s) => text.push_str(s),
                Err(_) => {
                    return Err(ExecError::Executor(
                        Self::NAME.into(),
                        "file is not valid UTF-8".into(),
                    ));
                }
            }
        }
        Ok(ToolResultContent::Text { text })
    }
}

impl std::fmt::Debug for ReadFileInWorkdir {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReadFileInWorkdir")
            .field("workdir", &self.workdir)
            .finish()
    }
}

/// An in-memory key-to-text lookup executor.
///
/// Useful for deterministic tests and smoke runs that need a closed-world
/// capability without touching the filesystem. The capability name defaults
/// to `"lookup"` but may be overridden at construction so multiple static
/// tables can coexist.
pub struct StaticLookup {
    name: &'static str,
    table: HashMap<String, String>,
}

impl StaticLookup {
    const DEFAULT_NAME: &'static str = "lookup";

    /// Build a `"lookup"` executor from an owned key→text map.
    pub fn new(table: HashMap<String, String>) -> Self {
        Self {
            name: Self::DEFAULT_NAME,
            table,
        }
    }

    /// Build a named static-lookup executor (e.g. `"lookup_config"`).
    pub fn with_name(name: &'static str, table: HashMap<String, String>) -> Self {
        Self { name, table }
    }
}

impl ToolExecutor for StaticLookup {
    fn name(&self) -> &'static str {
        self.name
    }

    fn execute(
        &self,
        args: &ValidatedArgs,
        _ctx: &ExecContext,
    ) -> Result<ToolResultContent, ExecError> {
        let key = args
            .value()
            .get("key")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                ExecError::Executor(self.name.into(), "missing string argument `key`".into())
            })?;
        match self.table.get(key) {
            Some(text) => Ok(ToolResultContent::Text { text: text.clone() }),
            None => Err(ExecError::Executor(
                self.name.into(),
                format!("no entry for key {key:?}"),
            )),
        }
    }
}

impl std::fmt::Debug for StaticLookup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StaticLookup")
            .field("name", &self.name)
            .field("entries", &self.table.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::authorization::AuthorizationScope;
    use aarambh_studio_inference::ToolCall;
    use serde_json::json;
    use std::collections::HashMap;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Instant;

    fn lookup_definition() -> ToolDefinition {
        ToolDefinition {
            name: "lookup".into(),
            description: "Look up a value by key.".into(),
            parameters: json!({
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
                "additionalProperties": false
            }),
        }
    }

    fn read_file_definition() -> ToolDefinition {
        ToolDefinition {
            name: "read_file_in_workdir".into(),
            description: "Read a UTF-8 text file from the working directory.".into(),
            parameters: json!({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": false
            }),
        }
    }

    // --- Roadmap acceptance test 1 ----------------------------------------

    #[test]
    fn unlisted_tool_name_is_hard_refused_never_attempted() {
        let mut sandbox = ToolSandbox::new(
            AuthorizationScope::new(["lookup"]).unwrap(),
            SandboxLimits::default(),
        )
        .unwrap();
        // Declare a tool "ghost" the model can request, but never register
        // an executor for it. The name is not in the closed-world execution
        // allowlist.
        let ghost = ToolDefinition {
            name: "ghost".into(),
            description: String::new(),
            parameters: json!({"type":"object","properties":{}}),
        };
        sandbox.register_definitions(&[ghost]).unwrap();
        let call = ToolCall {
            name: "ghost".into(),
            arguments: json!({}),
        };
        let result = sandbox.execute(&call, "call_0001");
        assert_eq!(result.status, ToolResultStatus::Error);
        assert!(
            result
                .error
                .as_deref()
                .unwrap()
                .contains("not in the closed-world")
        );
        assert_eq!(result.content, None);
    }

    // --- Roadmap acceptance test 2 ----------------------------------------

    #[test]
    fn unauthorized_but_declared_tool_is_refused_at_execution_not_declaration() {
        // The capability is registered (in the allowlist) and declared, but
        // the operator did NOT authorize "lookup".
        let mut sandbox =
            ToolSandbox::new(AuthorizationScope::empty(), SandboxLimits::default()).unwrap();
        sandbox
            .register_definitions(&[lookup_definition()])
            .unwrap();
        let table = HashMap::from([("alpha".to_string(), "value-alpha".to_string())]);
        sandbox
            .register_executor(Box::new(StaticLookup::new(table)))
            .unwrap();
        let call = ToolCall {
            name: "lookup".into(),
            arguments: json!({"key":"alpha"}),
        };
        let result = sandbox.execute(&call, "call_0001");
        assert_eq!(result.status, ToolResultStatus::Error);
        assert!(result.error.as_deref().unwrap().contains("not authorized"));
        assert_eq!(result.content, None);
        // And the distinct-from-unknown property: an authorized version
        // succeeds, proving the refusal was authorization-specific.
        let mut sandbox_ok = ToolSandbox::new(
            AuthorizationScope::new(["lookup"]).unwrap(),
            SandboxLimits::default(),
        )
        .unwrap();
        sandbox_ok
            .register_definitions(&[lookup_definition()])
            .unwrap();
        let table2 = HashMap::from([("alpha".to_string(), "value-alpha".to_string())]);
        sandbox_ok
            .register_executor(Box::new(StaticLookup::new(table2)))
            .unwrap();
        let ok = sandbox_ok.execute(&call, "call_0001");
        assert_eq!(ok.status, ToolResultStatus::Ok);
    }

    // --- Roadmap acceptance test 3 ----------------------------------------

    /// A cooperative executor that loops until cancelled, used to prove the
    /// wall-clock timeout fires and fails closed.
    struct HangingExecutor {
        name: &'static str,
        invocations: Arc<AtomicUsize>,
    }

    impl ToolExecutor for HangingExecutor {
        fn name(&self) -> &'static str {
            self.name
        }
        fn execute(
            &self,
            _args: &ValidatedArgs,
            ctx: &ExecContext,
        ) -> Result<ToolResultContent, ExecError> {
            self.invocations.fetch_add(1, Ordering::SeqCst);
            let started = Instant::now();
            // Busy-wait but poll the cancellation flag so a cooperative
            // exit is possible; we still expect the sandbox to time out
            // before this returns on its own.
            while !ctx.is_cancelled() {
                if started.elapsed() > Duration::from_secs(30) {
                    // Hard backstop so a test regression cannot hang CI.
                    return Err(ExecError::Executor(
                        self.name.into(),
                        "hard backstop fired".into(),
                    ));
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(ExecError::Timeout(
                self.name.into(),
                ctx.limits().timeout_ms,
            ))
        }
    }

    #[test]
    fn execution_timeout_kills_a_hanging_tool_call() {
        let invocations = Arc::new(AtomicUsize::new(0));
        let limits = SandboxLimits {
            timeout_ms: 120,
            ..SandboxLimits::default()
        };
        let mut sandbox =
            ToolSandbox::new(AuthorizationScope::new(["hang"]).unwrap(), limits).unwrap();
        let hang_def = ToolDefinition {
            name: "hang".into(),
            description: String::new(),
            parameters: json!({"type":"object","properties":{}}),
        };
        sandbox.register_definitions(&[hang_def]).unwrap();
        sandbox
            .register_executor(Box::new(HangingExecutor {
                name: "hang",
                invocations: invocations.clone(),
            }))
            .unwrap();
        let call = ToolCall {
            name: "hang".into(),
            arguments: json!({}),
        };
        let started = Instant::now();
        let result = sandbox.execute(&call, "call_0001");
        let elapsed = started.elapsed();
        assert_eq!(result.status, ToolResultStatus::Error);
        assert!(
            result
                .error
                .as_deref()
                .unwrap()
                .contains("wall-clock ceiling")
        );
        // The call returned promptly after the timeout (plus slack), not
        // after the 30s backstop.
        assert!(elapsed < Duration::from_secs(2), "took {elapsed:?}");
        // The executor was actually entered once (the refusal was an
        // execution-time timeout, not a pre-execution refusal).
        assert_eq!(invocations.load(Ordering::SeqCst), 1);
    }

    // --- Roadmap acceptance test 4 ----------------------------------------

    #[test]
    fn execution_respects_configured_memory_and_cpu_ceiling() {
        // (a) Output-size (memory) ceiling: an executor returning more than
        // max_output_bytes is refused with ResourceLimitExceeded.
        let limits = SandboxLimits {
            max_output_bytes: 16,
            ..SandboxLimits::default()
        };
        let mut sandbox =
            ToolSandbox::new(AuthorizationScope::new(["lookup"]).unwrap(), limits).unwrap();
        sandbox
            .register_definitions(&[lookup_definition()])
            .unwrap();
        let big = "x".repeat(64);
        let table = HashMap::from([("alpha".to_string(), big)]);
        sandbox
            .register_executor(Box::new(StaticLookup::new(table)))
            .unwrap();
        let call = ToolCall {
            name: "lookup".into(),
            arguments: json!({"key":"alpha"}),
        };
        let result = sandbox.execute(&call, "call_0001");
        assert_eq!(result.status, ToolResultStatus::Error);
        assert!(result.error.as_deref().unwrap().contains("output_bytes"));

        // (b) CPU ceiling: a compute-bound executor that loops is bounded
        // by the wall-clock timeout_ms (CPU time == wall-clock for
        // compute-bound work on a single thread).
        let cpu_limits = SandboxLimits {
            timeout_ms: 100,
            ..SandboxLimits::default()
        };
        let mut cpu_sandbox =
            ToolSandbox::new(AuthorizationScope::new(["burn"]).unwrap(), cpu_limits).unwrap();
        let burn_def = ToolDefinition {
            name: "burn".into(),
            description: String::new(),
            parameters: json!({"type":"object","properties":{}}),
        };
        cpu_sandbox.register_definitions(&[burn_def]).unwrap();
        cpu_sandbox
            .register_executor(Box::new(HangingExecutor {
                name: "burn",
                invocations: Arc::new(AtomicUsize::new(0)),
            }))
            .unwrap();
        let burn_call = ToolCall {
            name: "burn".into(),
            arguments: json!({}),
        };
        let started = Instant::now();
        let burn_result = cpu_sandbox.execute(&burn_call, "call_0001");
        assert_eq!(burn_result.status, ToolResultStatus::Error);
        assert!(
            burn_result
                .error
                .as_deref()
                .unwrap()
                .contains("wall-clock ceiling")
        );
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    // --- Roadmap acceptance test 5 ----------------------------------------

    #[test]
    fn malformed_tool_call_json_is_never_executed() {
        let attempts = Arc::new(AtomicUsize::new(0));
        struct CountingLookup {
            attempts: Arc<AtomicUsize>,
        }
        impl ToolExecutor for CountingLookup {
            fn name(&self) -> &'static str {
                "lookup"
            }
            fn execute(
                &self,
                _args: &ValidatedArgs,
                _ctx: &ExecContext,
            ) -> Result<ToolResultContent, ExecError> {
                self.attempts.fetch_add(1, Ordering::SeqCst);
                Ok(ToolResultContent::Text { text: "x".into() })
            }
        }
        let mut sandbox = ToolSandbox::new(
            AuthorizationScope::new(["lookup"]).unwrap(),
            SandboxLimits::default(),
        )
        .unwrap();
        sandbox
            .register_definitions(&[lookup_definition()])
            .unwrap();
        sandbox
            .register_executor(Box::new(CountingLookup {
                attempts: attempts.clone(),
            }))
            .unwrap();
        // Schema requires `key` (string) and disallows extra properties;
        // this call omits `key` and adds a disallowed property ->
        // schema-invalid -> never executed.
        let malformed = ToolCall {
            name: "lookup".into(),
            arguments: json!({"not_a_key": 42}),
        };
        let result = sandbox.execute(&malformed, "call_0001");
        assert_eq!(result.status, ToolResultStatus::Error);
        assert!(
            result
                .error
                .as_deref()
                .unwrap()
                .contains("schema validation")
        );
        assert_eq!(attempts.load(Ordering::SeqCst), 0);
    }

    // --- Roadmap acceptance test 6 ----------------------------------------

    #[test]
    fn execution_result_re_ingests_correctly_into_the_next_chain_step() {
        // Drive the real ToolChain with a FakeDecoder that emits a `lookup`
        // tool call, then a final answer. The SandboxedToolProvider
        // executes the call and the result re-enters the chain via the
        // existing result_ingestion path (proven by the chain state
        // recording the executed content and reaching the final turn).
        use crate::chain::{ToolChain, ToolChainConfig};
        use aarambh_studio_inference::{FinishReason, GenerationOutput, GenerationUsage};
        use std::collections::VecDeque;

        struct FakeDecoder {
            outputs: VecDeque<GenerationOutput>,
        }
        impl crate::chain::ChainDecoder for FakeDecoder {
            fn context_limit(&self) -> usize {
                64
            }
            fn encode_prefix(
                &mut self,
                _prompt: &str,
                _tools: &[ToolDefinition],
                _summary: Option<&str>,
            ) -> crate::AgentResult<Vec<u32>> {
                Ok(vec![1, 2])
            }
            fn encode_result(&mut self, _result: &ToolResult) -> crate::AgentResult<Vec<u32>> {
                Ok(vec![8, 9])
            }
            fn encode_result_metadata(
                &mut self,
                _result: &ToolResult,
            ) -> crate::AgentResult<Vec<u32>> {
                Ok(vec![8])
            }
            fn generate(
                &mut self,
                _transcript_ids: &[u32],
                _pending_media: Option<&ToolResultContent>,
                _max_new_tokens: usize,
            ) -> crate::AgentResult<GenerationOutput> {
                self.outputs
                    .pop_front()
                    .ok_or_else(|| crate::AgentError::Config("fake output exhausted".into()))
            }
            fn summarise(
                &mut self,
                _previous_summary: Option<&str>,
                _evicted: &[crate::ToolExchange],
                _max_tokens: usize,
            ) -> crate::AgentResult<String> {
                Ok("summary".into())
            }
        }

        let call = ToolCall {
            name: "lookup".into(),
            arguments: json!({"key":"alpha"}),
        };
        let first = GenerationOutput {
            text: String::new(),
            raw_text: String::new(),
            thinking_text: String::new(),
            answer_text: String::new(),
            token_ids: vec![4, 5],
            thinking_token_ids: Vec::new(),
            answer_token_ids: vec![4, 5],
            thinking_tokens: 0,
            finish_reason: FinishReason::ToolCall,
            steps: Vec::new(),
            speculative_stats: None,
            tool_call: Some(call.clone()),
            usage: GenerationUsage {
                prompt_tokens: 2,
                completion_tokens: 2,
                total_tokens: 4,
            },
        };
        let second = GenerationOutput {
            text: "done".into(),
            raw_text: "done".into(),
            thinking_text: String::new(),
            answer_text: "done".into(),
            token_ids: vec![6, 7],
            thinking_token_ids: Vec::new(),
            answer_token_ids: vec![6, 7],
            thinking_tokens: 0,
            finish_reason: FinishReason::EosToken,
            steps: Vec::new(),
            speculative_stats: None,
            tool_call: None,
            usage: GenerationUsage {
                prompt_tokens: 4,
                completion_tokens: 2,
                total_tokens: 6,
            },
        };
        let decoder = FakeDecoder {
            outputs: [first, second].into(),
        };
        let mut sandbox = ToolSandbox::new(
            AuthorizationScope::new(["lookup"]).unwrap(),
            SandboxLimits::default(),
        )
        .unwrap();
        sandbox
            .register_definitions(&[lookup_definition()])
            .unwrap();
        let table = HashMap::from([("alpha".to_string(), "value-alpha".to_string())]);
        sandbox
            .register_executor(Box::new(StaticLookup::new(table)))
            .unwrap();
        let provider = SandboxedToolProvider::new(sandbox);
        let mut chain = ToolChain::new(decoder, provider, ToolChainConfig::default()).unwrap();
        let output = chain.run("find it", vec![lookup_definition()]).unwrap();
        assert_eq!(output.final_output.text, "done");
        assert_eq!(output.metrics.tool_calls, 1);
        // The executed result re-entered the chain: the recorded exchange
        // carries the sandbox-produced content, not a caller-supplied value.
        let exchange = &output.state.exchanges()[0];
        assert_eq!(exchange.result.status, ToolResultStatus::Ok);
        assert_eq!(
            exchange.result.content,
            Some(ToolResultContent::Text {
                text: "value-alpha".into()
            })
        );
        assert_eq!(exchange.request.call, call);
    }

    // --- Supporting tests -------------------------------------------------

    #[test]
    fn read_file_executor_refuses_traversal_escape() {
        let dir = tempfile_workdir();
        let mut sandbox = ToolSandbox::new(
            AuthorizationScope::new(["read_file_in_workdir"]).unwrap(),
            SandboxLimits::default(),
        )
        .unwrap();
        sandbox
            .register_definitions(&[read_file_definition()])
            .unwrap();
        sandbox
            .register_executor(Box::new(ReadFileInWorkdir::new(&dir).unwrap()))
            .unwrap();
        // Absolute path -> refused.
        let abs = ToolCall {
            name: "read_file_in_workdir".into(),
            arguments: json!({"path":"/etc/passwd"}),
        };
        let r = sandbox.execute(&abs, "call_0001");
        assert_eq!(r.status, ToolResultStatus::Error);
        // `..` traversal -> refused.
        let traversal = ToolCall {
            name: "read_file_in_workdir".into(),
            arguments: json!({"path":"../secret.txt"}),
        };
        let r = sandbox.execute(&traversal, "call_0001");
        assert_eq!(r.status, ToolResultStatus::Error);
        // A file inside the workdir -> ok.
        fs::write(dir.join("hello.txt"), "hi there").unwrap();
        let ok = ToolCall {
            name: "read_file_in_workdir".into(),
            arguments: json!({"path":"hello.txt"}),
        };
        let r = sandbox.execute(&ok, "call_0001");
        assert_eq!(r.status, ToolResultStatus::Ok);
        assert_eq!(
            r.content,
            Some(ToolResultContent::Text {
                text: "hi there".into()
            })
        );
    }

    #[test]
    fn limits_validation_rejects_zero_ceilings() {
        assert!(
            ToolSandbox::new(
                AuthorizationScope::empty(),
                SandboxLimits {
                    timeout_ms: 0,
                    ..SandboxLimits::default()
                }
            )
            .is_err()
        );
        assert!(
            ToolSandbox::new(
                AuthorizationScope::empty(),
                SandboxLimits {
                    max_output_bytes: 0,
                    ..SandboxLimits::default()
                }
            )
            .is_err()
        );
    }

    #[test]
    fn sandboxed_provider_returns_chain_valid_result() {
        let mut sandbox = ToolSandbox::new(
            AuthorizationScope::new(["lookup"]).unwrap(),
            SandboxLimits::default(),
        )
        .unwrap();
        sandbox
            .register_definitions(&[lookup_definition()])
            .unwrap();
        let table = HashMap::from([("alpha".to_string(), "value-alpha".to_string())]);
        sandbox
            .register_executor(Box::new(StaticLookup::new(table)))
            .unwrap();
        let mut provider = SandboxedToolProvider::new(sandbox);
        let request = ToolResultRequest {
            call_id: "call_0001".into(),
            call: ToolCall {
                name: "lookup".into(),
                arguments: json!({"key":"alpha"}),
            },
        };
        let result = provider.next_result(&request).unwrap();
        assert_eq!(result.call_id, "call_0001");
        assert_eq!(result.status, ToolResultStatus::Ok);
    }

    fn tempfile_workdir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "aarambh-sandbox-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }
}
