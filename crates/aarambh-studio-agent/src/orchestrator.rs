//! Multi-agent orchestration (Phase 48, `ARCHITECTURE_V4.md` §62).
//!
//! One top-level orchestrating reasoning process delegates independent
//! sub-tasks to multiple sandboxed tool-execution sub-chains (each governed
//! entirely by Phase 47's boundaries), then merges their results back into
//! its own context. Orchestration is only as safe as the execution sandbox
//! underneath it — this module adds *delegation and aggregation*, it does
//! not widen Phase 47's execution envelope by a single byte.
//!
//! ## Hard, non-negotiable bounds
//!
//! Three ceilings are enforced as operator-set configuration, never as
//! something the orchestrator's own output can influence:
//!
//! 1. **Maximum sub-agent count** — [`OrchestrationLimits::max_sub_agents`].
//!    A [`DelegationPlan`] with more sub-tasks than this is rejected at
//!    [`Orchestrator::validate_plan`] time, before any sub-chain runs. The
//!    model cannot request unbounded fan-out by emitting a larger plan.
//! 2. **Maximum total execution time budget** —
//!    [`OrchestrationLimits::max_total_time_ms`]. The sum across all
//!    sub-chains, not per sub-chain, so many small sub-agents cannot
//!    collectively exceed the same ceiling one large one would hit. Once
//!    the budget is exhausted, every not-yet-started sub-task is refused
//!    with [`SubChainStatus::BudgetExceeded`] rather than started.
//! 3. **Sandbox scope containment** — a sub-agent's [`AuthorizationScope`]
//!    may only be a **subset** of its orchestrator's. Verified at
//!    `validate_plan` time by `parent.intersect(&child) == child` (true iff
//!    `child ⊆ parent`). Additionally, every tool name a sub-task declares
//!    must be authorized in that sub-task's own scope, so a sub-agent cannot
//!    declare a tool its narrowed scope does not include.
//!
//! ## Failure isolation
//!
//! One sub-agent's failure or execution error is contained to that
//! sub-chain's own result — it does not corrupt or silently swallow sibling
//! sub-agents' results, and the orchestrator's aggregation step receives an
//! explicit failure marker for that sub-chain rather than a missing or
//! malformed entry. Each sub-chain runs inside a `catch_unwind` boundary;
//! panics become [`SubChainStatus::Failed`] with the panic payload rendered
//! into the fail-closed [`ToolResult::error`] text.
//!
//! ## Composability: zero chain changes
//!
//! Each sub-chain is a [`ToolChain`] backed by a [`SandboxedToolProvider`]
//! constructed with the sub-task's narrowed [`AuthorizationScope`] and
//! per-sub-task [`SandboxLimits`]. The chain does not know or care that its
//! results came from sandboxed execution — the same `ToolResultProvider`
//! path v3 §46 / v4 §61 already use. Sub-chain outputs re-enter the
//! orchestrator's own context as [`ToolResultContent::Text`] (or media, for
//! multimodal sub-chains) via the **same** `result_ingestion` path, applied
//! recursively — not a new mechanism.
//!
//! ## CPU-first honesty
//!
//! Sub-chains run sequentially. The spec's wording — *"Sub-chains run
//! (conceptually parallel; actual concurrency bounded by configured limits
//! below)"* — is honored: `max_sub_agents` and `max_total_time_ms` together
//! bound the total work even when run sequentially. True parallelism would
//! require a `ChainDecoder` whose implementor is `Send + Sync`, which is out
//! of scope for the source release because the underlying `InferenceEngine`
//! holds a Candle device that is not safely cloneable across threads. The
//! deterministic unit tests use a `FakeChainDecoder` mirroring the pattern in
//! `chain.rs::tests` and `sandbox.rs::tests`, so they run in milliseconds.

use std::panic::AssertUnwindSafe;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::authorization::AuthorizationScope;
use crate::sandbox::{SandboxLimits, SandboxedToolProvider};
use crate::state::{ToolResult, ToolResultContent, ToolResultStatus};
use crate::{AgentError, AgentResult, ChainDecoder, ToolChain, ToolChainConfig};

/// Default maximum number of sub-agents an orchestrator may delegate to.
pub const DEFAULT_MAX_SUB_AGENTS: usize = 4;
/// Default total wall-clock budget across all sub-chains, in milliseconds.
pub const DEFAULT_MAX_TOTAL_TIME_MS: u64 = 30_000;
/// Minimum meaningful ceiling for `max_sub_agents` (a plan with zero
/// sub-tasks is a no-op, but a ceiling of zero would reject every plan).
pub const MIN_MAX_SUB_AGENTS: usize = 1;
/// Maximum accepted value for `max_sub_agents`. Matches the per-chain
/// `max_steps` ceiling in [`ToolChainConfig`] so an orchestrator cannot
/// fan out wider than a single chain could step.
pub const MAX_MAX_SUB_AGENTS: usize = 64;

/// Operator-set, non-model-influenceable ceilings on orchestration.
///
/// All fields are operator configuration supplied at [`Orchestrator::new`]
/// time; an orchestrator cannot influence them. Mirrors the design of
/// [`SandboxLimits`] and [`ToolChainConfig`]: hard bounds enforced by the
/// runtime, not by the model's own output.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OrchestrationLimits {
    /// Hard ceiling on the number of sub-agents in any [`DelegationPlan`].
    /// A plan with more sub-tasks is rejected at `validate_plan` time.
    pub max_sub_agents: usize,
    /// Hard ceiling on the sum of sub-chain wall-clock times, in
    /// milliseconds. The sum across all sub-chains, not per sub-chain, so
    /// many small sub-agents cannot collectively exceed the same ceiling
    /// one large one would hit.
    pub max_total_time_ms: u64,
}

impl Default for OrchestrationLimits {
    fn default() -> Self {
        Self {
            max_sub_agents: DEFAULT_MAX_SUB_AGENTS,
            max_total_time_ms: DEFAULT_MAX_TOTAL_TIME_MS,
        }
    }
}

impl OrchestrationLimits {
    /// Validate the ceilings are positive and bounded.
    ///
    /// `max_sub_agents` must be in `1..=64` (matching the `ToolChainConfig`
    /// `max_steps` range); `max_total_time_ms` must be greater than zero.
    pub fn validate(&self) -> AgentResult<()> {
        if !(MIN_MAX_SUB_AGENTS..=MAX_MAX_SUB_AGENTS).contains(&self.max_sub_agents) {
            return Err(AgentError::Config(format!(
                "max_sub_agents must be in the range {MIN_MAX_SUB_AGENTS}..={MAX_MAX_SUB_AGENTS}"
            )));
        }
        if self.max_total_time_ms == 0 {
            return Err(AgentError::Config(
                "max_total_time_ms must be greater than zero".into(),
            ));
        }
        Ok(())
    }
}

/// One sub-task the orchestrator delegates to an independent sub-chain.
///
/// Each sub-task becomes its own sub-chain — a full [`ToolChain`]-backed
/// instance with its own [`ChainDecoder`] (supplied by the orchestrator's
/// `build_decoder` callback), its own sandboxed tool scope (Phase 47), and
/// its own execution timeout budget ([`SandboxLimits::timeout_ms`] applies
/// per call within the sub-chain; the orchestrator's
/// [`OrchestrationLimits::max_total_time_ms`] bounds the sum across all
/// sub-chains).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelegatedSubTask {
    /// Operator- or model-assigned sub-task identifier. Becomes the
    /// `call_id` of the aggregated [`ToolResult`] so the orchestrator's
    /// own context can reference each sub-chain's output by id.
    pub sub_task_id: String,
    /// The user-visible prompt for the sub-chain.
    pub prompt: String,
    /// Tool definitions available to the sub-chain. Every name must be
    /// authorized in `authorization` (a sub-agent cannot declare a tool
    /// its narrowed scope does not include).
    pub tools: Vec<aarambh_studio_inference::ToolDefinition>,
    /// The sub-agent's authorization scope. Must be a subset of the
    /// orchestrator's scope (verified at `validate_plan` time via
    /// `AuthorizationScope::intersect`).
    pub authorization: AuthorizationScope,
    /// Per-sub-chain sandbox limits. The `timeout_ms` here bounds each
    /// individual tool call inside the sub-chain; the orchestrator's
    /// `max_total_time_ms` bounds the sum across all sub-chains.
    pub limits: SandboxLimits,
    /// Per-sub-chain tool-chain configuration (max steps, eviction policy,
    /// summary tokens, etc). Inherited from the orchestrator's own chain
    /// config by default.
    pub chain: ToolChainConfig,
}

/// The orchestrator's delegation plan, validated before any sub-chain runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelegationPlan {
    /// The sub-tasks the orchestrator is delegating to, in execution order.
    pub sub_tasks: Vec<DelegatedSubTask>,
}

impl DelegationPlan {
    /// Build an empty plan (no sub-tasks). Useful for tests; a real plan is
    /// typically deserialized from JSON.
    pub fn empty() -> Self {
        Self {
            sub_tasks: Vec::new(),
        }
    }

    /// Number of sub-tasks in the plan.
    pub fn len(&self) -> usize {
        self.sub_tasks.len()
    }

    /// Whether the plan contains no sub-tasks.
    pub fn is_empty(&self) -> bool {
        self.sub_tasks.is_empty()
    }
}

/// Outcome category for one delegated sub-chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SubChainStatus {
    /// The sub-chain ran to a final answer and its output was aggregated
    /// into a `status = Ok` [`ToolResult`].
    Completed,
    /// The sub-chain could not start because the orchestrator's total
    /// execution time budget was already exhausted. The aggregated result
    /// is a fail-closed `ToolResult` with `status = Error`.
    BudgetExceeded,
    /// The sub-task declared a tool its narrowed authorization scope does
    /// not include, or its scope was not a subset of the orchestrator's.
    /// The aggregated result is a fail-closed `ToolResult`.
    ScopeViolation,
    /// The sub-chain panicked or returned an `Err(AgentError)`. The
    /// aggregated result is a fail-closed `ToolResult` carrying the
    /// error text. Sibling sub-chains are unaffected.
    Failed,
}

/// The outcome of one delegated sub-chain, ready for aggregation.
///
/// `aggregated_result` is always present and is always a valid
/// [`ToolResult`]: `status = Ok` with the sub-chain's final text for
/// [`SubChainStatus::Completed`], or `status = Error` with a bounded error
/// message for every failure variant. The orchestrator's aggregation step
/// therefore never sees a missing or malformed entry, exactly as
/// `ARCHITECTURE_V4.md` §62 requires.
#[derive(Debug, Clone, Serialize)]
pub struct SubChainOutcome {
    /// The sub-task id this outcome belongs to.
    pub sub_task_id: String,
    /// Outcome category for the sub-chain.
    pub status: SubChainStatus,
    /// The aggregated [`ToolResult`] ready to re-ingest into the
    /// orchestrator's own context via the existing `result_ingestion`
    /// path. `call_id` matches `sub_task_id`.
    pub aggregated_result: ToolResult,
    /// Wall-clock time the sub-chain actually ran for, in milliseconds.
    /// Zero for sub-tasks that never started (`BudgetExceeded`,
    /// `ScopeViolation`).
    pub elapsed_ms: u64,
}

/// Maximum bytes of error text retained in a fail-closed aggregated result.
const MAX_AGGREGATED_ERROR_TEXT_BYTES: usize = 4 * 1024;

/// One top-level orchestrating reasoning process.
///
/// Built once from operator-set [`OrchestrationLimits`] and the
/// orchestrator's own [`AuthorizationScope`]. Each [`DelegationPlan`] is
/// validated against the three hard bounds before any sub-chain runs, then
/// executed sequentially with the budget tracked across sub-chains.
pub struct Orchestrator {
    limits: OrchestrationLimits,
    orchestrator_authorization: AuthorizationScope,
}

impl Orchestrator {
    /// Build an orchestrator from operator-set ceilings and the
    /// orchestrator's own authorization scope. The scope is the closed set
    /// of tool names the operator enabled at the top level; every
    /// sub-agent's scope must be a subset of it.
    pub fn new(
        limits: OrchestrationLimits,
        orchestrator_authorization: AuthorizationScope,
    ) -> AgentResult<Self> {
        limits.validate()?;
        Ok(Self {
            limits,
            orchestrator_authorization,
        })
    }

    /// Read-only access to the configured limits.
    pub fn limits(&self) -> OrchestrationLimits {
        self.limits
    }

    /// Read-only access to the orchestrator's own authorization scope.
    pub fn authorization(&self) -> &AuthorizationScope {
        &self.orchestrator_authorization
    }

    /// Validate the plan against all three hard bounds before any sub-chain
    /// runs. Returns the cause of the first violation.
    ///
    /// Bounds checked, in order:
    /// 1. **Sub-agent count** — `plan.sub_tasks.len() <= max_sub_agents`.
    /// 2. **Scope containment** — for every sub-task,
    ///    `orchestrator_authorization.intersect(&sub.authorization)
    ///    == sub.authorization` (true iff `sub.authorization ⊆
    ///    orchestrator_authorization`), and every declared tool name is
    ///    authorized in `sub.authorization`.
    /// 3. **Per-sub-task sandbox limits** — every sub-task's [`SandboxLimits`]
    ///    and [`ToolChainConfig`] must individually validate.
    pub fn validate_plan(&self, plan: &DelegationPlan) -> AgentResult<()> {
        // Bound 1: sub-agent count ceiling.
        if plan.sub_tasks.len() > self.limits.max_sub_agents {
            return Err(AgentError::Config(format!(
                "delegation plan has {} sub-tasks but the orchestrator's max_sub_agents ceiling is {}",
                plan.sub_tasks.len(),
                self.limits.max_sub_agents
            )));
        }

        // Bound 2: scope containment for every sub-task.
        for sub in &plan.sub_tasks {
            // A sub-agent's scope can only be a subset of its orchestrator's.
            // `parent.intersect(&child) == child` is true iff `child ⊆ parent`.
            let narrowed = self
                .orchestrator_authorization
                .intersect(&sub.authorization);
            if narrowed.allowed() != sub.authorization.allowed() {
                return Err(AgentError::Config(format!(
                    "sub-task {:?} authorizes tools outside the orchestrator's scope: {:?}",
                    sub.sub_task_id,
                    sub.authorization
                        .allowed()
                        .symmetric_difference(self.orchestrator_authorization.allowed())
                        .collect::<Vec<_>>(),
                )));
            }
            // A sub-agent cannot declare a tool its narrowed scope does not
            // include. This catches plans where the model declares a tool
            // name the operator did not narrow into the sub-agent's scope.
            for tool in &sub.tools {
                if !sub.authorization.is_authorized(&tool.name) {
                    return Err(AgentError::Config(format!(
                        "sub-task {:?} declares tool {:?} but the sub-agent's narrowed scope does not authorize it",
                        sub.sub_task_id, tool.name
                    )));
                }
            }
            // Bound 3: per-sub-task sandbox and chain config validation.
            sub.limits.validate()?;
            sub.chain.validate()?;
        }
        Ok(())
    }

    /// Run all sub-chains under the bounds; return one outcome per sub-task,
    /// in plan order. Each outcome's `aggregated_result` is ready to push
    /// back into the orchestrator's own [`ToolChain`] via the existing
    /// [`crate::ToolResultProvider`] path.
    ///
    /// The `build_decoder` closure is called once per sub-task and must
    /// return a fresh `(D, SandboxedToolProvider)` pair built from the
    /// sub-task's narrowed authorization and per-sub-task sandbox limits.
    /// The orchestrator owns neither the decoder nor the provider — each
    /// sub-chain is fully independent, exactly as the spec requires.
    pub fn run<D, F>(
        &self,
        plan: &DelegationPlan,
        mut build_decoder: F,
    ) -> AgentResult<Vec<SubChainOutcome>>
    where
        D: ChainDecoder,
        F: FnMut(&DelegatedSubTask) -> AgentResult<(D, SandboxedToolProvider)>,
    {
        // Defense-in-depth: validate_plan was supposed to be called already
        // by the operator, but re-check here so a programmatic caller cannot
        // bypass the bounds.
        self.validate_plan(plan)?;
        let budget = Duration::from_millis(self.limits.max_total_time_ms);
        let mut elapsed_so_far = Duration::ZERO;
        let mut outcomes = Vec::with_capacity(plan.sub_tasks.len());
        for sub in &plan.sub_tasks {
            // Bound 2 is checked per-sub-task at validate_plan time, but the
            // budget check is dynamic and must be done here, at the moment
            // the sub-chain would start.
            if elapsed_so_far >= budget {
                // Budget exhausted: refuse to start the sub-chain. Mark it
                // BudgetExceeded and continue so siblings still get their
                // chance if the budget allows — failure isolation.
                outcomes.push(SubChainOutcome {
                    sub_task_id: sub.sub_task_id.clone(),
                    status: SubChainStatus::BudgetExceeded,
                    aggregated_result: fail_closed_result(
                        &sub.sub_task_id,
                        "orchestration time budget exhausted before this sub-chain started",
                    ),
                    elapsed_ms: 0,
                });
                continue;
            }

            let started = Instant::now();
            let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
                self.run_one_sub_chain::<D, F>(sub, &mut build_decoder)
            }));
            let elapsed = started.elapsed();
            elapsed_so_far += elapsed;
            let status = match outcome {
                Ok(Ok(chain_output)) => SubChainOutcome {
                    sub_task_id: sub.sub_task_id.clone(),
                    status: SubChainStatus::Completed,
                    aggregated_result: ToolResult {
                        call_id: sub.sub_task_id.clone(),
                        status: ToolResultStatus::Ok,
                        content: Some(ToolResultContent::Text {
                            text: chain_output.final_output.text,
                        }),
                        error: None,
                    },
                    elapsed_ms: elapsed.as_millis() as u64,
                },
                Ok(Err(error)) => SubChainOutcome {
                    sub_task_id: sub.sub_task_id.clone(),
                    status: SubChainStatus::Failed,
                    aggregated_result: fail_closed_result(&sub.sub_task_id, &error.to_string()),
                    elapsed_ms: elapsed.as_millis() as u64,
                },
                Err(panic_payload) => {
                    let msg = panic_payload
                        .downcast_ref::<String>()
                        .map(String::as_str)
                        .or_else(|| panic_payload.downcast_ref::<&str>().copied())
                        .unwrap_or("sub-chain worker panicked");
                    SubChainOutcome {
                        sub_task_id: sub.sub_task_id.clone(),
                        status: SubChainStatus::Failed,
                        aggregated_result: fail_closed_result(
                            &sub.sub_task_id,
                            &format!("sub-chain panicked: {msg}"),
                        ),
                        elapsed_ms: elapsed.as_millis() as u64,
                    }
                }
            };
            outcomes.push(status);
        }
        Ok(outcomes)
    }

    /// Run one sub-chain to a final answer. Isolated in its own method so
    /// `run` can wrap it in `catch_unwind`. Errors propagate up; panics
    /// propagate up (caught by `catch_unwind`).
    fn run_one_sub_chain<D, F>(
        &self,
        sub: &DelegatedSubTask,
        build_decoder: &mut F,
    ) -> AgentResult<crate::chain::ChainOutput>
    where
        D: ChainDecoder,
        F: FnMut(&DelegatedSubTask) -> AgentResult<(D, SandboxedToolProvider)>,
    {
        let (decoder, provider) = build_decoder(sub)?;
        // The ToolChain constructor validates the chain config and that
        // context_reserve is smaller than the model context length; both
        // were already checked at validate_plan time, but re-validating
        // here is defense-in-depth against a buggy decoder factory that
        // returns a decoder with a different context limit.
        let mut chain = ToolChain::new(decoder, provider, sub.chain.clone())?;
        let tools = sub.tools.clone();
        let prompt = sub.prompt.clone();
        chain.run(prompt, tools)
    }
}

/// Build a fail-closed [`ToolResult`] with bounded error text.
///
/// The error text is truncated to [`MAX_AGGREGATED_ERROR_TEXT_BYTES`] so a
/// runaway error message cannot blow up the orchestrator's own context. The
/// `call_id` matches the sub-task id so the orchestrator can reference the
/// failed sub-chain by id in its own transcript.
fn fail_closed_result(sub_task_id: &str, message: &str) -> ToolResult {
    let bounded = if message.len() > MAX_AGGREGATED_ERROR_TEXT_BYTES {
        let mut head: String = message
            .chars()
            .take(MAX_AGGREGATED_ERROR_TEXT_BYTES - 1)
            .collect();
        head.push('…');
        head
    } else {
        message.to_string()
    };
    // Belt-and-braces: never produce an empty error text, which would violate
    // the ToolResult contract. The truncation above preserves at least one
    // char if the original was non-empty, but guard anyway.
    let error = if bounded.trim().is_empty() {
        "sub-chain failed".to_string()
    } else {
        bounded
    };
    ToolResult {
        call_id: sub_task_id.to_string(),
        status: ToolResultStatus::Error,
        content: None,
        error: Some(error),
    }
}

#[cfg(test)]
mod tests {
    //! Phase 48 acceptance tests.
    //!
    //! Five roadmap-named tests (one per acceptance criterion in
    //! `ROADMAP_V4.md` §"Phase 48 — Tests") plus supporting tests. All use a
    //! `FakeDecoder` mirroring `chain.rs::tests::FakeDecoder` and
    //! `sandbox.rs::tests::FakeDecoder` so they run in milliseconds.

    use std::collections::VecDeque;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use aarambh_studio_inference::{
        FinishReason, GenerationOutput, GenerationUsage, ToolCall, ToolDefinition,
    };
    use serde_json::json;

    use super::*;
    use crate::authorization::AuthorizationScope;
    use crate::sandbox::{SandboxLimits, StaticLookup, ToolSandbox};
    use crate::state::{ToolResultContent, ToolResultStatus};
    use crate::{ChainDecoder, EvictionPolicy, ToolExchange, ToolResult};

    /// A reusable fake decoder that pops pre-built `GenerationOutput`s from a
    /// queue, mirroring `chain.rs::tests::FakeDecoder` and
    /// `sandbox.rs::tests::FakeDecoder`.
    struct FakeDecoder {
        outputs: VecDeque<GenerationOutput>,
        context_limit: usize,
        /// Optional sleep duration to simulate work for budget tests.
        sleep_per_generate: Option<Duration>,
    }

    impl FakeDecoder {
        fn new(outputs: Vec<GenerationOutput>) -> Self {
            Self {
                outputs: outputs.into(),
                context_limit: 64,
                sleep_per_generate: None,
            }
        }

        fn with_sleep(outputs: Vec<GenerationOutput>, sleep: Duration) -> Self {
            Self {
                outputs: outputs.into(),
                context_limit: 64,
                sleep_per_generate: Some(sleep),
            }
        }
    }

    impl ChainDecoder for FakeDecoder {
        fn context_limit(&self) -> usize {
            self.context_limit
        }

        fn encode_prefix(
            &mut self,
            _prompt: &str,
            _tools: &[ToolDefinition],
            _summary: Option<&str>,
        ) -> AgentResult<Vec<u32>> {
            Ok(vec![1, 2])
        }

        fn encode_result(&mut self, _result: &ToolResult) -> AgentResult<Vec<u32>> {
            Ok(vec![8, 9])
        }

        fn encode_result_metadata(&mut self, _result: &ToolResult) -> AgentResult<Vec<u32>> {
            Ok(vec![8])
        }

        fn generate(
            &mut self,
            _transcript_ids: &[u32],
            _pending_media: Option<&ToolResultContent>,
            _max_new_tokens: usize,
        ) -> AgentResult<GenerationOutput> {
            if let Some(sleep) = self.sleep_per_generate {
                std::thread::sleep(sleep);
            }
            self.outputs
                .pop_front()
                .ok_or_else(|| AgentError::Config("fake output exhausted".into()))
        }

        fn summarise(
            &mut self,
            _previous_summary: Option<&str>,
            _evicted: &[ToolExchange],
            _max_tokens: usize,
        ) -> AgentResult<String> {
            Ok("summary".into())
        }
    }

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

    fn final_output(text: &str) -> GenerationOutput {
        GenerationOutput {
            text: text.into(),
            raw_text: text.into(),
            thinking_text: String::new(),
            answer_text: text.into(),
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
        }
    }

    /// Build a `SandboxedToolProvider` whose sandbox authorizes `lookup`
    /// and registers a `StaticLookup` executor returning `value-alpha`.
    fn lookup_provider(scope: AuthorizationScope) -> SandboxedToolProvider {
        let mut sandbox = ToolSandbox::new(scope, SandboxLimits::default()).unwrap();
        sandbox
            .register_definitions(&[lookup_definition()])
            .unwrap();
        let table =
            std::collections::HashMap::from([("alpha".to_string(), "value-alpha".to_string())]);
        sandbox
            .register_executor(Box::new(StaticLookup::new(table)))
            .unwrap();
        SandboxedToolProvider::new(sandbox)
    }

    /// Build a sub-task that runs a one-step lookup chain ending in `text`.
    fn sub_task(id: &str, scope: AuthorizationScope) -> DelegatedSubTask {
        sub_task_with_tool(id, lookup_definition(), scope)
    }

    /// Build a sub-task with an explicit tool definition. The tool's name
    /// must be authorized in `scope` — `validate_plan` checks this.
    fn sub_task_with_tool(
        id: &str,
        tool: ToolDefinition,
        scope: AuthorizationScope,
    ) -> DelegatedSubTask {
        DelegatedSubTask {
            sub_task_id: id.into(),
            prompt: format!("find {id}"),
            tools: vec![tool],
            authorization: scope,
            limits: SandboxLimits::default(),
            chain: ToolChainConfig {
                max_steps: 1,
                max_tokens_per_step: 256,
                context_reserve: 32,
                keep_recent: 4,
                summary_tokens: 128,
                eviction_policy: EvictionPolicy::DropOldest,
            },
        }
    }

    fn parent_scope() -> AuthorizationScope {
        AuthorizationScope::new(["lookup", "read_file_in_workdir"]).unwrap()
    }

    // --- Roadmap acceptance test 1 ----------------------------------------

    #[test]
    fn orchestrator_cannot_exceed_configured_max_sub_agent_count() {
        // Build a plan with max_sub_agents + 1 sub-tasks, each valid on its
        // own. The count ceiling is the first bound checked, so the extra
        // sub-task must trigger the rejection.
        let limits = OrchestrationLimits {
            max_sub_agents: 2,
            ..OrchestrationLimits::default()
        };
        let orchestrator = Orchestrator::new(limits, parent_scope()).unwrap();
        let plan = DelegationPlan {
            sub_tasks: vec![
                sub_task("a", AuthorizationScope::new(["lookup"]).unwrap()),
                sub_task("b", AuthorizationScope::new(["lookup"]).unwrap()),
                sub_task("c", AuthorizationScope::new(["lookup"]).unwrap()),
            ],
        };
        let err = orchestrator.validate_plan(&plan).unwrap_err();
        match err {
            AgentError::Config(msg) => {
                assert!(msg.contains("max_sub_agents"), "got: {msg}");
                assert!(msg.contains("3"), "got: {msg}");
                assert!(msg.contains("2"), "got: {msg}");
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    // --- Roadmap acceptance test 2 ----------------------------------------

    #[test]
    fn orchestrator_cannot_exceed_configured_total_execution_time_budget() {
        // Each sub-task's decoder sleeps 50 ms inside `generate` to simulate
        // work. With max_total_time_ms = 100, the first two sub-tasks run
        // (total 100 ms), and the third is refused with BudgetExceeded
        // *without* calling `generate` (proven by the per-decoder invocation
        // counter staying at 2).
        let invocations = Arc::new(AtomicUsize::new(0));

        let limits = OrchestrationLimits {
            max_sub_agents: 8,
            max_total_time_ms: 100,
        };
        let orchestrator = Orchestrator::new(limits, parent_scope()).unwrap();
        let plan = DelegationPlan {
            sub_tasks: vec![
                sub_task("a", AuthorizationScope::new(["lookup"]).unwrap()),
                sub_task("b", AuthorizationScope::new(["lookup"]).unwrap()),
                sub_task("c", AuthorizationScope::new(["lookup"]).unwrap()),
            ],
        };
        let invocations_for_closure = invocations.clone();
        let outcomes = orchestrator
            .run::<FakeDecoder, _>(&plan, |_sub| {
                invocations_for_closure.fetch_add(1, Ordering::SeqCst);
                let decoder =
                    FakeDecoder::with_sleep(vec![final_output("done")], Duration::from_millis(50));
                let provider = lookup_provider(AuthorizationScope::new(["lookup"]).unwrap());
                Ok((decoder, provider))
            })
            .unwrap();
        assert_eq!(outcomes.len(), 3);
        assert_eq!(outcomes[0].status, SubChainStatus::Completed);
        assert_eq!(outcomes[1].status, SubChainStatus::Completed);
        assert_eq!(outcomes[2].status, SubChainStatus::BudgetExceeded);
        // The third sub-chain never started: only two decoders were built.
        assert_eq!(invocations.load(Ordering::SeqCst), 2);
        // The aggregated result for the third is a fail-closed error.
        assert_eq!(
            outcomes[2].aggregated_result.status,
            ToolResultStatus::Error
        );
        assert!(
            outcomes[2]
                .aggregated_result
                .error
                .as_deref()
                .unwrap()
                .contains("budget")
        );
        // The first two aggregated results carry the sub-chain output.
        assert_eq!(outcomes[0].aggregated_result.status, ToolResultStatus::Ok);
        assert_eq!(
            outcomes[0].aggregated_result.content,
            Some(ToolResultContent::Text {
                text: "done".into()
            })
        );
    }

    // --- Roadmap acceptance test 3 ----------------------------------------

    #[test]
    fn sub_agent_sandbox_scope_is_never_wider_than_orchestrator_authorization() {
        // Parent authorizes {lookup, read_file_in_workdir}. One sub-task
        // authorizes {lookup, dangerous_shell} — dangerous_shell is not in
        // the parent's scope, so intersect(parent, child) != child and the
        // plan is rejected at validate_plan time.
        let orchestrator =
            Orchestrator::new(OrchestrationLimits::default(), parent_scope()).unwrap();
        let mut overreaching_scope = AuthorizationScope::empty();
        overreaching_scope.enable("lookup").unwrap();
        overreaching_scope.enable("dangerous_shell").unwrap();
        let plan = DelegationPlan {
            sub_tasks: vec![sub_task("rogue", overreaching_scope)],
        };
        let err = orchestrator.validate_plan(&plan).unwrap_err();
        match err {
            AgentError::Config(msg) => {
                assert!(msg.contains("rogue"), "got: {msg}");
                assert!(
                    msg.contains("outside the orchestrator's scope"),
                    "got: {msg}"
                );
            }
            other => panic!("expected Config error, got {other:?}"),
        }

        // A second plan where every sub-task's scope is a strict subset
        // passes validation. This is the positive control. Each sub-task
        // declares only the tool(s) its narrowed scope authorizes.
        let ok_plan = DelegationPlan {
            sub_tasks: vec![
                sub_task("a", AuthorizationScope::new(["lookup"]).unwrap()),
                sub_task_with_tool(
                    "b",
                    read_file_definition(),
                    AuthorizationScope::new(["read_file_in_workdir"]).unwrap(),
                ),
                sub_task("c", parent_scope()), // subset of itself is allowed
            ],
        };
        orchestrator.validate_plan(&ok_plan).unwrap();
    }

    // --- Roadmap acceptance test 4 ----------------------------------------

    #[test]
    fn result_aggregation_correctly_merges_multiple_sub_chain_outputs() {
        // Two sub-tasks with distinct ids and distinct final outputs. Both
        // outcomes must be Completed with the right aggregated content and
        // the right call_id (= sub_task_id), ready to re-ingest into the
        // orchestrator's own context.
        let orchestrator =
            Orchestrator::new(OrchestrationLimits::default(), parent_scope()).unwrap();
        let plan = DelegationPlan {
            sub_tasks: vec![
                sub_task("left", AuthorizationScope::new(["lookup"]).unwrap()),
                sub_task("right", AuthorizationScope::new(["lookup"]).unwrap()),
            ],
        };
        let outcomes = orchestrator
            .run::<FakeDecoder, _>(&plan, |sub| {
                // Each sub-chain produces a different final text bound to
                // its sub_task_id, so we can prove the aggregation is
                // per-sub-task and not mixed up.
                let text = format!("result-{}", sub.sub_task_id);
                let decoder = FakeDecoder::new(vec![final_output(&text)]);
                let provider = lookup_provider(AuthorizationScope::new(["lookup"]).unwrap());
                Ok((decoder, provider))
            })
            .unwrap();
        assert_eq!(outcomes.len(), 2);
        assert_eq!(outcomes[0].sub_task_id, "left");
        assert_eq!(outcomes[1].sub_task_id, "right");
        assert_eq!(outcomes[0].status, SubChainStatus::Completed);
        assert_eq!(outcomes[1].status, SubChainStatus::Completed);
        assert_eq!(outcomes[0].aggregated_result.call_id, "left");
        assert_eq!(outcomes[1].aggregated_result.call_id, "right");
        assert_eq!(
            outcomes[0].aggregated_result.content,
            Some(ToolResultContent::Text {
                text: "result-left".into()
            })
        );
        assert_eq!(
            outcomes[1].aggregated_result.content,
            Some(ToolResultContent::Text {
                text: "result-right".into()
            })
        );
        // Both `validate_for` pass — the aggregated results are valid input
        // to the orchestrator's own ToolChain.
        outcomes[0].aggregated_result.validate_for("left").unwrap();
        outcomes[1].aggregated_result.validate_for("right").unwrap();
    }

    // --- Roadmap acceptance test 5 ----------------------------------------

    #[test]
    fn one_sub_agent_failure_does_not_silently_corrupt_sibling_sub_agent_results() {
        // Three sub-tasks. The middle one's decoder has an empty output
        // queue, so its first `generate()` call returns
        // `Err("fake output exhausted")`. The orchestrator must mark the
        // middle outcome Failed (with a fail-closed aggregated result) and
        // still run the first and third to completion.
        let orchestrator =
            Orchestrator::new(OrchestrationLimits::default(), parent_scope()).unwrap();
        let plan = DelegationPlan {
            sub_tasks: vec![
                sub_task("first", AuthorizationScope::new(["lookup"]).unwrap()),
                sub_task("middle", AuthorizationScope::new(["lookup"]).unwrap()),
                sub_task("third", AuthorizationScope::new(["lookup"]).unwrap()),
            ],
        };
        let outcomes = orchestrator
            .run::<FakeDecoder, _>(&plan, |sub| {
                if sub.sub_task_id == "middle" {
                    // An empty queue makes `generate()` return Err on the
                    // first call, simulating a sub-chain that errors out
                    // before producing a final answer. The provider is
                    // still constructed (it is never called because
                    // generate() fails before the chain asks for a result).
                    let failing_decoder = FakeDecoder::new(Vec::new());
                    let provider = lookup_provider(AuthorizationScope::new(["lookup"]).unwrap());
                    return Ok((failing_decoder, provider));
                }
                let text = format!("result-{}", sub.sub_task_id);
                let decoder = FakeDecoder::new(vec![final_output(&text)]);
                let provider = lookup_provider(AuthorizationScope::new(["lookup"]).unwrap());
                Ok((decoder, provider))
            })
            .unwrap();
        assert_eq!(outcomes.len(), 3);
        assert_eq!(outcomes[0].status, SubChainStatus::Completed);
        assert_eq!(outcomes[1].status, SubChainStatus::Failed);
        assert_eq!(outcomes[2].status, SubChainStatus::Completed);
        // Siblings' aggregated results are intact.
        assert_eq!(
            outcomes[0].aggregated_result.content,
            Some(ToolResultContent::Text {
                text: "result-first".into()
            })
        );
        assert_eq!(
            outcomes[2].aggregated_result.content,
            Some(ToolResultContent::Text {
                text: "result-third".into()
            })
        );
        // The failed middle outcome is a fail-closed ToolResult, not missing.
        assert_eq!(
            outcomes[1].aggregated_result.status,
            ToolResultStatus::Error
        );
        assert!(
            outcomes[1]
                .aggregated_result
                .error
                .as_deref()
                .unwrap()
                .contains("fake output exhausted")
        );
        assert_eq!(outcomes[1].aggregated_result.call_id, "middle");
        // And the failed result is a valid chain input (status/error contract).
        outcomes[1]
            .aggregated_result
            .validate_for("middle")
            .unwrap();
    }

    // --- Supporting tests -------------------------------------------------

    #[test]
    fn limits_validation_rejects_zero_ceilings() {
        assert!(
            Orchestrator::new(
                OrchestrationLimits {
                    max_sub_agents: 0,
                    ..OrchestrationLimits::default()
                },
                parent_scope(),
            )
            .is_err()
        );
        assert!(
            Orchestrator::new(
                OrchestrationLimits {
                    max_total_time_ms: 0,
                    ..OrchestrationLimits::default()
                },
                parent_scope(),
            )
            .is_err()
        );
        // 65 exceeds the MAX_MAX_SUB_AGENTS (64) ceiling.
        assert!(
            Orchestrator::new(
                OrchestrationLimits {
                    max_sub_agents: 65,
                    ..OrchestrationLimits::default()
                },
                parent_scope(),
            )
            .is_err()
        );
        // A 64 ceiling is the upper bound and is accepted.
        assert!(
            Orchestrator::new(
                OrchestrationLimits {
                    max_sub_agents: 64,
                    ..OrchestrationLimits::default()
                },
                parent_scope(),
            )
            .is_ok()
        );
    }

    #[test]
    fn validate_plan_rejects_subtask_declaring_unauthorized_tool() {
        // The sub-task's narrowed scope authorizes {lookup} but the sub-task
        // declares a tool "ghost" that the scope does not include. This is a
        // scope violation distinct from "sub-agent wider than orchestrator":
        // the sub-agent's scope is a subset of the orchestrator's, but the
        // sub-task declares a tool the sub-agent itself is not authorized
        // to call.
        let orchestrator =
            Orchestrator::new(OrchestrationLimits::default(), parent_scope()).unwrap();
        let ghost = ToolDefinition {
            name: "ghost".into(),
            description: String::new(),
            parameters: json!({"type":"object","properties":{}}),
        };
        let mut sub = sub_task("ghosty", AuthorizationScope::new(["lookup"]).unwrap());
        sub.tools.push(ghost);
        let plan = DelegationPlan {
            sub_tasks: vec![sub],
        };
        let err = orchestrator.validate_plan(&plan).unwrap_err();
        match err {
            AgentError::Config(msg) => {
                assert!(msg.contains("ghost"), "got: {msg}");
                assert!(msg.contains("narrowed scope"), "got: {msg}");
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn intersect_equals_child_when_child_is_subset() {
        // The subset check `parent.intersect(&child) == child` is the
        // invariant `validate_plan` relies on. Prove it directly.
        let parent = parent_scope();
        let child = AuthorizationScope::new(["lookup"]).unwrap();
        let narrowed = parent.intersect(&child);
        assert_eq!(narrowed.allowed(), child.allowed());
        // A child that is not a subset produces a different narrowed set.
        let mut overreaching = AuthorizationScope::empty();
        overreaching.enable("lookup").unwrap();
        overreaching.enable("dangerous_shell").unwrap();
        let narrowed2 = parent.intersect(&overreaching);
        assert_ne!(narrowed2.allowed(), overreaching.allowed());
    }

    #[test]
    fn run_revalidates_plan_defense_in_depth() {
        // A programmatic caller might skip validate_plan and call run
        // directly with an invalid plan. run must re-validate.
        let orchestrator = Orchestrator::new(
            OrchestrationLimits {
                max_sub_agents: 1,
                ..OrchestrationLimits::default()
            },
            parent_scope(),
        )
        .unwrap();
        let plan = DelegationPlan {
            sub_tasks: vec![
                sub_task("a", AuthorizationScope::new(["lookup"]).unwrap()),
                sub_task("b", AuthorizationScope::new(["lookup"]).unwrap()),
            ],
        };
        let result = orchestrator.run::<FakeDecoder, _>(&plan, |sub| {
            let decoder = FakeDecoder::new(vec![final_output(&sub.sub_task_id)]);
            let provider = lookup_provider(AuthorizationScope::new(["lookup"]).unwrap());
            Ok((decoder, provider))
        });
        assert!(matches!(result, Err(AgentError::Config(_))));
    }

    /// A trivial sub-chain provider test: prove the orchestrator-built
    /// provider actually executes a tool call through the sandbox and
    /// re-ingests the result. This mirrors `sandbox.rs::tests::
    /// execution_result_re_ingests_correctly_into_the_next_chain_step`
    /// but at the orchestrator level: a single-sub-task plan whose
    /// sub-chain itself contains a tool call.
    #[test]
    fn orchestrator_sub_chain_can_execute_tools_through_sandbox() {
        use aarambh_studio_inference::{FinishReason, GenerationOutput, GenerationUsage};
        let orchestrator =
            Orchestrator::new(OrchestrationLimits::default(), parent_scope()).unwrap();
        let sub = DelegatedSubTask {
            sub_task_id: "worker-1".into(),
            prompt: "look up alpha".into(),
            tools: vec![lookup_definition()],
            authorization: AuthorizationScope::new(["lookup"]).unwrap(),
            limits: SandboxLimits::default(),
            chain: ToolChainConfig {
                max_steps: 1,
                max_tokens_per_step: 256,
                context_reserve: 32,
                keep_recent: 4,
                summary_tokens: 128,
                eviction_policy: EvictionPolicy::DropOldest,
            },
        };
        let plan = DelegationPlan {
            sub_tasks: vec![sub],
        };
        let call = ToolCall {
            name: "lookup".into(),
            arguments: json!({"key":"alpha"}),
        };
        let outcomes = orchestrator
            .run::<FakeDecoder, _>(&plan, |_sub| {
                // First decision: emit a tool call. Second decision: emit
                // the final answer. The SandboxedToolProvider executes the
                // call against `StaticLookup("alpha" -> "value-alpha")`
                // and re-ingests the result into the sub-chain.
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
                let second = final_output("alpha is value-alpha");
                let decoder = FakeDecoder::new(vec![first, second]);
                let provider = lookup_provider(AuthorizationScope::new(["lookup"]).unwrap());
                Ok((decoder, provider))
            })
            .unwrap();
        assert_eq!(outcomes.len(), 1);
        assert_eq!(outcomes[0].status, SubChainStatus::Completed);
        assert_eq!(
            outcomes[0].aggregated_result.content,
            Some(ToolResultContent::Text {
                text: "alpha is value-alpha".into()
            })
        );
    }
}
