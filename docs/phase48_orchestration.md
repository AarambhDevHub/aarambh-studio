# Phase 48 — Multi-Agent Orchestration

> From first principles. From zero. From Rust.
>
> Phase 48 (`ARCHITECTURE_V4.md` §62) extends the agent crate with a
> top-level **orchestrating reasoning process** that delegates independent
> sub-tasks to multiple parallel sandboxed tool-execution sub-chains
> (each governed entirely by Phase 47's boundaries), then merges their
> results back into its own context.

This is the runbook for the multi-agent orchestration capability shipped
in `v4.0.0-alpha.8`. It documents the design, the operator-facing CLI,
the hard bounds, the failure-isolation guarantee, the smoke workflow, and
the honesty boundary.

---

## Why this phase exists

Phase 47 made tool execution a first-class, opt-in capability of the
runtime itself — but every chain was still a single reasoning process
working through one tool at a time. Some tasks are genuinely parallel:
"read these three independent files and summarise each", "look up these
two unrelated keys", "fetch the configuration and the schema, then merge
the answer". Running them in one chain serialises work that has no
data dependency between the steps.

Phase 48 closes that gap, but **only on top of Phase 47's sandbox**.
Orchestration is only as safe as the execution sandbox underneath it: a
sub-agent inherits every Phase 47 boundary (closed-world allowlist,
operator authorization, schema re-validation, wall-clock timeout, output
ceiling) and adds **three orchestration-specific hard bounds** on top.

This is the second-highest-risk phase before Phase 51 (the public
inference server) and is deliberately placed *after* Phase 47 — never
before.

---

## The orchestration envelope

`Orchestrator::run` enforces the full `ARCHITECTURE_V4.md` §62 pipeline:

```
Operator configures OrchestrationLimits { max_sub_agents, max_total_time_ms }
and the orchestrator's own AuthorizationScope (from --allow-tool).
        │
        ▼
DelegationPlan { sub_tasks: Vec<DelegatedSubTask> } is loaded from JSON
        │
        ▼
Orchestrator::validate_plan checks the three hard bounds BEFORE any
sub-chain runs:
  1. plan.sub_tasks.len() <= max_sub_agents
  2. for each sub-task: parent.intersect(&sub.authorization) == sub.authorization
     (true iff sub.authorization ⊆ parent)
  3. for each sub-task: every declared tool name is_authorized in sub.authorization
  + per-sub-task SandboxLimits and ToolChainConfig validate
        │
        ▼
For each sub-task (sequentially, CPU-first honest default):
  - if elapsed_so_far >= max_total_time_ms: mark BudgetExceeded, skip
  - else build a fresh (ChainDecoder, SandboxedToolProvider) for this
    sub-task via the operator-supplied factory
  - the provider's ToolSandbox is constructed with the sub-task's
    narrowed AuthorizationScope and per-sub-task SandboxLimits
  - run ToolChain::run -> ChainOutput
  - wrap in catch_unwind so panics become SubChainStatus::Failed
        │
        ▼
Each SubChainOutcome.aggregated_result is a fully-formed ToolResult:
  - Completed → status=Ok, content=Text(final_output.text)
  - Failed / BudgetExceeded → status=Error, error=bounded message
  - The aggregated_result.call_id matches the sub_task_id
  - Every outcome is present, never missing — failure isolation
        │
        ▼
Outcomes re-enter the orchestrator's own context via the SAME
ToolResultProvider path v3 §46 / v4 §61 already use — applied
recursively, not a new mechanism.
```

Every bound is enforced as **operator-set configuration**, never as
something the orchestrator's own output can influence. The model cannot
widen its scope by emitting a sub-task with a broader authorization; it
cannot fan out wider than the configured ceiling by emitting a larger
plan; it cannot exceed the time budget by emitting many small sub-agents.

---

## The three hard, non-negotiable bounds

### 1. Maximum sub-agent count

```rust
pub struct OrchestrationLimits {
    pub max_sub_agents: usize,   // default 4, range 1..=64
    pub max_total_time_ms: u64,  // default 30_000
}
```

A `DelegationPlan` with more sub-tasks than `max_sub_agents` is rejected
at `validate_plan` time, before any sub-chain runs. The model cannot
request unbounded fan-out by emitting a larger plan.

The upper bound of 64 matches the per-chain `max_steps` ceiling in
`ToolChainConfig`, so an orchestrator cannot fan out wider than a single
chain could step.

### 2. Maximum total execution time budget

`max_total_time_ms` is the **sum** across all sub-chains, not per
sub-chain. The orchestrator accumulates `Instant::elapsed()` per
sub-chain; once the budget is exhausted, every not-yet-started sub-task
is refused with `SubChainStatus::BudgetExceeded` (and its
`aggregated_result` is a fail-closed `ToolResult` with `status = Error`).

This means many small sub-agents cannot collectively exceed the same
ceiling one large one would hit. A single sub-chain that itself runs
over the budget cannot be interrupted (the `ChainDecoder` trait does not
support cancellation), but no further sub-chains are started once the
budget is hit — the rest are all `BudgetExceeded`.

### 3. Sandbox scope containment

A sub-agent's `AuthorizationScope` may only be a **subset** of its
orchestrator's. Verified at `validate_plan` time by
`parent.intersect(&child) == child` — true iff `child ⊆ parent`.
Additionally, every tool name a sub-task declares must be
`is_authorized` in that sub-task's own scope, so a sub-agent cannot
declare a tool its narrowed scope does not include.

```rust
// In validate_plan:
let narrowed = self.orchestrator_authorization.intersect(&sub.authorization);
if narrowed.allowed() != sub.authorization.allowed() {
    return Err(AgentError::Config(format!(
        "sub-task {:?} authorizes tools outside the orchestrator's scope: ...",
        sub.sub_task_id
    )));
}
for tool in &sub.tools {
    if !sub.authorization.is_authorized(&tool.name) {
        return Err(AgentError::Config(format!(
            "sub-task {:?} declares tool {:?} but the sub-agent's narrowed scope does not authorize it",
            sub.sub_task_id, tool.name
        )));
    }
}
```

Orchestration can never be used as an escalation path to reach tools the
operator did not explicitly enable at the top level.

---

## Failure isolation

> *"One sub-agent's failure or execution error is contained to that
> sub-chain's own result — it does not corrupt or silently swallow
> sibling sub-agents' results, and the orchestrator's aggregation step
> receives an explicit failure marker for that sub-chain rather than a
> missing or malformed entry."*

Each sub-chain runs inside a `std::panic::catch_unwind` boundary (using
`AssertUnwindSafe` over the `&mut` decoder + provider pair, which is
safe because each sub-chain owns its own decoder and provider
independently). Three failure modes are all contained:

| Failure mode | Outcome | Siblings affected? |
|---|---|---|
| `build_decoder` closure returns `Err` | `SubChainStatus::Failed` with the error text | No |
| `ToolChain::run` returns `Err(AgentError)` | `SubChainStatus::Failed` | No |
| Sub-chain panics | `SubChainStatus::Failed` with the panic payload rendered into the error text | No |

The aggregated `ToolResult` for every failure variant is a fail-closed
`ToolResult{status: Error, error: bounded_message}` — never missing,
never malformed. The orchestrator's aggregation step therefore always
sees exactly one outcome per sub-task, in plan order.

---

## Composability: zero chain changes

The key architectural decision is that orchestration is **additive**,
exactly as Phase 47 was. Each sub-chain is a `ToolChain` backed by a
`SandboxedToolProvider` constructed with the sub-task's narrowed
`AuthorizationScope`:

```rust
pub struct Orchestrator {
    limits: OrchestrationLimits,
    orchestrator_authorization: AuthorizationScope,
}

impl Orchestrator {
    pub fn run<D, F>(&self, plan: &DelegationPlan,
                     build_decoder: F) -> AgentResult<Vec<SubChainOutcome>>
    where
        D: ChainDecoder,
        F: FnMut(&DelegatedSubTask)
            -> AgentResult<(D, SandboxedToolProvider)>;
}
```

The `ToolChain` does not know or care that its results came from a
sandboxed executor — the same `ToolResultProvider` path v3 §46 / v4 §61
already use. Sub-chain outputs re-enter the orchestrator's own context
as `ToolResultContent::Text` (or media, for multimodal sub-chains
inherited from Phase 35/36/42) via the **same** `result_ingestion` path,
applied recursively — not a new mechanism.

No file in `aarambh-studio-agent` was modified for correctness. The
only changes were:

- A new file: `crates/aarambh-studio-agent/src/orchestrator.rs`.
- One line added to `lib.rs`: `pub mod orchestrator;` plus the
  re-exports.
- One derive added to `chain.rs::ToolChainConfig`:
  `serde::Serialize, serde::Deserialize` (needed so `DelegatedSubTask`
  can be serialized to/deserialized from JSON; this is strictly
  additive — `ToolChainConfig` was already `Debug + Clone`).

---

## CLI: `agent --orchestrate`

The `agent` command gains five new flags, **all opt-in**:

| Flag | Default | Purpose |
|---|---|---|
| `--orchestrate` | off | Switch from single-chain mode to orchestration mode |
| `--delegation-plan <PATH>` | (none, required with `--orchestrate`) | JSON file describing the `DelegationPlan` |
| `--max-sub-agents N` | 4 | Hard ceiling on sub-agent count |
| `--max-orchestration-budget-ms MS` | 30,000 | Hard ceiling on summed sub-chain wall-clock |
| `--sub-agent-allow-tool <NAME>` | (inherits `--allow-tool`) | Per-sub-agent authorized tool name (repeatable). Sub-agents' scope is `intersect(orchestrator_scope, these names)` — never wider than the orchestrator's. |

When `--orchestrate` is **not** set, the command behaves exactly as in
Phase 47: it reads caller-supplied `ToolResult` JSON lines from stdin
(or a `--results` replay file, or `--execute-tools` sandbox). Orchestration
is strictly opt-in.

Example — delegate two independent file reads inside a fixed workdir:

```sh
target/release/aarambh-studio agent \
  --orchestrate \
  --config configs/tiny_shakespeare_smoke.toml \
  --model checkpoints/tiny_shakespeare_smoke/best/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare_smoke/tokenizer.json \
  --delegation-plan configs/orchestration_smoke.json \
  --allow-tool read_file_in_workdir \
  --sub-agent-allow-tool read_file_in_workdir \
  --exec-workdir ./data/sandbox_workdir \
  --max-sub-agents 4 \
  --max-orchestration-budget-ms 30000 \
  --jsonl
```

The CLI loads the plan, validates it against the three hard bounds
**before any model is loaded** (so a 5-sub-task plan with
`--max-sub-agents 4` errors immediately), then runs every sub-chain
under the bounds and prints one JSONL outcome per sub-task plus a final
`orchestration_metrics` summary.

---

## Delegation plan schema

A `DelegationPlan` is a JSON object with one field, `sub_tasks`, each
entry a `DelegatedSubTask`:

```json
{
  "sub_tasks": [
    {
      "sub_task_id": "reader-a",
      "prompt": "Read the file notes.txt and report its first line.",
      "tools": [
        {
          "name": "read_file_in_workdir",
          "description": "Read a UTF-8 text file from the working directory.",
          "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": false
          }
        }
      ],
      "authorization": {"enabled": ["read_file_in_workdir"]},
      "limits": {
        "timeout_ms": 5000,
        "max_output_bytes": 65536,
        "max_args_bytes": 8192
      },
      "chain": {
        "max_steps": 4,
        "max_tokens_per_step": 256,
        "context_reserve": 32,
        "keep_recent": 4,
        "summary_tokens": 128,
        "eviction_policy": "drop_oldest"
      }
    }
  ]
}
```

The `authorization.enabled` array is the sub-agent's narrowed scope. It
must be a subset of the orchestrator's `--allow-tool` scope; otherwise
`validate_plan` rejects the plan before any sub-chain runs.

---

## Tests

Five roadmap-named acceptance tests live in
`crates/aarambh-studio-agent/src/orchestrator.rs`, each with a real body:

| Test | Proves |
|---|---|
| `orchestrator_cannot_exceed_configured_max_sub_agent_count` | `validate_plan` rejects a plan with more sub-tasks than `max_sub_agents` |
| `orchestrator_cannot_exceed_configured_total_execution_time_budget` | Once the budget is exhausted, subsequent sub-tasks are `BudgetExceeded` without ever calling `generate` |
| `sub_agent_sandbox_scope_is_never_wider_than_orchestrator_authorization` | A sub-task whose `authorization` is not a subset of the orchestrator's is rejected at `validate_plan` time |
| `result_aggregation_correctly_merges_multiple_sub_chain_outputs` | Each `SubChainOutcome::aggregated_result` carries the sub-chain's final text, with `call_id` matching the sub-task id and `status = Ok` |
| `one_sub_agent_failure_does_not_silently_corrupt_sibling_sub_agent_results` | A failing sub-chain produces `Failed` for itself and `Completed` for its siblings |

Supporting tests cover `OrchestrationLimits` validation, the
`intersect == child when subset` invariant, defense-in-depth plan
re-validation in `run`, and an end-to-end
`orchestrator_sub_chain_can_execute_tools_through_sandbox` test that
proves the orchestrator-built provider actually executes a tool call
through the sandbox and re-ingests the result.

---

## Smoke workflow

```sh
scripts/phase48_smoke.sh
```

Runs the agent-crate orchestrator unit tests (the deterministic proof),
verifies `agent --help` surfaces the new flags, verifies
`--orchestrate` errors on missing `--delegation-plan` and missing
`--allow-tool`, verifies a plan exceeding `--max-sub-agents` is rejected
at validation time before any model is loaded, and writes a scorecard
to `artifacts/phase48_orchestration_smoke.json`.

The smoke follows the same honesty discipline as Phase 47: the unit
tests are the deterministic proof with fake decoders and real sandbox
executors; the CLI smoke verifies the operator-facing surface compiles,
the flags appear, and plan validation rejects out-of-bound plans early.
Whether orchestration produces *useful* model behaviour at scale is
measured by the eval harness, not asserted in prose.

---

## Honesty boundary

Phase 48's orchestration is **purely additive** and **CPU-first honest**:

- **Sequential execution.** Sub-chains run sequentially by default. The
  spec's wording — *"Sub-chains run (conceptually parallel; actual
  concurrency bounded by configured limits below)"* — is honored:
  `max_sub_agents` and `max_total_time_ms` together bound the total
  work even when run sequentially. True parallelism would require a
  `ChainDecoder` whose implementor is `Send + Sync`, which is out of
  scope for the source release because the `InferenceEngine` holds a
  Candle device that is not safely cloneable across threads. The CLI's
  per-sub-task decoder factory rebuilds a fresh `InferenceEngine` per
  sub-chain, so each sub-chain owns its own `&mut` decoder.
- **No new crate.** Phase 48 lives entirely in
  `crates/aarambh-studio-agent/src/orchestrator.rs`. No new crate, no
  new dependency.
- **No new ingestion mechanism.** Sub-chain outputs re-enter the
  orchestrator's own context via the **same** `ToolResultProvider` path
  v3 §46 / v4 §61 already use, applied recursively.
- **No execution envelope widening.** Phase 47's sandbox is unchanged:
  same closed-world allowlist, operator authorization, schema
  re-validation, wall-clock timeout, output ceiling. A sub-agent
  inherits every boundary and adds the three orchestration-specific
  hard bounds on top.

**Out of scope:** True parallel sub-chain execution (requires
`Send + Sync` `ChainDecoder`), inter-sub-chain communication (sub-tasks
are independent by design — if they need to share state, that belongs in
the orchestrator's own context, not in the sub-chains), and recursive
orchestration (an orchestrator whose sub-tasks are themselves
orchestrators) — all are documented extension points, not shipped
capabilities.

The safety-relevant property — *every configured bound holds under
test, and a failing sub-agent never corrupts its siblings'* — is
proven by the five roadmap-named acceptance tests plus supporting tests,
all running in milliseconds with fake decoders and real sandbox
executors.

---

## What this enables next

Phase 49 (RAG) is independent of Phase 48 — RAG augments the prompt
before generation; orchestration runs after the prompt is built. But
the two compose naturally: a RAG-augmented orchestrator can delegate
RAG-augmented sub-tasks to sub-chains, and the sub-chains' sandboxed
tool execution can include a retrieval tool (once Phase 49 ships its
`retrieve` executor). Phase 50 (model merging) and Phase 51 (public
inference server) build on the agentic substrate Phases 47 and 48
together established.
