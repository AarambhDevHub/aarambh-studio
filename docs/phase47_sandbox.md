# Phase 47 — Sandboxed Tool Execution

> From first principles. From zero. From Rust.
>
> Phase 47 (`ARCHITECTURE_V4.md` §61) closes the boundary v2 §30 opened
> (tool calls are emitted, never executed) and v3 §46 extended (multi-step
> chains, still emit-only): the model's tool calls can now be **actually
> executed** by aarambh-studio itself, but only inside a strict, explicit,
> closed-world sandbox.

This is the runbook for the sandboxed-execution capability shipped in
`v4.0.0-alpha.7`. It documents the design, the operator-facing CLI, the
reference executors, the smoke workflow, and the honesty boundary.

---

## Why this phase exists

Before Phase 47, aarambh-studio's tool-use story was **emit-only**:

- **v2 §30** — the model emits grammar-constrained JSON tool calls. A
  developer's own integration reads those calls and decides whether to
  execute them. aarambh-studio never executes anything.
- **v3 §46** — the model emits *sequences* of tool calls, using real
  intermediate results fed back by the caller. Still emit-only — the caller
  (a human, a script, an external service) does the execution.

Phase 47 makes execution a first-class, opt-in capability of the runtime
itself — but scoped as narrowly as the risk demands. There is **no generic
"run a shell command" or "eval this code" executor** anywhere in the crate,
by design. Every executor implements one specific, named capability, and an
unrecognised tool name is a hard refusal, never a best-effort fallback
attempt.

This is the highest-risk phase before Phase 51 (the public inference
server) and is deliberately placed before Phase 48 (multi-agent
orchestration), which depends on it completely — orchestration is only as
safe as the execution sandbox underneath it.

---

## The execution envelope

`ToolSandbox::execute` enforces the full `ARCHITECTURE_V4.md` §61 pipeline,
in order:

```
Model emits grammar-constrained JSON tool call (v2 §30)
        │
        ▼
1. Closed-world allowlist — the name must match a REGISTERED executor.
   Not registered → ExecError::UnknownTool (hard refusal, no attempt)
        │
        ▼
2. Operator authorization — the name must be in the operator's
   AuthorizationScope (set at startup via --allow-tool).
   Not authorized → ExecError::Unauthorized (distinct from UnknownTool:
   the capability exists, the operator simply did not enable it)
        │
        ▼
3. Argument-size ceiling — serialized args ≤ max_args_bytes (default 8 KiB).
   Exceeded → ExecError::ResourceLimitExceeded { kind: ArgumentBytes }
        │
        ▼
4. Schema re-validation — args re-validated against the declared JSON
   Schema (defense-in-depth on top of the grammar-constrained decoder).
   Invalid → ExecError::InvalidArgs (malformed calls are NEVER executed)
        │
        ▼
5. Bounded envelope — worker thread + recv_timeout wall-clock ceiling
   (default 5 s). Cooperative cancellation via an AtomicBool flag.
   Exceeded → ExecError::Timeout (thread detached, fail-closed)
        │
        ▼
6. Output-size ceiling — result payload ≤ max_output_bytes (default 64 KiB).
   Exceeded → ExecError::ResourceLimitExceeded { kind: OutputBytes }
        │
        ▼
ToolResult re-enters the chain via v3 §46's existing result_ingestion
path — no new ingestion mechanism, execution is purely additive.
```

Every failure yields a **fail-closed** `ToolResult{status:Error,
error:...}`. The chain records the refusal and continues — it never
silently drops a call or attempts a best-effort fallback.

---

## Composability: zero chain changes

The key architectural decision is that execution is **additive**.
`SandboxedToolProvider` implements the existing `ToolResultProvider` trait:

```rust
pub struct SandboxedToolProvider { sandbox: ToolSandbox }

impl ToolResultProvider for SandboxedToolProvider {
    fn next_result(&mut self, request: &ToolResultRequest)
        -> AgentResult<ToolResult>
    {
        let result = self.sandbox.execute(&request.call, &request.call_id);
        result.validate_for(&request.call_id)?;  // belt-and-braces
        Ok(result)
    }
}
```

The existing `ToolChain::run_with_callback` calls `next_result` — it does
not know or care whether the result came from a caller's stdin line, a
scripted replay file, or a sandboxed executor. The chain, the context
eviction, the multimodal result lifetime, and the safety checks are all
unchanged. Phase 47 adds a *provider*; it does not touch the chain.

---

## Operator authorization

Authorization is an **operator** decision, not a model decision. A model
can *declare* any tool in its schema and *request* execution of anything
it declares. Whether that request is ever carried out depends entirely on
what the operator explicitly enabled at startup:

```rust
pub struct AuthorizationScope { enabled: BTreeSet<String> }

impl AuthorizationScope {
    pub fn empty() -> Self;
    pub fn enable(&mut self, name: &str) -> Result<(), AgentError>;
    pub fn is_authorized(&self, name: &str) -> bool;
    pub fn intersect(&self, other: &Self) -> Self;  // Phase 48 sub-agents
}
```

`AuthorizationScope::intersect` supports Phase 48's multi-agent
orchestration: a sub-agent's authorized scope can only be a **subset** of
its orchestrator's. Orchestration can never escalate tool access beyond
what the operator enabled at the top level.

---

## Reference executors

Phase 47 ships two `ToolExecutor` implementations:

### `ReadFileInWorkdir` — the milestone executor

> "A read-only file lookup within a fixed working directory."

```rust
pub struct ReadFileInWorkdir { workdir: PathBuf }

impl ToolExecutor for ReadFileInWorkdir {
    fn name(&self) -> &'static str { "read_file_in_workdir" }
    fn execute(&self, args: &ValidatedArgs, ctx: &ExecContext)
        -> Result<ToolResultContent, ExecError>;
}
```

- Reads a UTF-8 text file by **relative** name from a fixed, canonicalized
  working directory.
- **Refuses** absolute paths and any `..` traversal component *before*
  joining, so a missing file cannot be probed for path-structure
  information and an absolute path cannot replace the workdir base.
- After canonicalization, double-checks the resolved path still starts with
  the workdir (defense-in-depth against symlink escapes).
- Caps the returned bytes at `max_output_bytes`.
- No network access, no write access — by construction.

### `StaticLookup` — deterministic in-memory executor

An in-memory `key → text` map, used by tests and smoke runs that need a
closed-world capability without touching the filesystem.

### Extension points (documented, not shipped)

The roadmap names `http_get_whitelisted_domain` as an example executor. It
is **not** shipped because adding an HTTP client dependency would widen the
attack surface and the dependency footprint for a source release; the
`ToolExecutor` trait is the extension point for an operator who needs it.
The trait is intentionally `Send + Sync` so executors can run on worker
threads under the timeout envelope.

---

## CLI: `agent --execute-tools`

The `agent` command gains five flags:

| Flag | Default | Purpose |
|---|---|---|
| `--execute-tools` | off | Switch from caller-executed stdin/replay results to sandboxed execution |
| `--allow-tool <NAME>` | (none) | Operator authorization (repeatable). At least one is required with `--execute-tools`. |
| `--exec-timeout-ms` | 5000 | Per-call wall-clock ceiling |
| `--exec-max-output-bytes` | 65536 | Maximum output payload bytes |
| `--exec-workdir <DIR>` | (none) | Binds the `read_file_in_workdir` executor to a directory |

Example — execute a read-only file lookup inside a fixed workdir:

```sh
target/release/aarambh-studio agent \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_shakespeare_smoke/best/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare_smoke/tokenizer.json \
  --tools data/tools_sandbox_smoke.json \
  --prompt "Read the file notes.txt and summarise it." \
  --execute-tools \
  --allow-tool read_file_in_workdir \
  --exec-workdir ./data/sandbox_workdir \
  --exec-timeout-ms 2000 \
  --max-steps 4
```

When `--execute-tools` is **not** set, the command behaves exactly as in
v3 §46: it reads caller-supplied `ToolResult` JSON lines from stdin (or a
`--results` replay file). Execution is strictly opt-in.

---

## Tool definition schema

Tools are declared with the same JSON Schema format every other phase uses.
The `read_file_in_workdir` executor expects a `path` string:

```json
{
  "name": "read_file_in_workdir",
  "description": "Read a UTF-8 text file from the working directory.",
  "parameters": {
    "type": "object",
    "properties": {
      "path": {"type": "string"}
    },
    "required": ["path"],
    "additionalProperties": false
  }
}
```

The sandbox compiles every declared schema at registration time and
re-validates arguments before execution — defense-in-depth on top of the
grammar-constrained decoder that already produced a schema-valid call.

---

## Tests

Six roadmap-named acceptance tests live in
`crates/aarambh-studio-agent/src/sandbox.rs`, each with a real body:

| Test | Proves |
|---|---|
| `unlisted_tool_name_is_hard_refused_never_attempted` | An unregistered name is a hard refusal, no execution attempt |
| `unauthorized_but_declared_tool_is_refused_at_execution_not_declaration` | Authorization is distinct from registration; an authorized copy succeeds |
| `execution_timeout_kills_a_hanging_tool_call` | A hanging executor is abandoned after `timeout_ms`, fail-closed, promptly |
| `execution_respects_configured_memory_and_cpu_ceiling` | Output-size and wall-clock (CPU) ceilings both fire as `ResourceLimitExceeded`/`Timeout` |
| `malformed_tool_call_json_is_never_executed` | Schema-invalid args never reach the executor (verified by a counting executor) |
| `execution_result_re_ingests_correctly_into_the_next_chain_step` | A full `ToolChain` + `FakeDecoder` + `SandboxedToolProvider` re-ingests the executed result into the chain state |

Supporting tests cover authorization `intersect`, read-file traversal
refusal, and `SandboxLimits` validation.

---

## Smoke workflow

```sh
scripts/phase47_smoke.sh
```

Runs the agent-crate sandbox unit tests (the deterministic proof), verifies
`agent --help` surfaces the new flags, and writes a scorecard to
`artifacts/phase47_sandbox_smoke.json`.

The smoke follows the same honesty discipline as Phase 46: the unit tests
are the deterministic proof with fake decoders and real executors; the CLI
smoke verifies the operator-facing surface compiles and the flags appear.
Whether sandboxed execution produces *useful* model behaviour at scale is
measured by the eval harness, not asserted in prose.

---

## Honesty boundary

Phase 47's sandbox is **pure-Rust and CPU-only**:

- **Wall-clock timeout** — cooperative cancellation via an `AtomicBool`
  flag, plus thread-abandonment on timeout. Safe Rust cannot force-kill a
  thread, so a non-cooperative executor that never polls `is_cancelled` is
  *detached* on timeout — it cannot block the chain, but it may continue
  running until it returns on its own. Every shipped executor polls the
  flag inside its loops.
- **Output-size ceiling** — enforced by `ToolSandbox` after the executor
  returns, defense-in-depth on top of the executor's own cap.
- **Argument-size ceiling** — enforced before schema validation.
- **Closed-world allowlist** — only registered executors run.
- **Operator authorization** — only `--allow-tool` names run.
- **Schema re-validation** — malformed calls never execute.

**Out of scope:** OS-level isolation (seccomp, cgroups, namespaces) is not
implemented, consistent with the project's CPU-first, source-release
posture. A general-purpose code-execution sandbox remains explicitly out
of scope — Phase 47's execution is strictly closed-world, named-capability
tool execution, never arbitrary code or shell execution.

The safety-relevant property — *a runaway or hung call never blocks the
chain and always produces a fail-closed result* — holds under every tested
failure condition (unknown tool, unauthorized tool, schema-invalid args,
hanging executor, oversized output, oversized args).

---

## What this enables next

Phase 48 (multi-agent orchestration) builds directly on this: an
orchestrator delegates sub-tasks to multiple parallel sandboxed
tool-execution chains, each governed entirely by Phase 47's boundaries,
with each sub-agent's authorized scope narrowed via
`AuthorizationScope::intersect` to a subset of the orchestrator's.
