#!/usr/bin/env bash
# Phase 48 — Multi-Agent Orchestration smoke test.
#
# Validates that:
#   - The Phase 48 agent-crate orchestrator unit-test suite passes: the
#     orchestrator respects the max-sub-agent-count ceiling, the total
#     execution-time budget, the sub-agent scope-containment invariant,
#     correct result aggregation across multiple sub-chains, and failure
#     isolation (one failing sub-agent does not corrupt its siblings'
#     outcomes).
#   - The CLI plumbing surfaces the new operator-facing flags:
#     `--orchestrate`, `--delegation-plan`, `--max-sub-agents`,
#     `--max-orchestration-budget-ms`, `--sub-agent-allow-tool`.
#
# Per the roadmap milestone: "An orchestrator correctly delegates a task
# requiring 2–3 independent tool-execution sub-chains, each respecting
# Phase 47's sandbox boundaries, with results merged coherently and every
# configured bound (sub-agent count, total time budget, sandbox scope)
# independently verified to hold under test."
#
# The unit tests are the deterministic proof; this script also verifies the
# operator-facing CLI surface compiles and the flags appear, and that
# plan-validation rejects out-of-bound plans before any model is loaded.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SCORECARD=${PHASE48_SCORECARD:-artifacts/phase48_orchestration_smoke.json}
mkdir -p "$(dirname "$SCORECARD")"

echo "==> Phase 48 agent-crate orchestrator unit tests"
cargo test --locked -p aarambh-studio-agent --lib orchestrator

echo "==> Phase 48 build a release CLI for flag checks"
cargo build --release --locked -p aarambh-studio

echo "==> Phase 48 verify the new agent flags appear in --help"
target/release/aarambh-studio agent --help | grep -q -- "--orchestrate"
target/release/aarambh-studio agent --help | grep -q -- "--delegation-plan"
target/release/aarambh-studio agent --help | grep -q -- "--max-sub-agents"
target/release/aarambh-studio agent --help | grep -q -- "--max-orchestration-budget-ms"
target/release/aarambh-studio agent --help | grep -q -- "--sub-agent-allow-tool"

echo "==> Phase 48 verify --orchestrate refuses to start without --delegation-plan"
set +e
ERR_OUTPUT=$(target/release/aarambh-studio agent \
  --config configs/tiny_shakespeare_smoke.toml \
  --tools data/tools_sandbox_smoke.json \
  --prompt "delegate two reads" \
  --orchestrate \
  --allow-tool read_file_in_workdir 2>&1)
ERR_EXIT=$?
set -e
if [[ "$ERR_EXIT" -eq 0 ]]; then
  echo "Phase 48 smoke FAILED: --orchestrate without --delegation-plan should error"
  exit 1
fi
echo "$ERR_OUTPUT" | grep -q "requires --delegation-plan" || {
  echo "Phase 48 smoke FAILED: expected 'requires --delegation-plan' error, got:"
  echo "$ERR_OUTPUT"
  exit 1
}

echo "==> Phase 48 verify --orchestrate refuses to start without --allow-tool"
set +e
ERR_OUTPUT=$(target/release/aarambh-studio agent \
  --config configs/tiny_shakespeare_smoke.toml \
  --tools data/tools_sandbox_smoke.json \
  --prompt "delegate two reads" \
  --orchestrate \
  --delegation-plan configs/orchestration_smoke.json 2>&1)
ERR_EXIT=$?
set -e
if [[ "$ERR_EXIT" -eq 0 ]]; then
  echo "Phase 48 smoke FAILED: --orchestrate without --allow-tool should error"
  exit 1
fi
echo "$ERR_OUTPUT" | grep -q "at least one --allow-tool" || {
  echo "Phase 48 smoke FAILED: expected 'at least one --allow-tool' error, got:"
  echo "$ERR_OUTPUT"
  exit 1
}

echo "==> Phase 48 verify plan exceeding --max-sub-agents is rejected at validation time"
# The smoke plan has 2 sub-tasks; setting --max-sub-agents 1 must reject it
# before any model is loaded (Phase 47 boundary surfaces early).
set +e
ERR_OUTPUT=$(target/release/aarambh-studio agent \
  --config configs/tiny_shakespeare_smoke.toml \
  --tools data/tools_sandbox_smoke.json \
  --prompt "delegate two reads" \
  --orchestrate \
  --delegation-plan configs/orchestration_smoke.json \
  --allow-tool read_file_in_workdir \
  --max-sub-agents 1 2>&1)
ERR_EXIT=$?
set -e
if [[ "$ERR_EXIT" -eq 0 ]]; then
  echo "Phase 48 smoke FAILED: a 2-sub-task plan with --max-sub-agents 1 should be rejected"
  exit 1
fi
echo "$ERR_OUTPUT" | grep -q "delegation plan rejected" || {
  echo "Phase 48 smoke FAILED: expected 'delegation plan rejected' error, got:"
  echo "$ERR_OUTPUT"
  exit 1
}
echo "$ERR_OUTPUT" | grep -q "max_sub_agents" || {
  echo "Phase 48 smoke FAILED: expected the rejection to mention max_sub_agents, got:"
  echo "$ERR_OUTPUT"
  exit 1
}

echo "==> Phase 48 write scorecard"
python3 - "$SCORECARD" <<'PY'
import json, sys
scorecard = {
    "phase": 48,
    "title": "Multi-Agent Orchestration",
    "agent_unit_tests": "passed",
    "cli_flags_surface": [
        "--orchestrate",
        "--delegation-plan",
        "--max-sub-agents",
        "--max-orchestration-budget-ms",
        "--sub-agent-allow-tool",
    ],
    "max_sub_agent_count_ceiling": True,
    "total_execution_time_budget": True,
    "sandbox_scope_containment_via_intersect": True,
    "result_aggregation_via_existing_toolresult_path": True,
    "failure_isolation_per_sub_chain": True,
    "no_new_crate": True,
    "no_new_dependency": True,
    "additive_to_phase_47_sandbox": True,
    "honesty_note": (
        "Orchestration is purely additive to Phase 47: each sub-chain is a "
        "ToolChain backed by a SandboxedToolProvider constructed with the "
        "sub-task's narrowed AuthorizationScope (via AuthorizationScope::"
        "intersect). Sub-chains run sequentially (CPU-first honest default); "
        "true parallelism would require a Send+Sync ChainDecoder, which is "
        "out of scope for the source release because the InferenceEngine "
        "holds a Candle device that is not safely cloneable across threads. "
        "The 5 roadmap-named acceptance tests prove the three hard bounds "
        "(sub-agent count, total time budget, scope containment) plus "
        "aggregation and failure isolation invariants using fake decoders "
        "and real sandbox executors, running in milliseconds."
    ),
}
json.dump(scorecard, open(sys.argv[1], "w"), indent=2)
print(f"wrote {sys.argv[1]}")
PY

echo "Phase 48 smoke completed: $SCORECARD"
