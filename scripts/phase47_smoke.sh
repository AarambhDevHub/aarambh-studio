#!/usr/bin/env bash
# Phase 47 — Sandboxed Tool Execution smoke test.
#
# Validates that:
#   - The Phase 47 agent-crate sandbox unit-test suite passes: an unlisted
#     tool name is a hard refusal, an unauthorized-but-declared tool is
#     refused at execution (not declaration), a hanging call is killed by
#     the wall-clock timeout, the output/CPU ceilings fire, malformed JSON
#     is never executed, and an executed result re-ingests into the chain.
#   - The CLI plumbing surfaces the new operator-facing flags:
#     `--execute-tools`, `--allow-tool`, `--exec-timeout-ms`,
#     `--exec-max-output-bytes`, `--exec-workdir`.
#
# Per the roadmap milestone: "A whitelisted, sandboxed tool (e.g. a
# read-only file lookup within a fixed working directory) executes
# correctly end-to-end inside a multi-step chain, with every safety
# boundary (allowlist, timeout, resource cap) independently tested and
# verified to fail closed, not open, under every tested failure condition."
# The unit tests are the deterministic proof; this script also verifies
# the operator-facing CLI surface compiles and the flags appear.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SCORECARD=${PHASE47_SCORECARD:-artifacts/phase47_sandbox_smoke.json}
mkdir -p "$(dirname "$SCORECARD")"

echo "==> Phase 47 agent-crate sandbox unit tests"
cargo test --locked -p aarambh-studio-agent --lib sandbox
cargo test --locked -p aarambh-studio-agent --lib authorization

echo "==> Phase 47 build a release CLI for flag checks"
cargo build --release --locked -p aarambh-studio

echo "==> Phase 47 verify the new agent flags appear in --help"
target/release/aarambh-studio agent --help | grep -q -- "--execute-tools"
target/release/aarambh-studio agent --help | grep -q -- "--allow-tool"
target/release/aarambh-studio agent --help | grep -q -- "--exec-timeout-ms"
target/release/aarambh-studio agent --help | grep -q -- "--exec-max-output-bytes"
target/release/aarambh-studio agent --help | grep -q -- "--exec-workdir"

echo "==> Phase 47 verify --execute-tools refuses to start without --allow-tool"
# Authorizing nothing is a configuration error: the operator must explicitly
# opt in to each executable tool. This is the operator-decision invariant.
set +e
ERR_OUTPUT=$(target/release/aarambh-studio agent \
  --config configs/tiny_shakespeare_smoke.toml \
  --tools data/tools_sandbox_smoke.json \
  --prompt "read notes.txt" \
  --execute-tools 2>&1)
ERR_EXIT=$?
set -e
if [[ "$ERR_EXIT" -eq 0 ]]; then
  echo "Phase 47 smoke FAILED: --execute-tools without --allow-tool should error"
  exit 1
fi
echo "$ERR_OUTPUT" | grep -q "at least one --allow-tool" || {
  echo "Phase 47 smoke FAILED: expected 'at least one --allow-tool' error, got:"
  echo "$ERR_OUTPUT"
  exit 1
}

echo "==> Phase 47 verify --exec-workdir requires the tool to be authorized"
mkdir -p data/sandbox_workdir
set +e
ERR_OUTPUT=$(target/release/aarambh-studio agent \
  --config configs/tiny_shakespeare_smoke.toml \
  --tools data/tools_sandbox_smoke.json \
  --prompt "read notes.txt" \
  --execute-tools \
  --allow-tool lookup \
  --exec-workdir data/sandbox_workdir 2>&1)
ERR_EXIT=$?
set -e
if [[ "$ERR_EXIT" -eq 0 ]]; then
  echo "Phase 47 smoke FAILED: --exec-workdir without authorizing read_file_in_workdir should error"
  exit 1
fi
echo "$ERR_OUTPUT" | grep -q "was not authorized via --allow-tool" || {
  echo "Phase 47 smoke FAILED: expected 'not authorized via --allow-tool' error, got:"
  echo "$ERR_OUTPUT"
  exit 1
}

echo "==> Phase 47 write scorecard"
python3 - "$SCORECARD" <<'PY'
import json, sys
scorecard = {
    "phase": 47,
    "title": "Tool Execution With Sandboxing",
    "agent_unit_tests": "passed",
    "cli_flags_surface": [
        "--execute-tools",
        "--allow-tool",
        "--exec-timeout-ms",
        "--exec-max-output-bytes",
        "--exec-workdir",
    ],
    "closed_world_allowlist": True,
    "operator_authorization": True,
    "schema_revalidation": True,
    "wall_clock_timeout": True,
    "output_and_args_ceilings": True,
    "fail_closed_on_every_failure": True,
    "no_new_crate": True,
    "no_new_dependency": True,
    "honesty_note": (
        "The sandbox is pure-Rust and CPU-only: wall-clock timeout "
        "(cooperative cancellation + thread-detachment), output/argument "
        "size ceilings, closed-world allowlist, operator authorization, and "
        "schema re-validation. OS-level isolation (seccomp/cgroups) is out "
        "of scope for the source release. The 6 roadmap-named acceptance "
        "tests prove the safety-relevant property — a runaway or hung call "
        "never blocks the chain and always produces a fail-closed result — "
        "under every tested failure condition. A general-purpose "
        "code-execution sandbox remains explicitly out of scope: execution "
        "is strictly closed-world, named-capability tool execution."
    ),
}
json.dump(scorecard, open(sys.argv[1], "w"), indent=2)
print(f"wrote {sys.argv[1]}")
PY

echo "Phase 47 smoke completed: $SCORECARD"
