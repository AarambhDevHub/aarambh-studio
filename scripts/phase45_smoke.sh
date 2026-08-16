#!/usr/bin/env bash
# Phase 45 — Test-Time Compute Scaling smoke test.
#
# Validates that:
#   - The Phase 45 inference-crate unit-test suite passes: N=1 reproduces
#     single-sample generation exactly, self-consistency majority-vote picks
#     the most common final answer, the heuristic process-reward scorer
#     correlates positively with the verifier on a labelled holdout, plus
#     the supporting selection-strategy and re-seed tests.
#   - The Phase 45 eval-crate unit-test suite passes: the scorecard
#     measurement plumbing carries single-sample and best-of-N accuracy in
#     its details map without asserting the delta improved.
#   - The CLI plumbing works end-to-end on CPU: a tiny trained checkpoint,
#     `infer --best-of-n 2 --selection self-consistency` produces a chosen
#     completion, and `infer --help` / `eval --help` list the new flags.
#
# i3 supports small N (2–4) for text tasks; larger N is Kaggle-scoped for
# cost reasons, per the milestone. This smoke keeps N=2 so it runs on CPU
# in well under a minute. Real accuracy deltas are reported only via the
# eval-harness scorecard, never asserted in prose.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SCORECARD=${PHASE45_SCORECARD:-artifacts/phase45_test_time_smoke.json}
mkdir -p "$(dirname "$SCORECARD")"

echo "==> Phase 45 best-of-N inference unit tests"
cargo test --locked -p aarambh-studio-inference --lib best_of_n
cargo test --locked -p aarambh-studio-inference --lib self_consistency
cargo test --locked -p aarambh-studio-inference --lib process_reward

echo "==> Phase 45 best-of-N eval-harness unit tests"
cargo test --locked -p aarambh-studio-eval --lib generation
cargo test --locked -p aarambh-studio-eval --lib tasks::gsm8k_subset

echo "==> Phase 45 ensure a tiny training fixture exists"
if [[ ! -f data/tiny_shakespeare.txt ]]; then
  mkdir -p data
  python3 - <<'PY'
from pathlib import Path
snippet = (
    "To be, or not to be, that is the question: "
    "whether tis nobler in the mind to suffer "
    "the slings and arrows of outrageous fortune, "
    "or to take arms against a sea of troubles "
    "and by opposing end them. "
)
text = (snippet * 400)
Path("data/tiny_shakespeare.txt").write_text(text)
print(f"wrote data/tiny_shakespeare.txt ({len(text)} bytes)")
PY
fi

echo "==> Phase 45 train a tiny checkpoint for best-of-N inference"
cargo run --quiet --locked -p aarambh-studio -- train \
  --config configs/best_of_n_smoke.toml

echo "==> Phase 45 best-of-N inference smoke (N=2, self-consistency)"
BEST_OF_N_OUTPUT=$(cargo run --quiet --locked -p aarambh-studio -- infer \
  --config configs/best_of_n_smoke.toml \
  --prompt "To be, or not to be" \
  --max-tokens 16 \
  --temperature 0.8 \
  --top-k 50 \
  --top-p 0.9 \
  --seed 42 \
  --best-of-n 2 \
  --selection self-consistency 2>&1) || {
    echo "$BEST_OF_N_OUTPUT"
    echo "Phase 45 best-of-N inference smoke FAILED"
    exit 1
  }
echo "$BEST_OF_N_OUTPUT" | head -5

echo "==> Phase 45 CLI --help surfaces the new flags"
cargo run --quiet --locked -p aarambh-studio -- infer --help | grep -q -- "--best-of-n"
cargo run --quiet --locked -p aarambh-studio -- infer --help | grep -q -- "--selection"
cargo run --quiet --locked -p aarambh-studio -- infer --help | grep -q -- "--ground-truth"
cargo run --quiet --locked -p aarambh-studio -- eval --help | grep -q -- "--best-of-n"

echo "==> Phase 45 write scorecard"
python3 - "$SCORECARD" <<'PY'
import json, sys
scorecard = {
    "phase": 45,
    "title": "Test-Time Compute Scaling",
    "smoke_n": 2,
    "smoke_selection": "self-consistency",
    "smoke_seed": 42,
    "cpu_fallback": True,
    "inference_unit_tests": "passed",
    "eval_unit_tests": "passed",
    "cli_help_surfaces_flags": True,
    "honesty_note": (
        "i3 supports small N (2-4) for text tasks; larger N is Kaggle-scoped "
        "for cost reasons. Whether best-of-N improves accuracy on a given "
        "task is measured by the eval-harness scorecard, not asserted here."
    ),
}
json.dump(scorecard, open(sys.argv[1], "w"), indent=2)
print(f"wrote {sys.argv[1]}")
PY

echo "Phase 45 smoke completed: $SCORECARD"
