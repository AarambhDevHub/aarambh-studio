#!/usr/bin/env bash
# Phase 54 — Model Card & Release Documentation Standard smoke test.
#
# Validates that:
#   - The two roadmap-named Phase 54 acceptance tests pass by name:
#       1. model_card_eval_scores_match_the_actual_eval_harness_run_exactly
#       2. model_card_generation_fails_loudly_if_no_redteam_report_is_present
#   - The supporting model-card unit tests pass (JSON round-trip, Markdown
#     seven-sections, metadata TOML round-trip, write produces md+json,
#     not-clean/metadata/scorecard fail-loudly corollaries).
#   - The CLI model-card tests pass (default paths, explicit overrides,
#     missing-redteam fail-loudly).
#   - The CLI end-to-end `eval --generate-model-card` produces a
#     MODEL_CARD.md with all seven §68 sections populated from real data
#     (a synthetic scorecard + the Phase 53 red-team report).
#   - The CLI surfaces the new `eval --generate-model-card` flag.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SCORECARD=${PHASE54_SCORECARD:-artifacts/phase54_model_card_smoke.json}
METADATA=${PHASE54_METADATA:-configs/model_card_metadata.toml}
REDTEAM=${PHASE54_REDTEAM:-artifacts/redteam_report.json}
MODEL_CARD_OUT=${PHASE54_OUTPUT:-artifacts/phase54_model_card_smoke.md}
SCORECARD_FIXTURE=${PHASE54_SCORECARD_FIXTURE:-artifacts/phase54_scorecard_fixture.json}
mkdir -p "$(dirname "$SCORECARD")" "$(dirname "$MODEL_CARD_OUT")"

echo "==> Phase 54 eval-crate acceptance tests (2 roadmap-named + supporting)"
cargo test --locked -p aarambh-studio-eval --lib model_card
EVAL_TESTS_OK="passed"

echo "==> Phase 54 CLI model-card tests"
cargo test --locked -p aarambh-studio cmd::eval_model_card
CLI_TESTS_OK="passed"

echo "==> Phase 54 build the CLI for the end-to-end run"
if [[ -n "${PHASE54_RELEASE:-}" ]]; then
  cargo build --release --locked -p aarambh-studio
  BIN=target/release/aarambh-studio
else
  cargo build --locked -p aarambh-studio
  BIN=target/debug/aarambh-studio
fi
BUILD_OK="passed"

echo "==> Phase 54 ensure a red-team report exists (run Phase 53 if missing)"
if [[ ! -f "$REDTEAM" ]]; then
  "$BIN" eval --redteam --redteam-report "$REDTEAM"
fi
REDTEAM_OK="present"

echo "==> Phase 54 write a synthetic eval-harness scorecard fixture"
# A minimal but valid Scorecard JSON (schema_version = 2) with three tasks.
# In a real release pipeline this file is produced by `aarambh-studio eval`.
python3 - "$SCORECARD_FIXTURE" <<'PY'
import json, sys
scorecard = {
    "schema_version": 2,
    "model_path": "checkpoints/v4/model.safetensors",
    "tokenizer_path": "checkpoints/v4/tokenizer.json",
    "config_path": "configs/train.toml",
    "tasks": [
        {"name": "mmlu", "metric": "accuracy", "value": 0.72,
         "higher_is_better": True, "examples": 100, "correct": 72,
         "loss": None, "ppl": None, "details": {}},
        {"name": "gsm8k", "metric": "pass@1", "value": 0.55,
         "higher_is_better": True, "examples": 50, "correct": 27,
         "loss": None, "ppl": None, "details": {}},
        {"name": "humaneval", "metric": "pass@1", "value": 0.40,
         "higher_is_better": True, "examples": 40, "correct": 16,
         "loss": None, "ppl": None, "details": {}},
    ],
    "context_len_used": 2048,
    "max_new_tokens": 128,
    "timestamp_unix": 1700000000,
}
with open(sys.argv[1], "w") as f:
    json.dump(scorecard, f, indent=2)
print("scorecard fixture written")
PY
FIXTURE_OK="written"

echo "==> Phase 54 end-to-end: eval --generate-model-card"
"$BIN" eval --generate-model-card \
  --model-card-metadata "$METADATA" \
  --model-card-scorecard "$SCORECARD_FIXTURE" \
  --model-card-redteam "$REDTEAM" \
  --model-card-output "$MODEL_CARD_OUT"
ENDTOEND_OK="passed"

echo "==> Phase 54 verify the generated card has all 7 §68 sections"
python3 - "$MODEL_CARD_OUT" <<'PY'
import sys
md = open(sys.argv[1]).read()
required = [
    "## Intended Use",
    "## Training Data & Licensing",
    "## Capabilities",
    "## Known Limitations",
    "## Red-Team Summary",
    "## Hardware Requirements",
    "## Version & Chat-Template Compatibility",
]
for section in required:
    assert section in md, f"missing section: {section}"
# Capabilities must carry the real scorecard task names, not placeholders.
for task in ["mmlu", "gsm8k", "humaneval"]:
    assert task in md, f"capabilities missing real {task} task"
# The fixture's mmlu accuracy is 0.72 → rendered as 0.7200 (4 dp).
assert "0.7200" in md, "capabilities missing real mmlu score (0.7200)"
# Red-team summary must be the real report (24 cases, clean).
assert "corpus size: 24" in md, "red-team summary missing real corpus size"
assert "All cases matched" in md, "red-team summary not clean"
print("ok")
PY
SECTIONS_OK="passed"

echo "==> Phase 54 verify the JSON companion round-trips"
python3 - "$MODEL_CARD_OUT" <<'PY'
import json, sys
json_path = sys.argv[1].replace(".md", ".json")
card = json.load(open(json_path))
assert card["schema_version"] == 1, f"schema_version != 1: {card['schema_version']}"
assert card["chat_template_version"] == 4, f"chat_template_version != 4"
assert len(card["training_data"]) >= 1, "training_data empty"
assert card["redteam_summary"]["failed"] == 0, "red-team not clean"
assert card["redteam_summary"]["corpus_size"] == 24, "corpus_size != 24"
assert len(card["capabilities"]["tasks"]) == 3, "wrong task count"
print("ok")
PY
JSON_OK="passed"

echo "==> Phase 54 verify the new eval --generate-model-card flag appears in --help"
"$BIN" eval --generate-model-card --help | grep -q -- "--generate-model-card"
HELP_OK="passed"

cat >"$SCORECARD" <<JSON
{
  "phase": 54,
  "version": "4.0.0",
  "eval_crate_model_card_tests": "$EVAL_TESTS_OK",
  "cli_model_card_tests": "$CLI_TESTS_OK",
  "cli_build": "$BUILD_OK",
  "redteam_report": "$REDTEAM_OK",
  "scorecard_fixture": "$FIXTURE_OK",
  "end_to_end_generate_model_card": "$ENDTOEND_OK",
  "all_seven_sections_present": "$SECTIONS_OK",
  "json_companion_round_trips": "$JSON_OK",
  "eval_help_generate_model_card_flag": "$HELP_OK",
  "model_card_path": "$MODEL_CARD_OUT"
}
JSON

echo "==> Phase 54 scorecard written to $SCORECARD"
cat "$SCORECARD"
