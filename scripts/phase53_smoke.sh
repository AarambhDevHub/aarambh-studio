#!/usr/bin/env bash
# Phase 53 — Red-Team / Adversarial Safety Evaluation smoke test.
#
# Validates that:
#   - The three roadmap-named Phase 53 acceptance tests pass by name:
#       1. every_redteam_case_has_a_labelled_expected_outcome
#       2. a_failing_redteam_case_is_surfaced_in_the_report_not_silently_dropped
#       3. redteam_corpus_sources_are_documented_and_free_public_only
#   - The supporting red-team unit tests pass (corpus size + uniqueness,
#     SafetyLayerTarget verdict mapping, report JSON round-trip + Markdown
#     failures-first, probe-error-as-Other-not-dropped).
#   - The CLI composite target drives all four v4.0 surfaces end-to-end and
#     every case matches its labelled expected outcome (zero failures).
#   - The CLI surfaces the new `eval --redteam` flag.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SCORECARD=${PHASE53_SCORECARD:-artifacts/phase53_redteam_smoke.json}
REPORT=${PHASE53_REPORT:-artifacts/phase53_redteam_report.json}
mkdir -p "$(dirname "$SCORECARD")"

echo "==> Phase 53 safety crate acceptance tests (3 roadmap-named + supporting)"
cargo test --locked -p aarambh-studio-safety --lib redteam
SAFETY_TESTS_OK="passed"

echo "==> Phase 53 CLI composite-target tests (all four surfaces)"
cargo test --locked -p aarambh-studio cmd::eval_redteam
CLI_TESTS_OK="passed"

echo "==> Phase 53 build the CLI for the end-to-end run"
if [[ -n "${PHASE53_RELEASE:-}" ]]; then
  cargo build --release --locked -p aarambh-studio
  BIN=target/release/aarambh-studio
else
  cargo build --locked -p aarambh-studio
  BIN=target/debug/aarambh-studio
fi

echo "==> Phase 53 end-to-end: eval --redteam against the v4.0 candidate build"
"$BIN" eval --redteam --redteam-report "$REPORT"
ENDTOEND_OK="passed"

echo "==> Phase 53 verify the report is clean (failed == 0, corpus_size == 24)"
python3 - "$REPORT" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
assert report["failed"] == 0, f"red-team report has failures: {report['failed']}"
assert report["corpus_size"] == 24, f"corpus_size != 24: {report['corpus_size']}"
assert report["passed"] == 24, f"passed != 24: {report['passed']}"
assert report["schema_version"] == 1, f"schema_version != 1: {report['schema_version']}"
assert len(report["outcomes"]) == 24, f"outcomes len != 24: {len(report['outcomes'])}"
print("ok")
PY
REPORT_OK="passed"

echo "==> Phase 53 verify the new eval --redteam flag appears in --help"
"$BIN" eval --redteam --help | grep -q -- "--redteam"
HELP_OK="passed"

cat >"$SCORECARD" <<JSON
{
  "phase": 53,
  "version": "4.0.0-alpha.13",
  "safety_crate_tests": "$SAFETY_TESTS_OK",
  "cli_composite_target_tests": "$CLI_TESTS_OK",
  "end_to_end_eval_redteam": "$ENDTOEND_OK",
  "report_clean": "$REPORT_OK",
  "eval_help_redteam_flag": "$HELP_OK",
  "report_path": "$REPORT"
}
JSON

echo "==> Phase 53 scorecard written to $SCORECARD"
cat "$SCORECARD"
