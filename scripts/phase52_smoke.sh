#!/usr/bin/env bash
# Phase 52 — System Role, Chat-Template Versioning, and Context Management
# smoke test.
#
# Validates that:
#   - The six roadmap-named Phase 52 acceptance tests pass:
#       1. session_with_no_system_turn_reproduces_v1_prompt_format_exactly
#       2. chat_template_version_mismatch_fails_server_startup_with_clear_error
#       3. user_message_content_can_never_populate_the_system_turn_position
#       4. sft_loss_mask_correctly_covers_a_leading_system_turn
#       5. context_policy_reject_refuses_rather_than_silently_drops_context
#       6. context_policy_sliding_window_never_evicts_the_system_turn
#   - The supporting module unit tests pass (tokenizer system token + migration
#     + version round-trip, serve system-role mapping, agent Reject + policy
#     mapping, safety two-halves defense).
#   - The CLI surfaces the new `infer --system` flag.
#   - The canonical v4 special-token table includes the ` IMS` system marker at
#     id 17 and the chat-template version constant is 4.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SCORECARD=${PHASE52_SCORECARD:-artifacts/phase52_smoke.json}
mkdir -p "$(dirname "$SCORECARD")"

echo "==> Phase 52 tokenizer crate tests (system token, version, migration)"
cargo test --locked -p aarambh-studio-tokenizer
TOKENIZER_TESTS_OK="passed"

echo "==> Phase 52 inference context-policy tests (Reject + SlidingWindow)"
cargo test --locked -p aarambh-studio-inference --lib context_policy
INFERENCE_TESTS_OK="passed"

echo "==> Phase 52 finetune SFT loss-mask test (leading system turn)"
cargo test --locked -p aarambh-studio-finetune --lib sft::tests::sft_loss_mask_correctly_covers_a_leading_system_turn
FINETUNE_TESTS_OK="passed"

echo "==> Phase 52 serve tests (system-role mapping, version gate)"
cargo test --locked -p aarambh-studio-serve --lib \
  session_with_no_system_turn_reproduces_v1_prompt_format_exactly
cargo test --locked -p aarambh-studio-serve --lib \
  chat_template_version_mismatch_fails_server_startup_with_clear_error
cargo test --locked -p aarambh-studio-serve --lib \
  user_message_content_can_never_populate_the_system_turn_position
SERVE_TESTS_OK="passed"

echo "==> Phase 52 agent tests (Reject + canonical policy mapping)"
cargo test --locked -p aarambh-studio-agent --lib \
  context_pressure_reject_refuses_rather_than_silently_dropping
cargo test --locked -p aarambh-studio-agent --lib \
  eviction_policy_maps_onto_canonical_context_truncation_policy
AGENT_TESTS_OK="passed"

echo "==> Phase 52 safety tests (two-halves system-turn defense)"
cargo test --locked -p aarambh-studio-safety --lib \
  injection_detection_is_the_user_input_side_half_of_system_turn_defense
SAFETY_TESTS_OK="passed"

echo "==> Phase 52 build the CLI for flag checks"
if [[ -n "${PHASE52_RELEASE:-}" ]]; then
  cargo build --release --locked -p aarambh-studio
  BIN=target/release/aarambh-studio
else
  cargo build --locked -p aarambh-studio
  BIN=target/debug/aarambh-studio
fi

echo "==> Phase 52 verify the new infer --system flag appears in --help"
"$BIN" infer --help | grep -q -- "--system"
INFER_HELP_OK="passed"

echo "==> Phase 52 verify the canonical v4 table + version constant"
python3 - <<'PY'
import re, pathlib, sys
src = pathlib.Path("crates/aarambh-studio-tokenizer/src/special.rs").read_text()
assert re.search(r'pub const SYSTEM_ID: u32 = 17;', src), "SYSTEM_ID must be 17"
assert re.search(r'pub const CURRENT_CHAT_TEMPLATE_VERSION: u32 = 4;', src), \
    "CURRENT_CHAT_TEMPLATE_VERSION must be 4"
assert "SYSTEM_SPECIAL_TOKENS" in src, "SYSTEM_SPECIAL_TOKENS table must exist"
print("ok")
PY
CONSTANTS_OK="passed"

cat >"$SCORECARD" <<JSON
{
  "phase": 52,
  "version": "4.0.0-alpha.12",
  "tokenizer_tests": "$TOKENIZER_TESTS_OK",
  "inference_context_policy_tests": "$INFERENCE_TESTS_OK",
  "finetune_sft_loss_mask_test": "$FINETUNE_TESTS_OK",
  "serve_tests": "$SERVE_TESTS_OK",
  "agent_tests": "$AGENT_TESTS_OK",
  "safety_tests": "$SAFETY_TESTS_OK",
  "infer_help_system_flag": "$INFER_HELP_OK",
  "canonical_constants": "$CONSTANTS_OK"
}
JSON

echo "==> Phase 52 scorecard written to $SCORECARD"
cat "$SCORECARD"
