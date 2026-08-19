# Phase 52 — System Role, Chat-Template Versioning, and Context Management

> **Status: shipped in v4.0.0-alpha.12.** All six roadmap acceptance tests pass.
> Commit: `feat: Phase 52 — system role, chat-template versioning, context
> management`.

Phase 52 is a **formalization / retrofit** phase — not a new capability. After
every feature phase in v1–v4 was done, three parts of the model's I/O contract
were still under-specified:

1. The system role token was referenced in the docs but never given a real,
   reserved id, a documented precedence rule, or a chat-template interaction.
2. The chat template changed shape four times (v1 base, v2 image, v3 video /
   document / tool, v4 audio) with no version tag anywhere.
3. Multi-turn context truncation was never documented, and is now a real
   concern given Phase 47–48 long agentic chains and Phase 49 RAG.

This phase closes all three, plus consolidates sampling-defaults guidance.

## 1. System role — ` IMS` at id 17

### The id-7 inconsistency (and the resolution)

`ROADMAP_V4.md` and `ARCHITECTURE_V4.md` §66 both state the system role token
was "reserved at id 7 since v1.0.0". **This was a documentation
inconsistency.** In the actual code (`aarambh-studio-tokenizer/src/special.rs`),
**id 7 is `IMAGE` (`<image>`)** since v2.0.0 — the system token did not exist
at any id.

Reassigning id 7 from `IMAGE` to `system` would break every v2/v3/v4 vision,
video, document, and audio checkpoint catastrophically. The project's invariant
since v2 is **append new special tokens, never reassign an existing id**.

**Resolution:** the system role marker ` IMS` is reserved at **id 17**, the
next free id immediately after `AUDIO_END` (16). This:

- does not change a single existing token id (satisfying "without changing a
  single token id" in spirit and letter),
- follows the exact append pattern used for image → video → document → audio,
- ships an `upgraded_for_system()` migration mirroring `upgraded_for_audio()`
  for upgrading an existing audio checkpoint to system-capable.

The canonical v4.0 special-token table is now `SYSTEM_SPECIAL_TOKENS` (18
tokens, ids 0–17), a strict superset of the Phase 42 `AUDIO_SPECIAL_TOKENS`.

### Documented role

```
 IMS\n{operator-set instructions}\n IMS\n{user turn}\n IMS\n...
```

- **Optional, single-use, leading position.** A session may include at most one
  ` IMS` turn, placed before any ` IMS` (user) turn. Omitting it entirely
  reproduces every prior version's ` IMS... IMS` format exactly — purely
  additive.
- **Loss masking.** `SftTrainer`'s existing rule (`build_loss_mask` masks every
  position before the ` IMS` position) already covers a leading system turn by
  construction — no training-code change was needed, only the documented
  `ChatTemplate::prefix_with_system()` prefix and the test
  `sft_loss_mask_correctly_covers_a_leading_system_turn`.
- **Precedence over user input.** System-turn content is always operator- or
  application-supplied. The serve layer's `assemble_chat_prompt` is the only
  place system turns are created, and it creates them exclusively from
  `role == "system" | "developer"` messages — a user message can only ever
  occupy the ` IMS` position. See §3.

## 2. Chat-template versioning

A `chat_template_version: u32` field is now stored in two places:

- **Tokenizer config** (`BpeTokenizer::chat_template_version: Option<u32>`):
  read from / written to the top-level `chat_template_version` field of
  `tokenizer.json`. `None` = pre-Phase-52 tokenizer that did not declare a
  version (legacy).
- **Checkpoint metadata** (`ModelConfig::chat_template_version: Option<u32>`):
  serde-defaulted, so every existing config TOML and checkpoint loads unchanged.

The version is bumped exactly once per template-shape change in the project's
history:

| version | template shape |
|---|---|
| `1` | v1.0.0 base ` IMS`/` IMS` chat format |
| `2` | v2.0.0 + image tokens |
| `3` | v3.0.0 + video / document / tool tokens |
| `4` | v4.0.0 + system role formalized (` IMS`) + audio tokens |

`CURRENT_CHAT_TEMPLATE_VERSION = 4`.

### Validation

`validate_chat_template_version(declared, expected, compatible)`:

- `Some(v)` where `v == expected` or `compatible.contains(&v)` → `Ok`.
- `Some(v)` otherwise → `Err` (clear mismatch message — never silent).
- `None` → `Ok` (undeclared / pre-Phase-52 legacy; absence is not a mismatch).

The serve server calls this at startup (`run_server` →
`validate_served_chat_template_version`); a self-learning session calls it at
session start (`SelfLearnLoop::from_paths`, `SELF_LEARNING_V4.md` §55). In both
places a mismatch is a clear startup error, never a silent misinterpretation of
prompt structure.

## 3. System-turn precedence — the two-halves defense

The defense against a user masquerading as the system has two halves
(`aarambh-studio-safety/src/input/injection.rs`):

- **User-input side** (`detect_injection`, since v1): flags patterns like
  `"new system prompt:"`, `"ignore previous instructions"`, and
  `"role":"system"` fragments inside user-supplied text. Everything in the
  ` IMS` (user) turn is treated as untrusted.
- **System-turn side** (Phase 52, structural): system-turn content is always
  operator-supplied — never derived from user input. The serve layer's
  `assemble_chat_prompt` creates system turns exclusively from
  `role == "system" | "developer"` messages, so a user message can only ever
  occupy the ` IMS` position.

Neither half alone is sufficient: the user-side detector could miss a novel
phrasing, and the system-side structural rule says nothing about *what* an
operator puts in the system turn. Together they make "user input becomes a
system instruction" a structural impossibility rather than a detection
probability. Pinned by
`user_message_content_can_never_populate_the_system_turn_position`.

## 4. Context-truncation policy

`aarambh-studio-inference/src/context_policy.rs` defines the canonical
`ContextTruncationPolicy`:

```rust
enum ContextTruncationPolicy {
    SlidingWindow, // drop oldest non-system turns first; system turn NEVER evicted
    Summarize,     // replace evicted turns with a generated summary turn
    Reject,        // refuse to proceed rather than silently drop context
}
```

One policy, referenced consistently by every long-context feature:

- The **agent** crate's `EvictionPolicy` maps one-to-one onto it
  (`DropOldest → SlidingWindow`, `Summarise → Summarize`, `Reject → Reject`)
  and now refuses loudly under `Reject` instead of silently evicting.
- **Self-learning** sessions default to `Reject`
  (`SELF_LEARNING_V4.md` §55) — silently dropping context mid-session would
  mean scoring and replaying turns whose verifier no longer has the full
  picture it originally reasoned over.
- **Sandboxed tool execution** (Phase 47) and **orchestration** (Phase 48)
  sessions are the canonical `Reject` use case.

Pinned by `context_policy_reject_refuses_rather_than_silently_drops_context`
and `context_policy_sliding_window_never_evicts_the_system_turn`.

## 5. Sampling defaults

`docs/SAMPLING_DEFAULTS.md` consolidates temperature / top-p / top-k guidance —
previously scattered across three architecture documents — into one canonical
table organized by use case (deterministic tool-call generation, open-ended
chat, creative writing, math/code verification, self-consistency, self-learning
rollout).

## Acceptance tests (all pass)

| Test | Crate |
|---|---|
| `session_with_no_system_turn_reproduces_v1_prompt_format_exactly` | serve |
| `chat_template_version_mismatch_fails_server_startup_with_clear_error` | serve |
| `user_message_content_can_never_populate_the_system_turn_position` | serve |
| `sft_loss_mask_correctly_covers_a_leading_system_turn` | finetune |
| `context_policy_reject_refuses_rather_than_silently_drops_context` | inference |
| `context_policy_sliding_window_never_evicts_the_system_turn` | inference |

Plus 20+ supporting tests across the tokenizer (system token, migration,
version round-trip), serve (system-role mapping, version gate), agent (Reject,
policy mapping), and safety (two-halves defense) crates.

## Files touched

- `crates/aarambh-studio-tokenizer/src/{special,bpe,lib}.rs`
- `crates/aarambh-studio-tokenizer/tests/tokenizer_tests.rs`
- `crates/aarambh-studio-core/src/config.rs`
- `crates/aarambh-studio-inference/src/{context_policy,lib}.rs`
- `crates/aarambh-studio-finetune/src/sft.rs`
- `crates/aarambh-studio-serve/src/server.rs`, `Cargo.toml`
- `crates/aarambh-studio-safety/src/input/injection.rs`
- `crates/aarambh-studio-agent/src/{state,chain}.rs`
- `crates/aarambh-studio-selflearn/src/learning_loop.rs`
- `crates/aarambh-studio-distill/src/{teacher_score,trainer,rollout}.rs`
- `aarambh-studio/src/cmd/{infer,serve}.rs`
- workspace-wide: 28 `ModelConfig` literals + ~15 `BpeTokenizer` test literals
  updated for the new fields.
- `docs/SAMPLING_DEFAULTS.md` (new), `docs/phase52_system_role_context.md` (this
  file), `scripts/phase52_smoke.sh` (new).
