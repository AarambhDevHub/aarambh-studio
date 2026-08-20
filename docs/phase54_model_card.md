# Phase 54 — Model Card & Release Documentation Standard

> **Status:** shipped in `v4.0.0-alpha.14` (Phase 54).
> **Architecture:** [`ARCHITECTURE_V4.md` §68](../ARCHITECTURE_V4.md).
> **Roadmap:** [`ROADMAP_V4.md` Phase 54](../ROADMAP_V4.md).
> **Smoke:** [`scripts/phase54_smoke.sh`](../scripts/phase54_smoke.sh).
> **Canonical smoke card:** `artifacts/phase54_model_card_smoke.md`.

---

## Why this phase exists

Every phase in v1–v4 ships its own unit-level tests and its own section in
`ARCHITECTURE*.md`, `ROADMAP*.md`, and `README.md`. But a released
checkpoint's capabilities, limitations, and provenance were described
piecemeal across those documents — never as a single, canonical artifact per
checkpoint configuration. A reader who wanted to know "what is this
checkpoint, what can it do, what are its limits, and is it safe to use?"
had to assemble the answer from four different files, none of which were
generated from the checkpoint's actual measured behavior.

Phase 54 fixes this with **one canonical, assembled — not hand-written —
document per released checkpoint configuration**: `MODEL_CARD.md`. The card
is generated from an eval-harness run (v2 §17), the red-team report (v4
§67), and a small set of static metadata fields (dataset list, license,
hardware requirements). Because the capabilities and red-team sections are
**pulled** from real runs rather than typed by hand, a model card cannot
silently drift out of sync with a checkpoint's actual measured behavior —
generation fails loudly if no red-team report is present for the checkpoint
being documented.

The seven sections, each pinned to its source so the card is auditable
against the guarantee it is making:

| # | Section | Source |
|---|---|---|
| 1 | Intended Use | static metadata (authored once per release) |
| 2 | Training Data & Licensing | static metadata, license-tagged per entry |
| 3 | Capabilities | **PULLED** from an actual eval-harness scorecard (v2 §17) |
| 4 | Known Limitations | static + eval-derived |
| 5 | Red-Team Summary | **PULLED** from v4 §67's actual red-team report |
| 6 | Hardware Requirements | static metadata |
| 7 | Version & Chat-Template Compatibility | **PULLED** from v4 §66's version tag |

---

## What ships

**No new crate. No new external dependency.** Phase 54 extends the existing
`aarambh-studio-eval` crate (workspace package count stays 21) with one new
module, adds one new CLI flag group, one config file, one JSON schema, and
two documentation files.

### `crates/aarambh-studio-eval/src/model_card.rs`

- **`DatasetEntry`** — `{ name, source_url, license, size_examples, split }`.
  The license tag is mandatory so a downstream reader can audit provenance
  without re-deriving it.
- **`ModelCardMetadata`** — the four static fields (`intended_use`,
  `training_data`, `known_limitations`, `hardware_requirements`), loadable
  from TOML or JSON. This is the **one hand-authored input** to model-card
  generation; everything else is pulled.
- **`ModelCardError`** — five variants naming the concrete loud-failure
  modes: `MissingRedTeamReport`, `RedTeamReportUnreadable`,
  `RedTeamReportNotClean { failed, corpus_size }`, `MissingScorecard`,
  `ScorecardUnreadable`, `MetadataUnreadable`.
- **`ModelCard`** — the seven §68 fields, in spec order:
  ```rust
  ModelCard {
      schema_version: u32,              // = 1
      generated_at_unix_ms: u128,
      intended_use: String,             // static
      training_data: Vec<DatasetEntry>, // static, license-tagged
      capabilities: Scorecard,          // PULLED from v2 §17
      known_limitations: Vec<String>,   // static + eval-derived
      redteam_summary: RedTeamReport,   // PULLED from v4 §67
      hardware_requirements: String,    // static
      chat_template_version: u32,       // PULLED from v4 §66
  }
  ```
  - `ModelCard::assemble(metadata, scorecard, redteam_report,
    chat_template_version)` — the in-memory entry point. Fails loudly if
    `redteam_report.is_clean()` is false.
  - `ModelCard::assemble_from_paths(metadata, scorecard, redteam,
    chat_template_version)` — the file-path entry point the CLI calls.
    Fails loudly if any file is missing or the red-team report is not clean.
  - `to_json()` — the machine-readable companion (validates against
    `schemas/model-card-v1.schema.json`).
  - `to_markdown()` — the canonical `MODEL_CARD.md`, seven sections in spec
    order, with the capabilities and red-team sections pulled verbatim from
    the real `Scorecard`/`RedTeamReport` Markdown renderers.
  - `write(markdown_path)` — writes both the `.md` and the `.json` companion.

### `aarambh-studio` (root CLI binary)

- **`src/cmd/eval_model_card.rs`** — the CLI runner. Resolves the four path
  flags to their defaults, calls `ModelCard::assemble_from_paths`, and writes
  the card. Mirrors `cmd/eval_redteam.rs`'s no-model short-circuit pattern.
- **`src/cmd/eval.rs`** — six new flags on `EvalArgs`:
  - `--generate-model-card` (the short-circuit trigger)
  - `--model-card-metadata <PATH>` (default: `configs/model_card_metadata.toml`)
  - `--model-card-scorecard <PATH>` (default: `artifacts/eval_scorecard.json`)
  - `--model-card-redteam <PATH>` (default: `artifacts/redteam_report.json`)
  - `--model-card-output <PATH>` (default: `MODEL_CARD.md`)
  - `--model-card-chat-template-version <N>` (default: 4, the current v4 tag)

### `configs/model_card_metadata.toml`

The one hand-authored input. Contains `intended_use`, `known_limitations`,
`hardware_requirements` (as bare top-level keys), and `[[training_data]]`
array-of-tables blocks (each with `name`, `source_url`, `license`,
`size_examples`, `split`). SPDX-style license identifiers are preferred.

### `schemas/model-card-v1.schema.json`

A JSON Schema (draft 2020-12) validating the `ModelCard::to_json()` companion.
Mirrors the `manas-forgetting-v1.schema.json` convention: `$id` under
`https://github.com/AarambhDevHub/aarambh-studio/schemas/...`,
`additionalProperties: false`, explicit `required` array.

### `docs/model_card_template.md`

The human-readable template + field guide. Documents each of the seven
sections, which fields are PULLED vs authored, and the fail-loudly
invariants. This is the tracked documentation file (NOT a generated
`MODEL_CARD.md`).

---

## Hard guarantees

1. **Capabilities match the eval-harness run exactly** — the `capabilities`
   field of an assembled card is the exact `Scorecard` produced by an
   eval-harness run, byte-for-byte equal. The Markdown capabilities section
   is the verbatim `Scorecard::to_markdown()` output, never re-rendered by
   hand. (Asserted by the
   `model_card_eval_scores_match_the_actual_eval_harness_run_exactly` test.)
2. **Generation fails loudly if no red-team report is present** — a
   checkpoint cannot get a model card without a clean red-team pass. The
   `assemble` entry point returns `RedTeamReportNotClean` if
   `is_clean()` is false; the `assemble_from_paths` entry point returns
   `RedTeamReportUnreadable` if the file is missing. (Asserted by the
   `model_card_generation_fails_loudly_if_no_redteam_report_is_present`
   test, which exercises both halves.)
3. **Missing metadata or scorecard fails loudly** — `MetadataUnreadable`
   and `ScorecardUnreadable` variants make every missing-input failure
   actionable. (Asserted by the
   `model_card_generation_fails_loudly_if_metadata_file_is_missing` and
   `model_card_generation_fails_loudly_if_scorecard_file_is_missing`
   tests.)
4. **The generated card has all seven §68 sections, in order** — the
   Markdown renderer produces exactly the seven spec sections, and the
   capabilities/red-team sections are the verbatim source renderers.
   (Asserted by the `model_card_markdown_contains_all_seven_sections`,
   `model_card_capabilities_section_matches_scorecard_markdown_verbatim`,
   and `model_card_redteam_section_matches_report_markdown_verbatim`
   tests.)
5. **The JSON companion round-trips** — `to_json()` → `from_str()` produces
   an equal card, so the machine-readable companion is always lossless.
   (Asserted by the `model_card_json_round_trips` test.)
6. **The CLI pass is clean** — `aarambh-studio eval --generate-model-card`
   produces a card with all seven sections populated from real data (a
   synthetic scorecard + the Phase 53 red-team report), and a non-clean
   red-team report exits non-zero.

---

## Usage

```sh
# Assemble a model card from the default paths:
#   - configs/model_card_metadata.toml   (static metadata)
#   - artifacts/eval_scorecard.json       (real eval-harness scorecard)
#   - artifacts/redteam_report.json       (real Phase 53 red-team report)
# Writes MODEL_CARD.md and MODEL_CARD.json.
aarambh-studio eval --generate-model-card

# Override each input path:
aarambh-studio eval --generate-model-card \
  --model-card-metadata configs/model_card_metadata.toml \
  --model-card-scorecard artifacts/eval_scorecard.json \
  --model-card-redteam artifacts/redteam_report.json \
  --model-card-output path/to/MODEL_CARD.md

# Override the chat-template version (defaults to 4, the current v4 tag):
aarambh-studio eval --generate-model-card \
  --model-card-chat-template-version 4 \
  --model-card-output MODEL_CARD.md

# The smoke harness (what CI runs):
scripts/phase54_smoke.sh
```

The static metadata TOML (`configs/model_card_metadata.toml`):

```toml
intended_use = "aarambh-studio v4.0 research checkpoint for ..."

known_limitations = [
  "Not fine-tuned for safety; rely on the safety layer (§13), not the model.",
  "Context window is bounded; long-context degradation is task-dependent (§41).",
  # ...
]

hardware_requirements = "CPU: 16 GB RAM (q4_k_m). GPU: 1x consumer GPU (bf16). ..."

[[training_data]]
name = "wikitext-103"
source_url = "https://huggingface.co/datasets/wikitext"
license = "CC-BY-3.0"
size_examples = 1801350
split = "train"
```

The generated `MODEL_CARD.md` (abridged):

```markdown
# Model Card

- schema version: `1`
- generated at (unix ms): `1724058419123`
- chat-template version: `4`

## Intended Use
aarambh-studio v4.0 research checkpoint for ...

## Training Data & Licensing
| Dataset | Source | License | Examples | Split |
|---|---|---|---:|---|
| wikitext-103 | https://... | CC-BY-3.0 | 1801350 | train |
...

## Capabilities
Pulled directly from the eval-harness scorecard (v2 §17) — never hand-entered.
| Task | Metric | Value | Examples |
|---|---:|---:|---:|
| mmlu | accuracy | 0.7200 | 100 |
...

## Known Limitations
- Not fine-tuned for safety; rely on the safety layer (§13), not the model.
...

## Red-Team Summary
Pulled directly from the Phase 53 red-team report (v4 §67) — never hand-entered.
# Red-Team Report
- corpus size: 24
- passed: 24
- failed: 0
All cases matched their labelled expected outcome.
...

## Hardware Requirements
CPU inference: 16 GB RAM (q4_k_m quantized). GPU inference: 1x consumer GPU (bf16). ...

## Version & Chat-Template Compatibility
Chat-template version: `4` (v4 §66). ...
| Version | Template shape |
|---|---|
| `1` | v1.0.0 base chat format |
| `2` | v2.0.0 + image tokens |
| `3` | v3.0.0 + video / document / tool tokens |
| `4` | v4.0.0 + system role formalized + audio tokens (current) |
```

---

## How it relates to self-learning

A self-learning session (V4 §67 / `SELF_LEARNING_V4.md`) records a checkpoint
id when it persists gradients. The model card is the **release boundary**
artifact: a checkpoint cannot be documented in a model card until its
red-team pass is clean (Phase 53) — enforced by `ModelCard::assemble`'s
`is_clean()` check. The same "measure, don't assume" discipline that has
governed every capability claim since v2 §17's eval harness now governs
every release-adjacent documentation claim: a card assembled today carries
today's eval-harness timestamp and today's red-team report, making any drift
from the checkpoint's actual measured behavior visible rather than hidden
behind hand-entered prose.

---

## Honesty boundary

A model card is an **assembled** document, not a substitute for reading the
eval-harness scorecard and the red-team report in full. The card's
capabilities section is the verbatim scorecard Markdown, and its red-team
section is the verbatim report Markdown — but a reader who needs the full
`outcomes` vector (every case's observed outcome) should consult the
red-team report JSON directly, not the card's summary.

The card does **not** claim the underlying model would refuse a novel
jailbreak it has never seen — that is a model-quality question, measured
separately by the existing eval harness (v2 §17), not by the model card.
The card's red-team section is a **structural** safety gate: it asserts
that the checkpoint's red-team pass is clean (every case matched its
labelled expected outcome), not that the model is adversarially robust in
general.

The static metadata fields (`intended_use`, `training_data`,
`known_limitations`, `hardware_requirements`) are hand-authored and are the
card's one non-pulled portion. They are the release operator's
responsibility and are auditable in `configs/model_card_metadata.toml`.

---

## What this enables next

- **Phase 55 (Final Release, v4.0.0)** treats a missing or non-clean model
  card as a release blocker — the final v4.0.0 release ships with a
  `MODEL_CARD.md` generated from the real v4.0 eval-harness run and the
  real v4.0 red-team report. The release audit (`scripts/phase28_release_audit.sh`)
  may be extended in Phase 55 to assert the card's presence.
- **Future checkpoints** (hypothetical v5+ work, not currently planned)
  would each get their own `MODEL_CARD.md` generated from their own
  scorecard + red-team report + metadata, making cross-checkpoint comparison
  a diff operation rather than a prose-reading exercise.
