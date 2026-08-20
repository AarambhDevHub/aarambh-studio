# Model Card Template & Field Guide

> **Phase 54** — the canonical, assembled model card for a released
> checkpoint configuration. See [`phase54_model_card.md`](phase54_model_card.md)
> for the runbook and [`ARCHITECTURE_V4.md` §68](../ARCHITECTURE_V4.md) for
> the design spec.

This document is the **human-readable template** for a model card. The
actual `MODEL_CARD.md` for a given checkpoint is **generated**, not
hand-written — see the usage section below.

---

## The seven sections (ARCHITECTURE_V4.md §68)

| # | Section | Source | Authored or pulled? |
|---|---|---|---|
| 1 | Intended Use | `configs/model_card_metadata.toml` → `intended_use` | **Authored** (static, once per release) |
| 2 | Training Data & Licensing | `configs/model_card_metadata.toml` → `[[training_data]]` | **Authored** (static, license-tagged) |
| 3 | Capabilities | `artifacts/eval_scorecard.json` (real eval-harness run, v2 §17) | **PULLED** (never hand-entered) |
| 4 | Known Limitations | `configs/model_card_metadata.toml` → `known_limitations` | **Authored** (static baseline) |
| 5 | Red-Team Summary | `artifacts/redteam_report.json` (real Phase 53 report, v4 §67) | **PULLED** (never hand-entered) |
| 6 | Hardware Requirements | `configs/model_card_metadata.toml` → `hardware_requirements` | **Authored** (static) |
| 7 | Version & Chat-Template Compatibility | `--model-card-chat-template-version` (v4 §66) | **PULLED** (from the version tag) |

---

## Field reference

### `intended_use` (string)

A one-paragraph statement of what the checkpoint is for, who should use it,
and what it is **not** for. Example:

```
aarambh-studio v4.0 research checkpoint for instruction-following,
tool-use, multimodal, and long-context experiments. Intended for research
and evaluation; not for production deployment.
```

### `training_data` (array of `DatasetEntry`)

Each entry records the dataset name, an optional public source URL, the
license (SPDX-style identifier), the approximate number of examples in the
split used, and the split name. The license tag is **mandatory** so a
downstream reader can audit provenance without re-deriving it.

```toml
[[training_data]]
name = "wikitext-103"
source_url = "https://huggingface.co/datasets/wikitext"
license = "CC-BY-3.0"
size_examples = 1801350
split = "train"
```

### `capabilities` (Scorecard, PULLED)

The full eval-harness scorecard (v2 §17), serialized as JSON and rendered
as the verbatim `Scorecard::to_markdown()` table. Never hand-entered — the
card is assembled from a real `artifacts/eval_scorecard.json` produced by
`aarambh-studio eval`.

### `known_limitations` (array of strings)

A bulleted list of known limitations. Static-authored here as the canonical
baseline; eval-derived limitations may be appended at render time in a
future phase, but the authored list is the source of truth today.

### `redteam_summary` (RedTeamReport, PULLED)

The full Phase 53 red-team report (v4 §67), serialized as JSON and rendered
as the verbatim `RedTeamReport::to_markdown()` output. Never hand-entered.
**Generation fails loudly if the report is not clean** — a checkpoint with a
known, unaddressed red-team failure cannot get a model card.

### `hardware_requirements` (string)

A one-paragraph summary of the hardware needed to run the checkpoint. Example:

```
CPU inference: 16 GB RAM (q4_k_m quantized). GPU inference: 1x consumer
GPU (bf16). Training: see ARCHITECTURE_V4.md §69 for the full hardware
matrix across phases.
```

### `chat_template_version` (u32, PULLED)

The chat-template version tag (v4 §66). Defaults to `4` (the current v4.0
tag). The checkpoint's declared version must match (or be explicitly
declared compatible with) this version, or the server refuses to load it to
avoid silently misinterpreting prompt structure.

| Version | Template shape |
|---|---|
| `1` | v1.0.0 base `<imas>`/`</imas>` chat format |
| `2` | v2.0.0 + image tokens |
| `3` | v3.0.0 + video / document / tool tokens |
| `4` | v4.0.0 + system role formalized + audio tokens (current) |

---

## Usage — generating a card

```sh
# From the default paths:
aarambh-studio eval --generate-model-card

# With explicit paths:
aarambh-studio eval --generate-model-card \
  --model-card-metadata configs/model_card_metadata.toml \
  --model-card-scorecard artifacts/eval_scorecard.json \
  --model-card-redteam artifacts/redteam_report.json \
  --model-card-output MODEL_CARD.md
```

The command writes `MODEL_CARD.md` (human-readable) and `MODEL_CARD.json`
(machine-readable companion, validates against
`schemas/model-card-v1.schema.json`).

---

## Fail-loudly invariants

1. **No red-team report** → `RedTeamReportUnreadable` error, non-zero exit.
2. **Red-team report not clean** → `RedTeamReportNotClean { failed, corpus_size }`
   error, non-zero exit.
3. **No scorecard** → `ScorecardUnreadable` error, non-zero exit.
4. **No metadata** → `MetadataUnreadable` error, non-zero exit.

These invariants make the model card a real safety gate rather than
decorative documentation — a checkpoint cannot get a card without a clean
red-team pass, and the capabilities section cannot be hand-entered to
override what the eval harness actually measured.
