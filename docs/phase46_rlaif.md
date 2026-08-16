# Phase 46 — RLAIF (Reinforcement Learning from AI Feedback)

> v4.0.0-alpha.6 · `aarambh-studio-finetune` (`rlaif.rs`, new) + `aarambh-studio` CLI (`finetune rlaif`, new) · depends on v1 §11 (GRPO), v2 §28 (DPO), v1 §12 (self-learning N-completion sampling)

Phase 46 adds a third alignment signal, alongside GRPO (v1 §11, verifier-based) and
DPO (v2 §28, human-preference-based): a frozen judge model scores pairs of
self-sampled completions, automatically generating preference data that feeds
the existing DPO training pipeline **unchanged** — useful for open-ended quality
dimensions where neither a hard verifier nor a static human preference dataset
is available.

## Why this matters

GRPO (v1 §11) needs a hard verifier — it works when correctness is checkable
(math, code, format compliance). DPO (v2 §28) needs a preference dataset —
static pairs of `(chosen, rejected)` completions, whether from a public dataset
or hand-labelled. Neither covers "generate fresh preference signal automatically,
for qualities that are neither checkable nor already labelled" — open-ended chat
quality being the clearest example. RLAIF fills exactly that gap.

RLAIF is deliberately architected as a **data-generation front end**, not a new
training objective — `dpo_loss` (v2 §28) does not change at all. This keeps the
numerically-stable two-class log-softmax formulation v2 §28 already got right,
rather than re-deriving a new loss function with its own numerical edge cases.

## Mechanism

```
For each prompt:
  │
  ▼
Self-sample N candidate completions (reuses v1 §12's N-completion
sampling pattern: seeds base+i, top-k/top-p, no thinking mode)
  │
  ▼
Form all C(N, 2) unordered candidate pairs
  │
  ▼
Judge each pair TWICE, in both A/B and B/A orderings:
  JudgeGenerator::generate_verdict(judge_prompt, max_tokens)
    → {"preferred": "A"|"B"|"tie", "margin": <0.0-1.0>, "reason": "..."}
  Judges have a documented first-position bias; judging both orderings
  is how that bias is detected and corrected.
  │
  ▼
Position-swap bias correction (resolve_preference):
  ├─ Agreement   (both orderings pick the same winner)
  │     → weight 1.0 (or down-weighted by margin if < agreement_margin)
  │     → emit (chosen, rejected) using the agreed winner
  │
  ├─ Tie         (either ordering returned "tie")
  │     → weight 0.0, discarded (no clear preference)
  │
  └─ Disagreement (orderings disagree on the winner)
        ├─ bias_discard = true  → weight 0.0, discarded
        └─ bias_discard = false → weight 0.25 (DISAGREEMENT_WEIGHT),
              emit using the MORE-CONFIDENT ordering's verdict
              (larger margin wins; equal margins → discarded as ambiguous)
  │
  ▼
Output: (chosen, rejected) pairs — the EXACT SAME SCHEMA v2 §28's DPO
pipeline already consumes ({prompt, chosen, rejected} JSONL)
  │
  ▼
Feed directly into the existing, UNMODIFIED `finetune dpo` / `finetune qdpo`
training path (DpoDataset::from_jsonl → DpoTrainer::train_step, unchanged)
```

### The `JudgeGenerator` trait

```rust
// aarambh-studio-finetune/src/rlaif.rs
pub trait JudgeGenerator {
    fn generate_verdict(&mut self, judge_prompt: &str, max_tokens: usize) -> Result<String>;
}
```

Deliberately free of any `aarambh-studio-inference` types so the finetune crate
(Layer 4) does not depend on the inference crate (Layer 5) — the same
architectural boundary Phase 45's `CompletionVerifier` trait established. The
`InferenceEngine` implementation lives in the CLI binary, alongside
`MathVerifierAdapter`.

### The `CandidateSampler` trait

```rust
// aarambh-studio-finetune/src/rlaif.rs
pub trait CandidateSampler {
    fn sample_candidates(&mut self, prompt: &str, n: usize, config: &RlaifConfig)
        -> Result<Vec<String>>;
}
```

Abstracts N-completion sampling (v1 §12 pattern) so RLAIF is testable with a
deterministic fake sampler without a real `InferenceEngine`. The
`InferenceEngine` implementation (in the CLI binary) samples N candidates with
seeds `base + i`, exactly as the self-learning loop's `online_grpo.rs` does for
GRPO grouping.

### The judge prompt template

The default template (`default_judge_template`) asks the judge to compare two
candidates for the same prompt and reply with ONLY valid JSON:

```json
{"preferred": "A" | "B" | "tie", "margin": <float 0.0-1.0>, "reason": "<one sentence>"}
```

`parse_judge_verdict` robustly parses this: malformed JSON, unknown `preferred`
values, or non-finite margins all fall back to a neutral `Tie` with margin
`0.0` — the pair is then discarded downstream rather than trusted at face value,
matching the roadmap's "down-weighted or discarded rather than trusted naively"
discipline.

### Position-swap bias correction

`judge_pair_both_orderings` judges a pair in both `(prompt, a, b)` and
`(prompt, b, a)` orderings, then `resolve_preference` translates each verdict
back into the original frame (in the BA ordering, "A" means `b` wins) and
classifies the result:

- **Agreement** — both orderings pick the same original-frame winner. Weight
  `1.0`, unless both margins are below `agreement_margin` (low-confidence
  agreement), in which case the weight is the margin itself.
- **Tie** — either ordering returned `Tie`. Weight `0.0`, discarded.
- **Disagreement** — orderings disagree. Discarded if `bias_discard`; otherwise
  down-weighted to `DISAGREEMENT_WEIGHT` (0.25) and the more-confident ordering's
  verdict is chosen (equal margins → discarded as genuinely ambiguous).

## CPU/CUDA honesty policy

Everything in `rlaif.rs` is pure Rust over the existing
`InferenceEngine`/`Sampler`/`DpoDataset` surface — zero `unsafe` blocks, zero
CUDA calls. It compiles and is unit-tested on CPU without the `cuda` feature.
The CUDA path is unchanged: when the policy or judge engine is on a CUDA
device, the existing `generate` calls run on GPU automatically; RLAIF adds no
new device-specific code.

## Backward compatibility

RLAIF is purely additive: `dpo_loss` (v2 §28) is byte-for-byte unchanged, the
`DpoDataset`/`DpoTrainer`/`DpoDataLoader` types are unchanged (only the
`DpoTrainer.train_loader` field was widened from private to `pub(crate)` so the
RLAIF integration test in `rlaif.rs` can pull one batch and prove the pairs feed
through the unmodified `train_step`). Existing `finetune dpo`/`qdpo`/`grpo`/`sft`
commands are untouched — `finetune rlaif` is a new subcommand, opt-in only.

## An honest scope constraint

RLAIF is text-only in Phase 46: the judge prompt template and candidate
sampling operate on text completions. Multimodal RLAIF (judging image/video/
document-grounded completions) is future work, not a half-implementation. The
judge is a frozen checkpoint — either the same model at an earlier stage, or
the Large scale judging Small/Tiny outputs, per the roadmap — never a trained
reward model (no reward-model checkpoint ships, consistent with the release
audit forbidding tracked model artifacts).

## Measured, not assumed

Whether an RLAIF-tuned checkpoint's win rate actually improves is an
eval-harness question, answered via v2 §28's existing `preference` eval task
(`eval --task preference`), not assumed from the technique's general reputation.
The fourth acceptance test
(`rlaif_dpo_run_reports_non_negative_win_rate_delta_on_preference_eval_task`)
enforces that the win-rate is *measured* and the delta is *non-negative* — the
honest floor, not a claimed win. A negative delta would be a real regression;
non-negative means RLAIF helped or was neutral. This is the same "measure, don't
assume" discipline every alignment phase since v2 §17 has held.

## An honest hardware constraint

RLAIF judge passes are Kaggle-class inference workloads (judge-model inference
at self-sampling scale), per `SELF_LEARNING_V4.md` §50's hardware-gating table.
The smoke script keeps N=2 on a tiny CPU checkpoint so it runs in well under a
minute; real RLAIF runs at scale are Kaggle-scoped for cost reasons, following
v1 §12's existing i3 self-learning N-completion budget precedent.

## Tests

| Test | Gate |
|---|---|
| `position_swap_disagreement_is_downweighted_not_silently_trusted` | disagreement weight < 1.0 (down-weighted, not trusted); `bias_discard` discards |
| `rlaif_generated_pairs_match_existing_dpo_pair_schema_exactly` | output is `{prompt, chosen, rejected}` and round-trips through `DpoExample` + JSONL |
| `rlaif_preference_pairs_fed_into_unmodified_dpo_pipeline_train_successfully` | generated pairs → `DpoDataset::from_examples` → real `DpoTrainer::train_step` (finite loss) |
| `rlaif_dpo_run_reports_non_negative_win_rate_delta_on_preference_eval_task` | measured win-rate ≥ 0.5 baseline (non-negative delta), not asserted improvement |
| `parse_judge_verdict_parses_valid_json` / `_handles_malformed_json_as_tie` / `_clamps_margin` / `_treats_unknown_preferred_as_tie` | robust judge-verdict parsing |
| `form_pairs_generates_all_combinations` | C(N,2) index pairs, handles n<2 |
| `build_judge_prompt_substitutes_all_placeholders` | prompt-template substitution |
| `agreement_low_margin_is_downweighted` | low-margin agreement down-weighted by margin |
| `disagreement_with_equal_margins_is_discarded` | equal-margin disagreement discarded as ambiguous |
| `tie_pairs_are_discarded` | tie verdicts produce weight 0.0 |
| `rlaif_config_rejects_fewer_than_two_candidates` | config validation |
| `read_prompts_jsonl_round_trips` / `_rejects_empty_file` | prompts JSONL I/O |

The four roadmap-named tests are the Phase 46 acceptance tests; the rest are
the supporting CPU unit tests that exercise the new code paths without CUDA
hardware or a real model checkpoint.

## Configs

- `configs/rlaif_smoke.toml` — CPU smoke training config (tiny Shakespeare,
  8 steps) that produces a checkpoint the smoke script runs RLAIF against
  (policy == judge, self-judging). The RLAIF surface is exercised via the
  `finetune rlaif` subcommand (not a TOML section), per the roadmap's
  explicit CLI-first scope.

## Smoke script

`scripts/phase46_smoke.sh` runs the `rlaif` finetune-crate unit tests, trains a
tiny checkpoint on `rlaif_smoke.toml`, writes a small `prompts.jsonl` fixture,
runs `finetune rlaif --n-candidates 2` end-to-end on CPU to generate a
preference-pair JSONL, verifies the generated JSONL is valid DPO schema
(`{prompt, chosen, rejected}`), feeds it into the unmodified `finetune dpo`
pipeline (1 step, reference-free), verifies the new flags appear in
`finetune rlaif --help` and `finetune --help`, and writes a scorecard to
`artifacts/phase46_rlaif_smoke.json`.

## Milestone

RLAIF-generated preference pairs, fed through the existing (unmodified)
`finetune dpo` pipeline, produce a checkpoint whose held-out preference win-rate
(v2 §28's eval task) is reported against the pre-RLAIF baseline — an honest
delta, not a claimed win, consistent with every other "measure, don't assume"
phase since v2 §17.

```
git commit -m "feat: Phase 46 — RLAIF"
git tag v4.0.0-alpha.6
```
