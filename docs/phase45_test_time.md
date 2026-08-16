# Phase 45 — Test-Time Compute Scaling

> v4.0.0-alpha.5 · `aarambh-studio-inference` (`best_of_n.rs`, `self_consistency.rs`, `process_reward.rs`, new) + `aarambh-studio-eval` (`generation.rs`, `harness.rs`, extended) · depends on v1 §7 (thinking engine), v2 §29 (speculative decoding), v1 §11 / v2 §22 (verifiers)

Phase 45 adds a genuinely new inference-time axis, distinct from the
thinking engine (v1 §7): instead of controlling *how many tokens* a
single generation spends reasoning, this phase generates *multiple
candidate completions* and selects among them — the Best-of-N /
self-consistency / verifier-guided-selection pattern that sits alongside,
not inside, the existing thinking-mode budget system.

## Why this matters

The thinking engine (v1 §7) controls the *depth* of one generation's
reasoning (None/Low/Medium/High/Max — v3 §48 added Max). Test-time compute
scaling controls the *breadth*: how many independent generations are
produced and how the best one is chosen. The two compose freely — each of
the N candidates can itself use any thinking mode. This is the axis the
2026 generation of frontier models use to trade extra inference compute
for accuracy on hard verifiable tasks (math, code) without retraining.

## Mechanism

```
Prompt
  │
  ▼
Generate N candidates in parallel:
  - InferenceEngine::prepare_session prefills the prompt KV-cache once
  - GenerationSession::fork_with_config clones the cache N times
  - Each fork gets an independent sampler; candidate 0 keeps the input
    seed (N=1 reproduces single-sample byte-for-byte), candidates 1..N
    are re-seeded base_seed + i so they diverge
  - InferenceEngine::decode_sessions advances all pending forks in one
    batched target forward pass (the existing v2 §29 batched-decode path)
  │
  ▼
SelectionStrategy selects one candidate:
  │
  ├─ Verifier        → CompletionVerifier.verify(candidate, ground_truth)
  │                    picks the highest-scoring candidate (ties: first)
  │
  ├─ SelfConsistency → extract each candidate's final answer
  │                    (number or last line), majority-vote
  │
  ├─ Majority        → majority-vote on raw completion strings
  │
  └─ ProcessReward   → ProcessRewardScorer.score(prompt, candidate)
                       picks the highest-scoring reasoning trace
```

### The `SelectionStrategy` enum

```rust
// aarambh-studio-inference/src/best_of_n.rs
pub enum SelectionStrategy {
    Verifier,
    SelfConsistency,
    Majority,
    ProcessReward,
}
```

`Verifier` and `SelfConsistency` are the two roadmap-named strategies for
verifiable tasks (math/code); `Majority` is the no-extraction baseline;
`ProcessReward` is the open-ended-task fallback from ARCHITECTURE_V4 §59
(used when neither a hard verifier nor a clean final-answer extraction
exists).

### The `BestOfNEngine` wrapper

```rust
// aarambh-studio-inference/src/best_of_n.rs
pub struct BestOfNEngine {
    target: InferenceEngine,
    config: BestOfNConfig,
}

impl BestOfNEngine {
    pub fn generate(&mut self, prompt: &str, config: GenerationConfig)
        -> Result<BestOfNOutput>;
}

pub struct BestOfNOutput {
    pub chosen: GenerationOutput,
    pub chosen_index: usize,
    pub candidates: Vec<GenerationOutput>,
    pub selection: SelectionStrategy,
    pub rationale: SelectionRationale,
}
```

This mirrors the wrapper-struct pattern `MtpSpeculativeEngine` and
`SpeculativeEngine` already use: the target engine is owned and reused for
prompt prefill + batched decode, and `GenerationConfig` is left untouched
so the `serve` crate's `GenerationRequest` (which wraps `GenerationConfig`)
is unchanged — best-of-N is a CLI/eval surface only, per the roadmap's
explicit scope.

### The local `CompletionVerifier` trait

```rust
// aarambh-studio-inference/src/best_of_n.rs
pub trait CompletionVerifier: Send + Sync {
    fn extract_answer(&self, completion: &str) -> Option<String>;
    fn verify(&self, completion: &str, ground_truth: &str) -> f32;
}
```

This trait is local to the inference crate (which is architecturally
lower-level than the finetune crate that owns `Verifier` /
`MathVerifier` / `CodeVerifier`). The CLI binary provides a thin
`MathVerifierAdapter` that wraps `aarambh_studio_finetune::MathVerifier`
into `CompletionVerifier` at the call site, so the inference crate never
depends on the finetune crate — the same layering discipline the rest of
the workspace already follows.

### Self-consistency answer extraction

`self_consistency.rs` re-declares `extract_final_number` byte-identically
to `aarambh_studio_finetune::extract_final_number` (with an attribution
doc-comment) so the inference crate does not pull the finetune crate into
its dependency graph. `extract_final_answer` extends this to non-numeric
completions by returning the last non-empty trimmed line — a reasonable
fallback for code-completion or short-answer tasks.

### The process-reward scorer

```rust
// aarambh-studio-inference/src/process_reward.rs
pub trait ProcessRewardScorer: Send + Sync {
    fn score(&self, prompt: &str, completion: &str) -> f32;
}

pub struct HeuristicProcessRewardScorer { /* ... */ }
pub struct ProcessRewardHead { /* placeholder for a future trained head */ }
```

The roadmap describes "a small classifier head trained on GRPO/DPO-style
contrastive step data". Phase 45 ships the `ProcessRewardScorer` trait
and a built-in `HeuristicProcessRewardScorer` that approximates the
trained head with a transparent scoring function (rewards a non-empty
thinking block, a final-answer marker, a parsable numeric answer, and a
non-trivial step count). A `ProcessRewardHead` placeholder is documented
as "loadable trained head — not yet trained; returns
`AarambhError::Unsupported` until a checkpoint exists", so the not-yet-
trained status is explicit at the call site rather than silently degrading
or panicking with `todo!()`. No trained checkpoint ships (the release audit
forbids tracked model artifacts).

## CPU/CUDA honesty policy

Everything in the three new inference modules is pure Rust over the
existing `InferenceEngine` / `Sampler` / `GenerationSession` surface —
zero `unsafe` blocks, zero CUDA calls. It compiles and is unit-tested on
CPU without the `cuda` feature, exactly as the speculative-decoding and
thinking-engine modules are structured. The CUDA path is unchanged: when
the target engine is on a CUDA device, the existing `prepare_session` and
`decode_sessions` calls run on GPU automatically; best-of-N adds no new
device-specific code.

## Backward compatibility

`N = 1` reproduces single-sample generation byte-for-byte. The first
acceptance test
(`best_of_n_with_n_equal_one_matches_single_sample_generation_exactly`)
enforces this: candidate 0 inherits the input sampler's seed unchanged,
the forked-session decode path is proven equivalent to the non-forked
path by the existing `forked_prefill_matches_independent_generation`
test (engine.rs), and for seeded `TopKTopP` the cloned `Box<StdRng>` has
identical state so the first sample matches. `infer` without `--best-of-n`
is untouched — the best-of-N branch only activates when the flag is set.

## An honest scope constraint

Best-of-N is text-only in Phase 45: combining `--best-of-n` with
`--image`/`--video`/`--document`/`--audio`/`--tools` returns
`AarambhError::Unsupported` with a clear message. This mirrors
`fork_with_config`'s existing no-tools constraint (the forked session
path does not support tool-calling prompts) and keeps the surface
honest — multimodal best-of-N is future work, not a half-implementation.

## Measured, not assumed

Whether Best-of-N with a given selection strategy actually improves
accuracy on a given task is an eval-harness question, answered per task
via `eval --best-of-n`, not assumed from the technique's general
reputation. The fourth acceptance test
(`best_of_n_accuracy_on_gsm8k_subset_is_measured_not_assumed_to_improve`)
enforces that the scorecard *records* the single-sample vs best-of-N delta
in its `details` map without asserting the delta is positive — different
tasks and selection strategies are expected to show different, sometimes
negligible, deltas; the scorecard is the source of truth, not the
roadmap's prose.

## An honest hardware constraint

i3 supports small N (2–4) for text tasks; larger N is Kaggle-scoped for
cost reasons, following v1 §12's existing i3 self-learning N-completion
budget precedent. This smoke keeps N=2 so it runs on CPU in well under a
minute. Real accuracy deltas are reported only via the eval-harness
scorecard, never asserted in prose.

## Tests

| Test | Gate |
|---|---|
| `best_of_n_with_n_equal_one_matches_single_sample_generation_exactly` | N=1 backward compat (byte-identical to single-sample) |
| `self_consistency_majority_vote_selects_the_most_common_final_answer` | self-consistency majority vote on extracted answers |
| `process_reward_score_correlates_positively_with_verifier_score_on_labelled_holdout` | heuristic PR scorer correlates with verifier on synthetic labelled data |
| `best_of_n_accuracy_on_gsm8k_subset_is_measured_not_assumed_to_improve` | eval scorecard records the delta (not asserts it improved) |
| `best_of_n_generates_n_distinct_candidates_with_stochastic_sampler` | re-seeding produces divergent candidates |
| `best_of_n_greedy_candidates_are_identical` | greedy best-of-N is degenerate (documented) |
| `verifier_selection_picks_first_fully_correct_candidate` | verifier selection picks the highest-scoring candidate |
| `extract_final_number_matches_gsm8k_marker` / `extract_final_answer_prefers_number_then_last_line` | answer extraction |
| `majority_vote_breaks_ties_by_first_occurrence` | tie-breaking determinism |
| `heuristic_scorer_rewards_thinking_block_and_answer_marker` | PR heuristic monotonicity |
| `selection_strategy_round_trips_through_display` | strategy parse + display |
| `selection_strategy_parses_kebab_and_snake_aliases` | CLI alias parsing |
| `rejects_zero_candidates` / `rejects_verifier_strategy_without_verifier` | config validation |

The four roadmap-named tests are the Phase 45 acceptance tests; the rest
are the supporting CPU unit tests that exercise the new code paths
without CUDA hardware.

## Configs

- `configs/best_of_n_smoke.toml` — CPU smoke training config (tiny
  Shakespeare, 8 steps) that produces a checkpoint the smoke script runs
  best-of-N inference against. The best-of-N surface is exercised via
  CLI flags (`--best-of-n`, `--selection`, `--ground-truth` on `infer`;
  `--best-of-n`, `--best-of-n-selection`, `--best-of-n-seed` on `eval`),
  not a TOML section, per the roadmap's explicit CLI-first scope.

## Smoke script

`scripts/phase45_smoke.sh` runs the `best_of_n`, `self_consistency`, and
`process_reward` inference-crate unit tests, the `generation` and
`gsm8k_subset` eval-crate unit tests, trains a tiny checkpoint on
`best_of_n_smoke.toml`, runs `infer --best-of-n 2 --selection
self-consistency` end-to-end on CPU, verifies the new flags appear in
`infer --help` and `eval --help`, and writes a scorecard to
`artifacts/phase45_test_time_smoke.json`.

## Milestone

`infer --best-of-n 8 --selection verifier` produces a measured, reported
accuracy delta versus single-sample generation on the GSM8K/HumanEval-
lite eval-harness subsets, with the delta included in a scorecard rather
than asserted in prose. i3 supports small N (2–4) for text tasks; larger
N is Kaggle-scoped for cost reasons, following v1 §12's existing i3
self-learning N-completion budget precedent.

```
git commit -m "feat: Phase 45 — test-time compute scaling"
git tag v4.0.0-alpha.5
```
