# Phase 50 — Model Merging / Weight Averaging

> **Status: shipped in v4.0.0-alpha.10 (Phase 50).**
> Proving tests: `crates/aarambh-studio-weights/tests/merge.rs`
> (13 tests, 4 roadmap-named acceptance tests + 9 supporting).
> Smoke: `scripts/phase50_smoke.sh` → `artifacts/phase50_merge_smoke.json`.
> Design: `ARCHITECTURE_V4.md` §64. Roadmap entry: `ROADMAP_V4.md` §"Phase 50".

## Why this phase exists

By the end of Phase 49, the project had produced genuinely different checkpoint
variants from five independent tuning tracks — DoRA (v2 §23), DPO (v2 §28),
RLAIF (v4 §46), fine-grained MoE (v3 §40), and distillation (v3 §42) — but
had no way to **combine** them into a single checkpoint. Phase 50 closes that
gap with a from-scratch, pure-Rust model-merging toolkit. No external merge
library (e.g. mergekit) is used, and none is needed: merging is, at its core,
tensor arithmetic on disk-loaded checkpoints with hard shape/schema
validation.

## What ships

A new module in the existing `aarambh-studio-weights` crate — **no new crate,
no new external dependency** — exposing five standard algorithms:

| Method | CLI | Inputs | Formula |
|---|---|---|---|
| Linear (Model Soups) | `merge linear` | N inputs + weights | `out = Σ wᵢ·Mᵢ` (normalized) |
| SLERP | `merge slerp` | N inputs + weights | pairwise chained spherical lerp |
| Task Arithmetic | `merge task-arithmetic` | base + N deltas + scales | `out = base + Σ sᵢ·(Mᵢ−base)` |
| TIES-Merging | `merge ties` | base + N deltas + scales + density | trim → elect sign → disjoint merge |
| DARE | `merge dare` | base + N deltas + scales + density + seed | drop-and-rescale → linear combine |

A new top-level CLI subcommand `aarambh-studio merge` (distinct from the
existing `finetune merge`, which only folds LoRA/DoRA *adapters* into a base).
The public Rust API lives in `aarambh_studio_weights::merge`:
`MergeMethod`, `MergeConfig`, `MergeReport`, `merge_models_from_paths`.

## Hard guarantees (non-negotiable)

1. **Validate before any write.** Architecture, tensor-name-set, per-tensor
   shape, and per-tensor dtype are all checked **before** a single output byte
   is produced. Mismatches fail loudly via `AarambhError::Config` / `::Shape`
   / `::Checkpoint` — no partial output, no silent corruption.
2. **SLERP identity floor.** At `weight = 1.0` (or `0.0`), SLERP reproduces an
   input checkpoint bit-for-bit (f32). This is the backward-compatibility
   floor for the merge tool itself — proven by
   `slerp_with_weight_one_zero_reproduces_the_first_input_exactly`.
3. **Measured, not assumed.** A `MergeReport` carries only structural facts
   (tensor counts, SLERP fallback counts, TIES conflict counts, DARE dropped
   fraction). Any quality claim is measured separately by the `eval` command
   against the merged artifact — the same discipline every capability claim has
   held since v2 §26 (MoE).
4. **MoE/MLA/MTP transparent.** Merging operates on raw
   `HashMap<String, Tensor>` name/shape-matched maps, so expert weights,
   router weights, MLA projections, and MTP heads all merge identically to any
   other tensor. There is no architecture-specific special-casing and no
   `reject_*` guard, because none is needed at the tensor level.
5. **CPU-first.** All math runs on `Device::Cpu` in `f32`. Merging is an
   offline, not latency-sensitive, operation. The crate's existing `cuda`
   feature is forwarded but not required.
6. **Deterministic.** DARE's drop mask is derived from a seeded xorshift64 PRNG
   (no `rand` dependency, no system randomness), so a merge is fully
   reproducible from its `--seed`.

## Usage

### Linear (Model Soups)

```bash
aarambh-studio merge linear \
  --inputs model_a.safetensors,model_b.safetensors,model_c.safetensors \
  --weights 0.5,0.3,0.2 \
  --output soup.safetensors
```

### SLERP

```bash
aarambh-studio merge slerp \
  --inputs model_a.safetensors,model_b.safetensors \
  --weights 0.7,0.3 \
  --output slerped.safetensors
```

For N>2 inputs, SLERP folds left-to-right: step *i* interpolates between the
running accumulator and input *i* with `t = wᵢ / Σⱼ≤ᵢ wⱼ`. Near-parallel
tensors (cos θ > `1 − 1e-6`) fall back to linear interpolation to avoid
division by `sin(θ) ≈ 0`; the fallback count is reported.

### Task Arithmetic

```bash
aarambh-studio merge task-arithmetic \
  --base base.safetensors \
  --deltas math_dora.safetensors,chat_dpo.safetensors \
  --scales 1.0,0.5 \
  --output merged.safetensors
```

Each delta's task vector is `δᵢ = Mᵢ − base`; the output is
`base + Σ sᵢ·δᵢ`. This lets you combine, e.g., a math-focused DoRA delta and a
chat-focused DPO delta onto the same base, each scaled independently.

### TIES-Merging

```bash
aarambh-studio merge ties \
  --base base.safetensors \
  --deltas d1.safetensors,d2.safetensors,d3.safetensors \
  --scales 1.0,1.0,1.0 \
  --density 0.5 \
  --normalize true \
  --output ties.safetensors
```

Per delta, per tensor: (1) **trim** to the top-`density` magnitude entries;
(2) **elect sign** by weighted majority of surviving deltas; (3) **disjoint
merge** — average only the deltas whose sign agrees with the elected sign.
`--normalize` rescales survivors by `1/density` to preserve expected
magnitude.

### DARE

```bash
aarambh-studio merge dare \
  --base base.safetensors \
  --deltas d1.safetensors,d2.safetensors \
  --scales 1.0,0.5 \
  --density 0.5 \
  --seed 42 \
  --output dare.safetensors
```

Per delta: **drop** each parameter with probability `1 − density` using a
deterministic seeded mask; **rescale** survivors by `1/density`; then combine
the surviving (rescaled) deltas linearly with `--scales` and add to the base.

## How it relates to self-learning

Model merging is **offline preprocessing**, deliberately kept separate from
the online self-learning loop (the same framing RLAIF §46 uses). The
self-learning loop (`SELF_LEARNING_V4.md` §51) generates experience-replay
entries and runs online GRPO updates; merging is something an operator does
*between* sessions to combine independently-tuned checkpoints. A merged
checkpoint can be loaded as the base for a future self-learning session, but
merging itself never touches the replay buffer or the gradient path.

## Honesty boundary

- Merging does **not** guarantee improvement. A merged checkpoint may be
  better, worse, or equal to its inputs depending on whether the task vectors
  are compatible. The `eval` command measures this; the merge command only
  produces a valid artifact and reports structural facts.
- All inputs must share an identical tensor-name set, per-tensor shape, and
  per-tensor dtype. Cross-architecture merging (different hidden sizes,
  different attention schedules, mismatched vocabularies) is rejected.
- Merging operates on SafeTensors only (the project's primary checkpoint
  format). GGUF inputs are not supported by the merge path; convert to
  SafeTensors first via `aarambh-studio convert`.
- The output is always written as `f32`, regardless of input dtype. This
  matches `load_model`'s default precision and the standard practice of
  merging in fp32.

## What this enables next

Phase 51 (Public/Hosted Inference Server) can serve a merged checkpoint
without special handling — it is just another SafeTensors file. A future
operator workflow might: fine-tune three specialist adapters (math, code,
chat), merge them onto one base via task-arithmetic, then serve the single
merged model instead of routing between three. Whether that is actually better
is, as always, measured by the eval harness — not assumed.
