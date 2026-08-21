# ROADMAP_V4.md — aarambh-studio v4.0

> From first principles. From zero. From Rust.
>
> Step-by-step build plan for v4.0 — the final planned version of
> aarambh-studio as an application. Builds on the completed v3.0.0 base
> (Phases 0–40, all ✅). No pretrained checkpoints are released as part
> of v4.0 — this is a source/engineering release, same policy as v1.0.0,
> v2.0.0, and v3.0.0. aarambh-studio is an application, not a library: v4.0
> ships as a GitHub source release with every crate `publish = false`,
> **not** a crates.io publish — this corrects the direction implied by
> v3's own Phase 40 and is the final, confirmed policy going forward.

---

## How to Read This Roadmap

Each phase has:
- **Goal** — exactly what you will have when this phase is done
- **Tasks** — the checklist to follow, in order, grouped by crate
- **Tests** — what you write to prove it works
- **Milestone** — how you know you are done, with the git tag to cut

Work top to bottom. Do not skip phases. Phases 41 (MLA) leads because it
completes the attention family v3 started (Gated DeltaNet, DSA) — every
phase after it trains on the resulting stable attention stack. Phases
42–44 are scale/modality phases. Phases 45–46 are reasoning-quality
phases, independent of each other but grouped together. Phases 47–48 are
the agentic arc and must run in that order. Phases 49–50 are independent
utility phases. Phase 51 is the highest-risk phase in the roadmap and is
placed deliberately last before the release. Phase 52 is the final
v4.0.0 release.

---

## Phase Map (Quick Reference)

```
Phase 41 →  Multi-Head Latent Attention (MLA)              (10–14 days)  [Kaggle]
Phase 42 →  Audio modality                                  (14–18 days)  [Kaggle]
Phase 43 →  Sparse/grouped MoE dispatch                     (10–14 days)  [Kaggle]
Phase 44 →  Multi-node distributed training                 (10–14 days)  [Kaggle*]
Phase 45 →  Test-time compute scaling                       (7–10 days)   [i3 + Kaggle]
Phase 46 →  RLAIF (AI-feedback alignment)                   (7–10 days)   [Kaggle]
Phase 47 →  Tool execution with sandboxing                  (10–14 days)  [i3 + Kaggle]
Phase 48 →  Multi-agent orchestration                        (10–14 days)  [i3 + Kaggle]
Phase 49 →  Retrieval-augmented generation (RAG)             (10–14 days)  [i3]
Phase 50 →  Model merging / weight averaging                 (5–7 days)    [i3]
Phase 51 →  Public/hosted inference server + prefix caching  (10–14 days)  [Kaggle]
Phase 52 →  System role, chat-template versioning, context mgmt (7–10 days) [i3]
Phase 53 →  Red-team / adversarial safety evaluation          (10–14 days)  [i3 + Kaggle]
Phase 54 →  Model card & release documentation standard        (3–5 days)   [i3]
Phase 55 →  crates-free source release (v4.0.0, final)         (5–7 days)   [all]
```

`[Kaggle*]` — see Phase 44's honesty note on real multi-node validation
limits under a free-tier hardware budget.

**Total realistic estimate: 128–172 days (~4.3–5.7 months)**

---

## Why This Order

1. **41 (MLA) leads v4** because it is the third and final leg of the
   attention family v3 began — Gated DeltaNet (v3 §29, linear attention)
   and DSA (v3 §30, sparse attention) are already in place. Adding MLA
   (latent KV compression) completes the pattern current frontier
   open-weight labs actually ship — some full-attention layers replaced
   by a compressed-latent variant, some by linear attention, some kept
   as sparse full attention — rather than leaving v4 with two of three
   proven techniques. As with v3 §29–30, attention surgery goes first so
   nothing downstream has to re-validate against a moving target.
2. **42 (audio)** follows the same pattern v3 used for video/document
   (§35–36 there): extend the existing frozen-encoder + trainable-
   projector fusion approach (v2 §24–25) to a new sense, once the
   attention stack underneath it is settled.
3. **43 (sparse MoE dispatch)** finally resolves the "documented future
   optimisation" carried forward unresolved through both v2 §26 and v3
   §40. It is sequenced after 41 because routing efficiency work
   benefits from tuning against the final attention stack, not an
   interim one.
4. **44 (multi-node)** depends on 43 — the whole point of sparse dispatch
   is unlocking larger effective MoE configurations, and multi-node is
   the scale that makes those configurations trainable at all.
5. **45–46 (test-time compute scaling, RLAIF)** are grouped as reasoning-
   quality phases. Both are new categories not covered by anything in
   v1–v3: 45 is inference-time (no training change), 46 is a third
   training-time alignment signal alongside GRPO (v1 §11) and DPO
   (v2 §28). They are independent of each other and of 41–44, placed
   here because they benefit from MTP's multi-candidate infrastructure
   (v3 §32) already existing.
6. **47–48 (tool execution, multi-agent orchestration)** must run in this
   order and only in this order: sandboxed execution (47) is the
   prerequisite that makes orchestrating multiple execution chains (48)
   safe to build at all. This is the closing arc of a boundary v2 §30
   opened (emit-only) and v3 §46 extended (multi-step, still emit-only).
7. **49–50 (RAG, model merging)** are independent utility phases. RAG is
   placed after the agentic work because tool-use chains (48) are a
   natural consumer of retrieval results. Model merging is placed here
   because by this point in the roadmap, DoRA, DPO, RLAIF, MoE, and
   distillation have all produced genuinely different checkpoint
   variants worth merging — earlier in the roadmap there would be
   nothing meaningful to merge yet.
8. **51 (public inference server)** is deliberately the last feature
   phase. It is the single biggest risk/scope jump in the entire v1–v4
   history — v2 §27 and v3 both stayed local-only on purpose. Opening
   the server to real multi-tenant traffic only makes sense once the
   model behind it (attention, MoE, agentic tool use) is fully settled,
   not while it is still changing underneath.
9. **52 (system role, chat-template versioning, context management)**
   is placed right after every feature phase is done and right before
   the safety/release phases, because it is a documentation-and-
   formalization pass on the model's I/O contract as it now stands
   after Phases 41–51 — not a new capability, a retrofit of what was
   under-specified: the `<|system|>` special token existed since v1 but
   was never given a documented role, the chat template has now grown
   several times (v2 image tokens, v3 video/document/tool tokens, v4
   audio tokens) with no version tag anywhere, and long agentic chains
   (Phases 47–48) and RAG (Phase 49) both make multi-turn context
   truncation a real, not theoretical, question for the first time.
10. **53 (red-team / adversarial safety evaluation)** comes right after
    51's public server and right before the release, deliberately —
    this is the last chance to systematically test the safety layer
    (`ARCHITECTURE.md` §13), the sandboxed execution boundary (Phase
    47), and the newly-public server (Phase 51) against adversarial
    input before anything ships, rather than relying only on the
    unit-level safety tests each individual phase already wrote.
11. **54 (model card)** comes last among the documentation phases
    because it summarizes everything above it — capabilities,
    limitations, eval scores, and now the red-team findings from 53 —
    into one canonical reference. Writing it before 53 would mean
    writing it twice.
12. **55 (final release)** closes the project the same way v1's Phase
    15 and v2's Phase 28 did: ship the *code* as source once it is
    proven, never ship unproven code or unreleased weights. v4.0.0 is
    confirmed as the **final planned version** — no v5 roadmap exists
    as of this release.

---

## Workspace `Cargo.toml` Additions

```toml
[workspace]
members = [
    # ...existing v1.0.0 + v2.0.0 + v3.0.0 members unchanged...
    "crates/aarambh-studio-core",
    "crates/aarambh-studio-tokenizer",
    "crates/aarambh-studio-data",
    "crates/aarambh-studio-nn",
    "crates/aarambh-studio-kernel",
    "crates/aarambh-studio-model",
    "crates/aarambh-studio-weights",
    "crates/aarambh-studio-quant",
    "crates/aarambh-studio-train",
    "crates/aarambh-studio-finetune",
    "crates/aarambh-studio-inference",
    "crates/aarambh-studio-safety",
    "crates/aarambh-studio-selflearn",
    "crates/aarambh-studio-eval",
    "crates/aarambh-studio-vision",
    "crates/aarambh-studio-serve",
    "crates/aarambh-studio-distill",
    "crates/aarambh-studio-agent",

    # new in v4.0
    "crates/aarambh-studio-audio",       # Phase 42
    "crates/aarambh-studio-retrieve",    # Phase 49

    "aarambh-studio",
]
```

Two new crates (`aarambh-studio-audio`, `aarambh-studio-retrieve`). Everything
else extends existing crates — most heavily `aarambh-studio-nn` (MLA, sparse
dispatch), `aarambh-studio-agent` (execution, orchestration, both already
scaffolded in v3 §37 for chain orchestration), `aarambh-studio-finetune`
(RLAIF), `aarambh-studio-inference` (test-time scaling), and `aarambh-studio-
serve` (public hosting, prefix cache). No new external dependencies
beyond what each phase's Dependency Policy note allows.

---

## Phase 41 — Multi-Head Latent Attention (MLA)

**Duration:** 10–14 days | **Hardware:** Kaggle (free quota)

> **Status: Implemented in v4.0.0-alpha.1.** The `mla.rs` module, `MlaConfig`,
> the three-way `HybridAttentionSchedule`, the partial-checkpoint retrofit path,
> `--kv-cache-report`, the smoke/retrofit scripts, and the full test suite
> (reconstruction tolerance, decoupled-RoPE split, cache-size, partial-load,
> backward-reachability) are all in place. See `docs/phase41_mla.md` and
> `CHANGELOG.md` §4.0.0-alpha.1. The checkbox list below is the original plan,
> preserved for traceability.

### Goal
A third attention kind — latent KV compression — addable to the
`HybridAttentionSchedule` v3 §29 introduced, so a model can now mix
Full, GatedDeltaNet, and LatentMLA layers in whatever ratio the config
specifies. MLA layers cache a single low-rank latent vector per token
instead of full per-head keys and values, cutting KV cache memory per
token substantially at long context, without discarding per-head
expressiveness.

### Tasks

**`aarambh-studio-nn`:**
```
[x] src/mla.rs
      down_proj: d_model -> latent_dim (e.g. 512), producing c_kv per
      token — this is the only thing cached, not per-head K/V
      per-head up-projection: c_kv -> K^(h), V^(h) via per-head learned
      matrices (weights, not cache — this is what preserves per-head
      diversity despite the shared cached latent)
      Decoupled RoPE: split each head's query/key into a "nope" half
      (no positional rotation, derived straight from the latent) and a
      "rope" half (small, separately cached, carries rotary position) —
      a compressed latent cannot naively carry an already-rotated key,
      so position has to be re-introduced on a small dedicated slice
      MlaCache — stores {c_kv latent, rope-half cache} per token,
      replacing the full per-head KV cache for MLA layers specifically

[x] src/attention.rs
      AttentionKind enum extended: Full | GatedDeltaNet | LatentMLA
      HybridAttentionSchedule extended to a 3-way per-layer assignment
      Full and GatedDeltaNet layers completely unchanged from v1/v3
```

**`aarambh-studio-model`:**
```
[x] Model config's attention_schedule accepts LatentMLA entries
[x] Backward compatible: a schedule with zero LatentMLA entries
    reproduces exact v3.0.0 behaviour
[x] New hybrid variants of Medium/Large configs mixing all three kinds
      configs/medium_hybrid_mla.toml
      configs/large_hybrid_mla.toml
```

**`aarambh-studio-train`:**
```
[x] Continued-pretraining retrofit recipe, same pattern as v3 §29:
    load an existing v3 checkpoint, reinitialise scheduled layers as
    LatentMLA, keep everything else loaded as-is, train at reduced
    learning rate
[x] Retrofit validation against the eval harness (v2 §17) before/after,
    same tolerance-band discipline as v3 §29
```

**`aarambh-studio-weights`:**
```
[x] Partial-checkpoint loading extended to support MLA-layer weights
    (down_proj, per-head up-projections, rope-half projections)
    alongside the existing Full/GatedDeltaNet partial-load path
```

**`aarambh-studio-inference`:**
```
[x] KV cache allocation per layer now depends on AttentionKind — MLA
    layers allocate {latent_dim + rope_half_dim} per token instead of
    {2 × num_kv_heads × head_dim} — measurably smaller at long context
[x] Memory report tooling: `aarambh-studio eval --kv-cache-report` prints
    bytes/token per attention kind in the active schedule
```

### Data Setup
```bash
# Same continued-pretraining corpus style as v3 Phase 29 — long
# documents, since MLA's payoff (like Gated DeltaNet's) shows up at
# long context, not short prompts.
scripts/phase41_prepare_mla_retrofit.sh data
```

### Tests
```rust
#[test]
fn schedule_with_zero_mla_layers_matches_v3_exactly() {}

#[test]
fn mla_reconstructed_kv_matches_reference_full_attention_within_tolerance() {
    // Latent down/up-projection round trip must approximate full K/V
    // closely enough that swapping a Full layer for LatentMLA at
    // equivalent training does not regress eval-harness score beyond
    // the documented tolerance band.
}

#[test]
fn decoupled_rope_nope_split_preserves_relative_position_encoding() {}

#[test]
fn mla_kv_cache_bytes_per_token_is_smaller_than_full_or_gqa_baseline() {
    // Assert the measured reduction factor against the documented
    // target range for the configured latent_dim.
}

#[test]
fn partial_checkpoint_load_preserves_non_mla_layer_weights_exactly() {}
```

### Milestone
```
Hybrid Medium/Large configs mixing Full, GatedDeltaNet, and LatentMLA
layers retrofit successfully from a v3.0.0 checkpoint, eval-harness
scores within tolerance of the pre-retrofit baseline, and measured
KV-cache bytes/token at 16K+ context reduced versus the v3 all-Full
baseline by the documented factor.

git commit -m "feat: Phase 41 — Multi-Head Latent Attention"
git tag v4.0.0-alpha.1
```

---

## Phase 42 — Audio Modality

**Duration:** 14–18 days | **Hardware:** Kaggle (free quota)

> **Status: Implemented in v4.0.0-alpha.2.** The new `aarambh-studio-audio`
> crate (frozen `FrozenAudioEncoder`, pure-Rust WAV decode + mel-spectrogram,
> trainable `AudioProjector`, `interleave_audio_tokens`, `AudioQaExample`
> JSONL), the `<audio>`/`<audio_end>` tokenizer tokens (IDs 15/16) with
> `validate_audio_special_tokens` / `upgraded_for_audio`, the
> `convert --upgrade-audio-vocab` migration, the `[vision.audio]` config block,
> the `finetune audio-dora`/`audio-qdora` two-stage trainer, the `infer --audio`
> flag, the `audio-qa` eval task, the smoke/fixture/data-prep scripts, and the
> full test suite (frozen-encoder gradient isolation, projector-only stage,
> fusion length, thinking composability) are all in place. See
> [docs/phase42_audio.md](docs/phase42_audio.md).

### Goal
A frozen, pretrained audio encoder plus a trainable projector gives the
model the ability to hear and reason about audio clips — speech and
non-speech — following the exact same frozen-encoder-plus-projector
pattern v2 §24 established for vision, extended to a new modality
rather than reinvented.

### Tasks

**`aarambh-studio-audio`** *(new crate, Layer 3)*:
```
[x] src/lib.rs
[x] src/encoder.rs
      FrozenAudioEncoder — a small (~40–90M param), permissively-
      licensed, pretrained speech/audio transformer encoder, loaded as
      SafeTensors through candle-core, same loading path
      aarambh-studio-weights already uses (identical policy to v2 §24's
      CLIP loading — no PyTorch bindings, no ONNX, no Python FFI)
[x] src/preprocess.rs
      Mel-spectrogram extraction from raw audio, using a pure-Rust or
      permissively-licensed system-library audio decode path — same
      dependency discipline v3 §35 established for video container
      decode: no Python-based audio ML tooling, local decode only
[x] src/projector.rs
      Projector MLP: audio_d_model -> hidden -> llm_d_model, trainable,
      mirrors vision's projector.rs exactly in structure
[x] src/fusion.rs
      interleave_audio_tokens() — generalises v2 §24's
      interleave_image_tokens() into a shared modal-token-splicing
      pattern, spliced at the <audio> special token position
[x] src/instruct_data.rs
      AudioQaExample, JSONL schema, mirrors vision's instruct_data.rs
```

**`aarambh-studio-tokenizer`:**
```
[x] <audio> / <audio_end> reserved special token strings, IDs,
    validation — same pattern as v2's <image>/<image_end> and v3's
    <video>/<document> tokens
```

**`aarambh-studio-finetune`:**
```
[x] vlm_dora.rs extended to accept audio-token-prefixed sequences —
    same DoRA-adapted-LLM + frozen-encoder two-stage recipe as v2 §25,
    substituting audio for image
```

**`aarambh-studio-eval`:**
```
[x] tasks/audio_qa_subset.rs — free, public audio-QA benchmark subset,
    implements the shared EvalTask trait (v2 §22)
```

**`aarambh-studio-inference`:**
```
[x] --audio CLI flag on infer, parallel to --image (v2) and
    --video/--document (v3)
```

### Data Setup
```bash
# Free public audio-caption / audio-QA dataset subset, sized for
# Kaggle's free storage/compute quota — same "free and public only"
# policy as every prior modality phase.
scripts/phase42_prepare_audio_data.sh data
```

### Tests
```rust
#[test]
fn frozen_audio_encoder_never_receives_gradients() {}

#[test]
fn projector_pretrain_stage_trains_only_projector_weights() {}

#[test]
fn audio_token_fusion_produces_expected_sequence_length() {}

#[test]
fn thinking_controller_behaves_identically_after_audio_context() {
    // Same composability guarantee v2 §25 established for vision —
    // <think> after audio tokens is indistinguishable from <think>
    // after text-only context to ThinkingController.
}
```

### Milestone
```
Audio projector pretrains on captioning data, then instruction-tunes
alongside a DoRA-adapted LLM for open-ended audio QA. `infer --audio`
produces coherent, on-topic responses to spoken/audio prompts on a
held-out set, and the audio_qa_subset eval task reports a documented
non-trivial score versus a random baseline.

git commit -m "feat: Phase 42 — audio modality"
git tag v4.0.0-alpha.2
```

---

## Phase 43 — Sparse/Grouped MoE Dispatch

**Duration:** 10–14 days | **Hardware:** Kaggle (free quota)

### Goal
Resolves the "documented future optimisation" carried forward unresolved
since v2 §26 and v3 §40: real sparse dispatch, where each token's
forward pass only computes its assigned top-k experts, rather than every
expert computing on every token and being masked afterward.

### Tasks

**`aarambh-studio-nn`:**
```
[x] src/dispatch.rs (extended)
      Token-to-expert grouping: sort/group tokens by router assignment
      into per-expert contiguous batches (gather)
      Per-expert matmul executes only on its assigned token group —
      not the full sequence
      Scatter results back into original token order
      DispatchKind enum: DenseMasked (v2/v3 behaviour, kept as the CPU
      fallback and as a correctness reference) | Sparse (new)
[x] CUDA grouped-GEMM path for the Sparse dispatch kind — this is
    where the real throughput win lives; the CPU path continues to use
    DenseMasked regardless of configuration, documented plainly as "GPU
    only pays off," not silently downgraded
```

**`aarambh-studio-model`:**
```
[x] MoeConfig gains `dispatch: DispatchKind`, default DenseMasked for
    exact backward compatibility with every existing MoE checkpoint
```

**`aarambh-studio-train`:**
```
[x] Load-balancing auxiliary loss unchanged — Sparse dispatch changes
    compute path only, not the loss the router is trained against
```

### Tests
```rust
#[test]
fn sparse_dispatch_output_matches_dense_masked_reference_within_tolerance() {
    // Correctness first, exactly like v2 §26 shipped DenseMasked first
    // — Sparse must reproduce the same numbers, just faster.
    // ✓ Implemented in aarambh-studio-nn/src/dispatch.rs (max abs diff < 1e-5).
}

#[test]
fn dispatch_kind_dense_masked_is_bit_identical_to_v2_v3_behaviour() {
    // ✓ Implemented in aarambh-studio-nn/src/moe.rs (diff == 0.0).
}

#[test]
fn sparse_dispatch_cuda_throughput_exceeds_dense_masked_at_kaggle_gpu_scale() {
    // Wall-clock, not a correctness gate — same honesty discipline
    // v2 §29 used for speculative decoding's speed claim.
    // ✓ Implemented in aarambh-studio-nn/src/moe.rs (skips on CPU, runs on CUDA).
}

#[test]
fn load_balancing_loss_value_is_unaffected_by_dispatch_kind() {
    // ✓ Implemented in aarambh-studio-nn/src/moe.rs.
}
```

### Milestone
```
Sparse dispatch produces numerically equivalent output to DenseMasked
on the same MoE config, with measured throughput improvement on Kaggle
GPU hardware at realistic batch sizes, verified via an eval-harness
before/after scorecard confirming no quality regression.

git commit -m "feat: Phase 43 — sparse/grouped MoE dispatch"
git tag v4.0.0-alpha.3
```

---

## Phase 44 — Multi-Node Distributed Training

**Duration:** 10–14 days | **Hardware:** Kaggle-adjacent, see note below

### Goal
Extends v2 §23's single-node NCCL data parallelism to multiple nodes —
still data-parallel only, not model/pipeline-parallel — so training can
scale past whatever a single machine's GPU count offers.

**Honesty note on hardware:** Kaggle notebooks do not provide true
multi-node access. This phase is realistically validated using either
(a) two or more externally-provisioned machines/VMs on a free or
low-cost cloud tier tunnelled together for NCCL, or (b) a documented
single-machine simulation (multiple processes, loopback networking)
that exercises the multi-node code path without genuinely separate
hardware. Both validation paths are documented plainly in the milestone
below — this phase does not claim a Kaggle-native multi-node benchmark,
because that claim would not be honest.

### Tasks

**`aarambh-studio-train`:**
```
[x] src/distributed.rs (extended)
      Node rank vs local rank distinction (world_size = nodes ×
      gpus_per_node)
      Multi-node NCCL initialisation over TCP rendezvous
      Gradient all-reduce unchanged in math from v2 §23 — only the
      topology it runs over changes
      Sharded data loader accounts for global world_size, not just
      single-node GPU count
      Minimal fault handling: a single retry on a transient NCCL
      timeout before failing loudly — this phase does not attempt full
      elastic/fault-tolerant training, which is explicitly out of scope
[x] Rank-zero logging/checkpointing extended to first-node-rank-zero
    specifically, so multi-node runs do not produce duplicate
    checkpoints from every node's local rank zero
```

### Tests
```rust
#[test]
fn world_size_one_node_reproduces_v2_single_node_behaviour_exactly() {}

#[test]
fn gradient_all_reduce_correctness_across_simulated_multi_node_topology() {}

#[test]
fn rank_zero_checkpoint_writes_from_exactly_one_process_globally() {}

#[test]
fn transient_nccl_timeout_triggers_single_retry_then_fails_loudly() {}
```

### Milestone
```
Multi-node data-parallel training runs correctly on the documented
validation path (external multi-VM tunnel or single-machine
simulation), with gradient correctness verified against the
single-node v2 baseline on identical data. Real-hardware multi-node
throughput numbers are reported only where genuinely available and are
clearly labelled as such — never implied from the simulation path.

git commit -m "feat: Phase 44 — multi-node distributed training"
git tag v4.0.0-alpha.4
```

---

## Phase 45 — Test-Time Compute Scaling

**Duration:** 7–10 days | **Hardware:** i3 (small N) + Kaggle (larger N)

### Goal
A genuinely new inference-time capability, distinct from the thinking
engine (v1 §7): instead of controlling *how many tokens* a single
generation spends reasoning, this phase generates *multiple candidate
completions* and selects among them — the 2026 test-time-scaling
pattern (Best-of-N, self-consistency, verifier-guided selection) that
sits alongside, not inside, the existing thinking-mode budget system.

### Tasks

**`aarambh-studio-inference`:**
```
[x] src/best_of_n.rs
      Parallel N-sample generation, reusing the existing sampler and
      (where enabled) speculative decoding infrastructure (v2 §29) for
      each candidate independently
      SelectionStrategy enum: Verifier | SelfConsistency | Majority
[x] src/self_consistency.rs
      For verifiable tasks (math/code): generate N candidates, extract
      final answers, majority-vote — reuses MathVerifier/CodeVerifier
      (v1 §11, v2 §22) purely for answer extraction/comparison, not
      scoring
[x] src/process_reward.rs
      Optional lightweight process-reward scoring: a small classifier
      head trained on GRPO/DPO-style contrastive step data, scores
      intermediate reasoning steps rather than only final answers, used
      to select the best trace among N candidates when a full verifier
      is unavailable (open-ended tasks)
```

**`aarambh-studio` CLI:**
```
[x] infer --best-of-n <N> --selection verifier|self-consistency|majority
```

**`aarambh-studio-eval`:**
```
[x] eval --compare gains a --best-of-n flag so scorecards can report
    "single-sample" vs "best-of-N" side by side — same measure-don't-
    assume discipline v2 §22 established
```

### Tests
```rust
#[test]
fn best_of_n_with_n_equal_one_matches_single_sample_generation_exactly() {}

#[test]
fn self_consistency_majority_vote_selects_the_most_common_final_answer() {}

#[test]
fn process_reward_score_correlates_positively_with_verifier_score_on_labelled_holdout() {}

#[test]
fn best_of_n_accuracy_on_gsm8k_subset_is_measured_not_assumed_to_improve() {
    // The eval-harness scorecard, not a hardcoded expectation, is the
    // source of truth for whether this phase actually helped.
}
```

### Milestone
```
`infer --best-of-n 8 --selection verifier` produces a measured, reported
accuracy delta versus single-sample generation on the GSM8K/HumanEval-
lite eval-harness subsets, with the delta included in a scorecard rather
than asserted in prose. i3 supports small N (2–4) for text tasks;
larger N is Kaggle-scoped for cost reasons, following v1 §12's existing
i3 self-learning N-completion budget precedent.

git commit -m "feat: Phase 45 — test-time compute scaling"
git tag v4.0.0-alpha.5
```

---

## Phase 46 — RLAIF (Reinforcement Learning from AI Feedback)

**Duration:** 7–10 days | **Hardware:** Kaggle (free quota)

> **Status: Implemented in v4.0.0-alpha.6.** `crates/aarambh-studio-finetune/src/rlaif.rs`
> ships the `JudgeGenerator`/`CandidateSampler` traits (Layer-4-clean, no inference-crate
> dependency — the `InferenceEngine` impls live in the CLI binary, mirroring Phase 45's
> `CompletionVerifier`/`MathVerifierAdapter` layering), the position-swap bias correction
> (every pair judged in both A/B orderings, disagreements down-weighted or discarded), and
> the `(chosen, rejected)` output schema that feeds the unmodified `finetune dpo` pipeline.
> The `finetune rlaif` CLI subcommand wires policy + judge `InferenceEngine`s. See
> `docs/phase46_rlaif.md`.

### Goal
A third alignment signal, alongside GRPO (v1 §11, verifier-based) and
DPO (v2 §28, human-preference-based): a judge model scores pairs of
self-sampled completions, automatically generating preference data that
feeds the existing DPO training pipeline unchanged — useful specifically
for open-ended quality dimensions where neither a hard verifier nor a
static human preference dataset is available.

### Tasks

**`aarambh-studio-finetune`:**
```
[x] src/rlaif.rs
      Judge prompt template: given a prompt and two candidate
      completions, the judge (a frozen checkpoint — either the same
      model at an earlier stage, or the Large scale judging Small/Tiny
      outputs) scores which is better and by how much
      Position-swap bias correction: every pair is judged twice, in
      both A/B orderings, and disagreements between the two orderings
      are down-weighted or discarded rather than trusted naively — a
      known failure mode where judges have a first-position bias
      Output format: identical (chosen, rejected) pair schema DPO
      already consumes (v2 §28) — RLAIF is a data-generation front end,
      not a new training objective
[x] Reuses v1 §12's self-learning N-completion sampling infrastructure
    to generate the candidate pairs before judging
```

### Tests
```rust
#[test]
fn position_swap_disagreement_is_downweighted_not_silently_trusted() {}

#[test]
fn rlaif_generated_pairs_match_existing_dpo_pair_schema_exactly() {}

#[test]
fn rlaif_preference_pairs_fed_into_unmodified_dpo_pipeline_train_successfully() {}

#[test]
fn rlaif_dpo_run_reports_non_negative_win_rate_delta_on_preference_eval_task() {
    // Uses v2 §28's existing `preference` eval task — measured, not
    // assumed, same discipline as every other v3/v4 alignment phase.
}
```

### Milestone
```
RLAIF-generated preference pairs, fed through the existing (unmodified)
`finetune dpo` pipeline, produce a checkpoint whose held-out preference
win-rate (v2 §28's eval task) is reported against the pre-RLAIF baseline
— an honest delta, not a claimed win, consistent with every other
"measure, don't assume" phase since v2 §17.

git commit -m "feat: Phase 46 — RLAIF"
git tag v4.0.0-alpha.6
```

---

## Phase 47 — Tool Execution With Sandboxing

**Duration:** 10–14 days | **Hardware:** i3 (orchestration) + Kaggle (multimodal tool results)

### Goal
Closes the boundary v2 §30 opened (tool calls are emitted, never
executed) and v3 §46 extended (multi-step chains, still emit-only): the
model can now have tool calls **actually executed**, but only inside a
strict, explicit sandbox — closed-world allowlisting, no filesystem or
network access unless specifically whitelisted per tool, hard timeouts,
and resource ceilings. This is the highest-risk phase before Phase 51
and is scoped conservatively on purpose.

### Tasks

**`aarambh-studio-agent`** *(extends the crate v3 §37 already scaffolded for chain orchestration)*:
```
[x] src/sandbox.rs
      ToolExecutor trait — implementors register one specific,
      named capability (e.g. "http_get_whitelisted_domain",
      "read_file_in_workdir") — there is no generic "run shell command"
      or "eval code" executor, ever, by design
      Execution is closed-world: an unrecognised tool name is a hard
      refusal, never a best-effort fallback attempt
      Every execution wrapped with an explicit timeout and a resource
      ceiling (memory, CPU, wall-clock) — a runaway or hung tool call
      is killed, not allowed to block the chain indefinitely
[x] src/authorization.rs
      Per-tool authorization is an operator decision, not a model
      decision — the model can request execution of anything in its
      declared tool schema, but only tools the operator has explicitly
      enabled at server/CLI startup are actually executable; everything
      else is refused regardless of what the model requests
```

**Composability:**
```
[x] Execution happens only after the grammar-constrained JSON (v2 §30)
    validates cleanly against the declared schema — malformed or
    partially-streamed tool calls are never executed
[x] Execution results re-enter the chain via the existing
    ToolResult/result_ingestion.rs path from v3 §46 — no new ingestion
    mechanism, execution is additive to what v3 already built
```

### Tests
```rust
#[test]
fn unlisted_tool_name_is_hard_refused_never_attempted() {}

#[test]
fn unauthorized_but_declared_tool_is_refused_at_execution_not_declaration() {}

#[test]
fn execution_timeout_kills_a_hanging_tool_call() {}

#[test]
fn execution_respects_configured_memory_and_cpu_ceiling() {}

#[test]
fn malformed_tool_call_json_is_never_executed() {}

#[test]
fn execution_result_re_ingests_correctly_into_the_next_chain_step() {}
```

### Milestone
```
A whitelisted, sandboxed tool (e.g. a read-only file lookup within a
fixed working directory) executes correctly end-to-end inside a
multi-step chain, with every safety boundary (allowlist, timeout,
resource cap) independently tested and verified to fail closed, not
open, under every tested failure condition. The 6 roadmap-named
acceptance tests in aarambh-studio-agent/src/sandbox.rs (plus
supporting tests) pass; the CLI `agent --execute-tools` path surfaces
the new flags; scripts/phase47_smoke.sh writes a scorecard.

git commit -m "feat: Phase 47 — sandboxed tool execution"
git tag v4.0.0-alpha.7
```

> **Status: Implemented in v4.0.0-alpha.7.** `crates/aarambh-studio-agent`
> ships `src/sandbox.rs` (`ToolExecutor` trait, `ToolSandbox`, `SandboxLimits`,
> `ExecContext`, `ExecError`, `SandboxedToolProvider`, and the reference
> `ReadFileInWorkdir` + `StaticLookup` executors) and `src/authorization.rs`
> (`AuthorizationScope`, with `intersect` for Phase 48 sub-agent scope
> narrowing). Execution is closed-world (registered executor + declared
> schema), operator-authorized, schema-re-validated, and bounded by a
> wall-clock timeout (worker thread + `recv_timeout`, cooperative
> cancellation flag, detached-on-timeout) and output/argument-size ceilings.
> `SandboxedToolProvider` implements `ToolResultProvider`, so execution plugs
> into the existing `ToolChain` with zero chain changes — results re-enter via
> the unchanged `result_ingestion` path. The CLI `agent --execute-tools`
> flag (with `--allow-tool`, `--exec-timeout-ms`, `--exec-max-output-bytes`,
> `--exec-workdir`) wires the operator's authorization. See
> `docs/phase47_sandbox.md` and `scripts/phase47_smoke.sh`. No new crate and
> no new dependency were added (std `thread`/`sync`/`mpsc` + existing
> `serde`/`thiserror` only); the release audit's 20-package invariant holds.

---

## Phase 48 — Multi-Agent Orchestration

**Duration:** 10–14 days | **Hardware:** i3 (orchestration) + Kaggle (multimodal sub-chains)

### Goal
One top-level orchestrating reasoning process delegates independent
sub-tasks to multiple parallel sandboxed tool-execution chains (each
governed entirely by Phase 47's boundaries), then merges their results
back into its own context. This depends completely on Phase 47 —
orchestration is only as safe as the execution sandbox underneath it.

### Tasks

**`aarambh-studio-agent`:**
```
[x] src/orchestrator.rs
      DelegationPlan — the orchestrator's own reasoning produces a set
      of sub-tasks, each assigned to an independent sub-chain
      Each sub-chain is its own GenerationSession-backed instance
      (reusing v2 §31's server session abstraction), with its own
      sandboxed tool scope (Phase 47) and its own timeout budget
      Hard bounds: maximum sub-agent count and maximum total execution
      time are configured ceilings, not model-controlled — an
      orchestrator cannot request unbounded fan-out
      Result aggregation: sub-chain outputs are merged into the
      orchestrator's context using the same ToolResult ingestion
      pattern (v3 §46, Phase 47) applied recursively
```

**Authorization boundary:**
```
[x] A sub-agent's sandbox scope can only be a subset of what the
    orchestrator itself was authorized for (Phase 47's
    authorization.rs) — orchestration can never be used to escalate
    tool access beyond what the operator explicitly enabled
```

### Tests
```rust
#[test]
fn orchestrator_cannot_exceed_configured_max_sub_agent_count() {}

#[test]
fn orchestrator_cannot_exceed_configured_total_execution_time_budget() {}

#[test]
fn sub_agent_sandbox_scope_is_never_wider_than_orchestrator_authorization() {}

#[test]
fn result_aggregation_correctly_merges_multiple_sub_chain_outputs() {}

#[test]
fn one_sub_agent_failure_does_not_silently_corrupt_sibling_sub_agent_results() {}
```

### Milestone
```
An orchestrator correctly delegates a task requiring 2–3 independent
tool-execution sub-chains, each respecting Phase 47's sandbox
boundaries, with results merged coherently and every configured bound
(sub-agent count, total time budget, sandbox scope) independently
verified to hold under test.

git commit -m "feat: Phase 48 — multi-agent orchestration"
git tag v4.0.0-alpha.8
```

> **Status: shipped in v4.0.0-alpha.8** — `crates/aarambh-studio-agent/src/
> orchestrator.rs` (`Orchestrator`, `DelegationPlan`, `DelegatedSubTask`,
> `SubChainOutcome`, `SubChainStatus`, `OrchestrationLimits`) ships the
> three hard bounds (max sub-agent count, total execution time budget,
> sandbox scope containment via `AuthorizationScope::intersect`), failure
> isolation via `catch_unwind`, and result aggregation via the existing
> `ToolResultProvider` path applied recursively. CLI: `agent --orchestrate
> --delegation-plan <PATH> --max-sub-agents N --max-orchestration-budget-ms
> MS --sub-agent-allow-tool NAME`. Sub-chains run sequentially (CPU-first
> honest default); true parallelism is gated on a future `Send + Sync`
> `ChainDecoder`. The five roadmap-named acceptance tests plus five
> supporting tests prove every invariant using fake decoders and real
> sandbox executors, running in milliseconds. See
> `docs/phase48_orchestration.md` for the runbook and the honesty
> boundary.

---

## Phase 49 — Retrieval-Augmented Generation (RAG)

**Duration:** 10–14 days | **Hardware:** i3

### Goal
A from-scratch retrieval pipeline in pure Rust — no external vector
database required, though one may optionally be plugged in later.
Retrieved context augments the prompt before generation; it does not
touch model internals, keeping this phase entirely additive and simple
to reason about.

### Tasks

**`aarambh-studio-retrieve`** *(new crate, Layer 4)*:
```
[x] src/lib.rs
[x] src/embedding.rs
      A small, dedicated text-embedding head — contrastively trained,
      CPU-capable, separate from the main decoder — turns a chunk of
      text into a fixed-size vector
[x] src/index.rs
      A from-scratch approximate-nearest-neighbour index (graph-based,
      pure Rust, no FFI to an external vector-search library) —
      insert(), search(top_k), persist-to-disk, load-from-disk
[x] src/chunking.rs
      Document chunking policy (fixed-size with overlap, configurable)
[x] src/retrieval.rs
      RetrievalPipeline::query() — embed the query, search the index,
      return top-k chunks
```

**Fusion (deliberately simple):**
```
[x] Retrieved chunks are spliced into the existing prompt-construction
    path as additional context ahead of the user's question — the same
    mechanism that already assembles system prompt + chat history +
    user turn, not a new model-level fusion mechanism. RAG augments the
    prompt; it does not change how the decoder processes it.
```

**`aarambh-studio` CLI:**
```
[x] aarambh-studio retrieve build-index --corpus docs/ --output my_index/
[x] aarambh-studio infer --rag --index my_index/ --prompt "..."
```

### Tests
```rust
#[test]
fn index_build_and_query_round_trip_returns_the_inserted_chunk() {}

#[test]
fn retrieval_recall_on_a_small_labelled_holdout_meets_a_documented_floor() {}

#[test]
fn rag_augmented_generation_measurably_improves_a_factual_eval_task_vs_no_retrieval() {
    // Measured via the eval harness, same discipline as every other
    // "does this actually help" phase since v2 §17.
}

#[test]
fn chunking_with_overlap_does_not_duplicate_index_entries_incorrectly() {}
```

### Milestone
```
`aarambh-studio retrieve build-index` and `infer --rag` work end-to-end on a
small local document corpus, with retrieval recall meeting a documented
floor on a held-out labelled set, and RAG-augmented generation showing a
measured, reported improvement on a factual eval-harness task versus
the no-retrieval baseline.

git commit -m "feat: Phase 49 — retrieval-augmented generation"
git tag v4.0.0-alpha.9
```

> **Status: shipped in v4.0.0-alpha.9** — the new
> `crates/aarambh-studio-retrieve` crate (Layer 4, 21st workspace member)
> ships the from-scratch pure-Rust retrieval pipeline: `chunking.rs`
> (fixed-size token-based chunking with overlap), `embedding.rs` (a
> weight-free `HashingEmbedder` as the default tested path plus a
> candle-backed `TextEmbedder` as the trained-head architecture),
> `index.rs` (a navigable small-world graph ANN with `insert`/`search`/
> `save`/`load`, no FFI to an external vector-search library), and
> `retrieval.rs` (`RetrievalPipeline::query()` and `augment_prompt()`
> splicing retrieved chunks into the existing prompt-construction path
> ahead of the user's question). CLI: `retrieve build-index` and
> `infer --rag --index <PATH> --rag-top-k N`. Eval: the `rag` task reports
> `no_retrieval_accuracy` / `rag_accuracy` / `rag_delta` details. Four
> roadmap-named acceptance tests plus 25 supporting tests run in
> milliseconds. No external vector-database dependency was added; an
> optional plug-in adapter is a documented extension point, not shipped.
> See `docs/phase49_rag.md` for the runbook and the honesty boundary.

---

## Phase 50 — Model Merging / Weight Averaging

**Duration:** 5–7 days | **Hardware:** i3

> **Status: shipped in v4.0.0-alpha.10** — a new `merge.rs` module in the
> existing `aarambh-studio-weights` crate (no new crate, no new external
> dependency) ships the five standard model-merging algorithms: linear/Model-
> Soups, SLERP, task-vector arithmetic, TIES-Merging, and DARE. A new
> top-level `aarambh-studio merge` CLI (distinct from the existing
> `finetune merge` adapter-folding command) drives every algorithm. Hard
> validation (identical tensor-name sets, per-tensor shape/dtype) runs before
> any arithmetic, so mismatched inputs are rejected without writing a single
> output byte. MoE/MLA/MTP checkpoints merge transparently because merging
> operates on raw `HashMap<String, Tensor>` maps. Four roadmap-named
> acceptance tests plus nine supporting tests run in milliseconds. A
> `MergeReport` carries only structural facts; any quality claim is measured
> separately by the `eval` command — the same "measured, not assumed"
> discipline every capability claim has held since v2 §26. See
> `docs/phase50_model_merging.md` for the runbook and `ARCHITECTURE_V4.md`
> §64 for the design.

### Goal
By this point in the roadmap, DoRA (v2 §23), DPO (v2 §28), RLAIF
(v4 §46), fine-grained MoE (v3 §40), and distillation (v3 §42) have
all produced genuinely different checkpoint variants. This phase adds
the tooling to merge compatible variants into one via SLERP
interpolation, task-vector arithmetic, and the broader standard
merging toolkit (linear/Model-Soups, TIES-Merging, DARE).

### Tasks

**`aarambh-studio-weights`:**
```
[x] src/merge.rs
      Hard shape/schema validation first — merging requires identical
      architecture and tensor shapes; incompatible checkpoints fail
      loudly at the validation step, never silently produce garbage
      weights
      Linear/Model-Soups weighted averaging of two or more compatible
      checkpoints, with a configurable weight per input (normalized to
      sum to one)
      SLERP (spherical linear interpolation) between two or more
      checkpoints, with a documented linear fallback for near-parallel
      tensors to avoid division by sin(θ) ≈ 0
      Task-vector arithmetic: delta = tuned_checkpoint - base_checkpoint,
      merged = base_checkpoint + sum(scaled deltas) — lets you combine
      independently-tuned deltas (e.g. a math-DoRA delta and a
      chat-DPO delta) onto one shared base
      TIES-Merging: trim each delta to its top-density magnitude entries,
      elect a sign per position by weighted majority, disjoint-merge
      only the agreeing deltas onto the base
      DARE: drop-and-rescale each task vector with a deterministic
      seeded mask (no rand dependency), then linearly combine the
      surviving (rescaled) deltas onto the base
      MoE/MLA/MTP transparency: merging operates on raw
      HashMap<String, Tensor> maps, so expert/router/MLA/MTP tensors
      merge identically to any other tensor — no special-casing, no
      reject_* guard
```

**`aarambh-studio` CLI:**
```
[x] aarambh-studio merge linear --inputs a.safetensors,b.safetensors \
      --weights 0.5,0.5 --output merged.safetensors
[x] aarambh-studio merge slerp --inputs a.safetensors,b.safetensors \
      --weights 0.5,0.5 --output merged.safetensors
[x] aarambh-studio merge task-arithmetic --base base.safetensors \
      --deltas math.safetensors,chat.safetensors --scales 1.0,0.5 \
      --output merged.safetensors
[x] aarambh-studio merge ties --base base.safetensors \
      --deltas math.safetensors,chat.safetensors --scales 1.0,1.0 \
      --density 0.5 --output merged.safetensors
[x] aarambh-studio merge dare --base base.safetensors \
      --deltas math.safetensors,chat.safetensors --scales 1.0,0.5 \
      --density 0.5 --seed 42 --output merged.safetensors
```

### Tests
```rust
#[test]
fn merge_rejects_checkpoints_with_incompatible_shapes_before_writing_output() {}

#[test]
fn slerp_with_weight_one_zero_reproduces_the_first_input_exactly() {}

#[test]
fn task_arithmetic_merge_of_two_independently_tuned_deltas_produces_valid_checkpoint() {}

#[test]
fn merged_checkpoint_eval_harness_score_is_reported_not_assumed_improved() {
    // Same honesty discipline as MoE (v2 §26): merging is measured,
    // not assumed to help, every time.
}

// Supporting tests (nine):
#[test]
fn linear_merge_of_two_identical_checkpoints_is_idempotent() {}
#[test]
fn linear_merge_weights_are_normalized_to_sum_one() {}
#[test]
fn slerp_parallel_vectors_fall_back_to_linear_interpolation() {}
#[test]
fn task_arithmetic_with_zero_scales_reproduces_the_base_checkpoint() {}
#[test]
fn ties_merge_resolves_sign_conflicts_by_weighted_majority() {}
#[test]
fn dare_drop_and_rescale_preserves_expected_magnitude() {}
#[test]
fn merge_rejects_mismatched_tensor_name_sets() {}
#[test]
fn merge_rejects_inconsistent_weight_counts() {}
#[test]
fn merge_output_is_loadable_by_safetensors_load_round_trip() {}
```

### Milestone
```
`aarambh-studio merge` produces a valid, loadable SafeTensors checkpoint
from all five algorithm paths (linear, slerp, task-arithmetic, ties,
dare), with shape-mismatch inputs correctly rejected before any output
is written, and the merged checkpoint's eval-harness scorecard reported
honestly against both input checkpoints' individual scores.

git commit -m "feat: Phase 50 — model merging"
git tag v4.0.0-alpha.10
```

---

## Phase 51 — Public/Hosted Inference Server + Prefix Caching

> **Status: shipped in v4.0.0-alpha.11.** All three modules (`auth.rs`,
> `prefix_cache.rs`, `tenant_isolation.rs`) are implemented, all five
> acceptance tests pass by name, and the loopback-only unauthenticated
> default from v2 §27 is preserved byte-for-byte when the new opt-in flags
> are not set. See `docs/phase51_public_serve.md` for the runbook and
> `CHANGELOG.md` `[4.0.0-alpha.11]` for the entry.

**Duration:** 10–14 days | **Hardware:** Kaggle (free quota, for load testing)

### Goal
The single biggest scope/risk jump in the v1–v4 history. v2 §27 and
v3 both stayed local-only deliberately. This phase opens the existing
`aarambh-studio-serve` (v2 §27) to genuinely multi-tenant, authenticated
traffic, and adds prefix caching — the highest-leverage serving
optimisation for exactly the agentic/tool-chain traffic pattern Phases
47–48 generate.

**Explicit non-goals, restated plainly:** this is still self-hosted —
you run it, Aarambh does not host it for you. There is no billing
system and no horizontal auto-scaling. This phase is about safe
multi-tenant exposure of a single running instance, not a hosted
product.

### Tasks

**`aarambh-studio-serve`:**
```
[x] src/auth.rs
      API key issuance/validation, replacing v2 §27's simple
      loopback-exemption/bearer-token model with real per-key identity
      Per-key rate limiting (requests/minute, tokens/minute), enforced
      at admission into the continuous batcher (v2 §27), not after
[x] src/prefix_cache.rs
      Prompt-prefix hashing, mapped to cached KV state so repeated
      system prompts / shared conversation prefixes across requests
      reuse computed KV state instead of recomputing it
      LRU eviction policy with a configurable memory ceiling
      Hit/miss metrics exposed via the existing /metrics endpoint
      (v2 §27)
[x] src/tenant_isolation.rs
      Per-tenant resource ceilings within the bounded-admission
      continuous batcher (v2 §27) — one tenant's burst of requests
      cannot starve another tenant's already-admitted requests
```

**Documentation:**
```
[x] Explicit deployment guidance: this phase makes public exposure
    *possible*, it does not make it a good default — the loopback-only,
    unauthenticated mode from v2 §27 remains the recommended default
    for local/single-user use
```

### Tests
```rust
#[test]
fn request_with_missing_or_invalid_api_key_is_rejected_before_admission() {}

#[test]
fn per_tenant_rate_limit_is_enforced_independently_per_key() {}

#[test]
fn prefix_cache_hit_measurably_reduces_latency_vs_a_cache_miss_baseline() {}

#[test]
fn prefix_cache_respects_the_configured_memory_ceiling_and_evicts_lru() {}

#[test]
fn one_tenants_request_burst_does_not_starve_another_tenants_admitted_queue() {}
```

### Milestone
```
A multi-tenant server correctly authenticates, rate-limits, and isolates
concurrent tenants under a simulated load test, and prefix caching
produces a measured latency/compute reduction on repeated-prefix traffic
versus a cold-cache baseline — the last, and most carefully tested,
phase before the final release.

git commit -m "feat: Phase 51 — public inference server, prefix caching"
git tag v4.0.0-alpha.11
```

---

## Phase 52 — System Role, Chat-Template Versioning, and Context Management

> **Status: shipped in v4.0.0-alpha.12.** The `<|system|>` system-role marker is
> reserved at **id 17** (not id 7 — id 7 is `IMAGE` since v2 and is never
> reassigned; the system marker takes the next free id, following the project's
> append-never-reassign discipline). All six roadmap acceptance tests pass. See
> `docs/phase52_system_role_context.md`.

**Duration:** 7–10 days | **Hardware:** i3

### Goal
Formalizes and version-tags the model's I/O contract as it stands after
every feature phase in v1–v4: a documented, first-class system role
(the `<|system|>` token has existed since v1 but was never given a
documented role or precedence rule), a version tag on the chat template
itself (which has now grown four times — v1's base format, v2's image
tokens, v3's video/document/tool tokens, v4's audio tokens — with no
way for a checkpoint or a server to declare which template shape it
expects), and a documented multi-turn context-truncation policy (never
written down, and now a real question given Phases 47–49's long
agentic chains and RAG-augmented prompts).

### Tasks

**`aarambh-studio-tokenizer`:**
```
[x] src/special.rs
      Document `<|system|>` (ID 7, already reserved since v1) as a
      first-class, optional turn: one system turn, placed before any
      user turn, carrying operator-set instructions
      chat_template_version: u32 field, stored in tokenizer config and
      checkpoint metadata — bumped exactly once per template-shape
      change in the project's history (v1=1, v2 image tokens=2, v3
      video/document/tool tokens=3, v4 audio tokens=4)
[x] Validation: a served checkpoint's chat_template_version must match
    (or be explicitly declared compatible with) the server's expected
    version — mismatch is a clear startup error, never a silent
    misinterpretation of prompt structure
```

**`aarambh-studio-safety`:**
```
[x] Precedence rule, made explicit and tested: system-turn content is
    always operator/application-supplied, never derived from user
    input — GenerationSession (v2 §31) must reject any code path that
    would let user-message content populate the system-turn position
[x] Existing prompt-injection detection (patterns like "new system
    prompt:") is documented as the *user-input-side* half of this
    defense; this phase adds and tests the *system-turn-side* half
```

**`aarambh-studio-serve`:**
```
[x] /v1/chat/completions system-role mapping, formalized: a request's
    {"role": "system", ...} message maps onto exactly one <|system|>
    turn at the start of the assembled prompt; a request without one
    assembles a prompt with no <|system|> turn, reproducing v1.0.0's
    original format exactly
```

**`aarambh-studio-inference` / `aarambh-studio-agent`:**
```
[x] src/context_policy.rs
      ContextTruncationPolicy enum: SlidingWindow (drop oldest
      non-system turns first, system turn is never evicted) |
      Summarize (replace evicted turns with a generated summary turn,
      reusing the existing self-critique-style summarization the
      project's SFT data already trains for) | Reject (refuse to
      proceed rather than silently drop context — the correct default
      for anything safety- or execution-sensitive, e.g. Phase 47/48
      sessions)
      Applied consistently across long agentic chains (v3 §46, v4
      §61–62) and RAG-augmented sessions (v4 §63) — this phase does not
      invent a new mechanism per feature, it documents and unifies one
      policy referenced by all of them
```

**`aarambh-studio-inference` (sampling defaults):**
```
[x] docs/SAMPLING_DEFAULTS.md — one canonical reference table:
    recommended temperature/top-p/top-k per use case (deterministic
    tool-call generation, open-ended chat, creative writing, math/code
    verification), consolidating guidance that was previously scattered
    across ARCHITECTURE.md, ARCHITECTURE_V2.md, and this roadmap
```

### Tests
```rust
#[test]
fn session_with_no_system_turn_reproduces_v1_prompt_format_exactly() {}

#[test]
fn chat_template_version_mismatch_fails_server_startup_with_clear_error() {}

#[test]
fn user_message_content_can_never_populate_the_system_turn_position() {}

#[test]
fn sft_loss_mask_correctly_covers_a_leading_system_turn() {}

#[test]
fn context_policy_reject_refuses_rather_than_silently_drops_context() {}

#[test]
fn context_policy_sliding_window_never_evicts_the_system_turn() {}
```

### Milestone
```
A conversation with a system turn, one with no system turn, and a
long agentic chain that exceeds context window under each
ContextTruncationPolicy all behave exactly as documented, with the
mismatch/rejection paths verified to fail loudly rather than silently.
SAMPLING_DEFAULTS.md gives one canonical answer to "what settings
should I use," replacing scattered guidance.

git commit -m "feat: Phase 52 — system role, chat-template versioning, context management"
git tag v4.0.0-alpha.12
```

---

## Phase 53 — Red-Team / Adversarial Safety Evaluation

> **Status: shipped in v4.0.0-alpha.13 (Phase 53).** The `redteam/` module
> lives in `aarambh-studio-safety`; the CLI ships `aarambh-studio eval
> --redteam --redteam-report <path>`. The corpus is 24 hand-authored /
> free-public-sourced cases across all four v4.0 surfaces; the composite
> target drives the real safety layer, sandbox, orchestrator, and
> server-auth boundaries end-to-end with zero failures. See
> `docs/phase53_redteam.md`.

**Duration:** 10–14 days | **Hardware:** i3 (text) + Kaggle (multimodal/execution)

### Goal
A systematic adversarial-testing pass — distinct from the unit-level
safety tests each individual phase already wrote — run once, near the
end, against the complete v4.0 surface: the safety layer
(`ARCHITECTURE.md` §13), the sandboxed tool-execution boundary (v4
§61), the newly-public inference server (v4 §65), and the
system-role/prompt-injection precedence rule (v4 §52).

### Tasks

**`aarambh-studio-safety`:**
```
[x] src/redteam/harness.rs
      AdversarialCase — a labelled (input, expected_outcome) pair;
      expected_outcome is one of {refused, sanitized, executed_safely}
      Corpus, hand-authored and free/public-sourced only (same dataset-
      licensing discipline as every other phase since v1): prompt-
      injection variants targeting the system-turn precedence rule
      (§52), jailbreak attempts against the safety layer, attempts to
      get an unauthorized tool executed (v4 §61), attempts to get the
      orchestrator (v4 §62) to exceed its configured bounds, attempts
      to exhaust rate limits or bypass auth on the public server (v4
      §65)
[x] src/redteam/report.rs
      RedTeamReport — pass/fail per case, with failures surfaced
      plainly, never silently excluded from the report
```

**`aarambh-studio` CLI:**
```
[x] aarambh-studio eval --redteam --report redteam_report.json
```

### Tests
```rust
#[test]
fn every_redteam_case_has_a_labelled_expected_outcome() {}

#[test]
fn a_failing_redteam_case_is_surfaced_in_the_report_not_silently_dropped() {}

#[test]
fn redteam_corpus_sources_are_documented_and_free_public_only() {}
```

### Milestone
```
`aarambh-studio eval --redteam` runs the complete adversarial corpus against
the v4.0 candidate build, producing a report with every case's outcome.
Any failing case is fixed and re-tested before Phase 55's release audit
— the release does not proceed with a known, unaddressed red-team
failure.

git commit -m "feat: Phase 53 — red-team adversarial safety evaluation"
git tag v4.0.0-alpha.13
```

---

## Phase 54 — Model Card & Release Documentation Standard

> **Status: shipped in v4.0.0-alpha.14 (Phase 54).** The `model_card.rs`
> module lives in `aarambh-studio-eval`; the CLI ships `aarambh-studio eval
> --generate-model-card --output MODEL_CARD.md`. The card is assembled from
> a real eval-harness scorecard (v2 §17), a real Phase 53 red-team report
> (v4 §67), and static metadata — capabilities and red-team sections are
> PULLED, never hand-entered. Generation fails loudly if no red-team report
> is present or the report is not clean. See `docs/phase54_model_card.md`.

**Duration:** 3–5 days | **Hardware:** i3

### Goal
One canonical document per released checkpoint configuration,
summarizing intended use, capabilities, known limitations, training
data provenance, eval-harness scores, and — since it is written after
Phase 53 — red-team findings. This has never existed as a single
artifact; the information was scattered across ARCHITECTURE*.md,
ROADMAP*.md, and README.md.

### Tasks

**`aarambh-studio-eval`:**
```
[x] src/model_card.rs
      ModelCard — generated from an eval-harness run plus the redteam
      report (v4 §53) plus static metadata (dataset list, license,
      hardware requirements) — assembled, not hand-written from
      scratch each time, so it cannot silently drift out of sync with
      actual eval numbers
```

**`aarambh-studio` CLI:**
```
[x] aarambh-studio eval --generate-model-card --output MODEL_CARD.md
```

**Documentation:**
```
[x] MODEL_CARD.md template: Intended Use, Training Data & Licensing,
    Capabilities (per eval-harness task), Known Limitations, Red-Team
    Summary (v4 §53), Hardware Requirements, Version & Chat-Template
    Compatibility (v4 §52)
```

### Tests
```rust
#[test]
fn model_card_eval_scores_match_the_actual_eval_harness_run_exactly() {
    // Implemented in crates/aarambh-studio-eval/src/model_card.rs.
    // Asserts card.capabilities == the scorecard passed to assemble(),
    // and that the Markdown capabilities section is the verbatim
    // Scorecard::to_markdown() output — never re-rendered by hand.
}

#[test]
fn model_card_generation_fails_loudly_if_no_redteam_report_is_present() {
    // Implemented in crates/aarambh-studio-eval/src/model_card.rs.
    // Half 1: a present-but-not-clean report returns
    //         RedTeamReportNotClean { failed, corpus_size }.
    // Half 2: a missing red-team report file returns
    //         RedTeamReportUnreadable via assemble_from_paths.
}
```

### Milestone
```
`MODEL_CARD.md` generates correctly from a real eval-harness run and
red-team report, with every section populated from actual data rather
than placeholder text.

git commit -m "feat: Phase 54 — model card and release documentation standard"
git tag v4.0.0-alpha.14
```

---

## Phase 55 — Final Source Release (v4.0.0)

**Duration:** 5–7 days | **Hardware:** all

### Goal
Freeze the complete workspace as application version 4.0.0 — **the
final planned version of aarambh-studio.** Same discipline as v1 §15, v2
§28, and v3 §40 in spirit, but with the release *target* corrected: a
GitHub source release with every crate `publish = false`, not a
crates.io publish. aarambh-studio is an application, not a library — this
is the confirmed, final policy, and it supersedes anything v3 §40
implied about crates.io.

### Tasks

```
[x] Every workspace package inherits version 4.0.0
[x] `Cargo.lock` committed, release commands use `--locked`
[x] Full production release audit extended to cover every v4 crate
    surface (aarambh-studio-audio, aarambh-studio-retrieve, extended
    aarambh-studio-agent, extended aarambh-studio-serve)
[x] Documentation completion: ARCHITECTURE_V4.md, ROADMAP_V4.md,
    SELF_LEARNING_V4.md finalised; CHANGELOG.md and README.md updated
    with the full v1 → v4 arc
[x] Release notes explicitly state: v4.0.0 is the final planned
    version — no v5 roadmap exists as of this release
[x] Release audit rejects unfinished markers, unchecked roadmap tasks,
    publishable packages, version drift, and tracked model artifacts —
    identical bar to every prior release
```

### Milestone
```
`aarambh-studio --version` reports `aarambh-studio 4.0.0`. The v4.0.0 GitHub
Release is tagged from a reviewed main branch, contains only GitHub's
automatic source archives, and no crate is published to crates.io —
consistent with every release before it. No pretrained checkpoint,
adapter, tokenizer, optimizer state, SafeTensors, or GGUF file is
attached.

git tag v4.0.0
git push origin v4.0.0
```

---

## Complete Phase Summary

| # | Phase | Key Deliverable | Hardware | Duration |
|---|---|---|---|---|
| 41 | Multi-Head Latent Attention | Third attention kind, latent KV compression | Kaggle | 10–14 days |
| 42 | Audio Modality | Frozen audio encoder + trainable projector | Kaggle | 14–18 days |
| 43 | Sparse MoE Dispatch | Real sparse dispatch, resolves v2/v3's deferred optimisation | Kaggle | 10–14 days |
| 44 | Multi-Node Training | Data-parallel training across nodes | Kaggle-adjacent | 10–14 days |
| 45 | Test-Time Compute Scaling | Best-of-N, self-consistency, process reward | i3 + Kaggle | 7–10 days |
| 46 | RLAIF | Third alignment signal, judge-scored preference pairs | Kaggle | 7–10 days |
| 47 | Sandboxed Tool Execution | Model-triggered execution, closed-world allowlist | i3 + Kaggle | 10–14 days |
| 48 | Multi-Agent Orchestration | Orchestrator delegating to sandboxed sub-chains | i3 + Kaggle | 10–14 days |
| 49 | RAG | From-scratch pure-Rust retrieval pipeline | i3 | 10–14 days |
| 50 | Model Merging | Linear/SLERP/TIES/DARE/task-arithmetic checkpoint merging (5 algorithms) | i3 | 5–7 days |
| 51 | Public Inference Server | Multi-tenant auth, rate limits, prefix caching | Kaggle | 10–14 days |
| 52 | System Role & Chat-Template Versioning | `<|system|>` formalized, template version tag, context-truncation policy | i3 | 7–10 days |
| 53 | Red-Team Evaluation | Systematic adversarial testing of safety, execution, and server surfaces | i3 + Kaggle | 10–14 days |
| 54 | Model Card | Canonical per-checkpoint documentation, assembled from real eval/red-team data | i3 | 3–5 days |
| 55 | Final Release | v4.0.0 source release, project's confirmed final version | all | 5–7 days |

**Total realistic estimate: 128–172 days (~4.3–5.7 months)**

---

## Dependency Policy Additions (v4.0)

| Dependency | Allowed crates | Reason |
|---|---|---|
| Permissively-licensed pure-Rust (or system-library-bound) audio decode crate | `aarambh-studio-audio` | Local mel-spectrogram extraction only, no network calls |
| Small contrastive text-embedding model, loaded as SafeTensors via `candle-core` | `aarambh-studio-retrieve` | Same local-SafeTensors-loading policy as every other encoder in the project |

**Still forbidden everywhere, unchanged from v1/v2/v3:** PyTorch bindings
(`tch-rs`), ONNX Runtime (`ort`), Python FFI, `llama.cpp` as a backend,
any external vector-database service dependency for the core RAG index
(an optional plug-in adapter may exist, but the from-scratch index
remains the default and the tested path). All computation goes through
`candle`.

**Version policy:** unchanged — pin major versions, test the whole
workspace on any `candle-core` upgrade.

---

## What's Explicitly Out of Scope for v4.0

- Releasing any pretrained checkpoint, adapter, or GGUF file — unchanged
  policy across all four versions.
- A hosted, Aarambh-operated version of the inference server — Phase 51
  makes multi-tenant self-hosting possible, not a managed product.
- Elastic/fault-tolerant multi-node training beyond the single-retry
  behaviour in Phase 44.
- Fine-grained per-step credit assignment in tool chains (flagged as a
  v4 candidate by v3 §39's known limitations) — not picked up in v4;
  chain-level outcome weighting from v3 §33 remains the mechanism.
- A general-purpose code-execution sandbox — Phase 47's execution is
  strictly closed-world, named-capability tool execution, never
  arbitrary code or shell execution.
- Any v5 roadmap. **v4.0.0 is the final planned version of aarambh-studio.**
  This document does not carry forward an "out of scope, natural next
  version" section the way v2 and v3 did — there is no v5 planned as of
  this release.
