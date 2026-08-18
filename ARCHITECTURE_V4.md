# ARCHITECTURE_V4.md — aarambh-studio v4.0

> From first principles. From zero. From Rust.
>
> Companion to `ARCHITECTURE.md`, `ARCHITECTURE_V2.md`, and
> `ARCHITECTURE_V3.md`. This document covers **only what v4.0 adds** on
> top of the completed v3.0.0 architecture. Sections continue numbering
> from v3's Section 52. Everything in the three prior documents is
> unchanged and continues to work exactly as documented. v4.0 is the
> **final planned version** of aarambh-studio as an application — see §69.

---

## Table of Contents

53. [What's New in v4.0](#53-whats-new-in-v40)
54. [Updated Workspace — 20 Library Crates](#54-updated-workspace--20-library-crates)
55. [Multi-Head Latent Attention (MLA)](#55-multi-head-latent-attention-mla)
56. [Audio Modality](#56-audio-modality)
57. [Sparse/Grouped MoE Dispatch](#57-sparsegrouped-moe-dispatch)
58. [Multi-Node Distributed Training](#58-multi-node-distributed-training)
59. [Test-Time Compute Scaling](#59-test-time-compute-scaling)
60. [RLAIF](#60-rlaif)
61. [Tool Execution With Sandboxing](#61-tool-execution-with-sandboxing)
62. [Multi-Agent Orchestration](#62-multi-agent-orchestration)
63. [Retrieval-Augmented Generation (RAG)](#63-retrieval-augmented-generation-rag)
64. [Model Merging / Weight Averaging](#64-model-merging--weight-averaging)
65. [Public Inference Server + Prefix Caching](#65-public-inference-server--prefix-caching)
66. [System Role, Chat-Template Versioning, and Context Management](#66-system-role-chat-template-versioning-and-context-management)
67. [Red-Team / Adversarial Safety Evaluation](#67-red-team--adversarial-safety-evaluation)
68. [Model Card](#68-model-card)
69. [Updated Dependency Layers](#69-updated-dependency-layers)
70. [Updated Memory & Compute Estimates](#70-updated-memory--compute-estimates)
71. [Updated Hardware Strategy](#71-updated-hardware-strategy)
72. [Final Release Contract — Why v4.0 Is the Last Version](#72-final-release-contract--why-v40-is-the-last-version)

---

## 53. What's New in v4.0

v4.0 closes out aarambh-studio with twelve additions across four themes:

**Attention completion:** Multi-Head Latent Attention (§55) joins Gated
DeltaNet and DSA (both v3) as the third attention kind, completing the
hybrid-attention family at the same maturity level current open-weight
frontier labs ship.

**Modality and scale completion:** audio (§56) closes the last major
input modality gap; sparse MoE dispatch (§57) resolves an optimisation
deferred since v2; multi-node training (§58) extends single-node
data-parallelism (v2) to genuinely larger scale.

**Reasoning-quality completion:** test-time compute scaling (§59) and
RLAIF (§60) round out the alignment/reasoning toolkit — GRPO
(verifier-based, v1), DPO (preference-based, v2), and now RLAIF
(AI-feedback-based, v4) as three complementary training-time signals,
plus a genuinely new inference-time axis in test-time scaling.

**Agency and deployment completion:** sandboxed tool execution (§61)
and multi-agent orchestration (§62) close the arc opened by v2's
emit-only tool calling; RAG (§63) and model merging (§64) are
independent utility completions; the public inference server (§65) is
the final, highest-risk deployment capability.

**Contract and safety completion:** §66–68 are a retrofit pass, not new
capability — formalizing the system role and chat-template version tag
that were reserved but under-specified since v1, a systematic red-team
evaluation of the complete v4.0 surface, and a canonical, assembled-
not-hand-written model card. These close gaps identified by auditing
the project's own documentation against what it actually shipped.

This is a **completion release**, not a foundation release the way v1
was, or a growth release the way v2 and v3 were. §72 explains why v4.0
is the final planned version.

## 54. Updated Workspace — 20 Library Crates

Two new crates. Everything else extends in place — no crate is removed
or renamed, matching the discipline every version has held since v1.

```
aarambh-studio/
├── Cargo.toml
├── ARCHITECTURE.md / ARCHITECTURE_V2.md / ARCHITECTURE_V3.md / ARCHITECTURE_V4.md
├── ROADMAP.md / ROADMAP_V2.md / ROADMAP_V3.md / ROADMAP_V4.md
├── SELF_LEARNING.md / SELF_LEARNING_V2.md / SELF_LEARNING_V3.md / SELF_LEARNING_V4.md
│
├── crates/
│   │   ...Layers 0–6 from v1.0.0/v2.0.0/v3.0.0, extended (see §55–65)...
│   │
│   ├── aarambh-studio-audio/             ← NEW, LAYER 3: Audio modality
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── encoder.rs            ← frozen pretrained audio encoder
│   │       ├── preprocess.rs         ← mel-spectrogram extraction
│   │       ├── projector.rs          ← trainable audio->llm projector
│   │       ├── fusion.rs             ← audio-token interleaving
│   │       └── instruct_data.rs      ← AudioQaExample schema
│   │
│   └── aarambh-studio-retrieve/          ← NEW, LAYER 4: RAG
│       └── src/
│           ├── lib.rs
│           ├── embedding.rs          ← contrastive text-embedding head
│           ├── index.rs              ← from-scratch ANN index
│           ├── chunking.rs           ← document chunking policy
│           └── retrieval.rs          ← RetrievalPipeline
│
└── aarambh-studio/                       ← LAYER 6: CLI binary
    └── src/cmd/
        ├── ...train.rs / infer.rs / finetune.rs / quantise.rs /
        │    convert.rs / eval.rs / serve.rs / agent.rs...
        ├── retrieve.rs               ← NEW: `aarambh-studio retrieve`
        └── merge.rs                  ← NEW: `aarambh-studio merge`
```

### Extended (not new) crates in v4.0

| Crate | v4.0 additions |
|---|---|
| `aarambh-studio-nn` | `mla.rs` (§55), `dispatch.rs` extended with `DispatchKind::Sparse` (§57) |
| `aarambh-studio-model` | `attention_schedule` accepts `LatentMLA` entries, `MoeConfig.dispatch: DispatchKind` |
| `aarambh-studio-train` | MLA retrofit recipe, multi-node `distributed.rs` extended (§58) |
| `aarambh-studio-weights` | Partial-checkpoint loading extended for MLA layers, `merge.rs` (§64) |
| `aarambh-studio-tokenizer` | `<audio>`/`<audio_end>` reserved tokens |
| `aarambh-studio-finetune` | `vlm_dora.rs` extended for audio, `rlaif.rs` (§60) |
| `aarambh-studio-inference` | `best_of_n.rs`, `self_consistency.rs`, `process_reward.rs` (§59), KV-cache report tooling (§55) |
| `aarambh-studio-eval` | `audio_qa_subset.rs`, `--best-of-n` comparison flag |
| `aarambh-studio-agent` | `sandbox.rs`, `authorization.rs` (§61), `orchestrator.rs` (§62) — extends the crate v3 §37 scaffolded |
| `aarambh-studio-serve` | `auth.rs`, `prefix_cache.rs`, `tenant_isolation.rs` (§65) |

### Updated Crate Count

```
v1.0.0: 14 crates (13 library + 1 binary)
v2.0.0: 17 crates (16 library + 1 binary)
v3.0.0: 19 crates (18 library + 1 binary)
v4.0.0: 21 crates (20 library + 1 binary)
```

Six library crates added across the project's entire history. Zero
renamed. Zero removed. This is the architectural claim v2's blog post
made explicit and v3/v4 continue to hold: growth in one direction has
never required touching the transformer core in a way that breaks
another.

---

## 55. Multi-Head Latent Attention (MLA)

**Crate:** `aarambh-studio-nn` (`mla.rs`) | **Depends on:** v1 §6.3 (GQA/RoPE), v2 §21 (YaRN/NTK), v3 §29 (`HybridAttentionSchedule`)

> **Status: Implemented in v4.0.0-alpha.1 (Phase 41).** `MlaAttention` and
> `MlaCache` ship in `crates/aarambh-studio-nn/src/mla.rs`; `AttentionKind::LatentMLA`
> and `MlaConfig` extend the schedule; the partial-checkpoint retrofit and
> `--kv-cache-report` are wired through. See `docs/phase41_mla.md` for usage.

### The Problem

v3 gave the model two ways to reduce the cost of a growing KV cache:
Gated DeltaNet (§29, linear attention — constant-size recurrent state)
and DSA (§30, sparse attention — attend to a learned subset of
positions). Neither directly attacks the *storage* cost of the
remaining full-attention layers, which still cache a full key and value
vector per head, per token.

### The Mechanism

MLA compresses what gets cached, not what gets computed. Instead of
caching per-head K and V directly, a token's hidden state is
down-projected once into a single shared latent vector; per-head keys
and values are then reconstructed from that one latent via small
per-head up-projection matrices — which are ordinary trainable weights,
not part of the cache.

```
hidden_state (d_model)
     │
     ▼
down_proj: d_model -> latent_dim         (e.g. 4096 -> 512)
     │
     ▼
c_kv  (this is the ONLY thing cached per token, for MLA layers)
     │
     ├──▶ per-head up_proj_K^(h): latent_dim -> head_dim   (weights, not cached)
     │        │
     │        ▼
     │      K^(h)  (reconstructed per head, at attention time)
     │
     └──▶ per-head up_proj_V^(h): latent_dim -> head_dim   (weights, not cached)
              │
              ▼
            V^(h)  (reconstructed per head, at attention time)
```

**Decoupled RoPE.** A naively-compressed latent cannot carry an
already-rotated (position-encoded) key — rotation is head-dimension-
specific and applying it before compression would defeat the point of
sharing one latent across heads. MLA splits each head's query and key
into two parts: a larger "nope" (no positional encoding) part derived
straight from the compressed latent, and a small separate "rope" part
that *is* rotary-encoded and cached on the side, at a much smaller
per-head width than a full key would need:

```
query/key (head_dim)
     │
     ├──▶ nope part (larger slice) — derived from c_kv, no RoPE applied
     │
     └──▶ rope part (smaller slice) — separately cached, RoPE applied
              normally (v2 §21's YaRN/NTK scaling still applies here
              unchanged)
```

The cache for an MLA layer, per token, is therefore `{c_kv (latent_dim)
+ rope_half (small, per-head or shared, per config)}` — substantially
smaller than a full per-head K and V cache at typical configurations,
while per-head expressiveness is preserved through the up-projection
weights at attention time.

### Composability With v3's Hybrid Schedule

`HybridAttentionSchedule` (v3 §29) already supported mixing Full and
GatedDeltaNet layers by index. v4 extends the schedule to a three-way
choice:

```rust
enum AttentionKind {
    Full,          // v1 GQA + RoPE/YaRN, unchanged
    GatedDeltaNet, // v3 §29, unchanged
    LatentMLA,     // v4, new
}
```

A schedule with zero `LatentMLA` entries reproduces v3.0.0 exactly —
the same backward-compatibility discipline every attention change since
v1 has held. A schedule can now, for example, use DSA-style sparse full
attention for one layer, Gated DeltaNet for the next three, and
LatentMLA for a fourth — the schedule is per-layer and fully
configurable, not a global architecture choice.

### KV Cache Retrofit

Following the exact pattern v3 §29 established: MLA layers are added to
an existing v3.0.0 checkpoint via continued pretraining, not a
from-scratch rebuild. Scheduled layers are reinitialised with fresh
MLA parameters; every other layer's weights load unchanged from the v3
checkpoint. Training proceeds at a reduced learning rate so the
untouched layers do not drift meaningfully while the new layers learn.

### Measured, Not Assumed

Same discipline as v2 §26 (MoE) and every attention change since:
retrofit success is judged by the eval harness (v2 §17) reporting
scores within a documented tolerance band of the pre-retrofit baseline,
and by a KV-cache memory report (`--kv-cache-report`) showing the
measured bytes/token reduction at long context — not by assuming
compression helps because the mechanism is theoretically sound.

---

## 56. Audio Modality

**Crate:** `aarambh-studio-audio` (new) | **Depends on:** v2 §24–25 (vision fusion pattern), v1 §7 (thinking engine)

### The Same Recipe, a New Sense

v2 §24 established a pattern for adding a modality without touching the
decoder: a frozen, pretrained encoder converts raw input into a grid of
embeddings; a small trainable projector maps those into the decoder's
`d_model` space; the result is spliced into the token sequence as
ordinary-looking tokens. v3 §35–36 reused this pattern for video and
documents. v4 reuses it a third time for audio, changing only the
domain-specific preprocessing step.

```
Raw audio waveform
     │
     ▼
Mel-spectrogram extraction (local, pure-Rust/system-library decode —
no network calls, no Python audio ML tooling)
     │
     ▼
FrozenAudioEncoder (pretrained, ~40-90M params, loaded as SafeTensors
via candle-core — same loading path as every other encoder in the
project)
     │
     ▼
Projector MLP (trainable): audio_d_model -> hidden -> llm_d_model
     │
     ▼
N "audio tokens" in llm_d_model space
     │
     ▼
Spliced into the input sequence at the <audio> special token position
     │
     ▼
...the rest of the decoder, completely unmodified...
```

### Two-Stage Training

Identical structure to v2 §25's vision recipe:

1. **Projector-only stage.** Everything else frozen; the projector
   trains alone on audio-captioning-style data, learning to map audio
   embeddings into a space the (frozen) decoder can already interpret
   reasonably.
2. **Instruction-tuning stage.** The projector continues training
   alongside a DoRA-adapted (v2 §23) LLM on open-ended audio-QA data —
   full fine-tuning at this stage remains out of reach of the free
   Kaggle compute budget, exactly the same constraint that made DoRA
   the right choice for vision in v2.

### Composability

Because fusion happens before the decoder sees the sequence, nothing
about the thinking engine (v1 §7), grammar-constrained tool calling
(v2 §30), or long-horizon tool chains (v3 §46) needs to change — a
`<think>` block or a tool call generated after audio tokens behaves
identically to one generated after text-only or image-only context.
This is the same composability guarantee every prior modality addition
has held, and it continues to hold because the fusion mechanism itself
never changes, only what feeds it.

### What This Does Not Do

Following the same explicit-scope discipline as every other modality
phase: this covers audio *understanding* (the model can be asked about
audio it's given), not audio *generation* — aarambh-studio does not produce
audio output. That capability, along with the broader question of a
dedicated audio-generation stack, belongs to a separate project
entirely and is intentionally out of scope here.

> **Delivered in v4.0.0-alpha.2 (Phase 42).** The `aarambh-studio-audio`
> crate, `<audio>`/`<audio_end>` tokenizer tokens, `convert --upgrade-audio-vocab`,
> `[vision.audio]` config, `finetune audio-dora`, `infer --audio`, the
> `audio-qa` eval task, and the full test suite are implemented. See
> [docs/phase42_audio.md](docs/phase42_audio.md). WAV PCM decode and
> mel-spectrogram extraction are pure-Rust from first principles (no
> `rustfft` or audio-ML dependency); MP3/FLAC/Ogg decode is future work.

---

## 57. Sparse/Grouped MoE Dispatch

**Crate:** `aarambh-studio-nn` (`dispatch.rs`, extended) + `aarambh-studio-core` (`DispatchKind`) | **Depends on:** v2 §26 (MoE), v3 §40 (fine-grained MoE) | **Status:** shipped in v4.0.0-alpha.3 (Phase 43)

### The Deferred Optimisation, Resolved

v2 §26 shipped MoE with dense masked-matmul dispatch — every expert
computes on every token, then gets masked and weighted by the router —
explicitly documented as "simplest-correct first, sparse dispatch is a
future optimisation." v3 §40 extended MoE to fine-grained routing with
a shared expert but kept the same dense dispatch, carrying the same
deferred note forward a second time. v4 finally resolves it.

```rust
enum DispatchKind {
    DenseMasked,  // v2/v3 behaviour — kept as CPU fallback and
                  // correctness reference, default for exact backward
                  // compatibility
    Sparse,       // v4, new — GPU-only real benefit
}
```

**Sparse dispatch**, conceptually:

```
Router produces top-k expert assignment per token (unchanged from v2)
        │
        ▼
Gather: sort/group tokens by assigned expert into contiguous
per-expert batches
        │
        ▼
Each expert's FFN computes ONLY on its assigned token group
(grouped GEMM on CUDA) — not the full sequence, unlike DenseMasked
        │
        ▼
Scatter: results written back into original token order
```

The load-balancing auxiliary loss (v2 §26) is completely unchanged —
`DispatchKind` only changes the compute path, never the objective the
router is trained against, so a checkpoint's routing behavior is
identical regardless of which dispatch kind executes it.

### Why the CPU Path Stays Dense

Sparse dispatch's benefit — skipping compute for unassigned experts —
only pays off when the grouped-GEMM kernels have enough tokens per
expert per batch to be worth dispatching separately, which is a
GPU-batch-scale assumption. On CPU, `DispatchKind::Sparse` is
documented to fall back to `DenseMasked` behaviour rather than attempt
a sparse path that would be slower, not faster, at CPU batch sizes.
This is stated plainly rather than silently downgraded.

### Measured, Not Assumed

Correctness is proven first — sparse output must numerically match the
dense-masked reference within tolerance on identical inputs — before
throughput is even discussed. The throughput claim itself is a
wall-clock measurement at Kaggle GPU scale, reported honestly, the same
discipline v2 §29 (speculative decoding) and every MoE-related claim
since have followed.

### Phase 43 Implementation

The shipped `sparse_grouped_dispatch` (`aarambh-studio-nn/src/dispatch.rs`)
is fully candle-native and differentiable, using no custom kernel:

```
flatten tokens → [N, H]; flatten assignments → [N*top_k]
arg_sort by expert id (no-grad permutation) → grouped order
gather token-ids + weights into grouped order (gather is differentiable)
per-expert boundaries scanned on host (O(N*top_k))
for each expert e with count_e > 0:
    index_select e's token group → [count_e, H]   (differentiable)
    expert.forward(group)                         (matmul on group ONLY)
    mul group_weights                             (differentiable)
    index_add scatter back into [N, H]            (differentiable)
reshape → [batch, seq, H]
```

The grouping permutation is a discrete index (no gradient), computed with
`arg_sort_last_dim` (a no-grad op); every value that flows into the loss
remains differentiable through candle's `gather`/`index_select`/`index_add`.
On CUDA, candle routes these ops plus the per-expert `matmul` to cuBLAS —
a genuine grouped-GEMM path that skips non-routed experts. A fused
single-kernel grouped-GEMM `.cu` file remains a documented future
optimisation; the current path already realises the throughput win without
it, and is verifiable on CPU for correctness.

`effective_dispatch_kind(configured, device)` selects `Sparse` only on a
CUDA device; CPU falls back to `DenseMasked` regardless of configuration
(the "GPU only pays off" policy). `MoeFfn::dispatch_kind()` exposes the
configured kind. QAT calibration (`forward_with_capture`) always uses the
dense reference to observe full per-expert activation distributions. See
`docs/phase43_sparse_moe.md` for the full test matrix and configs.

---

## 58. Multi-Node Distributed Training

**Crate:** `aarambh-studio-train` (`distributed.rs`, extended) | **Depends on:** v2 §27 (single-node NCCL data parallel)

### Scope

v2 §27 proved single-node, multi-GPU data-parallel training via NCCL.
v4 extends the same data-parallel approach — not model or pipeline
parallelism — across multiple nodes.

```
World: N nodes × M GPUs per node = world_size total ranks

Node rank (which machine) × local rank (which GPU on that machine)
     │
     ▼
NCCL rendezvous over TCP across nodes
     │
     ▼
Sharded data loader: each of the world_size ranks sees a disjoint
slice of the global batch — same principle as v2's single-node
sharding, extended to the larger world_size
     │
     ▼
Gradient all-reduce across ALL ranks, all nodes — same math as v2 §27,
different (larger) topology
     │
     ▼
Rank-zero of node-zero specifically logs and checkpoints — prevents
duplicate checkpoints from every node's own local rank zero
```

### An Honest Hardware Constraint

Kaggle notebooks do not provide genuine multi-node access — this is
stated plainly rather than glossed over. Validation of this phase
realistically happens one of two ways: externally-provisioned machines
(a free or low-cost cloud tier) tunnelled together for a real NCCL
multi-node run, or a documented single-machine simulation using
multiple processes over loopback networking, which exercises the code
path's correctness without genuinely separate hardware. Any throughput
numbers reported for this phase are explicitly labelled with which
validation path produced them — a simulation-derived number is never
presented as a real-hardware benchmark.

### Fault Tolerance — Deliberately Minimal

This phase implements exactly one fault-tolerance behavior: a single
retry on a transient NCCL rendezvous timeout, after which the run fails
loudly. Full elastic training (nodes joining/leaving mid-run,
checkpoint-and-resume on node failure) is explicitly out of scope — a
genuinely large feature in its own right that this project does not
attempt to half-implement.

### Implementation (Phase 44, v4.0.0-alpha.4)

The multi-node surface lives in `aarambh-studio-train/src/distributed.rs`:

- `MultiNodeTopology { num_nodes, gpus_per_node, node_rank, local_rank }`
  derives `global_world_size = num_nodes * gpus_per_node` and
  `global_rank = node_rank * gpus_per_node + local_rank`; `is_global_rank0()`
  is true only for the first node's first GPU, so exactly one process
  globally logs and checkpoints.
- `RendezvousTransport` enum (`File` default | `Tcp { endpoint }`):
  `File` reproduces v2 single-node behaviour byte-for-byte; `Tcp` (Phase 44)
  lets genuinely separate nodes exchange the 128-byte NCCL unique id over
  the network.
- `Rendezvous` trait + `FileRendezvous` + `TcpRendezvous`: pure
  standard-library I/O that exchanges a `Vec<u8>` blob, so the entire
  rendezvous layer compiles and is unit-tested on CPU without the `cuda`
  feature. The actual NCCL `Id` only enters at the call site, behind
  `#[cfg(feature = "cuda")]`.
- `RetryPolicy`: one retry on a transient (timeout / connection-refused)
  error, then fail loudly.
- Device-count fix: a multi-node worker needs only `gpus_per_node` devices
  locally (not the full global `world_size`); single-node runs still need
  `world_size`.
- `DistributedConfig` gains `num_nodes`, `node_rank`, `gpus_per_node`,
  `rendezvous`, `retry_attempts`, all defaulting to the single-node v2
  behaviour. Only `num_nodes >= 2` activates multi-node mode, deriving
  `world_size` and `rank` from the topology. Every existing single-node
  config deserialises to byte-identical v2 behaviour.

The gradient all-reduce math (`all_reduce_gradients`, `sync_bucket`,
`all_reduce_flat`) is unchanged from v2 §27 — only the topology it runs
over and the rendezvous that bootstraps it change.

---

## 59. Test-Time Compute Scaling

**Crate:** `aarambh-studio-inference` (`best_of_n.rs`, `self_consistency.rs`, `process_reward.rs`) | **Depends on:** v1 §7 (thinking engine), v2 §29 (speculative decoding), v1 §11/v2 §22 (verifiers)

### A New Axis, Not a Replacement

The thinking engine (v1 §7) controls how many tokens *one* generation
spends reasoning before answering (None/Low/Medium/High/Max —
v3 §48 added Max). Test-time compute scaling is a different axis
entirely: generate **multiple independent candidate completions**, then
select among them. The two compose freely — each of the N candidates
can itself use any thinking mode.

```
Prompt
  │
  ▼
Generate N candidates in parallel (each independently uses the
existing sampler; speculative decoding, v2 §29, accelerates each
candidate independently where enabled)
  │
  ▼
SelectionStrategy:
  │
  ├─ Verifier      → for checkable tasks (math/code): run the existing
  │                   MathVerifier/CodeVerifier (v1 §11, v2 §22)
  │                   against each candidate, select a verified-correct
  │                   one if any exists
  │
  ├─ SelfConsistency → extract each candidate's final answer, majority
  │                   vote across all N — no verifier required, works
  │                   for tasks with a well-defined final answer even
  │                   without a hard checker
  │
  └─ ProcessReward  → a small classifier head, trained on GRPO/DPO-style
                      contrastive step data, scores intermediate
                      reasoning steps (not just final answers) and
                      selects the highest-scoring trace — used for
                      open-ended tasks where neither a hard verifier
                      nor a clean final-answer extraction exists
```

### Cost Model

`N=1` reproduces single-sample generation exactly — this is the
backward-compatibility floor, same as every optional feature in the
project. Larger `N` scales compute roughly linearly (N independent
generations), which is why the roadmap gates larger N to Kaggle and
keeps small N (2–4) available on i3, following the exact budget
precedent v1 §12's self-learning N-completion sampling already
established for CPU-safe operation.

### Measured, Not Assumed

Whether Best-of-N with a given selection strategy actually improves
accuracy on a given task is an eval-harness question (v2 §17), answered
per task via `eval --compare --best-of-n`, not assumed from the
technique's general reputation. Different tasks and selection
strategies are expected to show different — sometimes negligible —
deltas; the scorecard is the source of truth, not the roadmap's prose.

### Implementation (Phase 45, v4.0.0-alpha.5)

The test-time-compute surface lives in three new modules of
`aarambh-studio-inference`:

- `best_of_n.rs`: `SelectionStrategy` enum
  (`Verifier | SelfConsistency | Majority | ProcessReward`), the local
  `CompletionVerifier` trait (kept local so the inference crate does not
  depend on the finetune crate that owns `Verifier` / `MathVerifier` /
  `CodeVerifier` — the CLI binary adapts at the call site),
  `BestOfNConfig`, `BestOfNEngine`, `BestOfNOutput`, and
  `SelectionRationale`. `BestOfNEngine` wraps an `InferenceEngine` and
  reuses `prepare_session` + `fork_with_config` + `decode_sessions` so the
  prompt KV-cache is prefilled once and the N forks are decoded together
  in one batched target forward pass. Candidate 0 inherits the input
  sampler's seed unchanged (N=1 reproduces single-sample byte-for-byte);
  candidates 1..N are re-seeded `base_seed + i`.
- `self_consistency.rs`: `extract_final_number` (byte-identical
  re-declaration of `aarambh_studio_finetune::extract_final_number`,
  attributed, so no cross-crate dependency), `extract_final_answer`,
  `majority_vote` (first-occurrence tie-breaking), and
  `self_consistency_select`.
- `process_reward.rs`: `ProcessRewardScorer` trait,
  `HeuristicProcessRewardScorer` (transparent structural scorer: rewards
  a non-empty thinking block, a final-answer marker, a parsable numeric
  answer, and a non-trivial step count), and `ProcessRewardHead`
  (placeholder for a future trained head; returns
  `AarambhError::Unsupported` until a checkpoint exists — no trained
  checkpoint ships, per the release audit).

The `aarambh-studio-eval` crate gains `best_of_n_generate` /
`sample_generate` / `BestOfNOptions` / `BestOfNResult` in `generation.rs`
and `best_of_n` / `best_of_n_selection` / `best_of_n_seed` fields on
`EvalConfig`. When `best_of_n` is set, the `gsm8k_subset` and
`humaneval_lite` tasks compute both single-sample and best-of-N accuracy
and record `single_sample_accuracy`, `best_of_n_accuracy`, and
`best_of_n_delta` in their `TaskScore::details` map.

The `aarambh-studio` CLI gains `--best-of-n` / `--selection` /
`--ground-truth` on `infer` and `--best-of-n` / `--best-of-n-selection` /
`--best-of-n-seed` on `eval`. Best-of-N is text-only: combining
`--best-of-n` with `--image` / `--video` / `--document` / `--audio` /
`--tools` returns `AarambhError::Unsupported`. The `serve` crate is
unchanged (its `GenerationRequest` wraps `GenerationConfig`, which Phase 45
leaves untouched — the wrapper-struct approach keeps the server surface
clean).

---

## 60. RLAIF

**Crate:** `aarambh-studio-finetune` (`rlaif.rs`) | **Depends on:** v1 §11 (GRPO), v2 §28 (DPO), v1 §12 (self-learning N-completion sampling)

### The Gap It Fills

GRPO (v1 §11) needs a hard verifier — it works when correctness is
checkable (math, code, format compliance). DPO (v2 §28) needs a
preference dataset — static pairs of (chosen, rejected) completions,
whether from a public dataset or hand-labelled. Neither covers
"generate fresh preference signal automatically, for qualities that are
neither checkable nor already labelled" — open-ended chat quality being
the clearest example. RLAIF fills exactly that gap.

```
Self-sample N candidate completions for a prompt (reuses v1 §12's
N-completion sampling infrastructure directly, unchanged)
        │
        ▼
Judge model (a frozen checkpoint — either an earlier stage of the same
model, or the Large scale judging Small/Tiny outputs) scores pairs of
candidates: "which is better, and roughly by how much"
        │
        ▼
Position-swap bias correction: EVERY pair is judged twice, in both
A/B orderings. Judges have a documented first-position bias; when the
two orderings disagree, the pair is down-weighted or discarded rather
than trusted at face value
        │
        ▼
Output: (chosen, rejected) pairs — the EXACT SAME SCHEMA v2 §28's DPO
pipeline already consumes
        │
        ▼
Feed directly into the existing, UNMODIFIED `finetune dpo`/`finetune
qdpo` training path
```

RLAIF is deliberately architected as a **data-generation front end**,
not a new training objective — `dpo_loss` (v2 §28) does not change at
all. This keeps the numerically-stable two-class log-softmax
formulation v2 §28 already got right, rather than re-deriving a new
loss function with its own numerical edge cases.

### Where RLAIF Sits Relative to GRPO and DPO

| Method | Signal source | Best for |
|---|---|---|
| GRPO (v1 §11) | Hard verifier | Math, code, format-checkable tasks |
| DPO (v2 §28) | Static human preference data | Chat quality, where labelled pairs exist |
| RLAIF (v4 §60) | AI judge, self-sampled | Chat quality, where no labelled pairs exist yet |

All three remain available and complementary — v4 does not deprecate or
replace GRPO or DPO, it adds a third signal source for the cases
neither of the first two covers well.

### Measured, Not Assumed

An RLAIF-tuned checkpoint's win rate is reported against the pre-RLAIF
baseline using v2 §28's existing `preference` eval task — an honest
delta, not a claimed win, the same discipline every alignment claim in
this project has held since v1.

### Implementation (Phase 46, v4.0.0-alpha.6)

`aarambh-studio-finetune/src/rlaif.rs` ships:

- `RlaifConfig` (serde, `Default`, `validate`): `n_candidates` (4),
  candidate sampling temperature/top-k/top-p/max-tokens, judge
  max-tokens, `bias_discard` (false), `agreement_margin` (0.1),
  `max_pairs_per_prompt`, base `seed`, judge prompt template.
- `JudgeGenerator` trait — deliberately free of `aarambh-studio-inference`
  types so the finetune crate (Layer 4) does not depend on the inference
  crate (Layer 5), mirroring Phase 45's `CompletionVerifier` layering.
  `generate_verdict(judge_prompt, max_tokens)` takes an already-built
  judge prompt so the finetune crate owns the template logic.
- `CandidateSampler` trait — abstracts v1 §12's N-completion sampling
  pattern (sample N candidates with seeds `base + i`).
- `JudgeVerdict` / `JudgeChoice` (`A`/`B`/`Tie`) / `parse_judge_verdict`
  (robust JSON parse; malformed JSON / unknown `preferred` / non-finite
  margin → neutral `Tie` with margin 0.0, discarded downstream).
- `BiasCorrectedPair` / `AgreementLevel` — `judge_pair_both_orderings`
  judges every pair in both A/B and B/A orderings; `resolve_preference`
  applies position-swap bias correction (agreement → weight 1.0 or
  margin-down-weighted; tie → discarded; disagreement → down-weighted
  to `DISAGREEMENT_WEIGHT` (0.25) or discarded, more-confident ordering's
  verdict wins, equal margins → discarded as ambiguous).
- `generate_rlaif_dataset` — the main entrypoint: sample N candidates per
  prompt, form all `C(N, 2)` pairs, judge both orderings, resolve
  preferences, return `Vec<DpoExample>` + `RlaifSummary`.
- `write_preference_jsonl` — writes the exact `{prompt, chosen, rejected}`
  schema `DpoDataset::from_jsonl` consumes.
- `RlaifPair` carries a `provenance: "rlaif_judge"` marker (§46's
  vocabulary) for downstream replay analysis.

The `InferenceEngine` implementations of `JudgeGenerator`/`CandidateSampler`
live in the CLI binary (`aarambh-studio/src/cmd/finetune.rs`:
`InferenceJudge`, `InferenceSampler`), alongside Phase 45's
`MathVerifierAdapter`. The `finetune rlaif` subcommand wires policy +
judge engines, supports self-judging (`--judge` defaults to `--base`),
and feeds the generated JSONL into the unmodified `finetune dpo` pipeline.
`dpo_loss` (v2 §28) is byte-for-byte unchanged; only the `DpoTrainer.train_loader`
field was widened from private to `pub(crate)` so the RLAIF integration test
in `rlaif.rs` can pull one batch and prove the pairs train successfully.
See `docs/phase46_rlaif.md` for the full runbook and `scripts/phase46_smoke.sh`
for the CPU smoke test.

---

## 61. Tool Execution With Sandboxing

**Crate:** `aarambh-studio-agent` (`sandbox.rs`, `authorization.rs`) | **Depends on:** v2 §30 (grammar-constrained emit-only tool calling), v3 §46 (multi-step chains, still emit-only)

### Closing an Arc, Carefully

The boundary has moved twice already: v2 §30 gave the model
grammar-constrained JSON tool calls it could *emit* but never execute.
v3 §46 let it emit *sequences* of tool calls, using real intermediate
results — but the execution of each call still happened outside
aarambh-studio entirely, by the developer integrating it. v4 §61 is the
first phase where aarambh-studio itself is permitted to execute a tool
call — and it is scoped as narrowly as the risk demands.

### Closed-World Execution

```rust
trait ToolExecutor {
    fn name(&self) -> &'static str;   // exact match required, no
                                        // pattern matching, no fuzzy
                                        // resolution
    fn execute(&self, args: ValidatedArgs) -> Result<ToolResult, ExecError>;
}
```

There is no generic "run a shell command" or "eval this code"
`ToolExecutor` anywhere in the crate, by design — every executor
implements one specific, named capability (e.g. a whitelisted read-only
file lookup within a fixed working directory, or an HTTP GET restricted
to an explicitly whitelisted domain list). An unrecognised tool name is
a **hard refusal**, never a best-effort fallback attempt at
interpreting what the model might have meant.

```
Model emits grammar-constrained JSON tool call (v2 §30)
        │
        ▼
JSON validated against declared schema — malformed or partially-
streamed calls are NEVER executed, matching v2 §30's existing
atomicity guarantee for tool-call output
        │
        ▼
Tool name checked against the closed allowlist:
  not in allowlist  → hard refusal, chain records the refusal, no
                       execution attempt of any kind
  in allowlist       → continue
        │
        ▼
Tool name checked against operator authorization (see below)
  not authorized     → hard refusal
  authorized         → continue
        │
        ▼
Execute inside a bounded envelope:
  - explicit timeout (execution killed if exceeded)
  - explicit memory/CPU ceiling (execution killed if exceeded)
  - no filesystem access beyond what the specific ToolExecutor exposes
  - no network access beyond what the specific ToolExecutor exposes
        │
        ▼
ToolResult re-enters the chain via v3 §46's existing
result_ingestion.rs — no new ingestion mechanism, execution is purely
additive to what v3 already built
```

### Authorization Is an Operator Decision, Not a Model Decision

A model can *declare* any tool in its schema and *request* execution of
anything it declares. Whether that request is ever actually carried out
depends entirely on what the operator explicitly enabled at server or
CLI startup (`authorization.rs`) — the model's own confidence, phrasing,
or reasoning about why a tool call is justified has no bearing on
whether it gets executed. This separation is deliberate: it keeps the
attack surface for "convince the model to execute something dangerous"
bounded by what a human operator pre-approved, not by what the model
can be talked into requesting.

> **Implemented in v4.0.0-alpha.7** as `crates/aarambh-studio-agent/src/{sandbox,authorization}.rs`.
> `SandboxedToolProvider` implements `ToolResultProvider`, so execution plugs
> into the existing `ToolChain` with zero chain changes. See
> `docs/phase47_sandbox.md` for the runbook and the honesty boundary
> (pure-Rust CPU sandbox: wall-clock timeout + output/argument-size ceilings;
> OS-level isolation is out of scope for the source release).

---

## 62. Multi-Agent Orchestration

**Crate:** `aarambh-studio-agent` (`orchestrator.rs`) | **Depends on:** v4 §61 (sandboxed execution) — hard dependency, must ship after

### One Reasoning Process, Several Sandboxed Sub-Chains

```
Orchestrator's own reasoning produces a DelegationPlan: a set of
independent sub-tasks
        │
        ▼
Each sub-task becomes its own sub-chain — a full GenerationSession
(reusing v2 §31's server session abstraction), with:
  - its own sandboxed tool scope (v4 §61)
  - its own execution timeout budget
        │
        ▼
Sub-chains run (conceptually parallel; actual concurrency bounded by
configured limits below)
        │
        ▼
Result aggregation: each sub-chain's output re-enters the
orchestrator's own context via the same ToolResult ingestion pattern
(v3 §46, v4 §61) — applied recursively, not a new mechanism
```

### Hard, Non-Negotiable Bounds

Three ceilings are enforced as operator-set configuration, never as
something the orchestrator's own output can influence:

1. **Maximum sub-agent count** — an orchestrator cannot request
   unbounded fan-out regardless of how it reasons about the task.
2. **Maximum total execution time budget** — the sum across all
   sub-chains, not per sub-chain, so many small sub-agents cannot
   collectively exceed the same ceiling one large one would hit.
3. **Sandbox scope containment** — a sub-agent's authorized tool scope
   can only be a **subset** of what the orchestrator itself was
   authorized for (v4 §61's `authorization.rs`). Orchestration can
   never be used as an escalation path to reach tools the operator did
   not explicitly enable at the top level.

### Failure Isolation

One sub-agent's failure or execution error is contained to that
sub-chain's own result — it does not corrupt or silently swallow
sibling sub-agents' results, and the orchestrator's aggregation step
receives an explicit failure marker for that sub-chain rather than a
missing or malformed entry.

> **Implemented in v4.0.0-alpha.8** as `crates/aarambh-studio-agent/src/
> orchestrator.rs`. Each sub-chain is a `ToolChain` backed by a
> `SandboxedToolProvider` constructed with the sub-task's narrowed
> `AuthorizationScope` (via `AuthorizationScope::intersect`), so
> orchestration plugs into the existing chain with **zero chain changes**
> — sub-chain outputs re-enter the orchestrator's own context via the
> unchanged `result_ingestion` path, applied recursively. The three hard
> bounds (max sub-agent count, total execution time budget, sandbox scope
> containment) are operator-set and enforced at `validate_plan` time
> before any sub-chain runs; failure isolation is via `catch_unwind`.
> Sub-chains run sequentially (CPU-first honest default — true parallelism
> is gated on a future `Send + Sync` `ChainDecoder`, out of scope for the
> source release because the `InferenceEngine` holds a Candle device that
> is not safely cloneable across threads). See
> `docs/phase48_orchestration.md` for the runbook and the honesty
> boundary.

---

## 63. Retrieval-Augmented Generation (RAG)

**Crate:** `aarambh-studio-retrieve` (new) | **Depends on:** none within the model itself — deliberately a prompt-level augmentation, not a model-internals change | **Status:** shipped in v4.0.0-alpha.9 (Phase 49) — `chunking.rs`, `embedding.rs` (`HashingEmbedder` default + `TextEmbedder` trained-head), `index.rs` (navigable small-world graph ANN, pure Rust), `retrieval.rs` (`RetrievalPipeline::query()`, `augment_prompt()`); CLI `retrieve build-index` and `infer --rag --index`; eval `rag` task reporting `no_retrieval_accuracy`/`rag_accuracy`/`rag_delta`; see `docs/phase49_rag.md`.

### Deliberately the Simplest Correct Design

RAG is implemented as **prompt augmentation**, not a new fusion
mechanism inside the decoder. This is a deliberate simplicity choice,
the same instinct that chose prefix fusion over cross-attention for
vision back in v2 §24: retrieved text chunks are spliced into the
existing prompt-construction pipeline (the same code path that already
assembles system prompt + chat history + user turn) as additional
context ahead of the user's question. The decoder never knows the
difference between retrieved context and any other text in its prompt.

```
Document corpus
        │
        ▼
chunking.rs — fixed-size chunks with configurable overlap
        │
        ▼
embedding.rs — a small, dedicated, CONTRASTIVELY-TRAINED text-embedding
head (separate from the main decoder, CPU-capable) turns each chunk
into a fixed-size vector
        │
        ▼
index.rs — a FROM-SCRATCH approximate-nearest-neighbour index
(graph-based, pure Rust, no FFI to an external vector-search library)
        │
        ▼
   [ index persisted to disk, reloadable ]
        │
        ▼
Query time: embed the query -> search the index -> top-k chunks
returned
        │
        ▼
Chunks spliced into the prompt ahead of the user's question, using the
EXISTING prompt-assembly path — no decoder-level change whatsoever
```

### Why From-Scratch, Not a Vector-DB Dependency

Consistent with the project's standing policy (no PyTorch, no ONNX
Runtime, no Python FFI, everything through `candle`), the default and
tested retrieval index is implemented in pure Rust rather than binding
to an external vector-database service. An external vector-DB adapter
may exist as an optional plug-in, but it is not the default path and is
not what the eval-harness retrieval tests validate against.

### Measured, Not Assumed

Retrieval quality (recall on a small labelled holdout) and end-to-end
usefulness (a factual eval-harness task's score with vs without RAG
enabled) are both reported as measured deltas, following the same
discipline every capability claim in this project has held since v2
§17's eval harness first shipped.

---

## 64. Model Merging / Weight Averaging

**Crate:** `aarambh-studio-weights` (`merge.rs`) | **Depends on:** v2 §23 (DoRA), v2 §28 (DPO), v4 §60 (RLAIF), v3 §40 (fine-grained MoE), v3 §42 (distillation) — needs their checkpoint variants to exist

### Why Now, Not Earlier

Model merging is only useful once genuinely different, independently-
trained checkpoint variants exist to merge. Earlier in the project's
history there was nothing meaningful to merge; by v4, DoRA-tuned,
DPO-tuned, RLAIF-tuned, MoE, and distilled variants all exist
side-by-side, making this the first point where merging has real
utility.

### Two Methods

**SLERP (spherical linear interpolation)** between two or more
compatible checkpoints:

```
merged = slerp(checkpoint_a, checkpoint_b, weight)
```

At `weight = 1.0` or `0.0`, this reproduces one input checkpoint
exactly — the backward-compatibility floor for the merge tool itself.

**Task-vector arithmetic**, for combining independently-tuned deltas
onto one shared base:

```
delta_i = tuned_checkpoint_i - base_checkpoint
merged  = base_checkpoint + Σ (scale_i × delta_i)
```

This lets you, for example, combine a math-focused DoRA delta and a
chat-focused DPO delta onto the same base checkpoint, each scaled
independently.

### Hard Validation Before Any Write

Both methods validate tensor shapes and architecture compatibility
**before** producing any output — merging checkpoints from
architecturally incompatible configs (different hidden sizes, different
attention schedules, mismatched vocabularies) fails loudly at the
validation step. This project has never silently produced a corrupted
or nonsensical checkpoint from a mismatched operation, and merging is
no exception.

### Measured, Not Assumed

A merged checkpoint's eval-harness scorecard is reported honestly
against both of its input checkpoints' individual scores — merging is
not assumed to be strictly additive or strictly beneficial, following
the identical "measure, don't assume" framing v2 §26 established for
MoE and every subsequent capability claim has held since.

---

## 65. Public Inference Server + Prefix Caching

**Crate:** `aarambh-studio-serve` (`auth.rs`, `prefix_cache.rs`, `tenant_isolation.rs`) | **Depends on:** v2 §31 (local OpenAI-compatible server)

### The Biggest Risk/Scope Jump in the Project's History

v2 §31 and v3 both kept the inference server local-only, deliberately.
v4 §65 is the first phase that opens it to genuinely multi-tenant,
authenticated traffic — which is why it is sequenced as the very last
feature phase in the entire v1–v4 roadmap, only after the model
underneath it (attention, MoE, agentic tool use) is fully settled.

```
Incoming request
        │
        ▼
auth.rs: API-key validation (replaces v2 §31's simple loopback-
exemption/bearer-token model with real per-key identity)
  invalid/missing key → rejected BEFORE admission into the continuous
                          batcher (v2 §31) — never queued, never
                          partially processed
        │
        ▼
Per-key rate limiting (requests/minute, tokens/minute) — enforced at
the same admission point
        │
        ▼
prefix_cache.rs: hash the prompt's prefix; check for a cached KV state
  hit  → reuse cached KV state, skip recomputing that portion of the
          forward pass
  miss → compute normally, cache the resulting prefix KV state (LRU
          eviction under a configurable memory ceiling)
        │
        ▼
tenant_isolation.rs: per-tenant resource ceilings within the EXISTING
bounded-admission continuous batcher (v2 §31, unchanged) — one
tenant's request burst cannot starve another tenant's already-admitted
requests
        │
        ▼
...rest of the existing v2 §31 serving pipeline, unchanged...
```

### Why Prefix Caching Specifically, and Why Now

Prefix caching is the single highest-leverage serving optimisation for
exactly the traffic pattern v4's own agentic features (§61–62) generate
— repeated system prompts, shared conversation prefixes across many
tool-execution sub-chains from one orchestrator. Placing it in the same
phase as multi-tenant auth is deliberate: the two are the pieces that
make multi-tenant *and* agentic traffic economically and operationally
viable on the same server.

### Explicit Non-Goals

This phase makes public, multi-tenant self-hosting **possible**; it
does not make it the recommended default, and it is not a hosted
product. There is no billing system. There is no horizontal
auto-scaling. The loopback-only, unauthenticated local mode from v2
§31 remains the documented default for single-user, local use — §65
adds a capability, it does not change the recommended starting point.

---

## 66. System Role, Chat-Template Versioning, and Context Management

**Crate:** `aarambh-studio-tokenizer`, `aarambh-studio-safety`, `aarambh-studio-serve`, `aarambh-studio-inference`/`aarambh-studio-agent` | **Depends on:** `<|system|>` token reservation (v1, ID 7), v2 §31 (server), v3 §46/v4 §61–62 (agentic chains), v4 §63 (RAG)

### Formalizing What Was Reserved but Undocumented

The `<|system|>` special token has been reserved at ID 7 since v1.0.0,
but no prior version documented a role, a precedence rule, or a chat-
template interaction for it. v4.0 closes that gap without changing a
single token ID:

```
<|system|>\n{operator-set instructions}\n<|user|>\n{user turn}\n<|assistant|>\n...
```

- **Optional, single-use, leading position.** A session may include at
  most one `<|system|>` turn, placed before any `<|user|>` turn.
  Omitting it entirely reproduces every prior version's
  `<|user|>...<|assistant|>` format exactly — this is purely additive.
- **Loss masking.** `SftTrainer`'s existing rule (mask everything before
  the `<|assistant|>` position) already covers a leading system turn
  correctly by construction — no training-code change was needed, only
  the documentation of why.
- **Precedence over user input.** System-turn content is always
  operator- or application-supplied. `GenerationSession` (v2 §31) never
  derives system-turn content from a user's own message — a user's
  message can only ever occupy the `<|user|>` position, which the
  existing prompt-injection guardrails (`aarambh-studio-safety`) already
  treat as untrusted. This is the system-side half of a defense whose
  user-side half (detecting `"new system prompt:"`-style injection
  attempts inside user input) has existed since v1.

### Chat-Template Versioning

The chat template's shape has changed with every version — v2 added
image tokens, v3 added video/document/tool tokens, v4 adds audio
tokens — with no prior version recording which shape a given checkpoint
expects. v4.0 adds a `chat_template_version` field to tokenizer config
and checkpoint metadata:

```
v1.0.0 template shape → chat_template_version = 1
v2.0.0 (+ image tokens) → chat_template_version = 2
v3.0.0 (+ video/document/tool tokens) → chat_template_version = 3
v4.0.0 (+ system role formalized, + audio tokens) → chat_template_version = 4
```

A server refuses to load a checkpoint whose declared version it does
not recognize, with a clear startup error — never a silent
misinterpretation of an older or newer prompt structure. This is the
same fail-loud-not-silent discipline every hardware and dispatch gate
in this project has held since v2.

### Context-Truncation Policy

Never previously documented, and made a real question (not a
theoretical one) by v4's own long agentic chains (§61–62) and
RAG-augmented prompts (§63):

```rust
enum ContextTruncationPolicy {
    SlidingWindow,  // drop oldest non-system turns first; the system
                     // turn, if present, is NEVER evicted
    Summarize,       // replace evicted turns with a generated summary
                     // turn, reusing the project's existing self-
                     // critique-style summarization capability
    Reject,          // refuse to proceed rather than silently drop
                     // context — the mandatory default for anything
                     // safety- or execution-sensitive, e.g. sandboxed
                     // tool-execution sessions (§61) and orchestration
                     // (§62)
}
```

One policy, referenced consistently by every long-context feature in
the project, rather than each feature inventing its own ad hoc
truncation behavior.

### Sampling Defaults Reference

`docs/SAMPLING_DEFAULTS.md` consolidates temperature/top-p/top-k
guidance — previously scattered informally across three prior
architecture documents — into one canonical table, organized by use
case (deterministic tool-call generation, open-ended chat, creative
writing, math/code verification).

---

## 67. Red-Team / Adversarial Safety Evaluation

**Crate:** `aarambh-studio-safety` (`redteam/`) | **Depends on:** `ARCHITECTURE.md` §13 (safety layer), v4 §61 (sandboxed execution), v4 §65 (public server), v4 §66 (system-role precedence)

### Distinct From Per-Phase Unit Tests

Every phase in this project ships its own unit-level safety tests (a
malformed tool call is rejected, an unauthorized execution is refused,
a PII pattern is redacted). Red-team evaluation is different in kind:
one systematic, end-to-end adversarial pass run against the *complete*
v4.0 surface near the end of the roadmap, specifically because Phase 65
(public server) and Phase 61 (execution) are the two highest-risk
capabilities in the project's history and deserve a dedicated
adversarial pass beyond what any single phase's own tests would think
to cover in isolation.

```rust
struct AdversarialCase {
    input: AdversarialInput,           // text, or a full request shape
                                         // targeting the server/execution
                                         // surface
    expected_outcome: ExpectedOutcome, // Refused | Sanitized | ExecutedSafely
    category: String,                  // e.g. "system_turn_injection",
                                         // "unauthorized_tool_execution",
                                         // "orchestrator_bound_bypass",
                                         // "auth_bypass_attempt"
}
```

**Corpus categories**, each targeting a specific v4.0 boundary:
- Prompt-injection variants specifically targeting the system-turn
  precedence rule (§66) — attempts to make user input masquerade as
  system-level instruction.
- Attempts to get an unauthorized tool executed despite the closed-
  world allowlist (§61).
- Attempts to get an orchestrator (§62) to exceed its configured
  sub-agent count or time-budget ceilings.
- Attempts to bypass authentication or exhaust rate limits on the
  public server (§65).

Every case carries a labelled expected outcome; a failing case is
surfaced plainly in the generated report, never silently excluded —
the same "measure, don't assume" discipline that has governed every
capability claim in this project since v2 §17's eval harness. Corpus
content is hand-authored or drawn from free/public sources only, the
same dataset-licensing policy every training/eval dataset in the
project has followed since v1.

---

## 68. Model Card

**Crate:** `aarambh-studio-eval` (`model_card.rs`) | **Depends on:** v2 §17 (eval harness), v4 §67 (red-team report)

### One Canonical, Assembled — Not Hand-Written — Document

Prior to v4.0, a released checkpoint's capabilities, limitations, and
provenance were described piecemeal across `ARCHITECTURE*.md`,
`ROADMAP*.md`, and `README.md`. `MODEL_CARD.md` consolidates this into
one document per released checkpoint configuration, generated from
real data rather than authored by hand each time:

```
ModelCard {
    intended_use: String,               // static metadata
    training_data: Vec<DatasetEntry>,   // static metadata, license-tagged
    capabilities: EvalHarnessScorecard, // PULLED from an actual eval run
                                          // (v2 §17), never hand-entered
    known_limitations: Vec<String>,     // static + eval-derived
    redteam_summary: RedTeamReport,     // PULLED from v4 §67's actual
                                          // report, never hand-entered
    hardware_requirements: String,      // static metadata
    chat_template_version: u32,         // PULLED from v4 §66's version tag
}
```

Because the capabilities and red-team sections are pulled directly from
real eval-harness and red-team runs rather than typed by hand, a model
card cannot silently drift out of sync with a checkpoint's actual
measured behavior — generation fails loudly if no red-team report is
present for the checkpoint being documented, rather than shipping a
model card with an empty or stale safety section.

---

## 69. Updated Dependency Layers

```
Layer 0: aarambh-studio-core
Layer 1: aarambh-studio-tokenizer, aarambh-studio-data
Layer 2: aarambh-studio-nn, aarambh-studio-kernel
Layer 3: aarambh-studio-model, aarambh-studio-vision, aarambh-studio-audio (NEW)
Layer 4: aarambh-studio-weights, aarambh-studio-quant, aarambh-studio-retrieve (NEW)
Layer 5: aarambh-studio-train, aarambh-studio-finetune, aarambh-studio-inference,
         aarambh-studio-safety, aarambh-studio-selflearn, aarambh-studio-eval,
         aarambh-studio-distill, aarambh-studio-agent
Layer 6: aarambh-studio-serve, aarambh-studio (CLI binary)
```

`aarambh-studio-audio` sits at Layer 3 alongside `aarambh-studio-vision` —
same role, a modality-specific encoder/fusion crate consumed by the
model layer above it. `aarambh-studio-retrieve` sits at Layer 4 — it
depends on tokenization and its own small embedding model, but produces
prompt-level context rather than model-internal state, placing it
alongside the weights/quant utilities rather than inside the core
model stack. Neither new crate required any Layer 0–2 change — the
same pattern every prior modality and utility addition has held since
v2.

**New allowed dependencies, scoped narrowly:**

| Dependency | Allowed crates | Reason |
|---|---|---|
| Permissively-licensed audio decode (pure-Rust or system-library-bound) | `aarambh-studio-audio` | Local mel-spectrogram extraction only — no network calls, no Python audio tooling |
| Small contrastive text-embedding architecture | `aarambh-studio-retrieve` | Loaded as SafeTensors via `candle-core`, same policy as every other encoder in the project |

**Still forbidden everywhere, unchanged since v1:** PyTorch bindings,
ONNX Runtime, Python FFI, `llama.cpp` as a backend. All computation
goes through `candle`. An external vector-database adapter for RAG may
exist as an optional plug-in, but the from-scratch pure-Rust index
remains the default and the tested path.

---

## 70. Updated Memory & Compute Estimates

| Addition | Approx. extra params (Small scale) | Approx. extra memory | Notes |
|---|---|---|---|
| MLA (per retrofitted layer) | Small — down/up-projections are narrow | KV cache **reduction**, not increase, at long context | Net memory win at the cache level despite added projection weights |
| Frozen audio encoder | ~40–90M (frozen, not trained) | Encoder weights + activation memory during forward pass only | Same class of cost as the frozen CLIP encoder (v2 §24) |
| Audio projector | Small (MLP) | Small | Same class as vision's projector |
| Sparse MoE dispatch | No parameter change | GPU: reduced active compute per token vs dense; CPU: unchanged (falls back to dense) | Compute-path change, not a parameter-count change |
| Test-time compute scaling (Best-of-N) | No parameter change | N× generation compute at inference time, scales with N | Inference-time cost only, no training-time or storage cost |
| Process reward model | Small (classifier head) | Small | Trained once, small additional checkpoint |
| RLAIF | No architecture change | Judge-model inference cost during data generation only | No cost at the trained checkpoint itself |
| RAG embedding head | Small, separate model | Small | CPU-capable by design |
| RAG index | N/A (not a model) | Scales with corpus size, disk-persisted | Not part of the model's own memory footprint |
| Model merging | No new architecture | Same footprint as any single input checkpoint | Merging does not increase parameter count |

Same headline as v3 §50: additions are additive and scoped, not
architecture-wide multipliers — MLA's KV-cache section is the one place
in v4 where the net effect is a **reduction**, not an addition.

---

## 71. Updated Hardware Strategy

| Workload | Hardware | Reasoning |
|---|---|---|
| MLA retrofit training | Kaggle | Continued-pretraining recipe, same class of cost as v3 §29's Gated DeltaNet retrofit |
| Audio projector/instruction training | Kaggle | Frozen encoder forward pass + DoRA training, same class as vision (v2 §25) |
| Sparse MoE dispatch (training/inference) | Kaggle (GPU) for the real benefit; i3 falls back to dense automatically | Sparse dispatch's payoff is GPU-batch-scale specific |
| Multi-node training | External multi-VM or documented simulation | Kaggle does not provide genuine multi-node access — stated plainly, not glossed over |
| Test-time compute scaling, small N (2–4) | i3 | Follows v1 §12's existing CPU-safe N-completion budget precedent |
| Test-time compute scaling, larger N | Kaggle | Cost scales with N; larger N gated to free GPU quota |
| RLAIF data generation | Kaggle | Judge-model inference at self-sampling scale |
| Sandboxed tool execution (text/tool-result only) | i3 | Lightweight orchestration overhead |
| Sandboxed tool execution (multimodal results) | Kaggle | Inherits the vision/video/document gate the moment any tool result is non-text |
| Multi-agent orchestration | i3 (orchestration) + Kaggle (any multimodal sub-chain) | Same inheritance rule as execution |
| RAG index build/query | i3 | CPU-capable embedding head and index by design |
| Model merging | i3 | Tensor arithmetic on disk-loaded checkpoints, no training involved |
| Public inference server load testing | Kaggle | Simulated concurrent-tenant load needs real GPU-backed serving to be meaningful |

Same discipline as every prior hardware table: a workload refuses to
start on hardware it isn't gated for, with a clear error message,
rather than silently degrading or producing misleading numbers.

---

## 72. Final Release Contract — Why v4.0 Is the Last Version

v4.0.0 is confirmed as **the final planned version of aarambh-studio as an
application.** This section states the reasoning plainly, the same way
every other design decision in this project has been stated rather than
left implicit.

**What "final" means here.** No `ROADMAP_V5.md`, `ARCHITECTURE_V5.md`,
or `SELF_LEARNING_V5.md` exists or is planned as of this release. The
project's roadmap arc — from v1's foundational pipeline, through v2's
first growth phase (vision, scale, tool calling, serving), v3's second
growth phase (hybrid attention, more modalities, forgetting
diagnostics), to v4's completion phase (attention family finished,
modality coverage finished, alignment toolkit finished, agentic
capability finished, deployment capability finished) — reaches a
natural, deliberately-declared end state here.

**Release policy, corrected and finalised.** v4.0.0 ships as a GitHub
source release. Every workspace crate remains `publish = false`. This
explicitly corrects the direction implied by v3 §40 (which described a
crates.io publish) — aarambh-studio is an **application**, not a **library**,
and the project's actual, confirmed policy going forward is: source
only, no crates.io publish, ever, consistent with v1.0.0's and
v2.0.0's original release policy. No pretrained checkpoint, adapter,
tokenizer, optimizer state, SafeTensors, or GGUF file is released at
any version, including this one.

**What "done" looks like for a from-scratch project.** Not every
software project needs an indefinitely-growing roadmap. v4.0.0
represents the point where the core engineering questions this project
set out to answer — can a complete LLM pipeline, including alignment,
multimodality, efficient serving, and safe tool use, be built from
scratch in Rust without Python — have been answered, demonstrated, and
documented in full, phase by phase, exactly as they were tackled. The
"prove it, document it, measure it before it ships" discipline that
shaped every phase from v1 §0 through v4 §65 applies to this decision
too: v4.0.0 is declared final because the roadmap's own stated goals
are met, not because momentum ran out.
