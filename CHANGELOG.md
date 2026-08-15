# Changelog

> From first principles. From zero. From Rust.

## [4.0.0-alpha.3] - 2026-08-15

### Added

- **Phase 43 — Sparse/Grouped MoE Dispatch:** Resolves the "documented
  future optimisation" carried forward unresolved since v2 §35 and v3's
  out-of-scope list. Each token's forward pass now computes only its
  assigned top-k experts, rather than every expert computing on every
  token and being masked afterward — the dense-masked-matmul trade-off v2
  Phase 22 and v3 Phase 31 deliberately shipped. The output is numerically
  equivalent to DenseMasked (same tokens, same weights, same per-expert
  reduction order), just faster, because the masked-away matmuls never run.
  - New `DispatchKind` enum (`aarambh-studio-core`, alongside `AttentionKind`):
    `DenseMasked` (v2/v3 behaviour, default, CPU fallback, correctness
    reference) | `Sparse` (new). Serialized as `dense_masked` / `sparse`.
  - `MoeConfig` gains a `dispatch: DispatchKind` field defaulting to
    `DenseMasked` for exact backward compatibility with every existing MoE
    checkpoint — old TOML/JSON without the field deserialises to the dense
    path, byte-identical to v2/v3.
  - `sparse_grouped_dispatch` (`aarambh-studio-nn/src/dispatch.rs`): tokens
    are grouped by router assignment into per-expert contiguous batches via
    `arg_sort` (no-grad permutation), then each expert's SwiGLU matmul
    executes only on its assigned token group via `index_select` →
    `expert.forward` → `index_add` scatter. Fully differentiable through
    candle's `gather`/`index_select`/`index_add` ops — router logits, expert
    parameters, and input activations all receive correct gradients.
  - `effective_dispatch_kind`: `Sparse` only activates on a CUDA device; the
    CPU path keeps `DenseMasked` regardless of configuration, documented
    plainly as "GPU only pays off" (the honesty discipline v2 §29 applied to
    speculative decoding's speed claim). The real throughput win lives on
    CUDA, where candle routes the per-expert gather/matmul/scatter to cuBLAS
    — a genuine grouped-GEMM path that skips non-routed experts.
  - `MoeFfn::dispatch_kind()` exposes the configured kind; the forward path
    selects dense vs sparse by the effective kind. QAT calibration
    (`forward_with_capture`) always uses the dense reference to observe full
    per-expert activation distributions.
  - Load-balancing auxiliary loss unchanged — computed in `top_k_gating`
    before dispatch, so identical for both kinds. Sparse changes the compute
    path only, not the loss the router is trained against.
  - New configs: `configs/moe_sparse_smoke.toml` (CPU smoke, dense fallback),
    `configs/large_sparse_moe.toml` (Kaggle GPU, the throughput win); new
    script `scripts/phase43_smoke.sh`; new doc `docs/phase43_sparse_moe.md`.
  - Tests: `sparse_dispatch_output_matches_dense_masked_reference_within_tolerance`
    (max abs diff < 1e-5), `dispatch_kind_dense_masked_is_bit_identical_to_v2_v3_behaviour`
    (diff == 0.0), `sparse_dispatch_supports_top_k_greater_than_one`,
    `sparse_dispatch_backward_reaches_router_and_expert_weights`,
    `sparse_dispatch_empty_expert_group_is_skipped`,
    `sparse_dispatch_matches_dense_with_shared_expert_summed_separately`,
    `load_balancing_loss_value_is_unaffected_by_dispatch_kind`,
    `sparse_configured_moe_falls_back_to_dense_masked_on_cpu`,
    `effective_dispatch_kind_uses_sparse_on_cuda` (CUDA-gated),
    `sparse_dispatch_cuda_throughput_exceeds_dense_masked_at_kaggle_gpu_scale`
    (wall-clock, CUDA-gated, skips on CPU), `moe_config_dispatch_defaults_to_dense_masked`,
    `moe_config_dispatch_serializes_as_snake_case`.

### Changed

- Workspace version bumped `4.0.0-alpha.2` → `4.0.0-alpha.3` (matches the
  Phase 43 milestone tag `v4.0.0-alpha.3`).

## [4.0.0-alpha.2] - 2026-08-09

### Added

- **Phase 42 — Audio Modality:** A fourth input sense — audio — following the
  exact frozen-encoder-plus-trainable-projector pattern v2 §24 established for
  vision and v3 §35–36 reused for video and documents. A frozen, pretrained
  audio spectrogram transformer converts a log-mel spectrogram into a grid of
  patch embeddings; a trainable projector maps them into the decoder's
  `d_model` space; the result is spliced into the token sequence at the
  `<audio>` special token position. Nothing about the decoder, the thinking
  engine, or tool calling changes.
  - New `aarambh-studio-audio` crate (Layer 3, workspace package count
    19 → 20): `FrozenAudioEncoder` (AST-style patchify + transformer blocks,
    `load_pretrained` via `VarBuilder::from_mmaped_safetensors`, same loading
    path as CLIP), `AudioPreprocessor` (pure-Rust WAV decode for PCM
    8/16/24/32-bit + 32/64-bit float, linear resampling, Hann window, radix-2
    Cooley-Tukey FFT, triangular mel filterbank, log-mel normalization — zero
    new dependencies), `AudioProjector` (two-layer GELU MLP mirroring
    `VisionProjector`), `interleave_audio_tokens`, `AudioQaExample` +
    `load_audio_qa_jsonl` (caption, QA, and LLaVA conversation formats).
  - Tokenizer: `<audio>`/`<audio_end>` reserved special tokens (IDs 15/16),
    `AUDIO_SPECIAL_TOKENS` table (17 entries, strict superset of the Phase 36
    document table), `BpeTokenizer::validate_audio_special_tokens` and
    `upgraded_for_audio` (insertion at ID 15, +2 rows, shifts learned IDs ≥ 15).
  - `convert --upgrade-audio-vocab` applies the vocabulary migration to a
    SafeTensors checkpoint and tokenizer together
    (`VocabularyExpansion { insertion_id: 15, source_ids: [IMAGE_ID, IMAGE_END_ID] }`).
  - `[vision.audio]` config block (`AudioTrainingConfig`: audio_root, encoder
    config/weights paths, `mel: MelSpectrogramConfig`, encoder_batch_size,
    feature_cache_entries) under the shared `[vision]` multimodal block.
  - `finetune audio-dora` / `audio-qdora` subcommands: a self-contained
    two-stage DoRA trainer (`run_audio_vlm_dora_from_config`) reusing
    `DoraAarambhModel`, `AdamW::from_varmap`, `CosineScheduleWithWarmup`,
    `accumulate_for_optimizer`, `cross_entropy_loss`, and `save_adapter`, with
    audio-specific `audio_example_loss` / `audio_labels_and_mask` and an
    `audio_adapter_config.json` artifact.
  - `infer --audio <path>` flag (conflicts with `--image`/`--video`/`--document`)
    with `AudioRuntime`, `build_audio_prompt_embeddings`,
    `project_audio_tokens`, `run_audio_infer`, and an `AudioSafetyAdapter`
    implementing `SafetyGenerator`.
  - `eval --tasks audio-qa` / `audio-qa-smoke` task implementing the shared
    `EvalTask` trait.
  - New configs: `configs/audio_qa_smoke.toml`,
    `configs/audio_qa_smoke_infer.toml`; new scripts
    `scripts/phase42_make_audio_smoke_fixture.py`,
    `scripts/phase42_prepare_audio_data.sh`, `scripts/phase42_smoke.sh`; new
    doc `docs/phase42_audio.md`.
  - Tests: `frozen_audio_encoder_never_receives_gradients`,
    `projector_pretrain_stage_trains_only_projector_weights`,
    `audio_token_fusion_produces_expected_sequence_length`,
    `thinking_controller_behaves_identically_after_audio_context`, plus
    WAV-decode, FFT-frequency-recovery, mel-frame-counting, projector-width,
    fusion-placeholder, JSONL-parsing, and tokenizer-table-consistency tests.
  - Audio **understanding** only (no generation); WAV PCM only (no MP3/FLAC/Ogg);
    mel extraction is pure-Rust from first principles. The same explicit-scope
    discipline every prior modality phase holds.

## [4.0.0-alpha.1] - 2026-07-31

### Added

- **Phase 41 — Multi-Head Latent Attention (MLA):** A third attention kind
  (`AttentionKind::LatentMLA`) joins Full and Gated DeltaNet in the
  `HybridAttentionSchedule`, completing the attention family v3 began (linear,
  sparse, latent-compressed). MLA layers cache a single low-rank latent vector
  (`c_kv`, width `latent_dim`) plus a small dedicated rotary key slice
  (`rope_head_dim`) per token, reconstructing per-head keys and values at
  attention time through trained up-projection weights that are never cached.
  - New `aarambh-studio-nn::mla` module (`MlaAttention`, `MlaCache`) with
    decoupled RoPE (nope half from the latent, rope half separately cached),
    inference/training/batched-decode/capture paths, and QAT-wrapped
    projections (`QatTarget::Mla`).
  - `HybridAttentionSchedule` extended with `mla_layers: Vec<usize>` and
    `mla: Option<MlaConfig>`; `mla_layers` takes precedence over the
    `full_attention_every_n` rule and the DSA override. A schedule with an
    empty `mla_layers` reproduces v3.0.0 exactly.
  - New `MlaConfig` (`latent_dim`, `nope_head_dim`, `rope_head_dim`, `n_heads`,
    `value_head_dim`) with dimension derivation and validation.
  - Model integration: per-layer MLA build, `HybridKvCache::Mla` allocation,
    named-tensor export (`blocks.{i}.mla.*`), and weight lookup.
  - Partial-checkpoint retrofit extended: `.mla.` tensors are freshly
    initialised alongside `.deltanet.` and `.dsa.` while every shared tensor
    loads bit-exactly (`RetrofitLoadReport.initialized_mla_tensors`).
  - `aarambh-studio eval --kv-cache-report` prints per-layer bytes/token by
    attention kind (no checkpoint required).
  - New configs: `configs/mla_smoke.toml`, `configs/medium_hybrid_mla.toml`,
    `configs/large_hybrid_mla.toml`; new scripts
    `scripts/phase41_prepare_mla_retrofit.sh`, `scripts/phase41_smoke.sh`;
    new doc `docs/phase41_mla.md`.
  - For the Medium hybrid MLA config, MLA per-token cache = 528 elements vs the
    1024-element GQA baseline — a ~1.94× reduction on retrofitted layers at
    long context.

## [3.0.0] - 2026-07-25

### Added

- Completed the v3 engineering roadmap across Phases 29–40:
  - **Gated DeltaNet + DeepSeek Sparse Attention (29–30):** Hybrid linear attention
    with decayed delta-rule recurrence and learned block-sparse attention for a
    mostly-linear, minimally-dense attention stack with zero fully-dense layers
    by default (dense available as a config fallback).
  - **Fine-grained MoE with shared expert (31):** Many small experts instead of
    few large ones plus always-active shared experts, with router warm-start
    from v2 coarse router weights.
  - **Multi-Token Prediction (32):** Auxiliary heads predicting multiple future
    tokens per position; doubles as a free draft source for speculative decoding.
  - **On-policy distillation (33):** New `aarambh-studio-distill` crate with student
    rollouts scored by a larger teacher, reducing train/inference distribution
    mismatch (KL-style and reward-style scoring behind a shared trait).
  - **Native QAT (34):** Device-native FakeQuantize for INT4/INT8 with identity
    STE, export parity with Q4_K_M blocks, and QAT-wrapped projections across
    attention, FFN, MoE, DeltaNet, DSA-indexer, and MTP heads.
  - **Native video and document understanding (35–36):** H.264 MP4 decoding,
    temporal fusion, PDF rasterisation, layout-aware projection, and shared VLM
    DoRA tuning — one training code path, three data types (image, video,
    document).
  - **Long-horizon tool-use chains (37):** New `aarambh-studio-agent` crate with
    multi-step tool orchestration, typed result ingestion (text/image/video/
    document), explicit step budgets, and multi-step SFT.
  - **Forgetting diagnostics (38):** Persistent per-capability forgetting curves,
    probe manifests, routing-drift diagnostics, and a shared JSONL export format
    directly importable by Manas's anti-forgetting tracking.
  - **Max thinking mode (39):** Fifth `ThinkingMode::Max` variant with a
    16,384-token budget, per-mode sampling defaults, and centralised
    `none|low|medium|high|max` vocabulary across all CLI commands.
  - **v3.0.0 source release (40):** CHANGELOG, RELEASE.md, release notes,
    CI extensions, and documentation — `publish = false`, crates.io deferred
    to v4.
- Added `.github/release-notes/v3.0.0.md` and an updated release workflow.
- Added Thinking Modes table to README.md (budget, sampling defaults, use cases).
- Added `HardProblemsComparison` helper for High-vs-Max accuracy validation.
- Added regression and structural tests for Max mode (budget, force-close,
  parser, GRPO rollouts, comparison logic).

### Changed

- Set the complete 19-package workspace to version 3.0.0 through shared
  package metadata (all remain `publish = false`; crates.io deferred to v4).
- RELEASE.md updated to v3.0.0 runbook with `agent`, `distill` CLI commands and
  `scripts/phase40_release_audit.sh`.
- CI extended to cover `aarambh-studio-agent`, `aarambh-studio-distill`, and the
  `agent`, `distill` CLI subcommands.
- `ROADMAP_V3.md` Phase 40 retitled from "crates.io Publish" to "v3.0.0 Source
  Release" with crates.io tasks removed.
- Updated architecture and roadmap docs for the completed v3 release line.
- v3.0.x is now the supported release line; v2.0.x is no longer maintained.

### Guarantees

- All v1/v2 dense model configurations remain valid when v3-only fields are
  absent.
- Existing v1/v2 tokenizer IDs, checkpoint paths, and adapter formats remain
  supported.
- v3 source release follows the same "no pretrained checkpoints, no model
  artifacts" policy as v1.0.0 and v2.0.0.
- Every exported Rust API is documented (`missing_docs` denied at the crate
  level) and every `unsafe` block has an explicit safety rationale.

### Release Policy

- v3.0.0 is a GitHub application source release only.
- All 19 workspace packages remain `publish = false`; crates.io publishing is
  deferred to v4.0.0.
- No pretrained checkpoints, adapters, tokenizer artifacts, GGUF files,
  optimizer state, or compiled CPU/CUDA binaries are attached.

## [3.0.0-alpha.11] - 2026-07-25

### Added

- **Phase 39 Max Thinking Mode**
  - Added a fifth `ThinkingMode::Max` variant with a 16,384-token nominal
    budget — the next step in the existing ~4x progression
    (0 → 256 → 1,024 → 4,096 → 16,384) and not a new reasoning algorithm.
  - Centralised thinking-mode parsing and display on `ThinkingMode` itself
    (`FromStr` + `Display`) so every CLI command, the serving API, GRPO, and
    distillation share one canonical `none|low|medium|high|max` vocabulary.
  - Added per-mode sampling defaults
    (`ThinkingMode::default_sampler()`): None (0.70/0.90), Low (0.75/0.92),
    Medium (0.80/0.95), High (0.80/0.95), Max (0.85/0.97). The server applies
    them only when the caller omits `temperature`/`top_p`; explicit parameters
    are never overridden.
  - Added `GrpoThinkingMode::Max` and `DistillThinkingMode::Max` mirroring the
    canonical variant, with no reward-shaping or objective changes.
  - Added a `thinking_mode` field to `EvalConfig` and an `aarambh-studio eval
    --thinking max` flag, plus a thinking-aware greedy generation helper in the
    eval harness that reuses the inference crate's `ThinkingController`.
  - Added a deterministic `hard-problems` eval task
    (`data/eval/hard_problems/data.jsonl`) that reports accuracy plus average
    thinking, completion, and total token counts for High-vs-Max comparison.
  - Added `scripts/phase39_smoke.sh` (infer/agent/eval High-vs-Max) and three
    optional Kaggle helper scripts for GRPO, distillation, and comparison.
  - Added `docs/phase39_max_thinking_results.md`.

### Changed

- `infer`, `agent`, `serve`, `finetune grpo`, `distill train`, `selflearn
  start`, and `eval` now all accept `--thinking max` through the same parser.
- The serving API accepts `reasoning_effort: "max"` and rejects unknown values.
- Runtime budget clamping is unchanged: the effective thinking budget still
  respects `max_new_tokens`, the answer reserve, and the model `max_seq_len`.
- Workspace packages now share version `3.0.0-alpha.11` and remain
  `publish = false`.

### Guarantees

- Max mode introduces zero structural changes to `ThinkingController` — the
  same `ForceOpen`/`ForceClose` forced-token mechanism, budget tracking, and
  collapse-on-force-close behaviour every existing mode already has.
- `None`/`Low`/`Medium`/`High` behaviour is byte-for-byte unchanged after Max
  is added (covered by regression tests).

### Tests (this commit)

- Added `existing_none_low_medium_high_modes_are_byte_for_byte_unchanged`
  regression test verifying budgets, sampling defaults, controller behaviour,
  parsing, and display for all four original modes are unaltered.
- Added `max_mode_accuracy_on_high_mode_unsolved_holdout_exceeds_high_mode_baseline`
  structural test validating the High-vs-Max accuracy comparison logic with a
  `HardProblemsComparison` helper.
- Added `max_mode_grpo_rollout_thinking_budget_is_16384`,
  `max_mode_grpo_rollout_force_closes_at_budget`, and
  `grpo_max_mode_rollout_budget_clamped_to_max_new_tokens` tests for GRPO
  Max-mode rollout thinking budget enforcement.
- Added `hard_problems_comparison_delta_is_max_minus_high` and
  `hard_problems_comparison_max_does_not_exceed_high_when_equal` tests for the
  comparison helper.

## [3.0.0-alpha.10] - 2026-07-24

### Added

- **Phase 38 Forgetting Diagnostics Tied to Manas**
  - Added a validated eight-capability probe manifest backed by existing
    math, code, reasoning, factual, vision, video, document, and tool-use
    evaluation tasks.
  - Added persistent multi-point forgetting curves with signed deltas,
    configurable significance, manifest/tokenizer fingerprints, atomic writes,
    idempotent point recording, and explicit unavailable-probe reporting.
  - Extended JSON and Markdown scorecards with capability deltas, skipped
    probes, and per-example MoE routing-drift summaries.
  - Added a read-only standard-training observer and post-commit
    self-learning hooks for inline GRPO, deferred-gradient flush, and replay
    updates.
  - Added standalone eval flags, `selflearn forgetting-report`, a CPU smoke
    config, preparation/smoke scripts, and a complete operating guide.
  - Added the exact seven-field
    `schemas/manas-forgetting-v1.schema.json` JSONL interchange contract.

### Changed

- Workspace packages now share version `3.0.0-alpha.10` and remain
  `publish = false`.
- MoE forwards expose sorted routed-expert sets for diagnostic collection;
  dense models retain the existing path without routing traces.
- Distributed training exposes a synchronization barrier so rank-0 probes
  cannot race subsequent optimizer work.
- v3 architecture and self-learning docs now describe the implemented
  adapter/KL/replay safeguards and no longer claim nonexistent gradient
  orthogonalization.

### Guarantees

- Forgetting probes are measurement-only: they do not alter loss, gradients,
  optimizer state, replay policy, or persisted model weights.
- Aarambh Studio has no source, runtime, or filesystem dependency on the sibling
  Manas project. JSONL transfer is explicit and operator controlled.
- The alpha ships source code and fixtures only; it includes no pretrained
  checkpoints and makes no capability-retention quality claim.

## [3.0.0-alpha.9] - 2026-07-24

### Added

- **Phase 37 Long-Horizon Tool-Use Chains**
  - Added `aarambh-studio-agent` with bounded repeated tool decisions, exact-token
    transcript state, explicit stop/max-step behavior, drop-oldest and
    summarising context policies, typed result validation, stdin ingestion,
    and deterministic replay.
  - Added `aarambh-studio agent` with strict safety by default, human and JSONL
    events, caller-controlled result roots, and immediate-next-turn native
    image/video/document result projection.
  - Added multi-step tool SFT masking across every call and the final answer,
    plus `tool-chain`/`agent-chain`/`bfcl-multistep` response-path evaluation.
  - Added three-call SFT/replay/eval fixtures, a source-only smoke script, a
    BFCL v1.3 explicit-response-path normalizer, and a complete runbook.

### Changed

- Virtual JSON tool tokens are encoded by one shared tokenizer protocol used
  by inference and fine-tuning.
- Inference accepts exact token transcripts and tool-constrained multimodal
  embedding prefixes for chain continuation.
- The workspace contains 19 non-publishable packages and is versioned
  `3.0.0-alpha.9`.

### Verified

- Focused tests cover stopping, max-step enforcement, replay mismatch,
  context eviction, result validation, shared virtual JSON, multi-step masks,
  and evaluation normalization.
- The fixture proves protocol and metric plumbing only. Held-out multi-step
  success requires a trained checkpoint and is not claimed by this alpha.

## [3.0.0-alpha.8] - 2026-07-23

### Added

- **Phase 36 Native Document Understanding**
  - Added resource-bounded PDF rendering through pinned pure-Rust Hayro 0.4,
    ordered scanned-page ingestion, aspect-preserving white-pad preprocessing,
    explicit page selection, and detached frozen-encoder feature caching.
  - Added learned or sinusoidal 2D row/column layout projection, canonical
    `<document>`/`<document_end>`/`<page_sep>` fusion, and deterministic
    tokenizer/SafeTensors vocabulary migration for IDs 12-14.
  - Added document DoRA/QDoRA instruction tuning through the shared VLM
    trainer, saved layout artifacts, CLI PDF inference with streaming safety,
    and `document-qa`/`docvqa` evaluation with ANLS, exact match, and optional
    table-subset metrics.
  - Added DocVQA-style JSONL normalization, a dependency-free four-PDF fixture,
    smoke configs, a complete end-to-end smoke script, and a Phase 36 runbook.

### Changed

- Image, video, and document instruction tuning now share optimizer
  accumulation, clipping, masking, artifact cadence, and DoRA model updates.
- The workspace version is now `3.0.0-alpha.8`.

### Verified

- CPU compile coverage includes PDF rendering, multimodal fusion, training,
  inference, and ANLS evaluation paths. Focused tests cover page bounds,
  2D positions, document token migration, page separators, masking, and ANLS.
- The smoke workflow validates mechanism and artifact plumbing only; useful
  document-answering quality requires real training and is not claimed.

## [3.0.0-alpha.7] - 2026-07-19

### Added

- **Phase 35 Native Video Understanding**
  - Added native H.264 MP4 decoding through bundled OpenH264, deterministic
    uniform and scene-aware fixed-count frame sampling, and a bounded cache of
    detached frozen-CLIP frame features.
  - Added learned and sinusoidal temporal position encodings, canonical
    `<video>`/`<video_end>`/`<frame_sep>` fusion, and exact single-frame
    compatibility with the existing image path.
  - Added normalized video-QA JSONL and official NExT-QA CSV loading, shared
    image/video DoRA and QDoRA instruction tuning, CLI video inference, and
    `video-qa`/`nextqa` evaluation tasks.
  - Added deterministic tokenizer and SafeTensors vocabulary migration for
    video token IDs 9-11, batched frame preprocessing/encoding, smoke configs,
    generated clips, an end-to-end smoke script, and a Phase 35 runbook.

### Changed

- VLM instruction tuning now uses one multimodal trainer for image and video
  examples; optimizer accumulation, clipping, artifact saving, and masking
  remain shared instead of being duplicated by modality.
- The workspace version is now `3.0.0-alpha.7`.

### Verified

- Token migration preserves every legacy token and clones compatible image
  rows for the new video markers; old image tokenizers retain their previous
  marker behavior until explicitly migrated.
- Unit coverage checks frame selection, scene boundaries, H.264 NAL parsing,
  temporal ordering, single-frame parity, video masking, and NExT-QA parsing.
- The local smoke workflow covers video generation, native decode, migration,
  two-step VLM tuning, video inference, and evaluation without claiming useful
  model quality from a two-step fixture.

## [3.0.0-alpha.6] - 2026-07-18

### Added

- **Phase 34 Native Quantization-Aware Training**
  - Added device-native INT4/INT8 fake quantization with an identity
    straight-through estimator and no host tensor round-trip.
  - Added exporter-aligned Q4_K_M 256-value blocks with f16 scale/min storage,
    global Q8 absmax simulation, per-tensor and per-output-channel policies,
    and forced Q8 DSA indexers under the export-aligned policy.
  - Added `QatLinear` coverage for attention, dense/routed/shared FFNs, MoE
    routers, Gated DeltaNet, DSA indexers, MTP heads, and the optional LM head.
  - Added one fake-quantized weight cache per projection and optimizer
    generation, exact SafeTensors initialization, QAT-policy checkpoint
    persistence, strict resume matching, and QAT coverage/cache metrics.
  - Added `eval --qat-compare` for matched baseline-FP, baseline-quantized,
    QAT-FP, and QAT-quantized scorecards with normalized robustness recovery.
  - Added CPU smoke and Tiny continuation configs, smoke/comparison scripts,
    Criterion benchmarks, and a Phase 34 implementation guide.

### Changed

- Calibration dataset/model iteration now lives in the CLI, leaving
  `aarambh-studio-quant` below model assembly and preventing a quant/model
  dependency cycle.
- Normal `AarambhModel::new` construction remains full precision even when a
  config records QAT history; only `new_for_training` activates fake
  quantization.
- The workspace version is now `3.0.0-alpha.6`.

### Verified

- Q4_K_M and Q8 fake-quantized values match the existing GGUF exporters,
  including padded Q4 tails.
- STE gradients are identity-valued, QAT caches refresh exactly once per
  optimizer generation, and a two-step trainer smoke produces finite losses
  and projection updates.
- Non-QAT configs preserve the existing construction path; old model JSON
  defaults `qat` to disabled.

## [3.0.0-alpha.5] - 2026-07-18

### Added

- **Phase 33 On-Policy Distillation**
  - Added `aarambh-studio-distill` with student-owned rollout collection, packed
    completion replay, local-checkpoint and scored-reference teacher backends,
    token-level forward KL, and group-normalized reward-policy objectives.
  - Added full-student AdamW training with MTP, MoE, and periodic DSA auxiliary
    loss blending, gradient accumulation/clipping, deterministic prompt order,
    exact model/optimizer/cursor resume, finite metrics, and final checkpoints.
  - Added static teacher-completion preparation and completion-only offline
    distillation as a matched control, plus fresh-rollout JSON/Markdown
    evaluation reports.
  - Added the `distill train`, `distill prepare-offline`, `distill
    train-offline`, and `distill evaluate` CLI workflows.
  - Added CPU smoke fixtures, Medium/Large CUDA recipes, corpus prompt
    preparation, a complete release smoke, and a matched on-policy/offline
    comparison harness.

### Changed

- Inference sessions can expose canonical prompt tokenization and fork an
  untouched prefilled cache into independent bounded samplers, avoiding one
  prompt prefill per rollout while retaining the existing decode path.
- The workspace version is now `3.0.0-alpha.5`.

### Verified

- Teacher logits are detached and gradients flow only through student logits;
  prompt, forced-token, and padding positions are excluded from replay loss.
- Local and dataset teachers, deterministic grouped rollout generation,
  forward KL, reward advantages, checkpoint manifest validation, and exact
  prefill-fork equivalence pass focused tests.
- The release binary completes local soft-KL training, scored-reference reward
  training, offline preparation/training, fresh-rollout evaluation, and exact
  resume on the checked-in Phase 33 smoke fixture.

## [3.0.0-alpha.4] - 2026-07-16

### Added

- **Phase 32 Multi-Token Prediction (MTP)**
  - Added optional MTP-2/MTP-3 future-token heads with independent
    normalization, one dense causal refinement block per offset, and a shared
    main LM-head projection.
  - Added offset-aligned auxiliary cross-entropy, mean auxiliary weighting,
    per-head training metrics, finite-loss checks, and two-step optimizer
    coverage.
  - Added one-checkpoint exact speculative decoding. Bare `--speculative` uses
    MTP heads; supplying draft model/config paths retains external speculation.
  - Added SafeTensors/GGUF head persistence and dense-checkpoint retrofit with
    complete-set initialization and partial-set rejection.
  - Added CPU smoke, Medium/Large continuation configs, matched training
    comparison, throughput benchmark, and Phase 32 implementation guide.

### Changed

- Cached model forwards can return final hidden states with logits, allowing
  MTP proposal heads to reuse one trunk prefill without an auxiliary KV cache.
- Fine-tuning projection selection freezes MTP tensors while preserving them
  through adapter merge workflows.
- Speculative statistics now identify external-draft versus MTP proposals and
  report auxiliary-head forward counts.
- Workspace version is now `3.0.0-alpha.4`.

### Verified

- MTP-disabled model and loss compatibility, output shapes, offset alignment,
  finite gradients, and auxiliary parameter updates.
- Exact greedy equivalence between ordinary and MTP speculative generation,
  committed-token callback behavior, safety integration, and context limits.
- SafeTensors/GGUF round trips, dense-to-MTP retrofit fidelity, and rejection
  of incomplete MTP checkpoint tensor sets.

## [3.0.0-alpha.3] - 2026-07-16

### Added

- **Phase 31 DeepSeek-style fine-grained MoE with shared experts**
  - Added configurable coarse-expert subdivision with conserved routed
    capacity, scaled top-k active width, and validation for exact expert-width
    divisibility.
  - Added always-active shared SwiGLU experts with independent checkpoint
    namespaces, calibration capture, differentiable training, and exclusion
    from routed load-balancing statistics.
  - Added function-preserving coarse-to-fine SafeTensors retrofit: replicated
    router rows, partitioned expert channels, scaled child down projections,
    and zero-output shared-path initialization.
  - Added matched coarse and fine-grained Medium/Large recipes, a combined CPU
    smoke config, and an expert-count sweep that emits evaluation scorecards
    and baseline-relative reports.

### Changed

- MoE dense dispatch now accumulates weighted expert outputs incrementally,
  avoiding a stacked all-expert output tensor while retaining the documented
  dense-compute behavior.
- Training and inference diagnostics now report routed pool size, active routed
  width, shared experts, fine expert width, utilization range, dead experts,
  and active parameter estimates.
- Workspace version is now `3.0.0-alpha.3`.

### Verified

- Exact Phase 22 default compatibility and unconditional shared-output
  addition without auxiliary-loss contamination.
- Shared-expert backward gradients, two-step fine-grained training, model
  tensor/capture naming, and fine-grained GGUF round trips.
- Coarse-to-fine retrofit output fidelity with a zero-start shared output path.

## [3.0.0-alpha.2] - 2026-07-16

### Added

- **Phase 30 DeepSeek Sparse Attention (DSA)**
  - Added learned block indexing on Phase 29's scheduled GQA layers, causal
    top-k block selection, mandatory current-block access, and exact dense
    fallback below the configured threshold.
  - Added compact pooled index-key cache state with cached prefill, decode,
    batched generation, and speculative snapshot compatibility.
  - Added periodic dense-attention-mass distillation, listwise KL indexer loss,
    top-k recall, selected-block/token counters, and dense-fallback metrics.
  - Added Rayon online-softmax sparse attention plus CUDA top-k, sparse forward,
    and teacher-mass PTX kernels for F32, F16, and BF16.
  - Added Phase 29 checkpoint retrofit, SafeTensors/GGUF round trips, Q8 indexer
    preservation, CPU/CUDA smoke configs, Medium/Large recipes, and 4K/16K/32K
    comparison tooling.

### Changed

- The default v3 hybrid schedule now turns its remaining full-attention slots
  into DSA layers when `[model.dsa_config]` is present.
- Inference `--stats` reports DSA stored-cache bytes separately from the
  selected K/V working set. DSA reduces compute and memory bandwidth, while
  total K/V storage remains linear in context length.
- Workspace version is now `3.0.0-alpha.2`.

### Verified

- Exact short-context dense fallback and full-sequence/cached sparse parity.
- Causal deterministic block selection and indexer-only teacher gradients.
- Phase 29 retrofit fidelity and DSA SafeTensors/GGUF compatibility.
- Two-step CPU training coverage for teacher and sparse-only optimizer steps.

## [3.0.0-alpha.1] - 2026-07-15

### Added

- **Phase 29 Gated DeltaNet hybrid linear attention**
  - Added configurable per-layer scheduling that keeps every Nth layer on the
    existing GQA/RoPE path and converts the remaining layers to Gated DeltaNet.
  - Added causal depthwise q/k/v convolutions, normalized q/k features,
    learnable decay and delta-rule gates, output gating, and fixed recurrent
    state with constant decode memory.
  - Added CPU-parallel and CUDA recurrent-update kernels with portable Candle
    fallback behavior and a Criterion recurrence benchmark.
  - Added hybrid cache snapshots for exact speculative rollback and independent
    batched decode state.
  - Added dense v2 SafeTensors retrofit loading, reduced-LR continued training,
    hybrid GGUF handling, calibration capture, and LoRA/QLoRA/DoRA/QDoRA targets.
  - Added CPU/CUDA smoke configs, Medium/Large 16K/32K retrofit configs,
    associative-recall evaluation data, and long-context benchmark scripts.

### Changed

- Training RMSNorm now uses Candle's differentiable implementation; the prior
  no-backward custom operation stopped gradients at normalization boundaries.
- Inference, serving, speculative decoding, quantisation, adapter merging, and
  checkpoint inspection now understand mixed full-attention/recurrent layers.
- Workspace version is now `3.0.0-alpha.1`; v2.0.0 remains the production
  source release while the v3 roadmap is implemented.

### Verified

- Full-sequence versus cached-token hybrid parity and constant recurrent-state size.
- Dense and hybrid last-block gradient propagation, including Gated DeltaNet LoRA/DoRA adapters.
- Partial checkpoint copy fidelity and hybrid GGUF round-trip behavior.
- CPU scalar/kernel parity and optional CUDA/CPU recurrence parity.

## [2.0.0] - 2026-07-12

### Added

- Completed the v2 engineering roadmap across Phases 16–28:
  - YaRN/NTK/linear long-context RoPE scaling and progressive context schedules
  - Evaluation harness with language, reasoning, preference, vision, and tool tasks
  - DoRA/QDoRA, VLM DoRA/QDoRA, DPO/QDPO, and tool-call fine-tuning
  - Frozen CLIP-style vision encoder, projector, image fusion, VQA training, and vision self-learning
  - Mixture-of-Experts FFNs and single-node NCCL data-parallel training
  - Exact speculative decoding and grammar-constrained function calling
  - Axum 0.8.9 OpenAI-compatible inference server with continuous batching
- Added `.github/release-notes/v2.0.0.md` and a source-only v2 tag workflow.
- Added `scripts/phase28_release_audit.sh` for version, publishing, artifact,
  roadmap, CLI, lockfile, and unfinished-marker validation.
- Added the held-out tiny preference fixture expected by the preference eval task.

### Changed

- Set the complete 17-package workspace to version 2.0.0 through shared package metadata.
- Raised the MSRV to Rust 1.89, the first release supporting the AVX-512 intrinsics used by the kernel path.
- Committed `Cargo.lock` and changed CI, release, installation, and validation commands to `--locked`.
- Aligned the direct `tokenizers` dependency with Candle's 0.22 line and removed an unused workspace dependency.
- Enabled a portable optimized release profile with `opt-level=3`, Thin LTO, one codegen unit, and stripped debug information.
- Updated README, architecture, roadmaps, guides, contributing, security, and release policy for the completed v2 release.
- v2.0.x is now the supported release line; v1.0.x is no longer maintained.

### Fixed

- Removed the final dead-code suppression and unused scalar helper.
- Added explicit safety rationale to memory-mapped checkpoint and CUDA/SIMD unsafe boundaries.
- Corrected stale Rust 1.80, v1-current, v2-in-progress, and crates.io-future wording.

### Security

- Streaming safety now evaluates rolling cross-token windows before SSE release,
  redacts PII, blocks toxic continuations, and keeps structured tool calls atomic.
- Release checks deny undocumented unsafe blocks, missing public API docs, known
  RustSec vulnerabilities, tracked model artifacts, and crates.io publishing steps.
- Non-loopback serving requires bearer authentication; request size, queue capacity,
  CORS, error exposure, and shutdown behavior remain bounded.

### Release Policy

- v2.0.0 is a GitHub application source release only.
- All workspace packages remain `publish = false` and are not released to crates.io.
- No pretrained checkpoints, adapters, tokenizer artifacts, GGUF files, optimizer
  state, or compiled CPU/CUDA binaries are attached.

## [2.0.0-alpha.12] - 2026-07-12

### Added

- **Phase 27 Inference Server**
  - Added `aarambh-studio-serve` with Axum 0.8.9 HTTP routing and OpenAI-compatible chat completions, legacy completions, and model listing
  - Added JSON and SSE responses, usage accounting, stop sequences, reasoning-effort mapping, function-call responses, and `[DONE]` termination
  - Added resumable `GenerationSession` state and shared batched decode passes with independent preallocated KV caches
  - Added bounded admission, chunked prefill, disconnect cancellation, strict request validation, health/readiness, metrics, and graceful shutdown
  - Added `aarambh-studio serve` with model ID, batching, safety, tool catalog, CORS, bind, and environment-key controls
  - Added a local release-mode server smoke script and OpenAI SDK/curl guide

### Changed

- Safety-enabled CLI streaming now uses a rolling cross-token filter instead of buffering the complete response
- Generation output reports prompt, completion, and total token usage
- Text generation supports up to four stop sequences and can omit retained per-step metadata for server workloads
- The transformer decode path can batch projections, normalization, and FFN/MoE work while keeping ragged attention caches isolated
- README, ROADMAP_V2, ARCHITECTURE, ARCHITECTURE_V2, and the complete guide document Phase 27 behavior

### Security

- Non-loopback server binds require bearer authentication; local loopback remains convenient by default
- Request bodies are capped at 1 MiB, queue capacity is bounded, CORS is opt-in, and internal failures are sanitized
- Streaming PII is redacted before release, toxic fragments terminate with `content_filter`, and structured tool calls remain atomic

### Verified

- Batched-versus-independent greedy parity and per-session cache isolation
- Split-token email/toxicity, stop holdback, OpenAI model-list, and SSE `[DONE]` tests
- Workspace formatting, check, Clippy, tests, rustdoc, release build, and local server smoke paths

## [2.0.0-alpha.11] - 2026-07-11

### Added

- **Phase 26 Tool Use / Function Calling**
  - Added a practical JSON Schema compiler, incremental token grammar, and schema post-validation
  - Added typed tool definitions, calls, choices, and a controller composed with thinking budgets
  - Added `infer --tools` and `--tool-choice auto|none|required|<name>`
  - Added direct-answer and tool-call protocol branches without changing reserved tokenizer IDs
  - Added grammar-constrained standard and exact speculative text decoding
  - Added atomic tool-call streaming, constrained predict-view candidates, and safety-aware blocking
  - Added LoRA/QLoRA `tool-sft` training with strict validated JSONL examples
  - Added tool-calling evaluation metrics, local fixtures, xLAM normalization, and pinned BFCL preparation

### Changed

- Generation output now carries an optional typed tool call and tool-specific finish/phase metadata
- Sampling can apply sparse allowed-token constraints before top-k/top-p filtering
- Evaluation scorecards support backward-compatible secondary task metrics
- README, ROADMAP_V2, ARCHITECTURE_V2, and the complete guide document Phase 26 behavior

### Security

- Phase 26 emits calls only and never executes commands, URLs, filesystem operations, or APIs
- PII-bearing structured calls are blocked instead of text-redacted into schema-invalid JSON

### Verified

- Grammar prefix, schema validation, unsupported-keyword, tool-data, and eval metric unit tests
- Workspace formatting, check, Clippy, tests, rustdoc, release build, and local CLI smoke paths

## [2.0.0-alpha.10] - 2026-07-11

### Added

- **Phase 25 Speculative Decoding**
  - Added exact Tiny-draft/Medium-or-Large-target decoding with configurable proposal width
  - Added modified rejection sampling, residual correction, and target bonus-token generation
  - Added block target verification with independent preallocated draft and target KV caches
  - Added `infer --speculative`, explicit draft model/config/tokenizer options, and generation telemetry
  - Added support for greedy/sampled decoding, thinking modes, streaming, predict view, and safety
  - Added statistical distribution, rejection, greedy parity, tokenizer compatibility, and cache rollback tests
  - Added a reproducible Kaggle benchmark script with output-equivalence verification

### Changed

- The sampler now exposes documented normalized-distribution operations used by exact decoding
- KV caches can truncate rejected suffixes without reallocating preallocated storage
- Generation outputs optionally include speculative acceptance and forward-pass counters
- README, ROADMAP_V2, and ARCHITECTURE_V2 now document Phase 25 commands and guarantees

### Verified

- `cargo fmt --all --check`
- `cargo check --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace --no-fail-fast`
- `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`
- `cargo build --release -p aarambh-studio`
- Local release-mode Tiny checkpoint target/draft smoke tests for greedy output, streaming, thinking, and telemetry

## [2.0.0-alpha.9] - 2026-07-11

### Added

- **Phase 24 DPO Preference Tuning**
  - Added canonical `{prompt, chosen, rejected}` JSONL loading, validation, truncation, and dynamic pair batching
  - Added numerically stable standard and reference-free DPO objectives with completion-only sequence scoring
  - Added `finetune dpo` for DoRA policies and `finetune qdpo` for quantized QDoRA policies
  - Added one-time frozen-reference log-probability precomputation so the reference model is not retained during optimizer steps
  - Added adapter-only DPO training with accumulation, clipping, cosine scheduling, periodic saves, and `dpo_config.json`
  - Added `preference` evaluation task and tracked local train/eval smoke pairs
  - Added HH-RLHF and UltraFeedback normalization scripts with deterministic held-out splits

### Changed

- `aarambh-studio-finetune` now exports documented DPO dataset, batch, loss, metrics, trainer, and run APIs
- The eval harness can report pairwise preference win rate using mean completion log-probability
- README, ROADMAP_V2, and ARCHITECTURE_V2 now document DPO/QDPO commands, reference behavior, and GRPO/DPO responsibilities

### Verified

- `cargo test -p aarambh-studio-finetune -p aarambh-studio-eval --no-fail-fast`
- `cargo fmt --all --check`
- `cargo check --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace --no-fail-fast`
- `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`
- Local two-step DoRA DPO run, adapter merge, preference eval, and one-step reference-free QDPO run

## [2.0.0-alpha.8] - 2026-07-10

### Added

- **Phase 23 Multi-GPU Training**
  - Added single-node NCCL data-parallel training context with env-worker launch support
  - Added `[distributed]` training config with `AARAMBH_STUDIO_WORLD_SIZE`, `AARAMBH_STUDIO_RANK`, `AARAMBH_STUDIO_LOCAL_RANK`, `AARAMBH_STUDIO_DIST_RUN_ID`, and `AARAMBH_STUDIO_DIST_RENDEZVOUS` overrides
  - Added deterministic sharded `DataLoader` construction with equal per-rank batch counts
  - Added bucketed F32 gradient all-reduce before gradient clipping and AdamW updates
  - Added rank-0-only logging, validation, checkpoint, and tokenizer creation behavior
  - Added `configs/wikitext103_small_2gpu.toml` for Kaggle 2×T4 runs

### Changed

- The `cuda` feature for training now enables Candle NCCL support
- Distributed runs fall back cleanly to rank-0 single-process training when the requested 2×T4 allocation is unavailable
- README, ROADMAP_V2, and ARCHITECTURE_V2 now document the Phase 23 env-worker launch path

### Verified

- `cargo fmt --all --check`
- `cargo check -p aarambh-studio-data -p aarambh-studio-train`
- `cargo check --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test -p aarambh-studio-data -p aarambh-studio-train --no-fail-fast`
- `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`

### Notes

- Local CUDA-feature verification is blocked on this machine because cudarc requires `nvcc`; run `cargo build --release -p aarambh-studio --features cuda` on Kaggle or another CUDA/NCCL host.

## [2.0.0-alpha.7] - 2026-07-10

### Added

- **Phase 22 Mixture of Experts**
  - Added optional `MoeConfig` on `ModelConfig` with every-N-layer placement
  - Added top-k router gating, dense masked expert dispatch, and MoE SwiGLU experts
  - Added differentiable load-balancing auxiliary loss and per-expert utilization stats
  - Added MoE tensor naming for router/expert checkpoints and GGUF roundtrips
  - Added `configs/moe_smoke.toml` and `configs/small_moe.toml`

### Changed

- Trainer loss now adds `aux_loss_weight * moe_aux_loss` when MoE layers are active
- MoE training logs include `ce_loss`, `moe_aux`, and `expert_util=[...]`
- LoRA/DoRA/self-learning adapter updates now reject MoE configs clearly in Phase 22

### Verified

- `cargo fmt --all --check`
- `cargo check -p aarambh-studio-nn -p aarambh-studio-model -p aarambh-studio-train`
- `cargo check --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test -p aarambh-studio-nn -p aarambh-studio-model`
- `cargo test -p aarambh-studio-train -p aarambh-studio-weights -p aarambh-studio-finetune -p aarambh-studio-selflearn`
- `cargo test --workspace`
- `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`
- `cargo run --release -p aarambh-studio -- train --config configs/moe_smoke.toml`
- `cargo run --release -p aarambh-studio -- infer --config configs/moe_smoke.toml --model checkpoints/moe_smoke/step_000002/model.safetensors --tokenizer checkpoints/moe_smoke/tokenizer.json --prompt "Hello" --max-tokens 4 --greedy --safety none`

## [2.0.0-alpha.6] - 2026-07-10

### Added

- **Phase 21 Vision-Aware Self-Learning**
  - Added `image_ref` replay entries with backward-compatible v1 JSONL loading
  - Added projected image-token cache under the self-learning state directory
  - Added grounded vision verifiers for count, color, yes/no presence, and exact VQA answers
  - Added CUDA-only vision self-learning gate so CPU text self-learning remains unchanged
  - Added multimodal LoRA generation, vision GRPO scoring, and cached vision replay SFT
  - Added `selflearn start --mode vision` and `selflearn stats --mode vision`

### Changed

- `infer --image --self-learn gpu` now runs the vision-aware self-learning path instead of rejecting the combination
- Open-ended vision prompts fall back to existing self-critique; checkable VQA prompts can use deterministic grounded rewards
- `SELF_LEARNING_V2.md`, README, and ROADMAP_V2 now document projected-token caching and the Kaggle/CUDA requirement

### Verified

- `cargo fmt --all --check`
- `cargo check -p aarambh-studio-selflearn -p aarambh-studio`
- `cargo check --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test -p aarambh-studio-selflearn`
- `cargo test --workspace`
- `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`
- Local CPU smoke gate for `selflearn start --mode vision` fails clearly with the Kaggle/CUDA requirement

## [2.0.0-alpha.5] - 2026-07-06

### Added

- **Phase 20 Vision-Language Training**
  - Added VQA instruction data loading for simple JSONL and LLaVA-style conversation records
  - Added `finetune vlm-dora` and `finetune vlm-qdora` for image-question-answer instruction tuning
  - Added embedding-level DoRA forward APIs so projected image tokens can train through the adapter decoder
  - Added VLM DoRA artifact saving with a normal DoRA adapter plus separate tuned `projector.safetensors`
  - Added VQA evaluation task and Phase 20 smoke/full data setup scripts

### Changed

- Phase 20 keeps the vision encoder frozen, trains DoRA/QDoRA adapter params, and optionally trains the projector
- README and ROADMAP_V2 now document the VQA instruction-tuning workflow

### Verified

- `cargo fmt --all --check`
- `cargo check --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace`

## [2.0.0-alpha.4] - 2026-07-05

### Added

- **Phase 19 Vision Encoder + Projector**
  - Added the `aarambh-studio-vision` crate with CLIP-style ViT encoding, image preprocessing, projector MLP, and `<image>` prefix fusion
  - Added public CLIP-B/32 SafeTensors loading with HuggingFace tensor-name normalization
  - Added `<image>` and `<image_end>` reserved special tokens for v2 multimodal tokenizers
  - Added embedding-prefix forward and generation paths so projected image tokens can enter the existing decoder without cross-attention changes
  - Added projector-only pretraining through `train --config configs/vision_projector_pretrain.toml`
  - Added `infer --image` support with streaming, predict-view, and safety guard integration
  - Added image-caption smoke evaluation and Phase 19 data/weight preparation scripts

### Changed

- Text-only tokenizer validation remains backward compatible with legacy checkpoints, while image inference/training require the v2 multimodal tokens
- README, ROADMAP_V2, and ARCHITECTURE_V2 now document the Phase 19 vision workflow

### Verified

- `cargo fmt --all`
- `cargo check --workspace`

## [2.0.0-alpha.3] - 2026-07-05

### Added

- **Phase 18 DoRA and QDoRA fine-tuning**
  - Added `DoraLinear`, `DoraConfig`, and `DoraAarambhModel` to `aarambh-studio-finetune`
  - Added row-normalized DoRA forward and merge math with trainable magnitude vectors
  - Added QDoRA support by reusing the existing packed INT4 base-weight path
  - Added `AdapterMethod` metadata with backward-compatible default loading for existing LoRA adapters
  - Added shared SFT adapter training over LoRA/QLoRA/DoRA/QDoRA models
  - Added `aarambh-studio finetune dora`, `aarambh-studio finetune qdora`, and merge method auto-detection
  - Added `docs/dora_vs_lora.md` with Phase 17 scorecard comparison commands

### Changed

- `finetune merge` can now merge LoRA or DoRA adapters through `--method auto|lora|dora`
- CI smoke checks now cover `finetune dora`, `finetune qdora`, and `finetune merge`
- README, ROADMAP_V2, and ARCHITECTURE_V2 now document the Phase 18 workflow

### Verified

- `cargo fmt --all --check`
- `cargo check -p aarambh-studio-finetune -p aarambh-studio`
- `cargo check --workspace`
- `cargo test -p aarambh-studio-finetune`
- `cargo run -p aarambh-studio -- finetune dora --help`
- `cargo run -p aarambh-studio -- finetune qdora --help`
- `cargo run -p aarambh-studio -- finetune merge --help`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace`
- `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`

## [2.0.0-alpha.2] - 2026-07-05

### Added

- **Phase 17 Evaluation Harness**
  - Added the `aarambh-studio-eval` crate with PPL, MMLU-lite, HellaSwag, GSM8K-subset, and HumanEval-lite task support
  - Added JSON and Markdown scorecards plus before/after scorecard comparison
  - Added `aarambh-studio eval` CLI with `--tasks`, `--data-dir`, `--out`, `--markdown`, and `--compare`
  - Added explicit `--allow-code-exec` gating for HumanEval-lite
  - Added `CodeVerifier` to `aarambh-studio-finetune` for sandboxed Python pass@1 checks
  - Added `scripts/phase17_prepare_eval_sets.sh` for preparing normalized public evaluation subsets

### Changed

- CI smoke checks now include `aarambh-studio eval --help`
- README, ROADMAP_V2, and ARCHITECTURE_V2 now document the Phase 17 workflow and scorecard contract

### Verified

- `cargo fmt --all --check`
- `cargo check --workspace`
- `cargo test -p aarambh-studio-eval -p aarambh-studio-finetune`
- `cargo run -p aarambh-studio -- eval --help`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace`
- `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`

## [2.0.0-alpha.1] - 2026-07-05

### Added

- **Phase 16 Long Context (RoPE Scaling)**
  - Added `RopeScalingConfig` and `RopeScalingMethod` to `aarambh-studio-core`
  - Added YaRN, NTK-aware, and linear RoPE inverse-frequency helpers in `aarambh-studio-nn`
  - Added `RopeCache::from_config()` for scaled and unscaled RoPE cache construction
  - Added Medium 16K, Large 16K, and long-context CUDA smoke training configs
  - Added `context_schedule` support for staged 4K to 8K to 16K continued pretraining
  - Added `scripts/phase16_prepare_longdoc.sh` for WikiText-103 long-document preparation

### Changed

- Main and LoRA model paths now use causal attention dispatch instead of storing a full max-size causal mask
- Inference KV caches now support preallocated fixed-capacity storage sized to the configured context length
- `rope_scaling = None` remains backward compatible with v1.0.0 unscaled RoPE behavior
- README, ROADMAP_V2, and ARCHITECTURE_V2 now document the Phase 16 workflow

### Verified

- `cargo check --workspace`
- `cargo fmt --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace`

## [1.0.0] - 2026-06-30

### Added

- **Phase 15 Production Release v1.0**
  - Added a GitHub source-release workflow for tag `v1.0.0`
  - Added `.github/release-notes/v1.0.0.md` as the full GitHub Release body
  - Added `RELEASE.md` with the v1.0.0 release checklist, validation commands, and release policy
  - Added strict public API documentation coverage across library crates with missing-docs denied
  - Added CLI version reporting through `aarambh-studio --version`

### Changed

- **Release policy**
  - Set every package manifest to `version = "1.0.0"`
  - Set every package manifest to `publish = false`; v1.0.0 does not publish to crates.io
  - Documented v1.0.0 as a GitHub source release with no pretrained checkpoints, adapters, tokenizer artifacts, GGUF files, or binary release assets
  - Removed YouTube and Discord launch items from the Phase 15 release scope

- **Documentation**
  - Updated README quickstart and production release sections for source builds and local install
  - Updated ROADMAP Phase 15 to reflect strict docs, CI, release workflow, release notes, and source-only release policy
  - Updated ARCHITECTURE distribution notes for source-built CLI usage from the GitHub v1.0 tag
  - Updated SECURITY, CONTRIBUTING, and CODE_OF_CONDUCT to remove Discord reporting paths
  - Updated contributor versioning guidance for the v1 release line

- **CI**
  - Expanded CI to run formatting, workspace check, clippy with all targets, tests, strict rustdoc, release binary build, and CLI help smoke checks
  - Added a release workflow that validates the default CPU build before creating the GitHub Release from `.github/release-notes/v1.0.0.md`

### Verified

- `cargo fmt --check`
- `cargo check --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace`
- `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`
- `cargo build --release -p aarambh-studio`
- CLI smoke checks for `--version`, `--help`, `train`, `infer`, `quantise`, `convert`, `finetune`, and `selflearn`

## [0.14.0] - 2026-06-30

### Added

- **Phase 14 Flash Attention CUDA kernels**
  - Replaced Phase 4 CUDA scaffolds with real `.cu` kernels for Flash Attention forward, Flash Attention backward source, fused RMSNorm, fused RoPE, and fused SwiGLU
  - Added NVCC-to-PTX build plumbing with `cfg(aarambh_cuda_kernels)` and graceful CPU/Candle fallback when NVCC is missing
  - Added Candle custom-op wrappers that load PTX into Candle's CUDA module cache at runtime
  - Added CUDA dispatch paths for supported contiguous F32/F16/BF16 FlashAttention and fused RMSNorm tensors
  - Added inference-only fused RoPE and fused SwiGLU hooks in `aarambh-studio-nn`
  - Added CUDA-gated kernel correctness tests against Candle references

### Changed

- **Kernel dispatch**
  - `KernelPath` now reports `CudaFlashAttention` and `CudaFusedRmsNorm` when CUDA PTX kernels are compiled and tensor shapes are supported
  - Attention dispatch detects project causal masks and routes supported CUDA cases to FlashAttention; arbitrary additive masks keep using Candle
  - Training attention uses a dedicated dispatch entry with Candle-compatible backward fallback behavior

- **Documentation**
  - Marked Phase 14 complete in README and ROADMAP
  - Updated architecture notes to describe PTX loading instead of CUDA scaffolding

### Verified

- `cargo fmt`
- `cargo check`
- `cargo test -p aarambh-studio-kernel`
- `cargo test -p aarambh-studio-nn`
- CUDA PTX tests are gated and must be run on a CUDA host with NVCC and `--features cuda`

## [0.13.0] - 2026-06-29

### Added

- **Phase 13 GPU scale-up implementation**
  - Added opt-in `cuda` feature forwarding across the workspace without changing default CPU builds
  - Added config-driven `dtype = "f32"|"f16"|"bf16"|"mixed"` parsing for train/infer paths
  - Added dtype-aware SafeTensors/GGUF model loading helpers for BF16 GPU inference and self-learning
  - Added WikiText-103 Small, Medium, Large, and CUDA smoke configs
  - Added Kaggle notebooks for Small/T4, Medium/P100, and Large/A100 training workflows
  - Added WikiText-103 preparation and checkpoint packaging scripts

### Changed

- **Training**
  - Trainer now builds model weights using the configured dtype instead of hardcoded F32
  - Cross-entropy casts logits to F32 for stable lower-precision training
  - AdamW keeps moment buffers and update math in F32 while writing params back to their model dtype
  - Training logs now include `tok/s` throughput for Phase 13 benchmarking

- **Model internals**
  - RoPE caches and causal masks are dtype-aware for BF16 model execution
  - Inference and self-learning model loaders now honor the run config dtype

- **Documentation**
  - Marked Phase 13 complete in README and ROADMAP
  - Updated architecture notes with CUDA feature commands and BF16 config behavior

### Verified

- `cargo fmt --all --check`
- `cargo check --workspace`
- `cargo test --workspace`
- `cargo test -p aarambh-studio-train`
- `cargo clippy --workspace --all-targets -- -D warnings`
- Phase 13 notebook JSON validation
- Phase 13 helper script syntax and dummy runtime checks
- CUDA training is prepared through notebooks/configs and must be executed on Kaggle or another CUDA host with `--features cuda`

## [0.12.0] - 2026-06-29

### Added

- **`aarambh-studio-selflearn` Phase 12 implementation**
  - Added CPU/GPU/disabled self-learning presets with online GRPO, replay, critique, metrics, and persistent state configuration
  - Added replay buffer JSONL persistence, score filtering, high-quality retention, score-squared sampling, topic diversity, and topic inference
  - Added stateless replay-only self-critique with robust JSON parsing, score clamping, malformed-output fallback, bounded rewrite generation, and rewrite support
  - Added LoRA-backed self-learning generation, deterministic-verifier online GRPO updates, CPU deferred pending gradients, GPU inline stepping, optimizer state, adapter state, pending-gradient persistence, and pending-gradient contribution accounting
  - Added replay SFT updates that train the live LoRA adapter from sampled replay entries with response-only loss masking
  - Added learning metrics with per-topic trend summaries

### Changed

- **CLI**
  - Added `--self-learn disabled|cpu|gpu`, `--replay-path`, `--self-learn-state-dir`, `--self-learn-verifier`, and `--self-learn-ground-truth` to `infer`
  - Added `aarambh-studio selflearn flush-gradients`, `replay`, `stats`, and `reset --yes`
  - Composed self-learning with safety so replay/gradient state commits only after safety allows the generated draft

- **Documentation**
  - Marked Phase 12 complete in README and ROADMAP
  - Updated self-learning CLI examples and architecture notes

### Verified

- `cargo check --workspace`
- `cargo test -p aarambh-studio-selflearn`
- `cargo test --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo run -p aarambh-studio -- infer --help`
- `cargo run -p aarambh-studio -- selflearn --help`
- `cargo run -p aarambh-studio -- selflearn replay --help`
- `cargo run -p aarambh-studio -- selflearn stats --replay-path /tmp/aarambh_phase12_empty_replay.jsonl --self-learn-state-dir /tmp/aarambh_phase12_empty_state`

## [0.11.0] - 2026-06-29

### Added

- **`aarambh-studio-safety` Phase 11 implementation**
  - Added prompt-injection and jailbreak detectors with weighted rule scoring, role-switch checks, leetspeak/confusable normalization, and Base64-like payload detection
  - Added PII detection/redaction for email, phone, SSN/national ID, credit cards with Luhn validation, known API-key prefixes, and high-entropy secrets
  - Added output toxicity scoring for hate speech, violence, sexual content, self-harm, and illegal activity
  - Added `SafetyPolicy` presets, `SafetyVerdict`, `SafetyGuard`, `SafeResponse`, and privacy-safe `SafetyEvent` audit logging with SHA-256 prompt hashes

### Changed

- **CLI**
  - `infer` now uses `SafetyGuard` by default
  - Added `--safety strict|permissive|research|none` and `--safety-audit-log`
  - Buffered safety-enabled streaming/predict-view callbacks until output checks pass, preventing unsafe text from being printed before guardrails run

- **Documentation**
  - Marked Phase 11 complete in README and ROADMAP
  - Added safety CLI examples and audit privacy notes
  - Aligned ARCHITECTURE safety policy fields with the implemented API

### Verified

- `cargo check -p aarambh-studio-safety`
- `cargo check -p aarambh-studio`
- `cargo test -p aarambh-studio-safety`

## [0.10.0] - 2026-06-29

### Added

- **`aarambh-studio-finetune` Phase 10 implementation**
  - Added GRPO dataset loading for `prompt`/`question` plus `ground_truth`/`answer` JSONL records
  - Added graph-free group rollout sampling from the live LoRA policy with temperature, top-k, top-p, and thinking-token forcing
  - Added differentiable replay of sampled completions through `LoraAarambhModel::forward_train()` for policy log-probs
  - Added frozen-reference KL loss using full generated-token distributions
  - Added `GrpoTrainer` with adapter-only AdamW, cosine warmup/decay, gradient accumulation, clipping, logging, and adapter checkpoints

- **Deterministic verifiers**
  - Added `MathVerifier` with GSM8K `#### answer` parsing, commas, negatives, decimals, and numeric tolerance
  - Added `FormatVerifier` for `<think>...</think>` structure rewards
  - Added `CompositeVerifier` and `math-format` verifier selection

- **CLI**
  - Added `aarambh-studio finetune grpo`
  - Added GRPO flags for reference checkpoint, verifier, group size, max new tokens, sampling controls, thinking mode, KL coefficient, LoRA rank/alpha/dropout, steps, LR, accumulation, logging, and save cadence

### Changed

- **Documentation**
  - Marked Phase 10 complete in README and ROADMAP
  - Added GRPO command examples, tiny local GRPO math data, and adapter output notes
  - Clarified that GRPO uses deterministic verifiers only; Self-Critique remains Phase 12 replay-buffer logic
  - Documented that GRPO training log-probs are recomputed by differentiable replay instead of cached inference

### Verified

- `cargo check -p aarambh-studio-finetune -p aarambh-studio`
- `cargo check --workspace`
- `cargo test -p aarambh-studio-finetune`
- `cargo test --workspace`

## [0.9.0] - 2026-06-29

### Added

- **`aarambh-studio-finetune` Phase 9 implementation**
  - Added `LoraConfig`, `LoraLinear`, frozen F32 base support, packed INT4 QLoRA base support, adapter dropout, target-module matching, and LoRA merge math
  - Added `LoraAarambhModel`, an adapter-aware decoder forward path that keeps the existing base model/inference code unchanged
  - Added adapter persistence with `adapter_config.json`, `adapter.safetensors`, and adapter train-state output
  - Added SFT JSONL loading for `{"instruction","response"}` and `{"instruction","thinking","response"}`
  - Added chat templates, thinking SFT formatting, shifted labels, prompt loss masking, and padded SFT batches
  - Added `SftTrainer` with adapter-only AdamW, cosine warmup/decay, gradient accumulation, clipping, logging, and adapter checkpoints
  - Added adapter merge into normal `model.safetensors` for the existing inference engine

- **CLI**
  - Added `aarambh-studio finetune sft`
  - Added `aarambh-studio finetune qlora`
  - Added `aarambh-studio finetune merge`
  - Added fine-tune overrides for LoRA rank/alpha/dropout, target modules, batch size, max steps, learning rate, accumulation, warmup, logging, and save cadence

### Changed

- **Documentation**
  - Marked Phase 9 complete in README and ROADMAP
  - Added LoRA/QLoRA/SFT command examples and adapter layout documentation
  - Clarified that DoRA is not part of Phase 9

### Verified

- `cargo fmt`
- `cargo check --workspace`
- `cargo test --workspace`

## [0.8.0] - 2026-06-28

### Added

- **`aarambh-studio-quant` crate implementation (Phase 8)**
  - INT8 absmax quantisation and dequantisation
  - Packed INT4 affine quantisation with per-group scales/zero-points
  - AWQ activation-scale computation and layer quantisation
  - GPTQ Hessian construction plus damped Cholesky inversion
  - GGUF Q4_K_M block quant/dequant helpers
  - QAT fake-quant nodes
  - INT8 `QuantisedKvCache`
  - Streaming calibration stats over real model linear inputs

- **`aarambh-studio-weights`**
  - Added GGUF save/load support for Q4_K_M, Q5_K_M, and Q8_0 formats
  - Added `.gguf` model loading through `load_any_model()`
  - Implemented HuggingFace safetensors conversion with standard key mapping and strict GQA K/V slicing

- **CLI**
  - Added `aarambh-studio quantise`
  - Added `aarambh-studio convert`
  - Added `aarambh-studio convert --gguf`
  - `aarambh-studio infer --model <path.gguf>` now loads GGUF checkpoints

### Changed

- **`aarambh-studio-model` / `aarambh-studio-nn`**
  - Added capture-aware forward methods for calibration inputs to attention and FFN linear layers

### Verified

- `cargo check -p aarambh-studio-quant`
- `cargo check -p aarambh-studio-weights`
- `cargo check -p aarambh-studio --all-targets`
- `cargo test -p aarambh-studio-quant`
- `cargo test -p aarambh-studio-weights`

## [0.7.0] - 2026-06-28

### Added

- **Thinking engine (Phase 7)**
  - `ThinkingController` now forces `<think>` once for `low`, `medium`, and `high` modes
  - Enforces mode budgets and force-injects `</think>` when the active budget is reached
  - Tracks started/closed state, effective generation budget, thinking-token count, and queued forced tokens
  - Added `GenerationPhase::{Thinking, Answer}` plus `forced` and `phase` metadata on each generation step

- **Inference output separation**
  - `GenerationOutput.text` is now the visible answer text
  - Added `raw_text`, `thinking_text`, `answer_text`, `thinking_token_ids`, `answer_token_ids`, and `thinking_tokens`
  - Preserves all forced tokens in `token_ids` while hiding thinking markers from user-visible answer output

- **CLI**
  - `aarambh-studio infer --thinking low|medium|high` now wraps prompts with user/assistant markers, prints thinking dimmed, and prints the final answer normally
  - Streaming output switches terminal styling between thinking and answer phases
  - Predict-view now shows token phase and forced-token metadata

- **`aarambh-studio-finetune`**
  - Added `ThinkingSftExample` and `format_thinking_sft()` as the Phase 9-compatible thinking SFT data format helper

### Changed

- **Documentation**
  - Marked Phase 7 complete in README and ROADMAP
  - Updated ARCHITECTURE to describe the implemented thinking controller and separated inference output

### Verified

- `cargo fmt --all -- --check`
- `cargo check --workspace --all-targets`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace`
- `cargo run --release -p aarambh-studio -- infer --config configs/tiny_shakespeare_smoke.toml --prompt "What is 2 + 2?" --max-tokens 48 --thinking low --greedy`
- `cargo run --release -p aarambh-studio -- infer --config configs/tiny_shakespeare_smoke.toml --prompt "What is 2 + 2?" --max-tokens 40 --thinking low --greedy --stream`
- `cargo run --release -p aarambh-studio -- infer --config configs/tiny_shakespeare_smoke.toml --prompt "What is 2 + 2?" --max-tokens 6 --thinking low --greedy --predict-view`

## [0.6.0] - 2026-06-28

### Added

- **`aarambh-studio-inference` crate** — Inference engine (Phase 6)
  - `InferenceEngine` with checkpoint loading, tokenizer validation, prompt prefill, cached one-token decode, EOS/max-token/context-limit stopping, and callback-based generation
  - `KvCache` wrapper over per-layer `aarambh-studio-nn::KVCache`
  - `Sampler` with greedy decode plus temperature/top-k/top-p sampling and top-candidate reporting for predict-view
  - `GenerationConfig`, `GenerationOutput`, `GenerationStep`, `FinishReason`, and `StreamEvent`
  - `ThinkingMode` and `ThinkingController` for Phase 7 budget tracking without token forcing

- **CLI**
  - Added `aarambh-studio infer` with `--config`, `--model`, `--tokenizer`, `--prompt`, `--max-tokens`, `--temperature`, `--top-p`, `--top-k`, `--seed`, `--thinking`, `--predict-view`, `--stream`, and `--greedy`
  - Defaults to `latest.json` or `best.json` from the configured checkpoint directory when `--model` is omitted
  - Added terminal predict-view rendering for top next-token candidates

- **Tokenizer**
  - Trained BPE tokenizers now reserve fixed project special-token IDs 0..6
  - Added special-token validation and special-aware encode support for `<|endoftext|>`, `<|pad|>`, `<|bos|>`, `<think>`, `</think>`, `<|user|>`, and `<|assistant|>`
  - Training automatically regenerates an owned stale tokenizer whose reserved IDs are invalid

### Changed

- **Binary crate structure**
  - Split CLI implementation into `cmd/train.rs`, `cmd/infer.rs`, and `ui/predict_view.rs`

- **Documentation**
  - Marked Phase 6 complete in README and ROADMAP
  - Updated ARCHITECTURE with tokenizer special-ID invariants and the implemented inference flow

### Verified

- `cargo check --workspace --all-targets`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace`
- `cargo run --release -p aarambh-studio -- train --config configs/tiny_shakespeare_smoke.toml`
- `cargo run --release -p aarambh-studio -- infer --config configs/tiny_shakespeare_smoke.toml --prompt "To be" --max-tokens 8 --greedy --predict-view`

## [0.5.0] - 2026-06-27

### Added

- **`aarambh-studio-train` crate** — Training loop (Phase 5)
  - Masked cross-entropy over `[batch, seq, vocab]` logits with padding masks
  - Project-owned AdamW with `beta1=0.9`, `beta2=0.95`, `eps=1e-8`, decoupled weight decay, and no-decay exclusions for embeddings, biases, and RMSNorm weights
  - Explicit gradient accumulation by parameter name, global norm clipping, cosine schedule with linear warmup, validation, logging, and full train loop
  - SafeTensors checkpointing for model weights and optimizer moments plus JSON train state, `latest.json`, and `best.json`
  - TOML run config loader, `configs/tiny_shakespeare.toml`, and `configs/tiny_shakespeare_smoke.toml`
  - 11 train tests covering loss masking, LR warmup/decay, AdamW defaults, weight decay policy, gradient clipping, checkpoint roundtrip, and synthetic tiny-model loss decrease

- **CLI**
  - Added `aarambh-studio train --config <path>` for Phase 5 training runs

- **Tokenizer**
  - Added `BpeTokenizer::save_pretrained()` to persist vocab and BPE merges in a reloadable tokenizer JSON
  - Supports both legacy string merges and modern array merges from HuggingFace `tokenizers`

### Changed

- **`aarambh-studio-core` crate**
  - Extended `TrainConfig` with `max_steps`, `min_lr_ratio`, and `seed`
  - Added serde defaults for backward-compatible config loading

- **`aarambh-studio-nn` and `aarambh-studio-model` crates**
  - Added `forward_train()` paths that use Candle autograd-compatible RMSNorm and attention instead of Phase 4 inference kernels
  - Changed token embedding initialization to `N(0, 0.02)` so tied LM heads start with sane logits and random-model loss near `ln(vocab)`

- **Training config**
  - Reuses an existing tokenizer JSON in the checkpoint directory instead of retraining BPE on every launch

### Verified

- `cargo check --workspace --all-targets`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace`
- `cargo run --release -p aarambh-studio -- train --config configs/tiny_shakespeare_smoke.toml`

## [0.4.0] - 2026-06-27

### Added

- **`aarambh-studio-kernel` crate** — Custom kernels (Phase 4)
  - Runtime dispatch API for RMSNorm and attention
  - Stable CPU SIMD RMSNorm with cached AVX2/FMA, AVX512, AVX2, and scalar fallback paths
  - Rayon parallel scaled dot-product attention for CPU F32 tensors
  - Candle fallback for unsupported devices, dtypes, shapes, and CUDA runtime paths
  - `build.rs` NVCC detection with graceful no-CUDA builds
  - CUDA scaffold kernels and FFI wrapper modules for Flash Attention, fused RMSNorm, fused RoPE, and fused SwiGLU
  - Criterion benchmark target for RMSNorm and attention kernels
  - 6 kernel tests covering dispatch, RMSNorm reference parity, parallel attention parity, masks, and CUDA scaffold availability
  - Local benchmark: RMSNorm SIMD ~1.43x faster than Candle; parallel attention ~2.94x faster than sequential

### Changed

- **`aarambh-studio-nn` crate**
  - `RMSNorm::forward()` now calls kernel dispatch
  - `GroupedQueryAttention::forward()` now calls kernel attention dispatch after Q/K/V preparation

- **Documentation**
  - Marked Phase 4 complete in README and ROADMAP
  - Updated ARCHITECTURE to match stable SIMD intrinsics and CUDA scaffold behavior

## [0.3.0] - 2026-06-27

### Added

- **`aarambh-studio-model` crate** — Full model forward pass (Phase 3)
  - `src/embedding.rs` — `TokenEmbedding` wrapping Candle embedding lookup with weight access for tied LM head
  - `src/head.rs` — `LmHead` supporting tied embedding weights and untied no-bias output projection
  - `src/model.rs` — `AarambhModel` with config validation, embedding, N transformer blocks, final RMSNorm, LM head, precomputed RoPE, precomputed causal mask, full-sequence `forward()`, cached `forward_with_cache()`, `empty_kv_cache()`, `named_tensors()`, and `get_weight()`
  - Implements `Configurable` and `Forward`
  - 8 active integration tests covering scale config validation, Tiny forward shape, finite logits, cached-vs-full forward equivalence, tied/untied LM head behavior, invalid config rejection, and README scale consistency
  - 1 ignored heavy test for full Tiny/Small/Medium/Large construction

- **`aarambh-studio-weights` crate** — SafeTensors I/O (Phase 3)
  - `save_model()` serializes `AarambhModel::named_tensors()` with `candle_core::safetensors::save`
  - `load_model()` loads SafeTensors through `VarBuilder::from_mmaped_safetensors`
  - `convert_hf()` is present as a Phase 8 unsupported conversion entrypoint
  - 2 integration tests covering SafeTensors weight/logit roundtrip and the Phase 8 conversion path

### Changed

- **`aarambh-studio-nn` crate**
  - Added read-only weight accessors on `GroupedQueryAttention`, `SwiGluFfn`, and `TransformerBlock` so higher layers can enumerate model tensors without making fields public

- **Documentation**
  - Updated README model scale table to match `ModelConfig`, `ARCHITECTURE.md`, and `ROADMAP.md`
  - Marked Phase 3 complete in README and ROADMAP

## [0.2.0] - 2026-06-25

### Added

- **`aarambh-studio-nn` crate** — Neural network primitives (Phase 2)
  - `src/norm.rs` — `RMSNorm` wrapping `candle_nn::ops::rms_norm` with learnable weight
  - `src/rope.rs` — `RopeCache` precomputing cos/sin tables for up to `max_seq_len`, applying rotary position embeddings to Q/K
  - `src/kvcache.rs` — `KVCache` with `update()` (catches K/V along seq dim), `clear()`, `seq_len()`
  - `src/attention.rs` — `GroupedQueryAttention` with Q/K/V projections, RoPE, K/V head expansion for GQA, causal masking, `softmax_last_dim`, output projection
  - `src/ffn.rs` — `SwiGluFfn` with gate/up/down projections and SiLU-gated activation
  - `src/block.rs` — `TransformerBlock` with pre-norm residual connections
  - 5 integration tests covering RMSNorm shape, RoPE magnitude, SwiGLU shape, GQA output, and full block output
  - `src/lib.rs` — flat re-exports of all modules

## [0.1.0] - 2026-06-24

### Added

- **`aarambh-studio-tokenizer` crate** — BPE tokeniser
  - `src/special.rs` — 7 special token ID constants
  - `src/vocab.rs` — `Vocab` struct with `HashMap`-backed token↔id lookup, JSON I/O
  - `src/bpe.rs` — `BpeTokenizer` with `train()` (delegates to `tokenizers` crate BpeTrainer), `from_pretrained()` (parses HuggingFace `tokenizer.json`), pure-Rust `encode()`/`decode()`, `save()`, `TokenizerLike` impl
  - 5 unit tests covering all paths

- **`aarambh-studio-data` crate** — Data pipeline
  - `src/dataset.rs` — `TextDataset` trait, `PlaintextDataset` (`.txt` files), `JsonlDataset` (`.jsonl` with `{"text": "..."}` format)
  - `src/preprocess.rs` — `chunk_and_tokenize(dataset, tokenizer, max_seq_len)` produces `(input, label)` pairs with labels shifted by 1
  - `src/loader.rs` — `Batch` struct (input_ids, labels, attention_mask tensors), `DataLoader` struct implementing `Iterator<Item=Result<Batch>>` with batching, shuffling, device placement, and epoch `reset()`
  - 8 unit tests

### Changed

- Updated all 14 crate `Cargo.toml` files to `edition = "2024"`

## [0.0.1] - 2026-06-24

### Added

- **Workspace scaffold**
  - Root `Cargo.toml` with all 14 workspace members and pinned dependency versions
  - `resolver = "2"` for modern feature resolution
  - Workspace-level dependencies: `candle-core`, `candle-nn`, `tokenizers`, `serde`, `thiserror`, `tokio`, `clap`, `tracing`, `safetensors`, `rayon`, `cc`, `which`

- **`aarambh-studio-core` crate** (Layer 0 — Foundation types)
  - `config.rs` — `ModelConfig` with `tiny()`/`small()`/`medium()`/`large()` presets, `head_dim()`, `from_json()`; `TrainConfig` with LLaMA-correct defaults (`beta2=0.95`, `batch_size=2`, `grad_accum_steps=16`)
  - `device.rs` — `Device` enum (`Cpu`, `Cuda`, `Metal`) with `to_candle()`, `best_available()`, `is_cpu()`
  - `dtype.rs` — `DType` (`F32`, `F16`, `BF16`) with `to_candle()`, `size_bytes()`; `Precision` with `weight_dtype()`
  - `error.rs` — `AarambhError` (8 variants using `thiserror`), `type Result<T>`
  - `traits.rs` — `Forward`, `Saveable`, `Loadable`, `Configurable`, `TokenizerLike`
  - `lib.rs` — flat re-exports of all public types
  - `tests/core_tests.rs` — 6 unit tests covering configs, device, dtype, and defaults

- **12 scaffold crates** — each with `Cargo.toml` + `lib.rs` doc-comment scaffold
  - `aarambh-studio-tokenizer`, `aarambh-studio-data`, `aarambh-studio-nn`, `aarambh-studio-kernel`, `aarambh-studio-model`, `aarambh-studio-weights`, `aarambh-studio-quant`, `aarambh-studio-train`, `aarambh-studio-finetune`, `aarambh-studio-inference`, `aarambh-studio-safety`, `aarambh-studio-selflearn`

- **Binary crate** — `aarambh-studio` with minimal `main.rs`

- **GitHub repository files**
  - `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`
  - `LICENSE` (Apache 2.0)
  - `.gitignore`, `.github/` (CI workflow, issue/PR templates, dependabot)
