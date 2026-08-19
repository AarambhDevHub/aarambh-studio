# aarambh-studio

> From first principles. From zero. From Rust.
>
> Sanskrit: *beginning* — a ground-up language model system in Rust.

[![CI](https://github.com/AarambhDevHub/aarambh-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/AarambhDevHub/aarambh-studio/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.89%2B-orange.svg)](https://www.rust-lang.org)

aarambh-studio is a decoder-only language model implementation built with Rust and
Candle. The repository covers the full engineering path: tokenization, model
construction, training, inference, quantization, adapter tuning, alignment,
evaluation, multimodal input, safety, and an OpenAI-compatible server.

The production source release is **v3.0.0**,
with hybrid Gated DeltaNet, DeepSeek Sparse Attention,
fine-grained MoE with shared experts, Multi-Token Prediction (MTP), on-policy
distillation, native quantization-aware training, native video/document input,
bounded long-horizon tool-use chains, persistent forgetting diagnostics, and
Max thinking mode (16,384-token budget). **v4.0.0-alpha.8** continues the v4 arc
with Multi-Head Latent Attention (Phase 41), a native Audio modality
(Phase 42), sparse/grouped MoE dispatch (Phase 43), multi-node
distributed training (Phase 44), test-time compute scaling
(Phase 45), RLAIF (Phase 46), sandboxed tool execution
(Phase 47), and **multi-agent orchestration (Phase 48)** — one
top-level orchestrating reasoning process delegates independent sub-tasks
to multiple parallel sandboxed tool-execution sub-chains (each governed
entirely by Phase 47's boundaries), then merges their results back into
its own context via the existing `ToolResult` ingestion path applied
recursively. Three hard, non-negotiable, operator-set bounds hold: maximum
sub-agent count, maximum total execution time budget, and sandbox scope
containment (a sub-agent's `AuthorizationScope` can only be a subset of
its orchestrator's, never wider) — and one sub-agent's failure is
contained to its own outcome, never corrupting sibling sub-agents'
results. **v4.0.0-alpha.9** adds **retrieval-augmented generation
(Phase 49)** — a from-scratch, pure-Rust retrieval pipeline (a new
`aarambh-studio-retrieve` crate with a navigable small-world graph ANN,
no external vector database) that augments the prompt with retrieved
context before generation, without touching model internals.
**v4.0.0-alpha.10** adds **model merging / weight averaging (Phase 50)**
— a from-scratch, pure-Rust merging toolkit (a new `merge.rs` module in
the existing `aarambh-studio-weights` crate, no new crate, no new
external dependency) that combines two or more compatible checkpoints
into one via five standard algorithms: linear/Model-Soups, SLERP,
task-vector arithmetic, TIES-Merging, and DARE.

**v4.0.0-alpha.11** adds **public/hosted inference server + prefix caching
(Phase 51)** — the existing `aarambh-studio-serve` server gains opt-in
multi-tenant API-key auth, per-key rate limiting (RPM + TPM), per-tenant
in-flight isolation, and prefix caching (longest-prefix KV reuse with LRU
eviction under a configurable memory ceiling). All three are opt-in; the
loopback-only, unauthenticated single-user mode from v2 §31 remains the
default. Still self-hosted — no billing, no auto-scaling.

**v4.0.0-alpha.12** adds **system role, chat-template versioning, and context
management (Phase 52)** — a formalization / retrofit pass on the model's I/O
contract: a documented, first-class system role (`<|system|>` at id 17, the next
free id — id 7 is `IMAGE` since v2 and is never reassigned), a
`chat_template_version` tag on tokenizer config and checkpoint metadata with a
fail-loud startup mismatch gate, a unified `ContextTruncationPolicy`
(SlidingWindow / Summarize / Reject) referenced by every long-context feature,
and a canonical `docs/SAMPLING_DEFAULTS.md` reference. Not a new capability — a
documentation and versioning pass on what was under-specified.

**v4.0.0-alpha.13** adds **red-team / adversarial safety evaluation (Phase
53)** — a single systematic, end-to-end adversarial-testing pass run once,
near the end of v4.0, against the complete v4.0 attack surface: the safety
layer (§13) and system-turn precedence (§66), the closed-world sandboxed
tool-execution boundary (§61), the orchestrator hard bounds (§62), and the
public-server auth / rate-limit / tenant-isolation surface (§65). A 24-case
corpus (hand-authored / free-public-sourced only) carries a labelled
expected outcome per case; a failing case is surfaced plainly in the report,
never silently dropped. New `aarambh-studio-safety/src/redteam/` module
tree (no new crate, no new external dependency) + `aarambh-studio eval
--redteam --redteam-report <path>` CLI flag. The pass runs without a trained
model — the four boundaries are exercisable with stub executors and an
in-memory key store, the same discipline every per-phase safety test already
uses.

> [!IMPORTANT]
> This is a source and engineering project. It does not publish crates to
> crates.io and does not ship pretrained checkpoints, adapters, GGUF files, or
> compiled binaries. You must train a model or provide compatible weights.

## What Is Included

| Area | Capabilities |
|---|---|
| Model | RMSNorm, RoPE, GQA, SwiGLU, KV cache, tied embeddings, Tiny to Large configs |
| Efficient architecture | YaRN/NTK/linear RoPE scaling, Gated DeltaNet, learned block-sparse DSA, Multi-Head Latent Attention (MLA), fine-grained MoE, sparse/grouped MoE dispatch, MTP |
| Training | BPE data pipeline, AdamW, cosine schedule, gradient accumulation/clipping, checkpoint resume, BF16 CUDA, single-node multi-GPU, on-policy distillation, native INT4/INT8 QAT |
| Fine-tuning | SFT, LoRA, QLoRA, DoRA, QDoRA, VLM adapters, GRPO, DPO, QDPO, RLAIF, tool-call tuning |
| Model merging | Linear/Model-Soups, SLERP, task-vector arithmetic, TIES-Merging, DARE (Phase 50) |
| Inference | Greedy/sampled decoding, streaming, thinking budgets, external or one-checkpoint MTP speculation, tool grammar, caller-executed chains, retrieval-augmented generation (Phase 49) |
| Model formats | SafeTensors, INT8, GPTQ/AWQ INT4, GGUF, Hugging Face conversion, quantized KV cache |
| Evaluation | Perplexity, MMLU-lite, HellaSwag, GSM8K, HumanEval-lite, preference, recall, multimodal/tool scorecards, capability forgetting curves, MoE routing drift, RAG vs no-retrieval delta (Phase 49) |
| Vision | Frozen CLIP-style encoder, image/video/document fusion, temporal and 2D layout encoding, multimodal DoRA/QDoRA tuning |
| Audio | Frozen audio spectrogram transformer, pure-Rust WAV decode + mel-spectrogram, `<audio>` token fusion, audio DoRA/QDoRA tuning (Phase 42) |
| Runtime | CPU SIMD, Rayon attention, optional custom CUDA PTX kernels, Axum 0.8.9 HTTP/SSE server |
| Guardrails | Prompt-injection checks, jailbreak checks, PII redaction, output scanning, streaming token safety, audit logs |
| Serving (Phase 51) | Multi-tenant API-key auth, per-key RPM/TPM rate limits, per-tenant in-flight isolation, prompt-prefix KV caching (LRU, memory-ceiling) — all opt-in; loopback-only remains the default |
| Self-learning | Opt-in critique, replay, verifier rewards, deferred CPU updates, CUDA vision mode, and post-commit forgetting probes |

The implementation history and proof obligations for each feature live in the
[roadmaps](#documentation). This README focuses on building and using the
project.

## Requirements

- Rust 1.89 or newer
- Linux or another platform supported by Candle
- A C/C++ build toolchain for the bundled OpenH264 decoder
- Optional NVIDIA GPU and CUDA toolkit for `--features cuda`
- `nvcc` available at build time for custom CUDA PTX kernels
- Python 3 only for dataset preparation scripts

CPU builds do not require CUDA. Tiny smoke configurations are designed for
local development; Medium and Large training require suitable GPU memory.

## Quick Start

```sh
git clone https://github.com/AarambhDevHub/aarambh-studio.git
cd aarambh-studio

cargo check --workspace --all-targets --locked
cargo test --workspace --locked
cargo build --release --locked -p aarambh-studio
target/release/aarambh-studio --help
```

Run a two-step CPU training smoke test using the checked-in Tiny Shakespeare
fixture:

```sh
target/release/aarambh-studio train \
  --config configs/tiny_shakespeare_smoke.toml
```

Train the normal Tiny recipe:

```sh
target/release/aarambh-studio train \
  --config configs/tiny_shakespeare.toml
```

Training creates a tokenizer, model checkpoints, optimizer state, and
`latest.json`/`best.json` pointers under the configured checkpoint directory.
Smoke checkpoints validate the pipeline; two optimizer steps are not enough to
produce useful language quality.

## CLI

```text
aarambh-studio train       Pretrain or continue a configured model
aarambh-studio infer       Generate text or answer an image/video/document/audio-grounded prompt
aarambh-studio agent       Orchestrate bounded caller-executed tool-use chains
aarambh-studio eval        Run evaluation tasks and compare scorecards
aarambh-studio quantise    Calibrate and export INT8/INT4 GGUF checkpoints
aarambh-studio convert     Convert SafeTensors, GGUF, or Hugging Face layouts
aarambh-studio finetune    Run SFT, adapters, GRPO, DPO, RLAIF, VLM, or merge workflows
aarambh-studio merge       Merge compatible checkpoints (linear/SLERP/TIES/DARE/task-arithmetic)
aarambh-studio distill     Train/evaluate on-policy or offline teacher distillation
aarambh-studio selflearn   Manage replay and persistent self-learning state
aarambh-studio serve       Start the OpenAI-compatible HTTP/SSE server
```

Use `aarambh-studio <command> --help` for the complete option set.

## Common Workflows

See the phase-specific docs for full walkthroughs with smoke fixtures:

| Workflow | Guide |
|---|---|
| Train (CPU/GPU, MTP, MoE, distillation, QAT) | `aarambh-studio train --help` + [configs/](configs/) TOML examples |
| Inference (text, thinking, image, video, document, audio) | [docs/aarambh-studio-complete-guide.md](docs/aarambh-studio-complete-guide.md) |
| Tool-use agent chains | [docs/phase37_agent.md](docs/phase37_agent.md) |
| Sandboxed tool execution | [docs/phase47_sandbox.md](docs/phase47_sandbox.md) |
| Multi-agent orchestration | [docs/phase48_orchestration.md](docs/phase48_orchestration.md) |
| Video understanding | [docs/phase35_video.md](docs/phase35_video.md) |
| Document understanding | [docs/phase36_document.md](docs/phase36_document.md) |
| Audio understanding | [docs/phase42_audio.md](docs/phase42_audio.md) |
| OpenAI-compatible server | [docs/inference-server.md](docs/inference-server.md) |
| Evaluation & forgetting diagnostics | [docs/phase38_forgetting.md](docs/phase38_forgetting.md) |
| Multi-Head Latent Attention (MLA) | [docs/phase41_mla.md](docs/phase41_mla.md) |
| Sparse/grouped MoE dispatch | [docs/phase43_sparse_moe.md](docs/phase43_sparse_moe.md) |
| Quantization (GPTQ, QAT, GGUF) | [docs/phase34_qat.md](docs/phase34_qat.md) |
| MLA hybrid attention & KV-cache report | `aarambh-studio eval --kv-cache-report` + [docs/phase41_mla.md](docs/phase41_mla.md) |
| Fine-tuning (SFT, adapters, GRPO, DPO) | `aarambh-studio finetune --help` |
| Model merging (linear/SLERP/TIES/DARE/task-arithmetic) | [docs/phase50_model_merging.md](docs/phase50_model_merging.md) |
| Multi-tenant serve + prefix caching (Phase 51) | [docs/phase51_public_serve.md](docs/phase51_public_serve.md) |
| System role, chat-template versioning, context policy (Phase 52) | [docs/phase52_system_role_context.md](docs/phase52_system_role_context.md) |
| Red-team / adversarial safety evaluation (Phase 53) | [docs/phase53_redteam.md](docs/phase53_redteam.md) |
| Sampling defaults reference | [docs/SAMPLING_DEFAULTS.md](docs/SAMPLING_DEFAULTS.md) |
| Self-learning | [SELF_LEARNING_V3.md](SELF_LEARNING_V3.md) |

```sh
# Minimal train-smoke → infer flow
cargo build --release --locked -p aarambh-studio
target/release/aarambh-studio train --config configs/tiny_shakespeare_smoke.toml
target/release/aarambh-studio infer --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_shakespeare_smoke/best/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare_smoke/tokenizer.json \
  --prompt "Hello" --max-tokens 16 --greedy
```

## Model Scales

| Scale | Parameters | Hidden | Layers | Heads | KV heads | FFN | Base context | RoPE theta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Tiny | 25M | 384 | 8 | 6 | 2 | 1,024 | 512 | 10,000 |
| Small | 117M | 768 | 12 | 12 | 4 | 2,688 | 1,024 | 10,000 |
| Medium | 360M | 1,024 | 24 | 16 | 8 | 3,392 | 2,048 | 500,000 |
| Large | 1.3B | 2,048 | 24 | 32 | 8 | 6,656 | 4,096 | 500,000 |

All standard scales use a 32,000-token vocabulary, RMSNorm epsilon `1e-5`,
GQA, SwiGLU, and tied embeddings. Long-context and hybrid variants are selected
through TOML without changing the base scale definitions.

## Thinking Modes

| Mode | Budget (tokens) | Default temperature | Default top-p | Use case |
|---|---|---|---|---|
| `none` | 0 | 0.70 | 0.90 | Simple/comparative evals, no reasoning overhead |
| `low` | 256 | 0.75 | 0.92 | Quick factual recall, short-answer tasks |
| `medium` | 1,024 | 0.80 | 0.95 | Standard reasoning, multi-step math/code |
| `high` | 4,096 | 0.80 | 0.95 | Complex multi-step proofs, long analysis |
| `max` | 16,384 | 0.85 | 0.97 | Hard problems unsolved by High (Phase 39) |

The thinking budget is a ceiling on the number of content tokens emitted inside
the `<think>` block before the controller force-closes it. The effective budget
is clamped to `min(mode.budget(), max_new_tokens - reserve)`, so Max never
exceeds the configured generation limit. Sampling defaults are applied only
when the caller does not supply explicit sampling parameters.

All five modes share the same `ThinkingController` mechanism — no structural
changes between modes. Every CLI command (`infer`, `agent`, `serve`, `eval`,
`finetune grpo`, `distill train`, `selflearn start`) accepts `--thinking` with
any of `none`, `low`, `medium`, `high`, or `max`. The server also accepts
`reasoning_effort: "max"` per request.

## Workspace

The workspace contains 20 internal library crates and one CLI package:

```text
aarambh-studio-core        Shared config, device, dtype, errors, and traits
aarambh-studio-tokenizer   BPE tokenizer and reserved special tokens
aarambh-studio-data        Datasets, preprocessing, sharding, and loaders
aarambh-studio-kernel      CPU SIMD and optional CUDA kernels
aarambh-studio-nn          Neural layers, attention, DeltaNet, DSA, MoE, and MTP
aarambh-studio-model       Full decoder model and cache integration
aarambh-studio-weights     SafeTensors, GGUF, conversion, and retrofit loading
aarambh-studio-quant       INT8/INT4, GPTQ, AWQ, QAT, and KV quantization
aarambh-studio-train       Optimizer, schedules, MTP loss, checkpoints, distributed train
aarambh-studio-finetune    Adapters, SFT, GRPO, DPO, VLM, and tool tuning
aarambh-studio-inference   Sampling, caching, thinking, MTP/external speculation, tools
aarambh-studio-agent       Bounded tool chains, exact state, and caller-result ingestion
aarambh-studio-safety      Input, output, streaming, PII, and audit policies
aarambh-studio-selflearn   Critique, replay, verifiers, and persistent update state
aarambh-studio-eval        Evaluation tasks, scorecards, and comparisons
aarambh-studio-vision      Image/video/document decode, preprocessing, temporal/layout fusion
aarambh-studio-audio       WAV decode, mel-spectrogram, frozen audio encoder, fusion (Phase 42)
aarambh-studio-distill     On-policy rollouts, teacher scoring, losses, and resume
aarambh-studio-serve       Axum HTTP/SSE serving and continuous batching
aarambh-studio-retrieve    From-scratch RAG: chunking, embeddings, graph ANN index, prompt augmentation (Phase 49)
aarambh-studio             Command-line application
```

Packages inherit one workspace version and use `publish = false`.

## Development Checks

Run the same primary gates used by CI:

```sh
cargo fmt --all --check
cargo check --workspace --all-targets --locked
cargo test --workspace --no-fail-fast --locked
cargo clippy --workspace --all-targets --locked -- \
  -D warnings -D clippy::undocumented_unsafe_blocks
RUSTDOCFLAGS="-D warnings -D missing_docs" \
  cargo doc --workspace --no-deps --locked
scripts/phase28_release_audit.sh
```

CUDA checks require a CUDA-capable environment and are intentionally opt-in.

## Documentation

| Document | Purpose |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | v1 model, training, inference, safety, and self-learning design |
| [ARCHITECTURE_V2.md](ARCHITECTURE_V2.md) | v2 long context, vision, MoE, distributed, tools, and serving additions |
| [ARCHITECTURE_V3.md](ARCHITECTURE_V3.md) | v3 hybrid attention, DSA, fine-grained MoE, MTP, agents, and forgetting diagnostics |
| [ARCHITECTURE_V4.md](ARCHITECTURE_V4.md) | v4 MLA, audio modality, sparse MoE dispatch, multi-node, test-time compute, RLAIF, sandboxed tools, multi-agent, RAG, merging, public server, system role, red-team, model card |
| [ROADMAP.md](ROADMAP.md) | Completed v1 phases |
| [ROADMAP_V2.md](ROADMAP_V2.md) | Completed v2 phases through the v2.0.0 release |
| [ROADMAP_V3.md](ROADMAP_V3.md) | Completed v3 phases |
| [ROADMAP_V4.md](ROADMAP_V4.md) | Current v4 delivery plan and status |
| [SELF_LEARNING.md](SELF_LEARNING.md) | Text self-learning design |
| [SELF_LEARNING_V2.md](SELF_LEARNING_V2.md) | Vision-aware self-learning design |
| [SELF_LEARNING_V3.md](SELF_LEARNING_V3.md) | v3 self-learning and forgetting-diagnostic integration |
| [SELF_LEARNING_V4.md](SELF_LEARNING_V4.md) | v4 self-learning scope across the v4 feature set |
| [docs/aarambh-studio-config-toml-guide.md](docs/aarambh-studio-config-toml-guide.md) | Configuration field reference |
| [docs/aarambh-studio-complete-guide.md](docs/aarambh-studio-complete-guide.md) | Beginner-oriented project walkthrough |
| [docs/aarambh-studio-math-formulas-guide.md](docs/aarambh-studio-math-formulas-guide.md) | Mathematical foundations and worked examples |
| [docs/inference-server.md](docs/inference-server.md) | Server endpoints, SDK usage, auth, safety, and limits |
| [docs/phase32_mtp.md](docs/phase32_mtp.md) | MTP training, retrofit, exact speculation, and benchmark method |
| [docs/phase33_distillation_results.md](docs/phase33_distillation_results.md) | On-policy distillation design, smoke proof, and comparison method |
| [docs/phase34_qat.md](docs/phase34_qat.md) | Native QAT configuration, continuation, export, and robustness validation |
| [docs/phase35_video.md](docs/phase35_video.md) | Video migration, decoding, tuning, inference, and NExT-QA evaluation |
| [docs/phase36_document.md](docs/phase36_document.md) | PDF/page ingestion, layout tuning, inference, and DocVQA ANLS evaluation |
| [docs/phase37_agent.md](docs/phase37_agent.md) | Tool-chain protocol, safety, context policy, SFT, and response-path evaluation |
| [docs/phase38_forgetting.md](docs/phase38_forgetting.md) | Capability curves, routing drift, training/self-learning hooks, and Manas JSONL |
| [docs/phase41_mla.md](docs/phase41_mla.md) | MLA configuration, retrofit, and KV-cache report |
| [docs/phase42_audio.md](docs/phase42_audio.md) | Audio encoder, mel-spectrogram, fusion, tuning, inference, and audio-QA evaluation |
| [docs/phase43_sparse_moe.md](docs/phase43_sparse_moe.md) | Sparse/grouped dispatch design, CPU/CUDA honesty, and equivalence proof |
| [docs/phase44_multi_node.md](docs/phase44_multi_node.md) | Multi-node topology, TCP rendezvous, single-retry fault policy, and validation paths |
| [docs/phase45_test_time.md](docs/phase45_test_time.md) | Best-of-N, self-consistency, verifier, and process-reward selection at inference time |
| [docs/phase46_rlaif.md](docs/phase46_rlaif.md) | RLAIF judge-model preference-pair generation feeding the existing DPO pipeline |
| [docs/phase47_sandbox.md](docs/phase47_sandbox.md) | Sandboxed tool execution envelope, closed-world allowlist, reference executors, and honesty boundary |
| [docs/phase48_orchestration.md](docs/phase48_orchestration.md) | Multi-agent orchestration: delegation plan, three hard bounds, failure isolation, and composability |
| [docs/phase49_rag.md](docs/phase49_rag.md) | Retrieval-augmented generation: chunking, embedding heads, navigable small-world graph ANN, prompt augmentation, and honesty boundary |
| [docs/phase50_model_merging.md](docs/phase50_model_merging.md) | Model merging / weight averaging: linear, SLERP, task-arithmetic, TIES-Merging, DARE, hard validation, and honesty boundary |
| [RELEASE.md](RELEASE.md) | Source-release process and artifact policy |
| [CHANGELOG.md](CHANGELOG.md) | Versioned implementation history |

## Current Boundaries

- No pretrained model, GGUF, adapter, or binary ships — you train your own.
- MoE uses dense masked dispatch on CPU (sparse dispatch is CUDA-only, Phase 43).
  Multi-node training is data-parallel only (Phase 44), not model/pipeline-parallel.
  Test-time compute scaling (Phase 45) is text-only and ships a heuristic
  process-reward scorer plus a trait for a future trained head.
- Tool chains were historically generated and orchestrated but never
  executed by the runtime; Phase 47 adds opt-in sandboxed execution
  (`agent --execute-tools`) that executes only operator-authorized,
  closed-world named capabilities (e.g. `read_file_in_workdir`) inside a
  bounded envelope. A general-purpose code-execution sandbox remains out
  of scope — execution is strictly closed-world, never arbitrary code or
  shell execution. Phase 48 adds opt-in multi-agent orchestration
  (`agent --orchestrate`) that delegates independent sub-tasks to multiple
  sandboxed tool-execution sub-chains under three hard, operator-set
  bounds (max sub-agent count, total execution time budget, sandbox scope
  containment via `AuthorizationScope::intersect`); sub-chains run
  sequentially by default (CPU-first honest — true parallelism would
  require a `Send + Sync` `ChainDecoder`, out of scope for the source
  release). Phase 49 adds retrieval-augmented generation (`infer --rag`)
  with a from-scratch pure-Rust navigable small-world graph index — no
  external vector database dependency is permitted for the core RAG index
  (an optional plug-in adapter may exist, but the from-scratch index
  remains the default and the tested path); the default tested embedder
  is weight-free (hashing) so the whole pipeline runs without a trained
  embedding checkpoint. Phase 50 adds model merging / weight averaging
  (`aarambh-studio merge`) with five standard algorithms (linear,
  SLERP, task-arithmetic, TIES-Merging, DARE) operating on raw
  `HashMap<String, Tensor>` maps; merging is offline, CPU-first, f32
  math, with hard shape/schema validation before any write. A merged
  checkpoint's quality is measured by the `eval` command, never assumed
  improved — and merging is deliberately kept separate from the online
  self-learning loop (the same framing RLAIF uses).
- Video is visual-only H.264 MP4; audio is WAV PCM only (no MP3/FLAC/Ogg).
- Documents are pixel-based (no OCR/table parser).
- The server is local/single-model by default; vision, audio, and
  self-learning are CLI workflows. Phase 51 (`v4.0.0-alpha.11`) adds opt-in
  multi-tenant API-key auth, per-key rate limits, per-tenant in-flight
  isolation, and prompt-prefix KV caching to `aarambh-studio-serve` — but
  this makes public multi-tenant self-hosting *possible*, not a hosted
  product. There is no billing system and no horizontal auto-scaling. The
  loopback-only, unauthenticated mode remains the recommended default for
  single-user, local use.

Full exclusions in the [versioned roadmaps](#documentation).

## Contributing And Security

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Use
[GitHub issues](https://github.com/AarambhDevHub/aarambh-studio/issues) for
reproducible bugs and scoped feature requests. Report vulnerabilities through
[SECURITY.md](SECURITY.md), not a public issue.

## Citation

```bibtex
@software{aarambh_studio_2026,
  title   = {aarambh-studio: A Ground-Up Language Model System in Rust},
  author  = {Aarambh Dev Hub},
  year    = {2026},
  url     = {https://github.com/AarambhDevHub/aarambh-studio},
  version = {4.0.0-alpha.13},
  license = {Apache-2.0}
}
```

## License

Licensed under the [Apache License 2.0](LICENSE).
