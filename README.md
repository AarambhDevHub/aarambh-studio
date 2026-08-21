# aarambh-studio

> From first principles. From zero. From Rust.
>
> Sanskrit: *beginning* — a ground-up language model system in Rust.

[![CI](https://github.com/AarambhDevHub/aarambh-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/AarambhDevHub/aarambh-studio/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.89%2B-orange.svg)](https://www.rust-lang.org)

aarambh-studio is a decoder-only language model implementation built with Rust
and Candle. The repository covers the full engineering path: tokenization,
model construction, training, inference, quantization, adapter tuning,
alignment, evaluation, multimodal input, safety, and an OpenAI-compatible
server.

**v4.0.0 is the final planned source release.** It completes the v4 arc
(Phases 41–55) on top of the finished v3.0.0 base and reaches a
deliberately-declared end state — no v5 roadmap exists as of this release.
Every workspace crate ships with `publish = false`; no crate is published to
crates.io, and no pretrained checkpoint, adapter, tokenizer, optimizer state,
SafeTensors, or GGUF file is attached. See the
[v4.0.0 release notes](.github/release-notes/v4.0.0.md).

### What v4.0.0 adds (Phases 41–55)

| Phase | Capability |
|---|---|
| 41 | Multi-Head Latent Attention (MLA) — the third attention kind, latent KV compression |
| 42 | Native Audio modality — WAV decode, mel-spectrogram, frozen encoder, `<audio>` fusion |
| 43 | Sparse/grouped MoE dispatch (CUDA); dense masked path retained on CPU |
| 44 | Multi-node data-parallel training via TCP rendezvous, single-retry fault policy |
| 45 | Test-time compute scaling — Best-of-N, self-consistency, verifier, process reward |
| 46 | RLAIF — judge-scored preference pairs feeding the existing DPO pipeline |
| 47 | Sandboxed closed-world tool execution (`agent --execute-tools`) |
| 48 | Multi-agent orchestration — delegating to sandboxed sub-chains under hard bounds |
| 49 | Retrieval-Augmented Generation — from-scratch pure-Rust navigable small-world graph |
| 50 | Model merging — linear, SLERP, task-arithmetic, TIES-Merging, DARE |
| 51 | Public inference server — opt-in multi-tenant auth, rate limits, prefix caching |
| 52 | System role, chat-template versioning, unified context-truncation policy |
| 53 | Red-team / adversarial safety evaluation (24-case, four-surface corpus) |
| 54 | Model card — assembled (not hand-written) from eval + red-team + metadata |
| 55 | Final source release — version freeze, audit extension, documentation finalised |

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
| Model merging | Linear/Model-Soups, SLERP, task-vector arithmetic, TIES-Merging, DARE |
| Inference | Greedy/sampled decoding, streaming, thinking budgets, external or one-checkpoint MTP speculation, tool grammar, caller-executed chains, retrieval-augmented generation |
| Model formats | SafeTensors, INT8, GPTQ/AWQ INT4, GGUF, Hugging Face conversion, quantized KV cache |
| Evaluation | Perplexity, MMLU-lite, HellaSwag, GSM8K, HumanEval-lite, preference, recall, multimodal/tool scorecards, capability forgetting curves, MoE routing drift, RAG delta |
| Vision | Frozen CLIP-style encoder, image/video/document fusion, temporal and 2D layout encoding, multimodal DoRA/QDoRA tuning |
| Audio | Frozen audio spectrogram transformer, pure-Rust WAV decode + mel-spectrogram, `<audio>` token fusion, audio DoRA/QDoRA tuning |
| Runtime | CPU SIMD, Rayon attention, optional custom CUDA PTX kernels, Axum 0.8.9 HTTP/SSE server |
| Guardrails | Prompt-injection checks, jailbreak checks, PII redaction, output scanning, streaming token safety, audit logs |
| Serving | Multi-tenant API-key auth, per-key RPM/TPM rate limits, per-tenant in-flight isolation, prompt-prefix KV caching — all opt-in; loopback-only remains the default |
| Self-learning | Opt-in critique, replay, verifier rewards, deferred CPU updates, CUDA vision mode, post-commit forgetting probes |

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

Run a two-step CPU training smoke test using the Tiny Shakespeare fixture:

```sh
target/release/aarambh-studio train \
  --config configs/tiny_shakespeare_smoke.toml
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

| Workflow | Guide |
|---|---|
| Train (CPU/GPU, MTP, MoE, distillation, QAT) | `aarambh-studio train --help` + [configs/](configs/) |
| Inference (text, thinking, image, video, document, audio) | [docs/aarambh-studio-complete-guide.md](docs/aarambh-studio-complete-guide.md) |
| Tool-use agent chains | [docs/phase37_agent.md](docs/phase37_agent.md) |
| Sandboxed tool execution | [docs/phase47_sandbox.md](docs/phase47_sandbox.md) |
| Multi-agent orchestration | [docs/phase48_orchestration.md](docs/phase48_orchestration.md) |
| Video / Document / Audio understanding | [docs/phase35_video.md](docs/phase35_video.md) · [docs/phase36_document.md](docs/phase36_document.md) · [docs/phase42_audio.md](docs/phase42_audio.md) |
| OpenAI-compatible server | [docs/inference-server.md](docs/inference-server.md) |
| Evaluation & forgetting diagnostics | [docs/phase38_forgetting.md](docs/phase38_forgetting.md) |
| MLA hybrid attention & KV-cache report | [docs/phase41_mla.md](docs/phase41_mla.md) |
| Sparse/grouped MoE dispatch | [docs/phase43_sparse_moe.md](docs/phase43_sparse_moe.md) |
| Quantization (GPTQ, QAT, GGUF) | [docs/phase34_qat.md](docs/phase34_qat.md) |
| Fine-tuning (SFT, adapters, GRPO, DPO) | `aarambh-studio finetune --help` |
| Model merging | [docs/phase50_model_merging.md](docs/phase50_model_merging.md) |
| Multi-tenant serve + prefix caching | [docs/phase51_public_serve.md](docs/phase51_public_serve.md) |
| System role, chat-template versioning, context policy | [docs/phase52_system_role_context.md](docs/phase52_system_role_context.md) |
| Red-team / adversarial safety evaluation | [docs/phase53_redteam.md](docs/phase53_redteam.md) |
| Model card & release documentation standard | [docs/phase54_model_card.md](docs/phase54_model_card.md) |
| Sampling defaults reference | [docs/SAMPLING_DEFAULTS.md](docs/SAMPLING_DEFAULTS.md) |
| Self-learning | [SELF_LEARNING_V4.md](SELF_LEARNING_V4.md) |

```sh
# Minimal train-smoke -> infer flow
cargo build --release --locked -p aarambh-studio
target/release/aarambh-studio train --config configs/tiny_shakespeare_smoke.toml
target/release/aarambh-studio infer --config configs/tiny_shakespeare_smoke.toml \
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
| `max` | 16,384 | 0.85 | 0.97 | Hard problems unsolved by High |

The thinking budget is a ceiling on content tokens emitted inside the thinking
block before the controller force-closes it, clamped to
`min(mode.budget(), max_new_tokens - reserve)`. Every CLI command (`infer`,
`agent`, `serve`, `eval`, `finetune grpo`, `distill train`, `selflearn start`)
accepts `--thinking` with any of `none`/`low`/`medium`/`high`/`max`. The server
also accepts `reasoning_effort: "max"` per request.

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
aarambh-studio-audio       WAV decode, mel-spectrogram, frozen audio encoder, fusion
aarambh-studio-distill     On-policy rollouts, teacher scoring, losses, and resume
aarambh-studio-serve       Axum HTTP/SSE serving and continuous batching
aarambh-studio-retrieve    From-scratch RAG: chunking, embeddings, graph ANN, prompt augmentation
aarambh-studio             Command-line application
```

Packages inherit one workspace version (`4.0.0`) and use `publish = false`.

## Development Checks

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
| [ARCHITECTURE_V2.md](ARCHITECTURE_V2.md) | v2 long context, vision, MoE, distributed, tools, and serving |
| [ARCHITECTURE_V3.md](ARCHITECTURE_V3.md) | v3 hybrid attention, DSA, fine-grained MoE, MTP, agents, forgetting |
| [ARCHITECTURE_V4.md](ARCHITECTURE_V4.md) | v4 MLA, audio, sparse MoE, multi-node, test-time, RLAIF, sandbox, multi-agent, RAG, merging, public server, system role, red-team, model card |
| [ROADMAP.md](ROADMAP.md) · [V2](ROADMAP_V2.md) · [V3](ROADMAP_V3.md) · [V4](ROADMAP_V4.md) | Completed phases per version |
| [SELF_LEARNING.md](SELF_LEARNING.md) · [V2](SELF_LEARNING_V2.md) · [V3](SELF_LEARNING_V3.md) · [V4](SELF_LEARNING_V4.md) | Self-learning design per version |
| [docs/aarambh-studio-config-toml-guide.md](docs/aarambh-studio-config-toml-guide.md) | Configuration field reference |
| [docs/aarambh-studio-complete-guide.md](docs/aarambh-studio-complete-guide.md) | Beginner-oriented project walkthrough |
| [docs/aarambh-studio-math-formulas-guide.md](docs/aarambh-studio-math-formulas-guide.md) | Mathematical foundations and worked examples |
| [docs/inference-server.md](docs/inference-server.md) | Server endpoints, SDK usage, auth, safety, and limits |
| [docs/phase32_mtp.md](docs/phase32_mtp.md) … [docs/phase54_model_card.md](docs/phase54_model_card.md) | Per-phase runbooks (see `docs/`) |
| [docs/model_card_template.md](docs/model_card_template.md) | Model card template & field guide |
| [RELEASE.md](RELEASE.md) | Source-release process and artifact policy |
| [CHANGELOG.md](CHANGELOG.md) | Versioned implementation history |

## Current Boundaries

- No pretrained model, GGUF, adapter, or binary ships — you train your own.
- MoE uses dense masked dispatch on CPU; sparse dispatch is CUDA-only.
- Multi-node training is data-parallel only, not model/pipeline-parallel.
- Test-time compute scaling is text-only; ships a heuristic process-reward scorer.
- Tool execution is strictly closed-world named capabilities — never arbitrary code or shell.
- Multi-agent sub-chains run sequentially by default (CPU-first honest).
- RAG's core index is the from-scratch pure-Rust graph; no external vector DB for the core index.
- Video is visual-only H.264 MP4; audio is WAV PCM only; documents are pixel-based (no OCR).
- The server is local/single-model by default; Phase 51 makes multi-tenant self-hosting *possible*, not a hosted product — no billing, no auto-scaling.
- v4.0.0 is the final planned version; no v5 roadmap exists.

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
  version = {4.0.0},
  license = {Apache-2.0}
}
```

## License

Licensed under the [Apache License 2.0](LICENSE).
