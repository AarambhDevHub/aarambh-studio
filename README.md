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
Max thinking mode (16,384-token budget). **v4.0.0-alpha.4** continues the v4 arc
with Multi-Head Latent Attention (Phase 41), a native Audio modality
(Phase 42), sparse/grouped MoE dispatch (Phase 43), and multi-node
distributed training (Phase 44) — a frozen audio
spectrogram transformer plus trainable projector that lets the model hear and
reason about audio clips (the same frozen-encoder-plus-projector recipe vision,
video, and documents use), real sparse expert dispatch where each token
computes only its assigned top-k experts rather than every expert on every
token then masked (numerically equivalent to the dense path, faster on CUDA),
and data-parallel training extended across multiple nodes over a TCP
rendezvous so the world can scale past a single machine's GPU count.

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
| Fine-tuning | SFT, LoRA, QLoRA, DoRA, QDoRA, VLM adapters, GRPO, DPO, QDPO, tool-call tuning |
| Inference | Greedy/sampled decoding, streaming, thinking budgets, external or one-checkpoint MTP speculation, tool grammar, caller-executed chains |
| Model formats | SafeTensors, INT8, GPTQ/AWQ INT4, GGUF, Hugging Face conversion, quantized KV cache |
| Evaluation | Perplexity, MMLU-lite, HellaSwag, GSM8K, HumanEval-lite, preference, recall, multimodal/tool scorecards, capability forgetting curves, and MoE routing drift |
| Vision | Frozen CLIP-style encoder, image/video/document fusion, temporal and 2D layout encoding, multimodal DoRA/QDoRA tuning |
| Audio | Frozen audio spectrogram transformer, pure-Rust WAV decode + mel-spectrogram, `<audio>` token fusion, audio DoRA/QDoRA tuning (Phase 42) |
| Runtime | CPU SIMD, Rayon attention, optional custom CUDA PTX kernels, Axum 0.8.9 HTTP/SSE server |
| Guardrails | Prompt-injection checks, jailbreak checks, PII redaction, output scanning, streaming token safety, audit logs |
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
aarambh-studio finetune    Run SFT, adapters, GRPO, DPO, VLM, or merge workflows
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

The workspace contains 19 internal library crates and one CLI package:

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
| [RELEASE.md](RELEASE.md) | Source-release process and artifact policy |
| [CHANGELOG.md](CHANGELOG.md) | Versioned implementation history |

## Current Boundaries

- No pretrained model, GGUF, adapter, or binary ships — you train your own.
- MoE uses dense masked dispatch on CPU (sparse dispatch is CUDA-only, Phase 43).
  Multi-node training is data-parallel only (Phase 44), not model/pipeline-parallel.
- Tool chains are generated and orchestrated but never executed by the runtime.
- Video is visual-only H.264 MP4; audio is WAV PCM only (no MP3/FLAC/Ogg).
- Documents are pixel-based (no OCR/table parser).
- The server is local/single-model; vision, audio, and self-learning are CLI workflows.

Full exclusions in the [versioned roadmaps](#documentation).

## Contributing And Security

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Use
[GitHub issues](https://github.com/AarambhDevHub/aarambh-studio/issues) for
reproducible bugs and scoped feature requests. Report vulnerabilities through
[SECURITY.md](SECURITY.md), not a public issue.

## Citation

```bibtex
@software{aarambh_ai_2026,
  title   = {aarambh-studio: A Ground-Up Language Model System in Rust},
  author  = {Aarambh Dev Hub},
  year    = {2026},
  url     = {https://github.com/AarambhDevHub/aarambh-studio},
  version = {4.0.0-alpha.4},
  license = {Apache-2.0}
}
```

## License

Licensed under the [Apache License 2.0](LICENSE).
