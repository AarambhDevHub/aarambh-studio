# Aarambh Studio — Docs

> From first principles. From zero. From Rust.

This folder holds the learning material behind Aarambh Studio — a from-scratch decoder-only LLM built in Rust using Candle. If you've ever looked at this repo and wondered *"okay but how does any of this actually work?"*, start here.

These docs aren't API references or code comments. They're written for someone coming in with **zero background in AI/ML** — a beginner who codes but has never touched a neural network before. The goal is that by the end of these guides, you understand not just *what* Aarambh Studio does, but *why* every piece exists and *how* the math underneath it actually works.

---

## What's in this folder

### 1. `aarambh-studio-complete-guide.md`
**The full project walkthrough — every phase, explained.**

This covers the 28 production phases of Aarambh Studio, from v1.0.0 through the
v2.0.0 source release. v3 implementation runbooks are listed separately below:

- **v1 (Phases 1–13):** Tokenizer, Data Pipeline, Neural Network Primitives, Full Model Forward Pass, Custom Kernels (CPU SIMD + GPU prep), Training Loop, Inference Engine + CLI, Thinking Engine, Quantization Stack, Fine-Tuning (LoRA/QLoRA/SFT), GRPO Reinforcement Learning, Safety Layer, Self-Learning.
- **v2 (Phases 14–28):** GPU Scale-Up, Flash Attention CUDA Kernels, Long Context (RoPE scaling), Evaluation Harness, DoRA Fine-Tuning, Vision Encoder + Projector, Vision-Language Training, Vision-Aware Self-Learning, Mixture of Experts, Multi-GPU Training, DPO Preference Tuning, Speculative Decoding, Tool Use / Function Calling, Inference Server, and the v2.0.0 production source release.

Each phase includes a plain-English definition, a beginner explanation, why it's needed, a worked example, and a diagram. Read this first — it's the map of the whole project.

### 2. `aarambh-studio-complete-guide-v3.md`
**The v3.0.0 walkthrough — Phases 29-40, explained in the same style.**

Picks up where the v2 complete guide ends, covering:
- **Phase 29-30:** Gated DeltaNet (hybrid linear attention) + DeepSeek Sparse Attention
- **Phase 31:** Fine-Grained MoE + Shared Expert
- **Phase 32:** Multi-Token Prediction (MTP)
- **Phase 33:** On-Policy Distillation
- **Phase 34:** Native QAT (Quantization-Aware Training)
- **Phase 35-36:** Native Video + Document Understanding
- **Phase 37:** Long-Horizon Tool-Use Chains
- **Phase 38:** Forgetting Diagnostics
- **Phase 39:** Max Thinking Mode
- **Phase 40:** v3.0.0 Source Release

Same format as the v2 guide: plain-English definition, beginner explanation, why we need it, worked example, diagram, and common questions. Read this after the v2 guide to see how the model evolved.

### 3. `phase35_video.md`
**The native-video implementation and operating guide.**

This documents the H.264 MP4 boundary, frame sampling, tokenizer/checkpoint
migration, temporal fusion, video DoRA/QDoRA tuning, inference, NExT-QA
evaluation, smoke workflow, and memory controls introduced in Phase 35.

### 3. `phase36_document.md`
**The native-document implementation and operating guide.**

This documents PDF/page rendering, 2D layout projection, vocabulary migration,
document DoRA/QDoRA tuning, inference, DocVQA-style import, ANLS evaluation,
the local smoke workflow, and resource limits introduced in Phase 36.

### 4. `phase37_agent.md`
**The long-horizon tool-chain protocol and operating guide.**

This documents caller-executed result ingestion, exact-token continuation,
context eviction/summarisation, multimodal result lifetime, safety checks,
multi-step SFT, scripted evaluation, and the BFCL response-path boundary.

### 4b. `phase47_sandbox.md`
**The sandboxed tool-execution runbook (Phase 47).**

This documents the closed-world `ToolExecutor` model, operator
authorization (`AuthorizationScope`), the bounded execution envelope
(wall-clock timeout + output/argument-size ceilings + schema
re-validation), the `SandboxedToolProvider` composability bridge into the
existing `ToolChain`, the reference `read_file_in_workdir` and `lookup`
executors, the `agent --execute-tools` CLI surface, the smoke workflow,
and the honesty boundary (pure-Rust CPU sandbox, no OS-level isolation).

### 4c. `cli-commands.md`
**The full CLI command reference — every command, every flag, a worked example for each.**

This is a human-readable index over every subcommand's `--help` output
(generated for `4.0.0-alpha.7`): `train`, `infer`, `agent` (incl. the
Phase 47 `--execute-tools` sandboxed-execution flags), `eval`, `quantise`,
`convert`, `finetune` (all subcommands incl. `rlaif`), `distill`,
`selflearn`, and `serve`. The verbatim `--help` output for every command
is also kept in `cli-commands-raw-help.txt` as an appendix.

### 5. `phase38_forgetting.md`
**Capability regression, MoE routing drift, and the Manas JSONL bridge.**

This documents the fixed probe manifest, persistent multi-point curves,
training and self-learning observers, standalone checkpoint comparisons,
significance semantics, routing diagnostics, and the dependency-free Manas
interchange contract introduced in Phase 38.

### 6. `phase39_max_thinking_results.md`
**Max thinking mode: the 16,384-token budget, sampling defaults, and High-vs-Max comparison.**

This documents the fifth `ThinkingMode` variant (`Max`, 16,384 tokens), the
centralised `none|low|medium|high|max` parsing/display, the per-mode sampling
defaults, the deterministic `hard-problems` eval task, the commands, expected
outputs, and the High-vs-Max comparison table introduced in Phase 39.

### 7. `aarambh-studio-math-formulas-guide.md`
**The math underneath v1-v2 phases, explained from zero.**

Once you know *what* each phase does, this file explains the actual formulas doing the work — Dot Product, Matrix Multiplication, Softmax, Scaled Dot-Product Attention, Layer Normalization, GELU Activation, Cross-Entropy Loss, Gradient Descent, Adam Optimizer, RoPE, LoRA Decomposition, Quantization, KL Divergence, and Perplexity.

Every formula comes with a symbol-by-symbol translation (so `Σ`, `∂`, `θ` stop looking scary) and **two fully solved numeric examples** worked by hand, step by step. Read this after the phases guide, whenever you want to understand the actual arithmetic behind a specific phase.

### 8. `aarambh-studio-math-formulas-guide-v3.md`
**The math behind v3 phases (Formulas 15-21).**

Covers the new arithmetic introduced in Phases 29-40:
- **Formula 15:** Gated DeltaNet Gating + Recurrence
- **Formula 16:** Top-k Sparse Attention
- **Formula 17:** Fine-Grained MoE Router (top-2 from 16 experts + shared expert)
- **Formula 18:** Multi-Token Prediction Loss (3-head joint cross-entropy)
- **Formula 19:** On-Policy KL Distillation
- **Formula 20:** Fake Quantization + Straight-Through Estimator
- **Formula 21:** Forgetting Delta (Probe Accuracy Drop)

Same format as the v2 math guide: symbol-by-symbol translation and two fully solved numeric examples per formula. Read this after the v3 phases guide when you want to see the actual numbers.

### 10. `ai-ml-dl-dataset-creation-guide.md`
**The foundation underneath everything — terminology and where the training data comes from.**

Two parts:
- **Part 1** untangles the terminology soup: AI, Machine Learning, Deep Learning, Neural Networks, NLP, LLMs, Generative AI, and the three types of ML (supervised, unsupervised, reinforcement) — how they all nest inside each other.
- **Part 2** walks through the practical pipeline of building a real training dataset: web scraping, public dumps (Common Crawl, Wikipedia, Gutenberg), APIs, cleaning, deduplication, filtering, PII removal, formatting into JSONL, train/val/test splitting, and licensing/ethics.

Read this first if you're completely new to AI in general, or read it alongside the other two whenever data-related phases (like Phase 2, Data Pipeline) come up.

### 11. `aarambh-studio-config-toml-guide.md`
**Every field in every `.toml` config file, explained — the practical "turn the dial" layer.**

This walks through the checked-in training and inference configurations in
`configs/` — Tiny/Small/Medium/Large, CUDA, long-context, MoE, distributed,
vision, video, and smoke configurations — field by field:

- **Top-level settings:** `dataset_path`, `tokenizer_path`, `vocab_size`, `validation_split`, `shuffle`, `resume`, `device`, `dtype`.
- **`[model]` architecture:** `hidden_dim`, `ffn_dim`, `n_layers`, `n_heads`/`n_kv_heads` (Grouped-Query Attention), `max_seq_len`, `rope_theta`, `norm_eps`, `tie_embeddings`.
- **`[model.rope_scaling]` (YaRN):** `method`, `factor`, `original_max_seq_len`, `beta_fast`/`beta_slow`, `attn_factor` — how the long-context configs stretch to 16K tokens.
- **`[[context_schedule]]`:** the staged sequence-length ramp-up used during long-context training.
- **`[vision]`:** image/projector paths and VLM limits.
- **`[vision.video]`:** video root, sampling budget, temporal encoding,
  frozen-feature cache, and encoder batch size.
- **`[vision.document]`:** document root, PDF DPI/page/pixel limits, 2D
  layout encoding, feature cache, and encoder page batch size.
- **`[train]` hyperparameters:** `lr`, `batch_size`/`grad_accum_steps`, `warmup_steps`, `min_lr_ratio`, `weight_decay`, Adam's `beta1`/`beta2`/`epsilon`, `clip_grad_norm`, checkpointing, and more — each one tied back to the exact formula it came from in the math guide.

Read this whenever you're about to write a new training config, or want to understand exactly what a specific field in an existing one actually does.

---

## Suggested reading order

If you're starting from zero:

```
ai-ml-dl-dataset-creation-guide.md       →  terminology + data origins
            │
            ▼
aarambh-studio-complete-guide.md             →  Aarambh Studio v1-v2 phases
            │
            ▼
aarambh-studio-math-formulas-guide.md        →  v1-v2 math (formulas 1-14)
            │
            ├─────────────────────────────
            │                             │
            ▼                             ▼
aarambh-studio-complete-guide-v3.md      aarambh-studio-math-formulas-guide-v3.md
(v3 phases 29-40)                    (v3 math formulas 15-21)
            │                             │
            └──────────┬──────────────────┘
                       ▼
        aarambh-studio-config-toml-guide.md   →  how to configure & run training
```

If you already know the basics and just want the project-specific details, jump to the guide matching your version — `aarambh-studio-complete-guide.md` for v1-v2, `aarambh-studio-complete-guide-v3.md` for v3.

---

## Who this is for

- Anyone reading the Aarambh Studio codebase for the first time and wondering what a given module actually does.
- Contributors who want to understand a phase deeply enough to help extend it.
- Future-me, six months from now, who forgot why a formula was written a certain way.

No prior ML background is assumed anywhere in these guides. If something is still unclear after reading, that's a gap in the doc, not a gap in you — feel free to open an issue.

---

## Keeping these docs updated

The v2 roadmap is complete. v3 changes must update the matching runbook,
configuration reference, architecture section, changelog, and V3 guides in the same pull request.

---

## Support Aarambh Studio

If these docs or the project itself helped you, consider supporting the work:

- ☕ [Buy Me a Coffee](https://www.buymeacoffee.com/aarambhdevhub)
- 💖 [GitHub Sponsors](https://github.com/sponsors/aarambh-darshan)
- 🎓 [Topmate](https://topmate.io/darshan_vichhi) — 1-on-1 mentoring and paid sessions
- 🪙 [Razorpay](https://razorpay.me/@aarambhdevhub) — for India-based support

Every bit helps keep this project — and the free educational content around it — going.

---

*Part of the Aarambh Dev Hub ecosystem. Built with Rust, one phase at a time.*
