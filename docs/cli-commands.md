# aarambh-studio CLI — Full Command Reference

> From first principles. From zero. From Rust.
>
> Every command, every flag, and a worked example for each — auto-generated
> from the binary's own `--help` output (`aarambh-studio 4.0.0-alpha.7`).
>
> Read this top to bottom the first time; use it as a reference thereafter.

---

## Quick start

```sh
git clone https://github.com/AarambhDevHub/aarambh-studio.git
cd aarambh-studio

# Build the CLI (CPU; CUDA is opt-in via --features cuda)
cargo build --release --locked -p aarambh-studio
target/release/aarambh-studio --version
target/release/aarambh-studio --help
```

The single binary is `target/release/aarambh-studio`. It has ten top-level
subcommands. Pass `--help` to any of them (or their sub-subcommands) for the
exact, current flag set:

```sh
target/release/aarambh-studio <command> --help
target/release/aarambh-studio finetune <subcommand> --help
target/release/aarambh-studio distill <subcommand> --help
target/release/aarambh-studio selflearn <subcommand> --help
```

---

## Command overview

| Command | Purpose |
|---|---|
| `train` | Pretrain or continue a configured model |
| `infer` | Generate text or answer an image/video/document/audio-grounded prompt |
| `agent` | Orchestrate bounded caller-executed tool-use chains (Phase 47: sandboxed execution) |
| `eval` | Run evaluation tasks and compare scorecards |
| `quantise` | Calibrate and export INT8/INT4 GGUF checkpoints |
| `convert` | Convert SafeTensors, GGUF, or Hugging Face layouts |
| `finetune` | SFT, adapters (LoRA/QLoRA/DoRA/QDoRA, VLM), GRPO, DPO, RLAIF, tool-call tuning, merge |
| `distill` | On-policy / offline teacher distillation (train + evaluate) |
| `selflearn` | Manage replay and persistent self-learning state |
| `serve` | Start the local OpenAI-compatible HTTP/SSE inference server |

Every command accepts `-h` / `--help` and (for the top-level) `-V` / `--version`.

---

## 1. `train` — pretrain or continue a model

**Purpose:** run the optimizer against a TOML-configured dataset + architecture,
producing checkpoints + optimizer state + `latest.json`/`best.json` pointers.

```sh
target/release/aarambh-studio train --config configs/tiny_shakespeare_smoke.toml
target/release/aarambh-studio train --config configs/wikitext103_small.toml
```

<details>
<summary><code>--help</code> output</summary>

```
Usage: aarambh-studio train --config <CONFIG>

Options:
      --config <CONFIG>
  -h, --help             Print help
```
</details>

---

## 2. `infer` — generate / answer a grounded prompt

**Purpose:** load a checkpoint + tokenizer and generate text, optionally
grounded in an image, video, document, or audio clip; with thinking budgets,
speculative decoding, tool calling, safety, self-learning, forgetting
diagnostics, and Best-of-N test-time compute scaling.

```sh
# Plain text generation
target/release/aarambh-studio infer \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_shakespeare_smoke/best/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare_smoke/tokenizer.json \
  --prompt "Hello" --max-tokens 16 --greedy

# Image-grounded, streaming, high thinking
target/release/aarambh-studio infer \
  --config configs/vision_vqa_smoke.toml \
  --prompt "Describe this image." --image data/sandbox_workdir/photo.jpg \
  --thinking high --stream --max-tokens 128

# Best-of-N test-time compute (Phase 45), verifier selection
target/release/aarambh-studio infer \
  --config configs/tiny_shakespeare.toml \
  --prompt "What is 17*23?" --best-of-n 8 --selection verifier \
  --ground-truth "391" --temperature 0.8
```

<details>
<summary><code>--help</code> output (full flag list)</summary>

```
Usage: aarambh-studio infer [OPTIONS] --prompt <PROMPT>

Options:
      --config <CONFIG>                      [default: configs/tiny_shakespeare.toml]
      --model <MODEL>
      --tokenizer <TOKENIZER>
      --image <IMAGE>
      --video <VIDEO>
      --document <DOCUMENT>
      --audio <AUDIO>
      --pages <PAGES>
      --document-dpi <DOCUMENT_DPI>
      --max-document-pages <MAX_DOCUMENT_PAGES>
      --frames <FRAMES>
      --frame-sampling <FRAME_SAMPLING>
      --prompt <PROMPT>
      --max-tokens <MAX_TOKENS>             [default: 256]
      --temperature <TEMPERATURE>           [default: 0.7]
      --top-p <TOP_P>                        [default: 0.9]
      --top-k <TOP_K>                        [default: 50]
      --seed <SEED>
      --thinking <THINKING>                 none|low|medium|high|max  [default: none]
      --predict-view
      --stream
      --greedy
      --speculative
      --draft-model <DRAFT_MODEL>
      --draft-config <DRAFT_CONFIG>
      --draft-tokenizer <DRAFT_TOKENIZER>
      --draft-tokens <DRAFT_TOKENS>
      --stats
      --tools <TOOLS>
      --tool-choice <TOOL_CHOICE>           [default: auto]
      --safety <SAFETY>                     [default: strict]
      --safety-audit-log <SAFETY_AUDIT_LOG> [default: safety_audit.jsonl]
      --self-learn <SELF_LEARN>             [default: disabled]
      --replay-path <REPLAY_PATH>
      --self-learn-state-dir <SELF_LEARN_STATE_DIR>      [default: adapters/selflearn]
      --self-learn-reference <SELF_LEARN_REFERENCE>
      --self-learn-verifier <SELF_LEARN_VERIFIER>        [default: none]
      --self-learn-vision-verifier <SELF_LEARN_VISION_VERIFIER>  [default: none]
      --self-learn-ground-truth <SELF_LEARN_GROUND_TRUTH>
      --forgetting-manifest <FORGETTING_MANIFEST>
      --forgetting-store <FORGETTING_STORE>
      --forgetting-jsonl <FORGETTING_JSONL>
      --forgetting-threshold <FORGETTING_THRESHOLD>       [default: 0.02]
      --forgetting-max-examples <FORGETTING_MAX_EXAMPLES> [default: 8]
      --forgetting-allow-code-exec
      --forgetting-require-all-probes
      --forgetting-baseline-id <FORGETTING_BASELINE_ID>
      --best-of-n <BEST_OF_N>               Phase 45: N independent candidates, select best
      --selection <SELECTION>               verifier|self-consistency|majority|process-reward  [default: self-consistency]
      --ground-truth <GROUND_TRUTH>         required with --selection verifier
  -h, --help
```
</details>

---

## 3. `agent` — bounded tool-use chains (Phase 47: sandboxed execution)

**Purpose:** orchestrate a multi-step, long-horizon tool-use chain. The model
emits grammar-constrained JSON tool calls; the chain feeds results back and
keeps exact-token state. By default the **caller** executes tools (stdin or a
`--results` replay file). With `--execute-tools` (Phase 47), aarambh-studio
**itself executes** the calls inside a closed-world sandbox.

### Emit-only (caller executes) — v3 §46 behavior

```sh
# Interactive: read one ToolResult JSON line from stdin per tool call
target/release/aarambh-studio agent \
  --config configs/tiny_shakespeare.toml \
  --tools data/agent_tools_smoke.json \
  --prompt "Look up customer C-42 and their latest order." \
  --max-steps 4 --jsonl

# Scripted replay (deterministic, for eval)
target/release/aarambh-studio agent \
  --config configs/tiny_shakespeare.toml \
  --tools data/agent_tools_smoke.json \
  --results data/agent_chain_smoke.jsonl \
  --prompt "Summarise the shipping quote."
```

### Sandboxed execution — Phase 47 (NEW)

The model's tool calls are executed by aarambh-studio itself, but only inside
a strict sandbox: a closed-world allowlist of named executors, operator
authorization, schema re-validation, a wall-clock timeout, and output/argument
size ceilings. There is **no generic shell/eval executor, ever**.

```sh
# Operator must authorize each executable tool by name (--allow-tool,
# repeatable). Without it, --execute-tools is a hard error before any model
# is loaded.
target/release/aarambh-studio agent \
  --config configs/tiny_shakespeare.toml \
  --tools data/tools_sandbox_smoke.json \
  --prompt "Read notes.txt and summarise it." \
  --execute-tools \
  --allow-tool read_file_in_workdir \
  --exec-workdir ./data/sandbox_workdir \
  --exec-timeout-ms 2000 \
  --exec-max-output-bytes 65536 \
  --max-steps 4
```

The five new flags (Phase 47):

| Flag | Default | Purpose |
|---|---|---|
| `--execute-tools` | off | Switch from caller-executed stdin/replay to sandboxed execution |
| `--allow-tool <NAME>` | (none, repeatable) | Operator authorization. **At least one is required** with `--execute-tools`. |
| `--exec-timeout-ms` | 5000 | Per-call wall-clock ceiling (CPU ceiling for compute-bound work) |
| `--exec-max-output-bytes` | 65536 | Maximum output payload bytes a tool may return |
| `--exec-workdir <DIR>` | (none) | Binds the `read_file_in_workdir` executor to a directory (read-only, traversal-refused) |

**Safety boundaries (all fail-closed):**
- *unrecognised tool name* → hard refusal (`UnknownTool`), no execution attempt
- *declared but not `--allow-tool`-authorized* → hard refusal (`Unauthorized`)
- *schema-invalid arguments* → never executed (`InvalidArgs`)
- *wall-clock timeout exceeded* → worker abandoned, `Timeout` result
- *output/args exceed ceiling* → `ResourceLimitExceeded` result

See `docs/phase47_sandbox.md` for the full runbook and honesty boundary.

<details>
<summary><code>agent --help</code> output (full flag list)</summary>

```
Run a bounded caller-executed long-horizon tool-use chain

Usage: aarambh-studio agent [OPTIONS] --tools <TOOLS> --prompt <PROMPT>

Options:
      --config <CONFIG>                Training/model configuration [default: configs/tiny_shakespeare.toml]
      --model <MODEL>                  Model checkpoint; defaults to the configured latest/best pointer
      --tokenizer <TOKENIZER>          Tokenizer JSON; defaults to the configured tokenizer path
      --tools <TOOLS>                  Native or OpenAI-compatible JSON tool definitions
      --prompt <PROMPT>                Initial user request
      --max-steps <MAX_STEPS>          Maximum caller-executed tool calls [default: 8]
      --max-tokens <MAX_TOKENS>        Maximum generated tokens per model decision [default: 256]
      --temperature <TEMPERATURE>      Sampling temperature [default: 0.7]
      --top-p <TOP_P>                  Nucleus sampling probability [default: 0.9]
      --top-k <TOP_K>                  Top-k sampling width [default: 50]
      --seed <SEED>                    Deterministic sampler seed
      --greedy                         Use greedy decoding
      --thinking <THINKING>            none|low|medium|high|max  [default: none]
      --results <RESULTS>              Scripted JSONL tool results; stdin JSONL when omitted
      --result-root <RESULT_ROOT>      Root for all image/video/document result paths [default: .]
      --eviction <EVICTION>            drop-oldest|summarise  [default: drop-oldest]
      --keep-recent <KEEP_RECENT>      Recent exchanges protected from eviction [default: 4]
      --summary-tokens <SUMMARY_TOKENS> Max summary tokens per eviction [default: 128]
      --jsonl                          Emit machine-readable lifecycle events on stdout
      --safety <SAFETY>                Safety mode [default: strict]
      --safety-audit-log <SAFETY_AUDIT_LOG>  [default: safety_audit.jsonl]
      --execute-tools                  Execute tool calls inside the sandbox (Phase 47)
      --allow-tool <NAME>             Operator-authorized tool name (repeatable)
      --exec-timeout-ms <MS>          Per-call wall-clock ceiling [default: 5000]
      --exec-max-output-bytes <N>     Max output payload bytes [default: 65536]
      --exec-workdir <DIR>            Working directory for read_file_in_workdir
  -h, --help                          Print help
```
</details>

---

## 4. `eval` — run evaluation tasks and compare scorecards

**Purpose:** run one or more evaluation tasks (perplexity, MMLU-lite, HellaSwag,
GSM8K, HumanEval-lite, preference, recall, multimodal/tool scorecards,
forgetting curves, MoE routing drift) and report/compare scorecards.

```sh
target/release/aarambh-studio eval --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_shakespeare_smoke/best/model.safetensors \
  --tasks ppl --tasks hellaswag
```

<details>
<summary><code>--help</code> output</summary>

```
Usage: aarambh-studio eval [OPTIONS] --config <CONFIG>

Options:
      --config <CONFIG>
      --model <MODEL>
      --tokenizer <TOKENIZER>
      --tasks <TASKS>                Repeatable task name
      --limit <LIMIT>
      --seed <SEED>
      --output <OUTPUT>
      --compare <COMPARE>
      --kv-cache-report
      --forgetting-manifest <FORGETTING_MANIFEST>
      --forgetting-store <FORGETTING_STORE>
      --forgetting-jsonl <FORGETTING_JSONL>
      --forgetting-threshold <FORGETTING_THRESHOLD>      [default: 0.02]
      --forgetting-max-examples <FORGETTING_MAX_EXAMPLES>  [default: 8]
      --forgetting-allow-code-exec
      --forgetting-require-all-probes
      --forgetting-baseline-id <FORGETTING_BASELINE_ID>
  -h, --help
```
</details>

---

## 5. `quantise` — INT8/INT4 GGUF export

**Purpose:** calibrate a checkpoint against a sample dataset and export an
INT8 or GPTQ/AWQ INT4 GGUF checkpoint.

```sh
target/release/aarambh-studio quantise \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_shakespeare_smoke/best/model.safetensors \
  --data data/tiny_shakespeare.txt --format int4-gguf \
  --output checkpoints/tiny_shakespeare_smoke/int4.gguf
```

<details>
<summary><code>--help</code> output</summary>

```
Usage: aarambh-studio quantise [OPTIONS] --model <MODEL> --data <DATA> --format <FORMAT> --output <OUTPUT>

Options:
      --config <CONFIG>
      --model <MODEL>
      --tokenizer <TOKENIZER>
      --data <DATA>
      --format <FORMAT>            int8|int4-gguf|gptq|awq
      --output <OUTPUT>
      --calibration-samples <N>
      --group-size <N>
  -h, --help
```
</details>

---

## 6. `convert` — checkpoint layout conversion

**Purpose:** convert between SafeTensors, GGUF, and Hugging Face layouts.

```sh
target/release/aarambh-studio convert \
  --input checkpoints/tiny_shakespeare_smoke/best/model.safetensors \
  --from safetensors --to hf --output ./hf_export/
```

<details>
<summary><code>--help</code> output</summary>

```
Usage: aarambh-studio convert [OPTIONS] --input <INPUT> --from <FROM> --to <TO> --output <OUTPUT>

Options:
      --config <CONFIG>
      --input <INPUT>
      --from <FROM>          safetensors|gguf|hf
      --to <TO>              safetensors|gguf|hf
      --output <OUTPUT>
      --tokenizer <TOKENIZER>
  -h, --help
```
</details>

---

## 7. `serve` — OpenAI-compatible HTTP/SSE server

**Purpose:** start a local, single-model OpenAI-compatible server
(`/v1/chat/completions`, `/v1/completions`, streaming SSE, continuous
batching, optional API-key auth, safety, tools, thinking).

```sh
target/release/aarambh-studio serve \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_shakespeare_smoke/best/model.safetensors \
  --model-id aarambh-studio-local --host 127.0.0.1 --port 8080

# Then from a client:
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"aarambh-studio-local","messages":[{"role":"user","content":"Hello"}]}'
```

<details>
<summary><code>--help</code> output</summary>

```
Start the local OpenAI-compatible inference server

Usage: aarambh-studio serve [OPTIONS] --model <MODEL>

Options:
      --config <CONFIG>                    [default: configs/tiny_shakespeare.toml]
      --model <MODEL>
      --tokenizer <TOKENIZER>
      --model-id <MODEL_ID>                [default: aarambh-studio-local]
      --host <HOST>                         [default: 127.0.0.1]
      --port <PORT>                         [default: 8080]
      --max-batch-size <MAX_BATCH_SIZE>     [default: 8]
      --queue-capacity <QUEUE_CAPACITY>     [default: 128]
      --batch-wait-ms <BATCH_WAIT_MS>       [default: 2]
      --prefill-chunk-size <PREFILL_CHUNK_SIZE>  [default: 128]
      --max-request-tokens <MAX_REQUEST_TOKENS>  [default: 2048]
      --thinking <THINKING>                none|low|medium|high|max  [default: none]
      --tools <TOOLS>
      --safety <SAFETY>                     [default: strict]
      --safety-audit-log <SAFETY_AUDIT_LOG> [default: safety_audit.jsonl]
      --api-key-env <API_KEY_ENV>           [default: AARAMBH_STUDIO_STUDIO_API_KEY]
      --cors-origin <CORS_ORIGIN>
  -h, --help
```
</details>

---

## 8. `finetune` — adapters, SFT, GRPO, DPO, RLAIF, VLM, merge

**Purpose:** supervised fine-tuning, LoRA/QLoRA/DoRA/QDoRA (including VLM
variants), GRPO reinforcement learning, DPO preference tuning, RLAIF
(AI-feedback) pair generation, tool-call SFT, and model merging.

### `finetune sft`
```sh
target/release/aarambh-studio finetune sft \
  --config configs/tiny_shakespeare.toml \
  --base checkpoints/tiny_shakespeare_smoke/best/model.safetensors \
  --data data/tool_sft_tiny.jsonl \
  --output checkpoints/tiny_shakespeare_smoke/sft --max-steps 8
```

### `finetune qlora` / `dora` / `qdora` / `qkdpo` / `qdpo` / `qkdpo` etc.
Same shape: `--base <ckpt> --data <jsonl> --output <dir>` plus adapter-specific
flags (rank, alpha, target modules). Run `finetune <sub> --help` for the exact
set.

### `finetune grpo` — verifier-based RL
```sh
target/release/aarambh-studio finetune grpo \
  --config configs/tiny_shakespeare.toml \
  --base checkpoints/tiny_shakespeare_smoke/best/model.safetensors \
  --data data/grpo_tiny_math.jsonl \
  --output checkpoints/tiny_shakespeare_smoke/grpo --max-steps 4
```

### `finetune dpo` — preference tuning
```sh
target/release/aarambh-studio finetune dpo \
  --config configs/rlaif_smoke.toml \
  --base checkpoints/rlaif_smoke/best/model.safetensors \
  --reference-free \
  --tokenizer checkpoints/rlaif_smoke/tokenizer.json \
  --data data/dpo_tiny_preferences.jsonl \
  --output checkpoints/rlaif_smoke/dpo --max-steps 1
```

### `finetune rlaif` — Phase 46, AI-feedback pair generation
```sh
# Sample N candidates per prompt, judge both orderings, emit DPO-schema pairs
target/release/aarambh-studio finetune rlaif \
  --config configs/rlaif_smoke.toml \
  --base checkpoints/rlaif_smoke/best/model.safetensors \
  --tokenizer checkpoints/rlaif_smoke/tokenizer.json \
  --prompts data/rlaif_smoke/prompts.jsonl \
  --output data/rlaif_smoke/rlaif_pairs.jsonl \
  --n-candidates 2 --max-new-tokens 24 --judge-max-tokens 48 --seed 42

# The output feeds the UNMODIFIED dpo pipeline:
target/release/aarambh-studio finetune dpo \
  --config configs/rlaif_smoke.toml \
  --base checkpoints/rlaif_smoke/best/model.safetensors --reference-free \
  --tokenizer checkpoints/rlaif_smoke/tokenizer.json \
  --data data/rlaif_smoke/rlaif_pairs.jsonl \
  --output checkpoints/rlaif_smoke/dpo_from_rlaif --max-steps 1
```

### `finetune tool-sft` / `tool-qlora` — tool-call tuning
```sh
target/release/aarambh-studio finetune tool-sft \
  --config configs/tiny_shakespeare.toml \
  --base checkpoints/tiny_shakespeare_smoke/best/model.safetensors \
  --data data/tool_sft_tiny.jsonl \
  --output checkpoints/tiny_shakespeare_smoke/tool_sft --max-steps 8
```

### `finetune merge` — model merging (Phase 50)
```sh
target/release/aarambh-studio finetune merge \
  --method slerp --inputs ckpt_a.safetensors ckpt_b.safetensors \
  --output merged.safetensors --alpha 0.5
```

<details>
<summary>Subcommands</summary>

`sft`, `qlora`, `tool-sft`, `tool-qlora`, `dora`, `qdora`, `vlm-dora`,
`vlm-qdora`, `grpo`, `dpo`, `qdpo`, `rlaif`, `merge`
</details>

---

## 9. `distill` — on-policy / offline distillation

**Purpose:** train a student against a teacher's on-policy rollouts, or
evaluate distillation quality.

### `distill train`
```sh
target/release/aarambh-studio distill train \
  --config configs/distill_smoke.toml \
  --student checkpoints/tiny_shakespeare_smoke/best/model.safetensors \
  --teacher data/distill_smoke_teacher.jsonl \
  --output checkpoints/distill_smoke/student
```

### `distill evaluate`
```sh
target/release/aarambh-studio distill evaluate \
  --config configs/distill_smoke.toml \
  --student checkpoints/distill_smoke/student/best/model.safetensors \
  --teacher data/distill_smoke_teacher.jsonl --limit 16
```

---

## 10. `selflearn` — persistent self-learning state

**Purpose:** manage the replay buffer, deferred CPU gradient updates, verifiers,
and forgetting-diagnostic integration.

### `selflearn start`
```sh
target/release/aarambh-studio selflearn start \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_shakespeare_smoke/best/model.safetensors \
  --state-dir adapters/selflearn --max-turns 16
```

### `selflearn flush-gradients`
```sh
target/release/aarambh-studio selflearn flush-gradients \
  --state-dir adapters/selflearn --config configs/tiny_shakespeare.toml
```

### `selflearn replay` / `stats` / `reset`
```sh
target/release/aarambh-studio selflearn stats --state-dir adapters/selflearn
target/release/aarambh-studio selflearn replay --state-dir adapters/selflearn --limit 8
target/release/aarambh-studio selflearn reset --state-dir adapters/selflearn
```

---

## Worked end-to-end: train → infer → agent (sandboxed)

```sh
# 1. Train a tiny CPU checkpoint
target/release/aarambh-studio train --config configs/tiny_shakespeare_smoke.toml

# 2. Generate text from it
target/release/aarambh-studio infer \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_shakespeare_smoke/best/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare_smoke/tokenizer.json \
  --prompt "To be, or" --max-tokens 32 --greedy

# 3. Run a sandboxed tool-use chain (Phase 47) that reads a file
mkdir -p data/sandbox_workdir
echo "Project notes: ship the release, tag v4.0.0-alpha.7." > data/sandbox_workdir/notes.txt
target/release/aarambh-studio agent \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_shakespeare_smoke/best/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare_smoke/tokenizer.json \
  --tools data/tools_sandbox_smoke.json \
  --prompt "Read notes.txt and summarise it." \
  --execute-tools \
  --allow-tool read_file_in_workdir \
  --exec-workdir ./data/sandbox_workdir \
  --exec-timeout-ms 2000 \
  --max-steps 4 --jsonl
```

---

## Common flags reference

| Flag | Used by | Meaning |
|---|---|---|
| `--config <TOML>` | all | model/training/device TOML (see `docs/aarambh-studio-config-toml-guide.md`) |
| `--model <PATH>` | infer/agent/eval/quantise/serve/finetune/distill/selflearn | checkpoint `.safetensors` path |
| `--tokenizer <PATH>` | most | tokenizer JSON path (defaults to checkpoint dir) |
| `--prompt <TEXT>` | infer/agent | the request |
| `--max-tokens <N>` | infer/agent | generation length |
| `--temperature`, `--top-p`, `--top-k`, `--seed`, `--greedy` | infer/agent | sampling |
| `--thinking <mode>` | infer/agent/serve/finetune/distill/selflearn | none\|low\|medium\|high\|max |
| `--safety <mode>` | infer/agent/serve | strict/audit/permissive |
| `--tools <JSON>` | infer/agent/serve | tool definitions |
| `--jsonl` | agent | machine-readable lifecycle events |
| `--execute-tools`, `--allow-tool`, `--exec-*` | agent | Phase 47 sandboxed execution |

---

## How to discover any flag yourself

```sh
# Top level
target/release/aarambh-studio --help

# Any command
target/release/aarambh-studio agent --help

# Any sub-subcommand
target/release/aarambh-studio finetune rlaif --help
target/release/aarambh-studio distill train --help
target/release/aarambh-studio selflearn start --help
```

Every `--help` output is the single source of truth — this document is a
human-readable index over it, generated for `4.0.0-alpha.7`.
