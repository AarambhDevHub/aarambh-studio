# Phase 49 — Retrieval-Augmented Generation (RAG)

> From first principles. From zero. From Rust.
>
> Phase 49 (`ARCHITECTURE_V4.md` §63) adds a from-scratch retrieval pipeline
> in pure Rust. No external vector database is required, and none is used —
> the approximate-nearest-neighbour index is a navigable small-world graph
> implemented entirely in the new `aarambh-studio-retrieve` crate (no FFI
> to an external vector-search library). Retrieved context augments the
> prompt *before* generation; it does not touch model internals.

This is the runbook for the retrieval-augmented generation capability
shipped in `v4.0.0-alpha.9`. It documents the design, the operator-facing
CLI, the hard bounds, the honesty boundary, the smoke workflow, and the
measured-improvement discipline.

---

## Why this phase exists

Phase 48 closed the multi-agent orchestration gap. The next natural
capability is factual grounding: the model produces answers from its
weights alone, and for facts that were never in the training set — or were
learned long enough ago that they have partially decayed — a retrieval
step that pulls relevant context into the prompt before generation is the
simplest, most composable way to improve factual accuracy without
retraining.

The roadmap places RAG *after* the agentic work (Phases 47–48) for two
reasons. First, tool-use chains are a natural consumer of retrieval
results: a RAG-augmented orchestrator can delegate RAG-augmented sub-tasks
to sub-chains, and the sub-chains' sandboxed tool execution can later
include a retrieval tool. Second, RAG is a *prompt-construction* concern,
not a model-internals concern — it composes with everything that consumes
a prompt, so building it after the prompt-construction path is mature
(Phases 47–48) rather than before means RAG plugs into a stable seam.

Phase 49 is **entirely additive**. It adds one new crate, wires two CLI
subcommands, and adds one eval task. No existing file was modified for
correctness — every change is a strictly additive match arm, flag, or
re-export.

---

## The retrieval envelope

The pipeline runs in five honest stages, each pure Rust:

```
Operator provides corpus_dir + RetrievalConfig (chunking, embedding, index)
        │
        ▼
Chunker::chunk_corpus(tokenizer, corpus_dir)
  → tokenizes each .txt/.md/.jsonl file, slices into overlapping token
    windows of chunk_size with chunk_size - overlap stride. Each chunk
    gets a monotonic id, a source path, and a byte offset — so overlap
    never duplicates index entries.
        │
        ▼
Embedder::embed(chunk.text)
  → a deterministic, L2-normalized vector per chunk. HashingEmbedder
    (default, weight-free) or TextEmbedder (candle head, weights-loaded).
    Both return unit vectors so cosine similarity is a plain dot product.
        │
        ▼
VectorIndex::insert(entry)
  → a navigable small-world graph (pure Rust, no FFI). Each new node is
    connected to its M nearest neighbours via a greedy beam search of
    width ef_construction; each neighbour's adjacency is pruned back to
    M nearest so the graph's out-degree stays bounded.
        │
        ▼
VectorIndex::save(dir) → index.json (human-readable, trivially loadable)
        │
        ▼
At query time: RetrievalPipeline::query(text)
  → embed the query, greedy beam search of width ef_search, return top_k
    RetrievedChunk { chunk_id, text, source, offset, score }
        │
        ▼
augment_prompt(prompt, retrieved)
  → splice retrieved chunks into the prompt AHEAD of the user's question
    via the SAME prompt-construction path the inference engine already
    uses (system prompt + chat history + user turn). RAG augments the
    prompt; it does not change how the decoder processes it.
```

Every bound is enforced as **operator-set configuration**, never as
something the retrieval pipeline's own output can influence. The index
cannot grow past the corpus the operator provides; the embedder dimension
is fixed at build time and validated at query time; the top-k is an
operator ceiling.

---

## The hard, non-negotiable bounds

### 1. No external vector database

```rust
pub struct VectorIndex {
    config: IndexConfig,
    entries: Vec<IndexEntry>,
    graph: Vec<Vec<usize>>,  // adjacency list, parallel to entries
}
```

The index is a navigable small-world graph implemented entirely in
`crates/aarambh-studio-retrieve/src/index.rs`. There is no FFI to
`hnswlib`, `faiss`, `annoy`, or any other external vector-search library,
and no dependency on an external vector-database service. This is the
forbidden-dependencies rule from `ARCHITECTURE_V4.md` §69 and
`ROADMAP_V4.md`'s "Still forbidden everywhere" block: *"any external
vector-database service dependency for the core RAG index (an optional
plug-in adapter may exist, but the from-scratch index remains the default
and the tested path)."*

An optional plug-in adapter to an external vector store is a documented
extension point and is **not** implemented here.

### 2. Embedder dimension is fixed and validated

`IndexConfig::dim` is set at index-build time. Every inserted vector and
every query vector must have exactly that length, or the call returns
`AarambhError::Shape`. The pipeline constructor additionally verifies
`embedder.dim() == index.config().dim` so a mismatched embedder cannot
silently produce zero-similarity results.

### 3. Chunk overlap never duplicates index entries

`ChunkingConfig` validates `overlap < chunk_size` (otherwise chunks would
never advance). Each chunk receives a monotonic, zero-based `id` and a
byte `offset` into its source — so two chunks that share an overlap window
are distinct index entries with distinct ids, never duplicates.

### 4. RAG is text-only

Combining `--rag` with `--image`, `--video`, `--document`, or `--audio`
returns `AarambhError::Unsupported`. Multimodal paths construct their own
prompts (image/video/document/audio tokens spliced at the embedding
level), which is a different fusion mechanism than prompt augmentation.
RAG augments the text prompt; it does not touch the decoder.

---

## Failure isolation

> *"Retrieval is prompt augmentation, not a model-level fusion mechanism.
> A retrieval miss degrades to the no-retrieval baseline — it does not
> corrupt the decoder, the prompt structure, or sibling queries."*

| Failure mode | Outcome | Generation affected? |
|---|---|---|
| Index directory missing or unreadable | `infer --rag` errors before any model is loaded | No model loaded |
| Index dimension ≠ embedder dimension | `RetrievalPipeline::new` returns `Shape` error | No generation |
| Query produces zero retrieved chunks | `augment_prompt` returns the prompt unchanged | Generation runs as if `--rag` were absent |
| Retrieved chunk is irrelevant | Prompt is longer but the decoder is unchanged; generation degrades gracefully to the no-retrieval baseline | Yes, but only as a weaker answer — never a crash |

The fourth row is the honesty boundary: a retrieval miss is *not* a
correctness failure, it is a quality regression to the baseline. The
`rag_delta` detail in the eval scorecard measures whether RAG actually
helped on a given task.

---

## Composability: zero decoder changes

The key architectural decision is that RAG is **prompt augmentation, not
model fusion**. The retrieval pipeline never touches the `InferenceEngine`,
the `ChainDecoder`, the KV cache, or any attention mechanism. It produces
a string, and that string is fed into the exact same prompt-construction
path the inference engine already uses:

```rust
// In infer.rs::run, after building the base prompt:
let prompt = if args.rag {
    let pipeline = RetrievalPipeline::load_hashing(&index_dir, args.rag_top_k)?;
    let retrieved = pipeline.query(&args.prompt)?;
    augment_prompt(&prompt, &retrieved)
} else {
    prompt
};
// ... then the existing generation path runs unchanged.
```

This means RAG composes with every other inference capability:
speculative decoding, best-of-N, self-learning, safety, tool calling, and
thinking mode all work on the augmented prompt exactly as they would on
the original. No decoder trait was widened, no new fusion mechanism was
introduced.

---

## CLI: `retrieve build-index` and `infer --rag`

### Building an index

```sh
aarambh-studio retrieve build-index \
  --corpus data/rag_smoke_corpus/ \
  --output checkpoints/rag_smoke_index/ \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --chunk-size 128 \
  --overlap 24 \
  --embedding-dim 128 \
  --top-k 4 \
  --max-neighbors 16 \
  --ef-construction 64 \
  --ef-search 32 \
  --embedder hashing
```

The `retrieve build-index` subcommand reads every `.txt`/`.md`/`.jsonl`
file under `--corpus` (recursively), chunks each, embeds each chunk, and
writes `index.json` to `--output`.

### Querying with `infer --rag`

```sh
aarambh-studio infer \
  --config configs/tiny_shakespeare_smoke.toml \
  --model checkpoints/tiny_shakespeare_smoke/best/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare_smoke/tokenizer.json \
  --prompt "What is the capital of France?" \
  --rag \
  --index checkpoints/rag_smoke_index/ \
  --rag-top-k 4 \
  --max-tokens 64
```

When `--rag` is set, the pipeline loads the index, embeds the query,
retrieves the top-k chunks, and splices them into the prompt ahead of the
user's question before any generation path runs. The retrieved chunks are
logged to stderr with their scores and sources for operator visibility.

### New flags on `infer`

| Flag | Default | Purpose |
|---|---|---|
| `--rag` | off | Enable retrieval-augmented generation (text-only) |
| `--index <PATH>` | (required with `--rag`) | Path to an index directory built by `retrieve build-index` |
| `--rag-top-k N` | 4 | Number of chunks to retrieve per query |

---

## Index schema

An index directory contains a single `index.json` file — a human-readable,
trivially loadable JSON object:

```json
{
  "config": {
    "dim": 128,
    "max_neighbors": 16,
    "ef_construction": 64,
    "ef_search": 32
  },
  "entries": [
    {
      "id": 0,
      "vector": [0.0, 0.1, ...],
      "metadata": {
        "chunk_id": 0,
        "source": "data/rag_smoke_corpus/geography.txt",
        "offset": 0,
        "text": "Aarambh Studio — Geography Reference\n\nThe capital of France is Paris..."
      }
    }
  ],
  "graph": [[1, 2, 3], [0, 2, 4], ...]
}
```

The `graph` field is the adjacency list parallel to `entries`:
`graph[i]` is the list of node indices reachable from node `i`.

---

## Tests

Four roadmap-named acceptance tests (one per acceptance criterion in
`ROADMAP_V4.md` §"Phase 49 — Tests") plus supporting tests, all running
in milliseconds:

| Test | Location | Proves |
|---|---|---|
| `index_build_and_query_round_trip_returns_the_inserted_chunk` | `index.rs` | A chunk queried with its own embedding is the top-1 result |
| `retrieval_recall_on_a_small_labelled_holdout_meets_a_documented_floor` | `index.rs` | Recall@1 ≥ 0.8 (documented floor) on a 10-query labelled holdout |
| `rag_augmented_prompt_contains_retrieved_context_and_preserves_user_query` | `retrieval.rs` | Augmentation precedes the user query with retrieved context and preserves the query verbatim — the deterministic proof that RAG augments without corrupting the prompt |
| `chunking_with_overlap_does_not_duplicate_index_entries_incorrectly` | `chunking.rs` | Overlap produces distinct, monotonic chunk ids sharing exactly the overlap window — never duplicate entries |

Supporting tests cover: `IndexConfig` / `ChunkingConfig` / `TextEmbedderConfig`
validation, dimension-mismatch rejection (insert, search, pipeline),
empty-index search, save/load round-trip, cosine similarity properties,
ANN-vs-brute-force recall on a small index, hashing-embedder determinism
and L2-normalization, hashing-embedder similarity tracks shared substrings,
`TextEmbedder` constructs and embeds on CPU, `augment_prompt` no-op on
empty results, `build_index` rejects `TextEmbedder` without weights, and
the `RagTask` eval task's answer-matching and accuracy helpers.

The `rag_augmented_generation_measurably_improves_a_factual_eval_task_vs_no_retrieval`
acceptance criterion is measured by the `rag` eval task, which reports
`no_retrieval_accuracy`, `rag_accuracy`, and `rag_delta` details — the
same "X vs baseline" reporting discipline the gsm8k best-of-N task uses.

---

## Smoke workflow

```sh
scripts/phase49_smoke.sh
```

Runs the retrieve-crate unit tests (the deterministic proof), verifies
`retrieve build-index --help` surfaces the new flags, builds an index
from `data/rag_smoke_corpus/` and queries it end-to-end via a small Rust
test harness, and writes a scorecard to
`artifacts/phase49_rag_smoke.json`.

The smoke follows the same honesty discipline as Phase 48: the unit tests
are the deterministic proof with the weight-free `HashingEmbedder`; the
CLI smoke verifies the operator-facing surface compiles, the flags
appear, and an index builds and loads end-to-end. Whether retrieval
produces *useful* model behaviour at scale is measured by the eval
harness (`eval --tasks rag`), not asserted in prose.

---

## Honesty boundary

Phase 49's RAG is **purely additive** and **CPU-first honest**:

- **No external vector database.** The index is a pure-Rust navigable
  small-world graph. An optional plug-in adapter to an external vector
  store is a documented extension point, not a shipped capability.
- **No new external dependency.** `aarambh-studio-retrieve` depends only
  on `aarambh-studio-core`, `aarambh-studio-tokenizer`, `candle-core`,
  `candle-nn`, `half`, `serde`, and `serde_json` — all already in the
  workspace.
- **Weight-free default.** The `HashingEmbedder` (deterministic byte
  3-gram feature hashing with signed contributions and L2 normalization)
  is the default tested path so the whole pipeline runs in milliseconds
  without a trained embedding checkpoint — mirroring the Phase 47/48
  "fake decoder for tests, real engine for production" discipline. The
  `TextEmbedder` (candle: token-embedding table → mean-pool → linear
  projection → L2-normalize) is the trained-head architecture the
  roadmap describes; when no embedding checkpoint is shipped, the hashing
  embedder remains the default.
- **No decoder changes.** RAG augments the prompt string; it does not
  touch the `InferenceEngine`, `ChainDecoder`, KV cache, or attention.
  Every existing inference capability (speculative, best-of-N,
  self-learning, safety, tool calling, thinking) composes with RAG
  unchanged.
- **CPU-first, sequential index build.** The corpus is chunked and
  indexed sequentially. True parallelism would require `Send + Sync`
  embedders, which is out of scope for the source release.

**Out of scope:** true parallel index build, an external vector-store
adapter, multimodal RAG (retrieving images/audio/video and fusing at the
embedding level rather than the prompt level), and learned re-ranking —
all are documented extension points, not shipped capabilities.

The retrieval-relevant property — *retrieved chunks are spliced into the
prompt ahead of the user's question, the user query is preserved verbatim,
and recall meets a documented floor on a labelled holdout* — is proven by
the four roadmap-named acceptance tests plus supporting tests, all
running in milliseconds with the weight-free `HashingEmbedder`.

---

## What this enables next

Phase 49 is independent of Phase 48 — RAG augments the prompt before
generation; orchestration runs after the prompt is built. But the two
compose naturally:

- A **RAG-augmented orchestrator** can delegate RAG-augmented sub-tasks
  to sub-chains. Each sub-chain builds its own prompt with its own
  retrieved context, exactly as the top-level orchestrator does.
- The sub-chains' sandboxed tool execution can later include a
  **retrieval tool** (a `ToolExecutor` that calls
  `RetrievalPipeline::query` and returns `ToolResultContent::Text`),
  letting the model retrieve on demand rather than only at prompt
  construction. This is a documented extension point, not shipped in
  Phase 49.
- Phase 50 (model merging) and Phase 51 (public inference server) build
  on the agentic + retrieval substrate Phases 47–49 together
  established. Phase 52 (system role / chat-template versioning /
  context-truncation policy) is the first phase that has to reason
  explicitly about RAG-augmented prompt length — the context-truncation
  policy will need to account for retrieved chunks as part of the
  assembled prompt.
