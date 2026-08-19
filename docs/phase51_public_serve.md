# Phase 51 — Public/Hosted Inference Server + Prefix Caching

> From first principles. From zero. From Rust.
>
> Phase 51 (`ARCHITECTURE_V4.md` §65) opens the existing
> `aarambh-studio-serve` server (v2 §31) to genuinely multi-tenant,
> authenticated traffic and adds prefix caching — the highest-leverage
> serving optimisation for the agentic/tool-chain traffic pattern Phases
> 47–48 generate.

This is the runbook for the public-serving capability shipped in
`v4.0.0-alpha.11`. It documents the design, the operator-facing CLI, the
hard bounds, the honesty boundary, the smoke workflow, and the
measured-improvement discipline.

---

## Why this phase exists

Phase 50 closed the model-merging gap. The natural next step before the
final release is the single biggest scope/risk jump in the project's
history: opening the inference server to traffic from more than one
caller. v2 §31 and v3 both stayed local-only deliberately — the model
underneath (attention, MoE, agentic tool use) was not yet settled enough
to expose. By the end of v4 that is no longer true, so the server can
finally grow real per-key identity, per-key rate limits, per-tenant
isolation, and prefix caching.

Prefix caching is placed in the *same* phase as multi-tenant auth on
purpose. The two are the pieces that make multi-tenant *and* agentic
traffic economically viable on the same server: agentic chains
(Phases 47–48) replay the same system prompt and shared conversation
prefix across many tool-execution sub-requests, and prefix caching lets
the server reuse the computed KV for that prefix instead of recomputing
it on every request. Without it, the per-request prefill cost would
make multi-tenant agentic traffic impractical.

Phase 51 is **strictly additive and opt-in**. It adds three modules to
the existing `aarambh-studio-serve` crate (no new crate, no new external
dependency; `EXPECTED_PACKAGES` stays 21) and one strictly-additive method
on the inference engine. When none of the new flags are set, the server
is byte-for-byte the `4.0.0-alpha.10` loopback-only single-user server.

---

## What ships

Three new modules in `crates/aarambh-studio-serve/src/`:

| Module | Responsibility |
|---|---|
| `auth.rs` | Per-key API identity + per-key rate limiting |
| `prefix_cache.rs` | Prompt-prefix → cached KV store with LRU eviction |
| `tenant_isolation.rs` | Per-tenant concurrent-in-flight ceiling |

One new method on `InferenceEngine` (`aarambh-studio-inference`):

```rust
pub fn prepare_session_with_prefix_cache<L, S>(
    &self,
    prompt: &str,
    config: GenerationConfig,
    chunk_size: usize,
    lookup: L,   // FnOnce(&[u32]) -> Option<(KvCache, matched_len)>
    store: S,    // FnOnce(&[u32], &KvCache)
) -> Result<GenerationSession>;
```

`prepare_session_with_chunk_size` now delegates to it with no-op closures,
so every existing caller (`infer`, `agent`, `eval`, the batcher default)
is unchanged.

---

## The request flow

```
Incoming HTTP request (chat / completion)
   |
   v
auth.rs: AuthGate::authorize(&headers)
   auth off (default)        -> synthetic "local" tenant (loopback-open)
   auth on, valid key        -> resolved TenantId
   auth on, missing/invalid  -> 401 (BEFORE admission)
   |
   v
RateLimiter::check(tenant, limits, est_tokens)
   over RPM or TPM ceiling -> 429 (BEFORE admission)
   |
   v
TenantLimiter::try_admit(tenant) -> TenantPermit (RAII)
   over ceiling -> 429 "tenant_busy" (BEFORE admission)
   |
   v
batcher.submit(GenerationRequest)   <- existing v2 §31 path, UNCHANGED
   |
   v
batcher worker::admit_job
   engine.prepare_session_with_prefix_cache(prompt, cfg, chunk,
       lookup = |ids| prefix_cache.lookup(ids),
       store  = |ids, kv| prefix_cache.store(ids, kv))
   |  hit  -> restore cached KV (truncated to matched prefix), prefill only the tail
   |  miss -> fresh KV, full prefill, then store the full-prefix KV (LRU-evict if over ceiling)
   |
   v
... existing v2 §31 decode/stream pipeline, UNCHANGED ...
```

---

## Hard guarantees

1. **Auth before admission.** A request with a missing/invalid API key is
   rejected with 401 *before* it is submitted to the continuous batcher —
   never queued, never partially processed.
2. **Per-key rate limiting.** Requests/minute and tokens/minute are
   enforced independently per key at admission (429 on breach).
3. **Prefix-cache correctness.** A cache hit restores the cached KV and
   prefills only the remaining tokens; the resulting KV is identical to
   a fresh full prefill (the model is deterministic), so a hit yields
   byte-identical output under greedy decoding — the test suite asserts
   this.
4. **Measured saving.** A hit's skipped prefill tokens are counted in
   `prefix_cache_prefill_tokens_saved`, not assumed.
5. **Memory ceiling + LRU.** The cache evicts least-recently-used entries
   to stay under both the byte and entry ceilings.
6. **Tenant isolation at admission.** One tenant's burst is rejected with
   429 `tenant_busy` while another tenant's request proceeds normally.
7. **Backward compatibility.** With auth off, prefix cache disabled, and
   unlimited tenants (the defaults), the server is byte-for-byte the v2
   §31 single-user server; every pre-existing serve test continues to
   pass unchanged.

---

## Usage

### Default (loopback-only, single-user) — unchanged

```sh
aarambh-studio serve \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_shakespeare/best/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --port 8080
```

No new flags → no auth, no prefix cache, no tenant ceiling. Identical to
`4.0.0-alpha.10`.

### Multi-tenant auth

Write a key file (`configs/serve_keys.example.json` is the reference):

```json
{
  "keys": [
    {
      "secret": "replace-with-a-long-random-secret-for-tenant-a",
      "tenant": "tenant-a",
      "limits": { "requests_per_minute": 60, "tokens_per_minute": 10000 }
    },
    {
      "secret": "replace-with-a-different-long-random-secret-for-tenant-b",
      "tenant": "tenant-b",
      "limits": { "requests_per_minute": 120, "tokens_per_minute": 50000 }
    }
  ]
}
```

Start the server with `--api-keys` (and a non-loopback bind, which now
requires either `--api-keys` or the legacy `AARAMBH_STUDIO_STUDIO_API_KEY`):

```sh
aarambh-studio serve \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_shakespeare/best/model.safetensors \
  --host 0.0.0.0 --port 8080 \
  --api-keys configs/serve_keys.example.json \
  --max-concurrent-per-tenant 4
```

Clients present their key:

```sh
curl http://0.0.0.0:8080/v1/chat/completions \
  -H 'Authorization: Bearer replace-with-a-long-random-secret-for-tenant-a' \
  -H 'Content-Type: application/json' \
  -d '{"model":"aarambh-studio-local","messages":[{"role":"user","content":"Hi"}],"max_tokens":8}'
```

A request with a missing or unknown key returns `401
authentication_error` before any inference work; a request that breaches
the per-key RPM or TPM returns `429 rate_limit_error`; a request that
would exceed the tenant's in-flight ceiling returns `429
tenant_concurrent_ceiling_reached`.

### Prefix caching

```sh
aarambh-studio serve \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_shakespeare/best/model.safetensors \
  --prefix-cache \
  --prefix-cache-max-bytes $((128 * 1024 * 1024)) \
  --prefix-cache-max-entries 64
```

Repeated system prompts and shared conversation prefixes now reuse the
computed KV state. Cache hit/miss/eviction counters and the
`prefix_cache_prefill_tokens_saved` total are exposed on `/metrics`
alongside the existing server telemetry.

---

## Honesty boundary

- The memory ceiling is an **estimate** computed from the model
  configuration (`n_layers * 2 * n_kv_heads * head_dim * dtype_bytes` per
  cached sequence token). It is deterministic and is the value the LRU
  evictor uses, but for non-standard attention layers (MLA, Gated
  DeltaNet) the actual per-layer storage may differ slightly — the
  estimate errs on the conservative side.
- A prefix-cache hit's saving is **counted**, not assumed: the
  `prefix_cache_prefill_tokens_saved` counter is incremented by the exact
  matched-prefix length, and a hit's output is asserted byte-identical to
  the cold-cache baseline under greedy decoding.
- Rate-limit token accounting uses the request's `max_tokens` as an
  admission-time estimate — the actual completion length is not known
  until generation finishes. This is the standard admission-control
  heuristic (you cannot know the actual length at admission).

---

## How it relates to self-learning

Prefix caching is a **serving** optimisation, not a self-learning-loop
change. It does not alter what the model learns or how the online
self-learning loop (`SELF_LEARNING_V4.md` §55) replays or updates. For
that reason `SELF_LEARNING_V4.md` is intentionally untouched by Phase 51
(the same framing RLAIF and model-merging use for offline operations
kept separate from the online loop). Prefix caching is transparent to
the model: it reuses KV state the model would have computed anyway.

---

## Smoke workflow

```sh
scripts/phase51_smoke.sh
```

runs the serve-crate integration tests (the five roadmap-named acceptance
tests plus supporting tests, all against a tiny in-memory `InferenceEngine`
— no checkpoint on disk), builds the CLI, verifies `serve --help` shows
the new flags, and writes a scorecard to
`artifacts/phase51_serve_smoke.json`.

The deterministic proof is the test suite; the smoke script additionally
verifies the operator-facing CLI surface compiles and the flags render.

---

## What this enables next

Phase 52 (System Role, Chat-Template Versioning, and Context
Management) formalises the model's I/O contract as it stands after every
feature phase in v1–v4. A multi-tenant, prefix-caching server is the
deployment target that contract is designed for: a documented system
role, a versioned chat template, and bounded context management are
exactly what a public server needs to evolve its prompt format without
breaking existing tenants.
