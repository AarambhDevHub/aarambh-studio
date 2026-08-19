# Aarambh AI Inference Server

Phase 27 exposes one local checkpoint through an OpenAI-compatible HTTP/SSE
API. The server does not download, publish, or execute model-generated tool
calls.

## Start The Server

```bash
cargo run --release -p aarambh-studio -- serve \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_shakespeare/step_000050/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --model-id aarambh-tiny \
  --port 8080
```

The default bind is `127.0.0.1:8080`. For a non-loopback bind, set a key before
starting the server:

```bash
export AARAMBH_STUDIO_STUDIO_API_KEY='replace-with-a-long-random-secret'
cargo run --release -p aarambh-studio -- serve \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_shakespeare/step_000050/model.safetensors \
  --model-id aarambh-tiny \
  --host 0.0.0.0
```

## Chat Completion

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "aarambh-tiny",
    "messages": [{"role": "user", "content": "To be, or not to be"}],
    "max_tokens": 32,
    "temperature": 0
  }'
```

Set `"stream": true` to receive `chat.completion.chunk` SSE events followed by
`data: [DONE]`. With safety enabled, generated fragments pass through a rolling
cross-token scanner before they are released. PII is redacted and toxic output
terminates with `finish_reason: "content_filter"`.

## OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
response = client.chat.completions.create(
    model="aarambh-tiny",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=32,
)
print(response.choices[0].message.content)
```

When local authentication is disabled, SDKs may still require a non-empty
client-side `api_key`; any placeholder is accepted because the local server does
not validate authorization unless `AARAMBH_STUDIO_STUDIO_API_KEY` is configured.

## Endpoints

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `GET /v1/models`
- `GET /healthz`
- `GET /readyz`
- `GET /metrics`

Phase 27 supports one text model and one generated choice per request. Vision,
self-learning, speculative server decoding, parallel tool calls, tool-result
history, and `/v1/responses` are not part of this phase.

## Multi-tenant auth, rate limits, and prefix caching (Phase 51)

Phase 51 (`v4.0.0-alpha.11`, `ARCHITECTURE_V4.md` §65) adds three opt-in
serving capabilities to the same server: multi-tenant API-key auth, per-key
rate limiting, per-tenant in-flight isolation, and prompt-prefix KV caching.
**None of this is on by default** — the loopback-only, unauthenticated
single-user mode documented above remains the recommended starting point.

Declare one key per tenant in a JSON file (see
`configs/serve_keys.example.json`), then start with `--api-keys`:

```bash
aarambh-studio serve \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_shakespeare/best/model.safetensors \
  --host 0.0.0.0 --port 8080 \
  --api-keys configs/serve_keys.example.json \
  --max-concurrent-per-tenant 4
```

- Missing/unknown bearer key → `401 authentication_error` before any work.
- Per-key RPM/TPM breach → `429 rate_limit_error`.
- Tenant in-flight ceiling → `429 tenant_concurrent_ceiling_reached`.

Prefix caching (opt-in):

```bash
aarambh-studio serve \
  --model checkpoints/tiny_shakespeare/best/model.safetensors \
  --prefix-cache \
  --prefix-cache-max-bytes $((128 * 1024 * 1024)) \
  --prefix-cache-max-entries 64
```

The `/metrics` endpoint reports `prefix_cache_hits`, `prefix_cache_misses`,
`prefix_cache_evictions`, and `prefix_cache_prefill_tokens_saved`. See
[`docs/phase51_public_serve.md`](phase51_public_serve.md) for the full
runbook.

## Smoke Test

```bash
bash scripts/phase27_server_smoke.sh
bash scripts/phase51_smoke.sh
```

Override `CONFIG`, `MODEL`, `TOKENIZER`, `HOST`, `PORT`, or `MODEL_ID` when the
checkpoint layout differs.
