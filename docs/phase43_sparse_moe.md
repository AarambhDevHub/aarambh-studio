# Phase 43 — Sparse/Grouped MoE Dispatch

> v4.0.0-alpha.3 · `aarambh-studio-nn` (dispatch.rs extended) + `aarambh-studio-core` (DispatchKind) · depends on v3 §40 (fine-grained MoE), v2 §26 (dense masked dispatch)

Phase 43 resolves the "documented future optimisation" carried forward
unresolved since v2 §35 and v3's out-of-scope list: **real sparse
dispatch**, where each token's forward pass only computes its assigned
top-k experts, rather than every expert computing on every token and being
masked afterward.

## Why this matters

v2 Phase 22 and v3 Phase 31 shipped only **dense masked dispatch**: every
routed expert runs on every token, then the router weights mask the
non-selected contributions. This is correct but wasteful — at fine-grained
MoE scale (v3 §40, e.g. 32 routed experts, top-k 8) each token pays for 32
expert feed-forwards but uses only 8. The compute that actually contributes
to the output is `top_k / num_experts` of the dense cost; the rest is
multiplied by zero and thrown away.

Sparse dispatch makes that ratio real: each expert's SwiGLU matmul now
executes only on the tokens it was routed to. The output is numerically
identical to dense masked dispatch (same tokens, same weights, same
per-expert reduction order) — just faster, because the masked-away matmuls
never run.

## Mechanism

```
Router logits [batch, seq, num_experts]
     │
     ▼
top_k_gating (UNCHANGED — computes aux loss + dispatch weights here)
     │
     ├─ indices  [batch, seq, top_k]   (selected expert ids)
     ├─ weights  [batch, seq, top_k]   (softmax weights, differentiable)
     └─ aux_loss (load-balancing term, dispatch-independent)
     │
     ▼
DispatchKind selection
     │
     ├─ DenseMasked (v2/v3 behaviour, CPU fallback, QAT calibration)
     │     every expert.forward(full_x) * dispatch_weights[:,:,e], summed
     │
     └─ Sparse (CUDA only — the throughput win)
           flatten tokens → [N, H]
           flatten assignments → [N*top_k]
           arg_sort by expert id (no-grad permutation) → grouped order
           gather token-ids + weights into grouped order (differentiable)
           per-expert boundaries (scanned on host, O(N*top_k))
           for each expert e with count_e > 0:
               index_select e's token group → [count_e, H]
               expert.forward(group)            ← matmul on group ONLY
               mul group_weights                ← differentiable
               index_add scatter back into [N, H]
           reshape → [batch, seq, H]
     │
     ▼
+ shared experts (always active, unchanged by dispatch kind)
     │
     ▼
output [batch, seq, H]
```

### Why this is differentiable end-to-end

The permutation that groups tokens by expert is a discrete index — it
carries no gradient, so it is computed with `arg_sort_last_dim` (a no-grad
operation). But every *value* that flows into the loss remains
differentiable:

- **Router weights**: gathered with `gather` (differentiable) →
  `group_weights` → multiplied into the expert output → scattered via
  `index_add` (differentiable). Router-logit gradients flow back through
  the gather's backward (scatter of the upstream gradient to original
  positions).
- **Expert parameters**: each `expert.forward(group)` is a normal SwiGLU
  matmul chain; its backward reaches `w_gate`/`w_up`/`w_down`.
- **Input activations**: `index_select` (differentiable) gathers the
  expert input; its backward scatters the expert-input gradient back to
  the original token positions.

The unit test `sparse_dispatch_backward_reaches_router_and_expert_weights`
verifies all three gradient paths.

### Why the output is numerically equivalent to dense masked

For a token routed to top-k experts {e₁,…,eₖ} with weights {w₁,…,wₖ}, both
paths compute `Σⱼ expert_eⱼ(x) · wⱼ`:

- **Dense**: loops experts 0..E-1, each computes on the full sequence,
  multiplies by `dispatch_weights[:,:,e]` (zero for non-routed tokens),
  accumulates. Non-routed contributions are exactly zero.
- **Sparse**: loops experts 0..E-1 in the same order, each computes only
  on its routed token group, multiplies by the matching slot weight,
  scatters via `index_add`. The same tokens, same weights, same per-expert
  accumulation order.

The reduction order is identical (expert index ascending in both), so the
floating-point result matches to within f32 rounding — verified by
`sparse_dispatch_output_matches_dense_masked_reference_within_tolerance`
(max abs diff < 1e-5) and `dispatch_kind_dense_masked_is_bit_identical_to_v2_v3_behaviour`
(diff == 0.0 for the dense path itself).

## The `DispatchKind` enum

```rust
// aarambh-studio-core/src/config.rs
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DispatchKind {
    /// v2/v3 behaviour: every expert on every token, masked by router weights.
    /// CPU fallback and correctness reference.
    #[default]
    DenseMasked,
    /// Tokens grouped by router assignment; each expert matmul runs only on
    /// its assigned token group. CUDA-only throughput win.
    Sparse,
}
```

`MoeConfig` gains a `dispatch: DispatchKind` field defaulting to
`DenseMasked`, so every existing MoE checkpoint and config is exactly
backward-compatible — old TOML/JSON without the field deserialises to the
dense path, byte-identical to v2/v3.

## CPU/CUDA honesty policy

```rust
// aarambh-studio-nn/src/dispatch.rs
pub fn effective_dispatch_kind(configured: DispatchKind, device: &Device) -> DispatchKind {
    match configured {
        DispatchKind::DenseMasked => DispatchKind::DenseMasked,
        DispatchKind::Sparse => if device.is_cuda() {
            DispatchKind::Sparse
        } else {
            DispatchKind::DenseMasked   // documented fallback, not silent
        },
    }
}
```

The real throughput win lives on CUDA, where candle routes the per-expert
`index_select`/`matmul`/`index_add` to cuBLAS — a genuine grouped-GEMM
path that skips non-routed experts. On CPU the sparse algorithm is
correct but not faster (candle's CPU scatter/gather has overhead that
eats the saving), so `MoeFfn` on CPU falls back to `DenseMasked`
**regardless of configuration**, documented plainly as "GPU only pays
off" — the same honesty discipline v2 §29 applied to speculative
decoding's speed claim. The configured kind is queryable via
`MoeFfn::dispatch_kind()`; the effective kind via
`effective_dispatch_kind()`.

QAT calibration (`forward_with_capture`) always uses the dense reference
path, so it observes full per-expert activation distributions regardless
of the configured dispatch kind.

## Load-balancing auxiliary loss — unchanged

The auxiliary loss is computed in `top_k_gating` **before** dispatch, so
it is identical for both kinds — Sparse changes the compute path only,
not the loss the router is trained against. The unit test
`load_balancing_loss_value_is_unaffected_by_dispatch_kind` verifies the
recorded `aux_loss` and `expert_utilization` match across dispatch kinds.

## Tests

| Test | Gate |
|---|---|
| `sparse_dispatch_output_matches_dense_masked_reference_within_tolerance` | correctness (max abs diff < 1e-5) |
| `dispatch_kind_dense_masked_is_bit_identical_to_v2_v3_behaviour` | backward compat (diff == 0.0) |
| `sparse_dispatch_supports_top_k_greater_than_one` | top_k > 1 equivalence |
| `sparse_dispatch_backward_reaches_router_and_expert_weights` | differentiability |
| `sparse_dispatch_empty_expert_group_is_skipped` | empty expert group handled |
| `sparse_dispatch_matches_dense_with_shared_expert_summed_separately` | shared expert interaction |
| `load_balancing_loss_value_is_unaffected_by_dispatch_kind` | aux loss dispatch-independent |
| `sparse_configured_moe_falls_back_to_dense_masked_on_cpu` | CPU fallback policy |
| `effective_dispatch_kind_uses_sparse_on_cuda` | CUDA selection (skips on CPU) |
| `sparse_dispatch_cuda_throughput_exceeds_dense_masked_at_kaggle_gpu_scale` | wall-clock throughput (skips on CPU, runs on CUDA) |

The CUDA-only tests use `Device::cuda_if_available(0)` and early-return on
CPU — they compile cleanly in CPU CI and exercise the real CUDA path when
GPU hardware is present.

## Configs

- `configs/moe_sparse_smoke.toml` — tiny CPU smoke with `dispatch = "sparse"`
  (runs through the dense fallback, validates config deserialisation +
  training).
- `configs/large_sparse_moe.toml` — Kaggle GPU config mirroring
  `large_finegrained_moe.toml` with `dispatch = "sparse"` (the real
  throughput win).

## Smoke script

`scripts/phase43_smoke.sh` runs the unit-test suite, a two-step CPU
training smoke on `moe_sparse_smoke.toml`, and verifies the saved
checkpoint contains the fine-grained routed + shared MoE expert tensors,
writing a scorecard to `artifacts/phase43_sparse_moe_smoke.json`.

## Milestone

Sparse dispatch produces numerically equivalent output to DenseMasked on
the same MoE config, with measured throughput improvement on Kaggle GPU
hardware at realistic batch sizes, verified via the eval harness before/
after scorecard confirming no quality regression.

```
git commit -m "feat: Phase 43 — sparse/grouped MoE dispatch"
git tag v4.0.0-alpha.3
```
