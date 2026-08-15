#!/usr/bin/env bash
# Phase 43 — Sparse/Grouped MoE dispatch smoke test.
#
# Validates that:
#   - The Phase 43 unit-test suite passes (sparse vs dense equivalence within
#     tolerance, DenseMasked bit-identical to v2/v3, aux loss
#     dispatch-independent, CPU fallback to DenseMasked, CUDA throughput
#     gated to skip on CPU).
#   - The new `dispatch = "sparse"` config field deserialises and a
#     two-step CPU training run completes through the documented
#     DenseMasked fallback — CPU never runs sparse, per the "GPU only pays
#     off" honesty policy.
#   - The saved checkpoint contains the fine-grained routed + shared MoE
#     expert tensors, proving the sparse-configured MoE layer built and
#     trained.
#
# The real sparse throughput win is measured on Kaggle GPU hardware with
# configs/large_sparse_moe.toml; the wall-clock gate lives in the
# `sparse_dispatch_cuda_throughput_exceeds_dense_masked_at_kaggle_gpu_scale`
# unit test (skipped on CPU).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SCORECARD=${PHASE43_SCORECARD:-artifacts/phase43_sparse_moe_smoke.json}
mkdir -p "$(dirname "$SCORECARD")"

echo "==> Phase 43 sparse/grouped MoE dispatch unit tests"
cargo test --locked -p aarambh-studio-nn dispatch
cargo test --locked -p aarambh-studio-nn moe
cargo test --locked -p aarambh-studio-core moe_config_dispatch

if [[ "${PHASE43_SKIP_TRAIN:-0}" == "1" ]]; then
  echo "PHASE43_SKIP_TRAIN=1; training smoke was skipped"
  python3 - "$SCORECARD" <<'PY'
import json, sys
json.dump({"phase": 43, "train_smoke": "skipped"}, open(sys.argv[1], "w"), indent=2)
print(f"wrote {sys.argv[1]} (train smoke skipped)")
PY
  echo "Phase 43 smoke completed (train skipped)"
  exit 0
fi

echo "==> Phase 43 ensure a tiny training fixture exists"
if [[ ! -f data/tiny_shakespeare.txt ]]; then
  mkdir -p data
  python3 - <<'PY'
from pathlib import Path
# A tiny public-domain-style text fixture so the BPE tokenizer and the
# two-step training smoke can run without the full wikitext corpus.
snippet = (
    "To be, or not to be, that is the question: "
    "whether tis nobler in the mind to suffer "
    "the slings and arrows of outrageous fortune, "
    "or to take arms against a sea of troubles "
    "and by opposing end them. "
)
text = (snippet * 400)
Path("data/tiny_shakespeare.txt").write_text(text)
print(f"wrote data/tiny_shakespeare.txt ({len(text)} bytes)")
PY
fi

echo "==> Phase 43 two-step CPU training smoke (moe_sparse_smoke.toml, dispatch=sparse -> dense fallback)"
cargo run --quiet --locked -p aarambh-studio -- train \
  --config configs/moe_sparse_smoke.toml

echo "==> Phase 43 verify the saved checkpoint contains sparse-MoE expert tensors"
python3 - "$SCORECARD" <<'PY'
import json, sys
from pathlib import Path
ptr = json.loads(Path("checkpoints/moe_sparse_smoke/latest.json").read_text())
model = Path(ptr["path"]) / "model.safetensors"
if not model.exists():
    raise SystemExit(f"checkpoint not found: {model}")
raw = model.read_bytes()
header_len = int.from_bytes(raw[:8], "little")
header = json.loads(raw[8:8 + header_len].decode("utf-8"))
names = list(header.keys())
# Layer 1 is the MoE layer (every_n_layers=2). With num_experts=4 and
# fine_grained_factor=2 there are 8 routed experts (0..7) plus one shared.
required = [
    "blocks.1.ffn.router.weight",
    "blocks.1.ffn.experts.0.w_gate.weight",
    "blocks.1.ffn.experts.7.w_down.weight",
    "blocks.1.ffn.shared_experts.0.w_up.weight",
]
missing = [n for n in required if n not in names]
assert not missing, f"checkpoint missing sparse-MoE tensors: {missing}"
routed = sum(1 for n in names if ".ffn.experts." in n and n.endswith(".w_down.weight"))
shared = sum(1 for n in names if ".ffn.shared_experts." in n and n.endswith(".w_down.weight"))
scorecard = {
    "phase": 43,
    "title": "Sparse/Grouped MoE Dispatch",
    "dispatch_configured": "sparse",
    "dispatch_effective_cpu": "dense_masked",
    "checkpoint": str(model),
    "tensor_count": len(names),
    "routed_expert_down_weights": routed,
    "shared_expert_down_weights": shared,
    "aux_loss_unchanged": True,
}
json.dump(scorecard, open(sys.argv[1], "w"), indent=2)
print(f"Phase 43 checkpoint OK: {len(names)} tensors, sparse-MoE layer 1 present")
print(f"  routed expert down projections: {routed}")
print(f"  shared expert down projections: {shared}")
print(f"wrote {sys.argv[1]}")
PY

echo "Phase 43 smoke completed: $SCORECARD"
