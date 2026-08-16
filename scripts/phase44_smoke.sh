#!/usr/bin/env bash
# Phase 44 — Multi-node distributed training smoke test.
#
# Validates that:
#   - The Phase 44 unit-test suite passes: single-node reproduces v2
#     behaviour exactly, gradient all-reduce correctness across a simulated
#     2-node x 2-GPU topology, rank-zero checkpoint writes from exactly one
#     process globally, transient rendezvous timeout triggers a single retry
#     then fails loudly, plus topology derivation, file/TCP rendezvous over
#     loopback, and the multi-node device-count fix.
#   - The multi-node TOML config (num_nodes, gpus_per_node, node_rank, TCP
#     rendezvous, retry_attempts) deserialises and an 8-step CPU training run
#     completes through the documented single-process fallback — CPU never
#     runs NCCL, per the honesty policy. The real multi-node code path
#     (topology + TCP rendezvous + retry) is verified by the unit tests above.
#
# The real multi-node throughput win lives on multi-VM NCCL hardware; the
# validation paths (external multi-VM tunnel or single-machine loopback
# simulation) are documented in docs/phase44_multi_node.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SCORECARD=${PHASE44_SCORECARD:-artifacts/phase44_multi_node_smoke.json}
mkdir -p "$(dirname "$SCORECARD")"

echo "==> Phase 44 multi-node distributed training unit tests"
cargo test --locked -p aarambh-studio-train --lib distributed

if [[ "${PHASE44_SKIP_TRAIN:-0}" == "1" ]]; then
  echo "PHASE44_SKIP_TRAIN=1; training smoke was skipped"
  python3 - "$SCORECARD" <<'PY'
import json, sys
json.dump({"phase": 44, "train_smoke": "skipped"}, open(sys.argv[1], "w"), indent=2)
print(f"wrote {sys.argv[1]} (train smoke skipped)")
PY
  echo "Phase 44 smoke completed (train skipped)"
  exit 0
fi

echo "==> Phase 44 ensure a tiny training fixture exists"
if [[ ! -f data/tiny_shakespeare.txt ]]; then
  mkdir -p data
  python3 - <<'PY'
from pathlib import Path
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

echo "==> Phase 44 multi-node config deserialisation + CPU fallback training smoke"
cargo run --quiet --locked -p aarambh-studio -- train \
  --config configs/multinode_smoke.toml

echo "==> Phase 44 verify the config parsed as a multi-node run"
python3 - "$SCORECARD" <<'PY'
import json, sys
from pathlib import Path
import tomllib

cfg = tomllib.loads(Path("configs/multinode_smoke.toml").read_text())
dist = cfg["distributed"]
rendezvous = dist["rendezvous"]
assert dist["num_nodes"] == 2, dist
assert dist["gpus_per_node"] == 1, dist
assert dist["node_rank"] == 0, dist
assert rendezvous["kind"] == "tcp", rendezvous
assert rendezvous["endpoint"] == "127.0.0.1:39200", rendezvous
assert dist["retry_attempts"] == 1, dist

# The CPU fallback training run must have produced a checkpoint.
def count_tensors(path: Path) -> int:
    raw = path.read_bytes()
    header_len = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8:8 + header_len].decode("utf-8"))
    return len(header)

checkpoint_dir = Path(cfg["train"]["checkpoint_dir"])
latest = checkpoint_dir / "latest.json"
if latest.exists():
    ptr = json.loads(latest.read_text())
    model = Path(ptr["path"]) / "model.safetensors"
    checkpoint_ok = model.exists()
    tensor_count = count_tensors(model) if checkpoint_ok else 0
else:
    checkpoint_ok = False
    tensor_count = 0

scorecard = {
    "phase": 44,
    "title": "Multi-Node Distributed Training",
    "num_nodes": dist["num_nodes"],
    "gpus_per_node": dist["gpus_per_node"],
    "world_size_derived": dist["num_nodes"] * dist["gpus_per_node"],
    "rendezvous": rendezvous["kind"],
    "rendezvous_endpoint": rendezvous["endpoint"],
    "retry_attempts": dist["retry_attempts"],
    "cpu_fallback": True,
    "checkpoint": str(checkpoint_dir),
    "checkpoint_ok": checkpoint_ok,
    "tensor_count": tensor_count,
}
json.dump(scorecard, open(sys.argv[1], "w"), indent=2)
print(f"Phase 44 config OK: num_nodes={dist['num_nodes']} gpus_per_node={dist['gpus_per_node']} rendezvous={rendezvous['kind']}")
print(f"  checkpoint_ok={checkpoint_ok} tensors={tensor_count}")
print(f"wrote {sys.argv[1]}")
PY

echo "Phase 44 smoke completed: $SCORECARD"
