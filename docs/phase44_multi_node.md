# Phase 44 — Multi-Node Distributed Training

> v4.0.0-alpha.4 · `aarambh-studio-train` (`distributed.rs`, extended) · depends on v2 §27 (single-node NCCL data parallel)

Phase 44 extends the single-node, multi-GPU NCCL data-parallel training
from v2 §27 to **multiple nodes** — still data-parallel only, not model or
pipeline parallelism — so training can scale past whatever a single
machine's GPU count offers.

## Why this matters

v2 §27 proved data-parallel training across the GPUs of *one* machine:
each GPU holds a full model replica, processes a disjoint slice of the
global batch, and the gradients are all-reduced (summed then divided by the
world size) so every replica steps the optimizer in lockstep. That ceiling
is the GPU count of a single box.

Phase 44 lifts only that ceiling. The gradient all-reduce math is
byte-for-byte unchanged from v2 — what changes is the *topology* it runs
over (the world is now `N nodes × M GPUs` instead of `1 node × M GPUs`)
and the *rendezvous* that shares the NCCL unique id (TCP across nodes, not
just a shared-filesystem file). Everything else — the optimizer, the
loss, the bucketed all-reduce, the checkpoint format — is identical.

## Mechanism

```
World: N nodes x M GPUs per node = world_size total ranks

MultiNodeTopology:
    node_rank (which machine) x local_rank (which GPU on that machine)
         |
         v   global_rank       = node_rank * gpus_per_node + local_rank
             global_world_size = num_nodes * gpus_per_node
         |
         v
NCCL rendezvous over TCP (rank 0 binds, others connect) OR over a
shared filesystem file (v2 default, still works single-node)
         |
         v
Sharded data loader: each of the world_size ranks sees a disjoint
slice of the global batch — same principle as v2's single-node sharding,
extended to the larger world_size
         |
         v
Gradient all-reduce across ALL ranks, all nodes — same math as v2 §27,
different (larger) topology
         |
         v
Rank-zero of node-zero specifically logs and checkpoints — prevents
duplicate checkpoints from every node's own local rank zero
```

### The node-rank / local-rank distinction

A multi-node world is `num_nodes × gpus_per_node` ranks. Two indices
identify each rank:

- **`node_rank`** — which machine this rank lives on (0 to `num_nodes - 1`).
- **`local_rank`** — which GPU on that machine (0 to `gpus_per_node - 1`).

These combine into the **global rank** that NCCL and the data loader see:

```rust
// aarambh-studio-train/src/distributed.rs
pub struct MultiNodeTopology {
    pub num_nodes: usize,
    pub gpus_per_node: usize,
    pub node_rank: usize,
    pub local_rank: usize,
}

impl MultiNodeTopology {
    pub fn global_world_size(&self) -> usize {
        self.num_nodes.saturating_mul(self.gpus_per_node)
    }
    pub fn global_rank(&self) -> usize {
        self.node_rank
            .saturating_mul(self.gpus_per_node)
            .saturating_add(self.local_rank)
    }
    /// Only the first node's first GPU (global rank 0) logs/checkpoints.
    pub fn is_global_rank0(&self) -> bool {
        self.node_rank == 0 && self.local_rank == 0
    }
}
```

The invariant `world_size = num_nodes * gpus_per_node` and
`rank = node_rank * gpus_per_node + local_rank` holds by construction, so
the global rank zero — the only rank that logs and checkpoints — is exactly
the first node's first GPU, never every node's local rank zero.

### The device-count fix

v2's device-count check required `device_count >= world_size` on every
worker. That is correct for single-node runs (the whole world is on one
box) but **wrong for multi-node**: a 2-node × 2-GPU world has
`world_size = 4`, yet each node only has 2 GPUs. Phase 44 fixes this so a
multi-node worker only needs `gpus_per_node` devices locally:

```rust
impl ResolvedDistributedConfig {
    /// Single-node runs need world_size devices locally; multi-node runs
    /// only need gpus_per_node (the rest live on other nodes).
    pub fn local_device_requirement(&self) -> usize {
        match &self.topology {
            Some(topology) => topology.gpus_per_node,
            None => self.world_size,
        }
    }
}
```

Single-node configs (`num_nodes` defaults to 1) are untouched — they still
require `world_size` local devices, byte-identical to v2.

## The `RendezvousTransport` enum

```rust
// aarambh-studio-train/src/distributed.rs
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum RendezvousTransport {
    /// File-based rendezvous over a shared filesystem (v2 default).
    #[default]
    File,
    /// TCP rendezvous (Phase 44): rank 0 binds `endpoint`, every other
    /// rank connects to receive the NCCL unique id.
    Tcp { endpoint: String },
}
```

`File` (the default) reproduces v2 single-node behaviour exactly: rank 0
writes the 128-byte NCCL unique id to `rendezvous_dir/run_id/nccl_id.bin`,
every other rank polls until it appears. This works for single-node runs
and for multi-node runs only when every node mounts the same
`rendezvous_dir` over a network share.

`Tcp` is the multi-node transport Phase 44 adds: rank 0 binds a TCP port on
`endpoint` (`host:port`), every other rank connects to it and reads the
128-byte id over the network — no shared filesystem required. This is what
genuinely separate nodes need.

## The TCP rendezvous

```rust
pub trait Rendezvous: Send + Sync {
    /// Rank 0 publishes `id_bytes` to every other rank.
    fn broadcast(&self, id_bytes: &[u8]) -> Result<()>;
    /// Non-zero ranks block until they receive the id bytes from rank 0.
    fn receive(&self) -> Result<Vec<u8>>;
}

pub struct TcpRendezvous {
    endpoint: String,
    world_size: usize,
    rank: usize,
    timeout: Duration,
}
```

Rank 0 binds a `TcpListener` on `endpoint`, sets it non-blocking, and
accepts `world_size - 1` connections; for each it writes the 128 id bytes.
Non-zero ranks `connect_timeout` to `endpoint` (retrying on
connection-refused until the deadline, since rank 0 may not be listening
yet) and `read_exact` 128 bytes.

The trait exchanges raw `Vec<u8>` — not the NCCL `Id` type — so the entire
rendezvous layer is pure standard-library I/O. It compiles and is unit-tested
on CPU without the `cuda` feature; the actual NCCL `Id` only enters at the
call site, behind `#[cfg(feature = "cuda")]`:

```rust
#[cfg(feature = "cuda")]
impl NcclGradientSync {
    fn new(config: &ResolvedDistributedConfig, device: &Device) -> Result<Self> {
        let timeout = Duration::from_secs(config.init_timeout_secs);
        let policy = RetryPolicy::with_retries(config.retry_attempts);
        let rendezvous = build_rendezvous(config, timeout);
        let comm = policy.run(|| -> Result<Comm> {
            if config.rank == 0 {
                let id = Id::new()?;
                rendezvous.broadcast(&id_to_bytes(&id))?;
                Comm::from_rank(stream, rank, world_size, id)?
            } else {
                let id = bytes_to_id(&rendezvous.receive()?)?;
                Comm::from_rank(stream, rank, world_size, id)?
            }
        })?;
        // ... bucketed all-reduce unchanged from v2
    }
}
```

The gradient all-reduce itself (`all_reduce_gradients`, `sync_bucket`,
`all_reduce_flat`) is unchanged from v2 §27 — only the topology it runs
over and the rendezvous that bootstraps it change.

## CPU/CUDA honesty policy

Everything outside the actual NCCL collective calls — the multi-node
topology math, the TCP/file rendezvous exchange, the single-retry fault
policy, and the global-rank-zero checkpointing decision — is pure
standard-library Rust. It compiles and is unit-tested on CPU **without**
the `cuda` feature, exactly as v2 structured its own distributed code. The
real NCCL collectives remain behind `#[cfg(feature = "cuda")]`. The CPU
CI validates every multi-node code path; only the CUDA hardware path is
gated behind the feature.

## Fault tolerance — deliberately minimal

This phase implements exactly one fault-tolerance behaviour: a single
retry on a transient NCCL rendezvous timeout (or connection-refused during
the brief window before rank 0 is listening), after which the run fails
loudly.

```rust
pub struct RetryPolicy {
    pub max_retries: usize,        // default 1 — exactly one retry
    pub backoff: Duration,
}

impl RetryPolicy {
    pub fn run<T, F: FnMut() -> Result<T>>(&self, mut op: F) -> Result<T> {
        // retry up to max_retries times when the error is transient
        // (a timeout or connection-refused); fail loudly otherwise.
    }
}
```

Full elastic training (nodes joining/leaving mid-run, checkpoint-and-resume
on node failure) is explicitly out of scope — a genuinely large feature in
its own right that this project does not attempt to half-implement. The
honesty discipline v2 applied to speculative decoding's speed claim applies
here to fault tolerance: ship the small, correct behaviour, label the rest
as future work, never imply more than was built.

## An honest hardware constraint

Kaggle notebooks do not provide genuine multi-node access — this is stated
plainly rather than glossed over. Validation of this phase realistically
happens one of two ways:

1. **External multi-VM tunnel** — two or more externally-provisioned
   machines on a free or low-cost cloud tier, tunnelled together for NCCL,
   running the real multi-node code path on genuinely separate hardware.
2. **Single-machine loopback simulation** — multiple processes on one
   machine over loopback networking, which exercises the multi-node code
   path's correctness without genuinely separate hardware.

Both validation paths are exercised by the `distributed` unit-test suite
(the TCP rendezvous test binds an ephemeral loopback port and runs four
threads as the four ranks of a 2-node × 2-GPU world). Any throughput
numbers reported for this phase are explicitly labelled with which
validation path produced them — a simulation-derived number is never
presented as a real-hardware benchmark.

## Backward compatibility

`DistributedConfig` gains five new fields — `num_nodes`, `node_rank`,
`gpus_per_node`, `rendezvous`, `retry_attempts` — all defaulting to the
single-node v2 behaviour (`num_nodes = 1`, `rendezvous = File`,
`retry_attempts = 1`). Every existing single-node config (e.g.
`configs/wikitext103_small_2gpu.toml`) deserialises to byte-identical v2
behaviour: `num_nodes <= 1` means single-node, so `world_size` and `rank`
are taken as explicitly configured and the topology is inactive. Only
`num_nodes >= 2` activates multi-node mode, deriving `world_size` and
`rank` from the topology.

## Tests

| Test | Gate |
|---|---|
| `world_size_one_node_reproduces_v2_single_node_behaviour_exactly` | backward compat (num_nodes=1 == v2) |
| `gradient_all_reduce_correctness_across_simulated_multi_node_topology` | all-reduce math across 2-node × 2-GPU (4 ranks) |
| `rank_zero_checkpoint_writes_from_exactly_one_process_globally` | only global rank 0 checkpoints |
| `transient_nccl_timeout_triggers_single_retry_then_fails_loudly` | single-retry fault policy |
| `multi_node_topology_derives_global_rank_and_world_size` | topology math (2×2, 3×4, single-node) |
| `invalid_multi_node_topology_rejected` | topology validation (zero gpus, bad node_rank, bad local_rank) |
| `multi_node_config_requires_gpus_per_node_devices_not_world_size` | device-count fix |
| `file_rendezvous_round_trips_id_bytes` | file transport |
| `file_rendezvous_receive_times_out_when_rank0_never_publishes` | file timeout |
| `tcp_rendezvous_broadcasts_id_bytes_across_loopback` | TCP transport (4 ranks, loopback) |
| `sharded_data_loader_partitions_across_global_world_size_not_local_gpus` | global world_size drives the shard count |
| `multi_node_topology_validate_requires_tcp_endpoint_when_configured` | TCP endpoint validation |
| `config_env_overrides_world_rank_and_local_rank` | v2 env override (unchanged) |
| `gradient_average_matches_two_rank_mean` | v2 all-reduce math (unchanged) |
| `invalid_rank_is_rejected` | v2 rank validation (unchanged) |

The four roadmap-named tests (`world_size_one_node...`,
`gradient_all_reduce_correctness...`, `rank_zero_checkpoint...`,
`transient_nccl_timeout...`) are the Phase 44 acceptance tests; the rest
are the supporting CPU unit tests that exercise the new code paths without
CUDA hardware.

## Configs

- `configs/multinode_smoke.toml` — CPU smoke with `num_nodes = 2`,
  `gpus_per_node = 1`, TCP rendezvous on `127.0.0.1:39200`, and
  `retry_attempts = 1`. Deserialises the multi-node fields and runs an
  8-step CPU training through the single-process fallback (CPU never runs
  NCCL, per the honesty policy).

## Smoke script

`scripts/phase44_smoke.sh` runs the `distributed` unit-test suite (the
real multi-node code paths on CPU: topology, TCP rendezvous over loopback,
retry policy, rank-zero decision, device-count fix), then a two-step CPU
training smoke on `multinode_smoke.toml` that validates the config
deserialisation and the single-process fallback, writing a scorecard to
`artifacts/phase44_multi_node_smoke.json`.

## Milestone

Multi-node data-parallel training runs correctly on the documented
validation path (external multi-VM tunnel or single-machine loopback
simulation), with gradient correctness verified against the single-node v2
baseline on identical data. Real-hardware multi-node throughput numbers are
reported only where genuinely available and are clearly labelled as such —
never implied from the simulation path.

```
git commit -m "feat: Phase 44 — multi-node distributed training"
git tag v4.0.0-alpha.4
```
