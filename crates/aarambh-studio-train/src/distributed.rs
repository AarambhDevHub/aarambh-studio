//! Data-parallel distributed training helpers.
//!
//! Phase 44 (v4) extends the original single-node NCCL data parallelism
//! (v2 §27) to multiple nodes. The gradient all-reduce math is unchanged
//! from v2 — only the topology it runs over grows, and the rendezvous that
//! shares the NCCL unique id now supports a TCP transport so nodes without
//! a shared filesystem can join the world.
//!
//! Everything outside the actual NCCL collective calls — the multi-node
//! topology math, the TCP/file rendezvous exchange, the single-retry fault
//! policy, and the global-rank-zero checkpointing decision — is pure
//! standard-library Rust, so it compiles and is unit-tested on CPU without
//! the `cuda` feature. The real NCCL collectives remain behind
//! `#[cfg(feature = "cuda")]`, exactly as in v2.

use std::env;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use aarambh_studio_core::{AarambhError, Result};
#[cfg(any(feature = "cuda", test))]
use candle_core::{DType, Tensor};
use serde::{Deserialize, Serialize};

use crate::optim::GradMap;

const DEFAULT_BUCKET_BYTES: usize = 64 * 1024 * 1024;
const DEFAULT_INIT_TIMEOUT_SECS: u64 = 120;
const DEFAULT_RUN_ID: &str = "aarambh-dist";
/// Size in bytes of the NCCL unique-id blob exchanged during rendezvous.
pub const NCCL_ID_BYTES: usize = 128;
/// Polling interval used while waiting for file or TCP rendezvous.
const RENDEZVOUS_POLL_INTERVAL: Duration = Duration::from_millis(50);
/// Default backoff between retry attempts of a transient failure.
const DEFAULT_RETRY_BACKOFF: Duration = Duration::from_millis(100);

/// Distributed collective backend.
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DistributedBackend {
    /// NVIDIA NCCL collectives through Candle/cudarc.
    #[default]
    Nccl,
}

/// How the NCCL unique id is shared between ranks during rendezvous.
///
/// The default `File` transport reproduces the v2 single-node behaviour
/// byte-for-byte: a shared-filesystem rendezvous directory that rank 0
/// writes the id to and every other rank polls until it appears. The `Tcp`
/// transport (Phase 44) lets genuinely separate nodes exchange the id
/// without a shared filesystem — rank 0 binds a TCP port, every other rank
/// connects to it to receive the id over the network.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum RendezvousTransport {
    /// File-based rendezvous over a shared filesystem (v2 default).
    ///
    /// Single-node default. Usable for multi-node only when every node
    /// mounts the same `rendezvous_dir` over a network share.
    #[default]
    File,
    /// TCP rendezvous (Phase 44): rank 0 binds `endpoint`, every other
    /// rank connects to receive the NCCL unique id. Required for
    /// multi-node runs whose nodes do not share a filesystem.
    Tcp {
        /// `host:port` rank 0 binds; non-zero ranks connect here.
        endpoint: String,
    },
}

impl RendezvousTransport {
    /// Return true when this is the multi-node TCP transport.
    pub fn is_tcp(&self) -> bool {
        matches!(self, Self::Tcp { .. })
    }
}

fn default_num_nodes() -> usize {
    1
}

fn default_gpus_per_node() -> usize {
    1
}

fn default_retry_attempts() -> usize {
    1
}

/// Resolved multi-node topology (Phase 44).
///
/// Combines `num_nodes`, `gpus_per_node`, `node_rank`, and `local_rank`
/// into the global rank and world size that NCCL and the data loader see.
/// The invariants are:
///
/// ```text
/// world_size = num_nodes * gpus_per_node
/// rank       = node_rank * gpus_per_node + local_rank
/// ```
///
/// so the global rank zero — the only rank that logs and checkpoints — is
/// exactly the first node's first GPU, never every node's local rank zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultiNodeTopology {
    /// Total number of nodes participating in the world.
    pub num_nodes: usize,
    /// Number of CUDA devices each node contributes.
    pub gpus_per_node: usize,
    /// Zero-based index of this node within the world.
    pub node_rank: usize,
    /// This process's GPU index within its node.
    pub local_rank: usize,
}

impl MultiNodeTopology {
    /// Build a topology from its four components.
    pub fn new(
        num_nodes: usize,
        gpus_per_node: usize,
        node_rank: usize,
        local_rank: usize,
    ) -> Self {
        Self {
            num_nodes,
            gpus_per_node,
            node_rank,
            local_rank,
        }
    }

    /// Global number of ranks: `num_nodes * gpus_per_node`.
    pub fn global_world_size(&self) -> usize {
        self.num_nodes.saturating_mul(self.gpus_per_node)
    }

    /// Global rank of this process: `node_rank * gpus_per_node + local_rank`.
    pub fn global_rank(&self) -> usize {
        self.node_rank
            .saturating_mul(self.gpus_per_node)
            .saturating_add(self.local_rank)
    }

    /// Return true only for the first node's first GPU (global rank 0).
    ///
    /// This is the rank that logs and checkpoints globally — multi-node
    /// runs do not produce duplicate checkpoints from every node's own
    /// local rank zero.
    pub fn is_global_rank0(&self) -> bool {
        self.node_rank == 0 && self.local_rank == 0
    }

    /// Return true when more than one node participates.
    pub fn is_multi_node(&self) -> bool {
        self.num_nodes >= 2
    }

    /// Validate the topology fields that do not depend on derived values.
    pub fn validate(&self) -> Result<()> {
        if self.num_nodes == 0 {
            return Err(AarambhError::Config(
                "distributed.num_nodes must be greater than zero".into(),
            ));
        }
        if self.gpus_per_node == 0 {
            return Err(AarambhError::Config(
                "distributed.gpus_per_node must be greater than zero".into(),
            ));
        }
        if self.node_rank >= self.num_nodes {
            return Err(AarambhError::Config(format!(
                "distributed.node_rank {} must be less than num_nodes {}",
                self.node_rank, self.num_nodes
            )));
        }
        if self.local_rank >= self.gpus_per_node {
            return Err(AarambhError::Config(format!(
                "distributed.local_rank {} must be less than gpus_per_node {}",
                self.local_rank, self.gpus_per_node
            )));
        }
        Ok(())
    }
}

/// TOML configuration for one data-parallel worker.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct DistributedConfig {
    /// Enable distributed data-parallel training.
    pub enabled: bool,
    /// Collective backend.
    pub backend: DistributedBackend,
    /// Total number of worker processes (global, across all nodes).
    ///
    /// When `num_nodes >= 2` this is derived as `num_nodes * gpus_per_node`
    /// and any explicitly-configured value is ignored in favour of the
    /// derived one.
    pub world_size: usize,
    /// Global rank for this worker.
    ///
    /// When `num_nodes >= 2` this is derived as
    /// `node_rank * gpus_per_node + local_rank`.
    pub rank: usize,
    /// CUDA device index local to this machine.
    pub local_rank: usize,
    /// Rendezvous run identifier used for NCCL unique-id sharing.
    pub run_id: String,
    /// Directory used for file-based NCCL rendezvous.
    pub rendezvous_dir: PathBuf,
    /// Maximum seconds nonzero ranks wait for rank 0 rendezvous.
    pub init_timeout_secs: u64,
    /// Maximum F32 gradient bucket size before an all-reduce call.
    pub bucket_bytes: usize,
    /// Fall back to rank-0 single-GPU training when requested GPUs are unavailable.
    pub fallback_single_gpu: bool,
    /// Number of nodes participating in the world (Phase 44).
    ///
    /// Defaults to `1`, which reproduces v2 single-node behaviour exactly.
    /// Values `>= 2` activate multi-node mode: `world_size` and `rank` are
    /// derived from the topology, and a TCP rendezvous (or a genuinely
    /// shared `rendezvous_dir`) is expected.
    #[serde(default = "default_num_nodes")]
    pub num_nodes: usize,
    /// Zero-based index of this node within the world (Phase 44).
    #[serde(default)]
    pub node_rank: usize,
    /// Number of CUDA devices each node contributes (Phase 44).
    #[serde(default = "default_gpus_per_node")]
    pub gpus_per_node: usize,
    /// Rendezvous transport used to share the NCCL unique id (Phase 44).
    #[serde(default)]
    pub rendezvous: RendezvousTransport,
    /// Number of retries after the first attempt when a transient
    /// rendezvous or all-reduce failure occurs (Phase 44). Defaults to
    /// `1` — exactly one retry, after which the run fails loudly.
    #[serde(default = "default_retry_attempts")]
    pub retry_attempts: usize,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            backend: DistributedBackend::Nccl,
            world_size: 1,
            rank: 0,
            local_rank: 0,
            run_id: DEFAULT_RUN_ID.to_string(),
            rendezvous_dir: PathBuf::from(".aarambh_dist"),
            init_timeout_secs: DEFAULT_INIT_TIMEOUT_SECS,
            bucket_bytes: DEFAULT_BUCKET_BYTES,
            fallback_single_gpu: true,
            num_nodes: 1,
            node_rank: 0,
            gpus_per_node: 1,
            rendezvous: RendezvousTransport::File,
            retry_attempts: 1,
        }
    }
}

impl DistributedConfig {
    /// Validate the distributed configuration values that do not depend on hardware.
    pub fn validate(&self) -> Result<()> {
        if self.bucket_bytes == 0 {
            return Err(AarambhError::Config(
                "distributed.bucket_bytes must be greater than zero".into(),
            ));
        }
        if self.init_timeout_secs == 0 {
            return Err(AarambhError::Config(
                "distributed.init_timeout_secs must be greater than zero".into(),
            ));
        }
        if self.num_nodes >= 2 {
            // Multi-node mode: rank/world_size are derived from topology in
            // resolve_runtime, so validate the topology here and only reject
            // a clearly-impossible explicit world_size.
            let topology = MultiNodeTopology::new(
                self.num_nodes,
                self.gpus_per_node,
                self.node_rank,
                self.local_rank,
            );
            topology.validate()?;
            if self.world_size > 1 && self.world_size != topology.global_world_size() {
                return Err(AarambhError::Config(format!(
                    "distributed.world_size {} does not match num_nodes*gpus_per_node = {}",
                    self.world_size,
                    topology.global_world_size()
                )));
            }
            if let RendezvousTransport::Tcp { endpoint } = &self.rendezvous
                && endpoint.trim().is_empty()
            {
                return Err(AarambhError::Config(
                    "distributed.rendezvous TCP endpoint must not be empty".into(),
                ));
            }
        } else {
            if self.world_size == 0 {
                return Err(AarambhError::Config(
                    "distributed.world_size must be greater than zero".into(),
                ));
            }
            if self.rank >= self.world_size {
                return Err(AarambhError::Config(format!(
                    "distributed.rank {} must be less than world_size {}",
                    self.rank, self.world_size
                )));
            }
        }
        Ok(())
    }

    /// Return the resolved multi-node topology, or `None` for single-node.
    pub fn topology(&self) -> Option<MultiNodeTopology> {
        if self.num_nodes >= 2 {
            Some(MultiNodeTopology::new(
                self.num_nodes,
                self.gpus_per_node,
                self.node_rank,
                self.local_rank,
            ))
        } else {
            None
        }
    }

    /// Return the number of CUDA devices this single node must provide.
    ///
    /// Single-node runs need `world_size` devices locally; multi-node runs
    /// only need `gpus_per_node` (the rest live on other nodes). This is
    /// the Phase 44 fix to the v2 device-count check, which required the
    /// full global world size on every node.
    pub fn local_device_requirement(&self) -> usize {
        if self.num_nodes >= 2 {
            self.gpus_per_node
        } else {
            self.world_size
        }
    }
}

/// Fully resolved distributed worker configuration after env overrides.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedDistributedConfig {
    /// Collective backend.
    pub backend: DistributedBackend,
    /// Total number of worker processes (global across all nodes).
    pub world_size: usize,
    /// Global rank for this worker.
    pub rank: usize,
    /// CUDA device index local to this machine.
    pub local_rank: usize,
    /// Rendezvous run identifier used for NCCL unique-id sharing.
    pub run_id: String,
    /// Directory used for file-based NCCL rendezvous.
    pub rendezvous_dir: PathBuf,
    /// Maximum seconds nonzero ranks wait for rank 0 rendezvous.
    pub init_timeout_secs: u64,
    /// Maximum F32 gradient bucket size before an all-reduce call.
    pub bucket_bytes: usize,
    /// Resolved multi-node topology when `num_nodes >= 2`, else `None`.
    pub topology: Option<MultiNodeTopology>,
    /// Rendezvous transport used to share the NCCL unique id.
    pub rendezvous: RendezvousTransport,
    /// Retry attempts applied to transient rendezvous/all-reduce failures.
    pub retry_attempts: usize,
}

impl ResolvedDistributedConfig {
    /// Return true when this worker is the global rank 0.
    ///
    /// In multi-node runs the global rank 0 is exactly the first node's
    /// first GPU, so only that process logs and checkpoints — never every
    /// node's local rank zero.
    pub fn is_rank0(&self) -> bool {
        self.rank == 0
    }

    /// Return true when this is a multi-node run.
    pub fn is_multi_node(&self) -> bool {
        self.topology.is_some_and(|t| t.is_multi_node())
    }

    /// Return the number of CUDA devices this single node must provide.
    ///
    /// Single-node runs need `world_size` devices locally; multi-node runs
    /// only need `gpus_per_node` (the rest live on other nodes). This is
    /// the Phase 44 fix to the v2 device-count check, which required the
    /// full global world size on every node.
    pub fn local_device_requirement(&self) -> usize {
        match &self.topology {
            Some(topology) => topology.gpus_per_node,
            None => self.world_size,
        }
    }
}

/// Runtime decision for the current process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DistributedRuntime {
    /// Distributed training is disabled.
    Disabled,
    /// This process participates in NCCL data-parallel training.
    Active(ResolvedDistributedConfig),
    /// Rank 0 should run the normal single-process path.
    SingleProcessFallback {
        /// Requested global rank.
        rank: usize,
        /// Requested world size.
        world_size: usize,
        /// Human-readable fallback reason.
        reason: String,
    },
    /// This worker should exit successfully because rank 0 is handling fallback.
    NonParticipant {
        /// Requested global rank.
        rank: usize,
        /// Requested world size.
        world_size: usize,
        /// Human-readable exit reason.
        reason: String,
    },
}

impl DistributedRuntime {
    /// Return true when this process is the logging/checkpointing rank.
    pub fn is_rank0(&self) -> bool {
        match self {
            Self::Active(config) => config.is_rank0(),
            Self::Disabled | Self::SingleProcessFallback { .. } => true,
            Self::NonParticipant { .. } => false,
        }
    }
}

/// Resolve distributed configuration from TOML plus environment variables.
pub fn resolve_runtime(config: Option<&DistributedConfig>) -> Result<DistributedRuntime> {
    let mut resolved = config.cloned().unwrap_or_default();
    apply_env_overrides(&mut resolved)?;
    resolve_multi_node_topology(&mut resolved)?;
    resolved.validate()?;

    if !resolved.enabled && resolved.world_size <= 1 {
        return Ok(DistributedRuntime::Disabled);
    }
    if resolved.world_size <= 1 {
        return Ok(DistributedRuntime::Disabled);
    }

    let available = cuda_device_count();
    let Some(device_count) = available else {
        return fallback_or_error(
            &resolved,
            "CUDA/NCCL support is not available in this build",
        );
    };
    let local_device_requirement = resolved.local_device_requirement();
    if device_count < local_device_requirement || resolved.local_rank >= device_count {
        return fallback_or_error(
            &resolved,
            &format!(
                "requested world_size={} local_rank={} num_nodes={} gpus_per_node={} but only {device_count} CUDA device(s) are visible",
                resolved.world_size,
                resolved.local_rank,
                resolved.num_nodes,
                resolved.gpus_per_node
            ),
        );
    }

    let topology = resolved.topology();
    Ok(DistributedRuntime::Active(ResolvedDistributedConfig {
        backend: resolved.backend,
        world_size: resolved.world_size,
        rank: resolved.rank,
        local_rank: resolved.local_rank,
        run_id: resolved.run_id,
        rendezvous_dir: resolved.rendezvous_dir,
        init_timeout_secs: resolved.init_timeout_secs,
        bucket_bytes: resolved.bucket_bytes,
        topology,
        rendezvous: resolved.rendezvous,
        retry_attempts: resolved.retry_attempts,
    }))
}

/// Derive global `world_size` and `rank` from the multi-node topology when
/// `num_nodes >= 2`. Single-node configs (`num_nodes <= 1`) are untouched,
/// preserving v2 behaviour byte-for-byte.
fn resolve_multi_node_topology(config: &mut DistributedConfig) -> Result<()> {
    if config.num_nodes <= 1 {
        return Ok(());
    }
    let topology = MultiNodeTopology::new(
        config.num_nodes,
        config.gpus_per_node,
        config.node_rank,
        config.local_rank,
    );
    topology.validate()?;
    config.world_size = topology.global_world_size();
    config.rank = topology.global_rank();
    config.enabled = true;
    Ok(())
}

fn apply_env_overrides(config: &mut DistributedConfig) -> Result<()> {
    if let Some(world_size) = env_usize("AARAMBH_STUDIO_WORLD_SIZE")? {
        config.world_size = world_size;
        config.enabled = world_size > 1;
    }
    if let Some(rank) = env_usize("AARAMBH_STUDIO_RANK")? {
        config.rank = rank;
    }
    if let Some(local_rank) = env_usize("AARAMBH_STUDIO_LOCAL_RANK")? {
        config.local_rank = local_rank;
    }
    if let Some(num_nodes) = env_usize("AARAMBH_STUDIO_NUM_NODES")? {
        config.num_nodes = num_nodes;
    }
    if let Some(node_rank) = env_usize("AARAMBH_STUDIO_NODE_RANK")? {
        config.node_rank = node_rank;
    }
    if let Some(gpus) = env_usize("AARAMBH_STUDIO_GPUS_PER_NODE")? {
        config.gpus_per_node = gpus;
    }
    if let Some(retries) = env_usize("AARAMBH_STUDIO_DIST_RETRIES")? {
        config.retry_attempts = retries;
    }
    if let Ok(run_id) = env::var("AARAMBH_STUDIO_DIST_RUN_ID")
        && !run_id.trim().is_empty()
    {
        config.run_id = run_id;
    }
    if let Ok(path) = env::var("AARAMBH_STUDIO_DIST_RENDEZVOUS")
        && !path.trim().is_empty()
    {
        config.rendezvous_dir = PathBuf::from(path);
    }
    if let Ok(endpoint) = env::var("AARAMBH_STUDIO_DIST_RENDEZVOUS_ENDPOINT")
        && !endpoint.trim().is_empty()
    {
        config.rendezvous = RendezvousTransport::Tcp { endpoint };
    }
    Ok(())
}

fn env_usize(name: &str) -> Result<Option<usize>> {
    match env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .map(Some)
            .map_err(|err| AarambhError::Config(format!("invalid {name} value '{value}': {err}"))),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(err) => Err(AarambhError::Config(format!("invalid {name}: {err}"))),
    }
}

fn fallback_or_error(config: &DistributedConfig, reason: &str) -> Result<DistributedRuntime> {
    if !config.fallback_single_gpu {
        return Err(AarambhError::Unsupported(format!(
            "distributed training requires usable CUDA/NCCL: {reason}"
        )));
    }
    if config.rank == 0 {
        Ok(DistributedRuntime::SingleProcessFallback {
            rank: config.rank,
            world_size: config.world_size,
            reason: reason.to_string(),
        })
    } else {
        Ok(DistributedRuntime::NonParticipant {
            rank: config.rank,
            world_size: config.world_size,
            reason: reason.to_string(),
        })
    }
}

#[cfg(feature = "cuda")]
fn cuda_device_count() -> Option<usize> {
    candle_core::cuda::cudarc::driver::CudaDevice::count()
        .ok()
        .map(|count| count as usize)
}

#[cfg(not(feature = "cuda"))]
fn cuda_device_count() -> Option<usize> {
    None
}

/// Single-retry policy for transient distributed failures (Phase 44).
///
/// Implements the roadmap's "exactly one retry on a transient NCCL
/// rendezvous timeout, then fail loudly" behaviour, without attempting
/// full elastic training (explicitly out of scope). The policy is pure
/// standard-library Rust and unit-tested on CPU without the `cuda`
/// feature.
#[derive(Debug, Clone, Copy)]
pub struct RetryPolicy {
    /// Number of retries after the first attempt (0 = no retry, 1 = one retry).
    pub max_retries: usize,
    /// Sleep duration between attempts.
    pub backoff: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: default_retry_attempts(),
            backoff: DEFAULT_RETRY_BACKOFF,
        }
    }
}

impl RetryPolicy {
    /// Build a policy with the given retry count and the default backoff.
    pub fn with_retries(max_retries: usize) -> Self {
        Self {
            max_retries,
            backoff: DEFAULT_RETRY_BACKOFF,
        }
    }

    /// Run `op`, retrying up to `max_retries` times when the error is
    /// transient (a rendezvous timeout or connection-refused during the
    /// brief window before rank 0 is listening). Non-transient errors
    /// propagate immediately on the first attempt.
    pub fn run<T, F>(&self, mut op: F) -> Result<T>
    where
        F: FnMut() -> Result<T>,
    {
        let mut last_err: Option<AarambhError> = None;
        for attempt in 0..=self.max_retries {
            if attempt > 0 {
                std::thread::sleep(self.backoff);
            }
            match op() {
                Ok(value) => return Ok(value),
                Err(err) => {
                    let transient = is_transient(&err);
                    last_err = Some(err);
                    if !transient || attempt == self.max_retries {
                        break;
                    }
                }
            }
        }
        Err(last_err.expect("retry loop always runs at least once"))
    }
}

/// Classify a distributed error as transient (retryable).
///
/// A transient error is a rendezvous timeout or a connection-refused
/// during the brief window before rank 0 is listening — both expected
/// during normal startup and worth a single retry. Other errors (shape
/// mismatch, unsupported build, invalid config) propagate immediately.
fn is_transient(err: &AarambhError) -> bool {
    let message = err.to_string().to_lowercase();
    message.contains("timed out")
        || message.contains("timeout")
        || message.contains("connection refused")
}

/// Exchange the NCCL unique-id blob between ranks during rendezvous.
///
/// The blob is [`NCCL_ID_BYTES`] (128) raw bytes. Rank 0 produces it and
/// every other rank receives an identical copy. Both implementations are
/// pure standard-library I/O, so they compile and are tested on CPU
/// without the `cuda` feature — the actual NCCL `Id` type only enters at
/// the call site, behind `#[cfg(feature = "cuda")]`.
pub trait Rendezvous: Send + Sync {
    /// Rank 0 publishes `id_bytes` to every other rank.
    fn broadcast(&self, id_bytes: &[u8]) -> Result<()>;
    /// Non-zero ranks block until they receive the id bytes from rank 0.
    fn receive(&self) -> Result<Vec<u8>>;
}

/// File-based rendezvous over a shared filesystem (v2 default).
pub struct FileRendezvous {
    dir: PathBuf,
    run_id: String,
    timeout: Duration,
}

impl FileRendezvous {
    /// Create a file rendezvous rooted at `dir/run_id/nccl_id.bin`.
    pub fn new(dir: impl Into<PathBuf>, run_id: impl Into<String>, timeout: Duration) -> Self {
        Self {
            dir: dir.into(),
            run_id: run_id.into(),
            timeout,
        }
    }

    fn path(&self) -> PathBuf {
        self.dir.join(&self.run_id).join("nccl_id.bin")
    }
}

impl Rendezvous for FileRendezvous {
    fn broadcast(&self, id_bytes: &[u8]) -> Result<()> {
        let path = self.path();
        let dir = path
            .parent()
            .ok_or_else(|| AarambhError::Config("invalid NCCL rendezvous path".into()))?;
        std::fs::create_dir_all(dir)?;
        let tmp = path.with_extension("bin.rank0.tmp");
        std::fs::write(&tmp, id_bytes)?;
        std::fs::rename(tmp, &path)?;
        Ok(())
    }

    fn receive(&self) -> Result<Vec<u8>> {
        let path = self.path();
        let deadline = Instant::now() + self.timeout;
        loop {
            match std::fs::read(&path) {
                Ok(bytes) => {
                    if bytes.len() == NCCL_ID_BYTES {
                        return Ok(bytes);
                    }
                    return Err(AarambhError::Config(format!(
                        "invalid NCCL id length in {}: {}",
                        path.display(),
                        bytes.len()
                    )));
                }
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                    if Instant::now() >= deadline {
                        return Err(AarambhError::Config(format!(
                            "timed out waiting for NCCL rendezvous file {}",
                            path.display()
                        )));
                    }
                    std::thread::sleep(RENDEZVOUS_POLL_INTERVAL);
                }
                Err(err) => return Err(err.into()),
            }
        }
    }
}

/// TCP rendezvous: rank 0 binds `endpoint`, every other rank connects
/// (Phase 44). Required for multi-node runs whose nodes do not share a
/// filesystem.
pub struct TcpRendezvous {
    endpoint: String,
    world_size: usize,
    rank: usize,
    timeout: Duration,
}

impl TcpRendezvous {
    /// Create a TCP rendezvous. Rank 0 binds `endpoint` and accepts
    /// `world_size - 1` connections; non-zero ranks connect to receive
    /// the id.
    pub fn new(
        endpoint: impl Into<String>,
        world_size: usize,
        rank: usize,
        timeout: Duration,
    ) -> Self {
        Self {
            endpoint: endpoint.into(),
            world_size,
            rank,
            timeout,
        }
    }

    fn socket_addr(&self) -> Result<std::net::SocketAddr> {
        self.endpoint.parse().map_err(|err| {
            AarambhError::Config(format!(
                "invalid TCP rendezvous endpoint '{}': {}",
                self.endpoint, err
            ))
        })
    }
}

impl Rendezvous for TcpRendezvous {
    fn broadcast(&self, id_bytes: &[u8]) -> Result<()> {
        if self.rank != 0 {
            return Err(AarambhError::Config(
                "TCP rendezvous broadcast may only be called by rank 0".into(),
            ));
        }
        let listener = TcpListener::bind(self.socket_addr()?).map_err(|err| {
            AarambhError::Config(format!(
                "failed to bind TCP rendezvous {}: {}",
                self.endpoint, err
            ))
        })?;
        listener
            .set_nonblocking(true)
            .map_err(|err| AarambhError::Config(format!("failed to set nonblocking: {err}")))?;
        let expected = self.world_size.saturating_sub(1);
        let deadline = Instant::now() + self.timeout;
        let mut accepted = 0usize;
        while accepted < expected {
            if Instant::now() >= deadline {
                return Err(AarambhError::Config(format!(
                    "timed out waiting for TCP rendezvous peers on {} (accepted {}/{})",
                    self.endpoint, accepted, expected
                )));
            }
            match listener.accept() {
                Ok((mut stream, _)) => {
                    stream
                        .set_write_timeout(Some(self.timeout))
                        .map_err(|err| AarambhError::Config(format!("set_write_timeout: {err}")))?;
                    if stream.write_all(id_bytes).is_ok() {
                        accepted += 1;
                    }
                }
                Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(RENDEZVOUS_POLL_INTERVAL);
                }
                Err(err) => {
                    return Err(AarambhError::Config(format!(
                        "TCP rendezvous accept failed: {err}"
                    )));
                }
            }
        }
        Ok(())
    }

    fn receive(&self) -> Result<Vec<u8>> {
        let addr = self.socket_addr()?;
        let deadline = Instant::now() + self.timeout;
        loop {
            let now = Instant::now();
            if now >= deadline {
                return Err(AarambhError::Config(format!(
                    "timed out connecting to TCP rendezvous {}",
                    self.endpoint
                )));
            }
            let remaining = deadline.saturating_duration_since(now);
            match TcpStream::connect_timeout(&addr, remaining) {
                Ok(mut stream) => {
                    stream
                        .set_read_timeout(Some(self.timeout))
                        .map_err(|err| AarambhError::Config(format!("set_read_timeout: {err}")))?;
                    let mut buffer = vec![0u8; NCCL_ID_BYTES];
                    match stream.read_exact(&mut buffer) {
                        Ok(()) => return Ok(buffer),
                        Err(err) => {
                            if Instant::now() >= deadline {
                                return Err(AarambhError::Config(format!(
                                    "timed out reading TCP rendezvous id from {}: {err}",
                                    self.endpoint
                                )));
                            }
                            std::thread::sleep(RENDEZVOUS_POLL_INTERVAL);
                        }
                    }
                }
                Err(err) => {
                    if Instant::now() >= deadline {
                        return Err(AarambhError::Config(format!(
                            "timed out connecting to TCP rendezvous {}: {err}",
                            self.endpoint
                        )));
                    }
                    std::thread::sleep(RENDEZVOUS_POLL_INTERVAL);
                }
            }
        }
    }
}

/// Build the rendezvous implementation selected by `config`.
///
/// `File` → [`FileRendezvous`] (v2 single-node default). `Tcp` →
/// [`TcpRendezvous`] (Phase 44 multi-node). Both are pure standard-library
/// I/O, so this dispatch is available without the `cuda` feature.
pub fn build_rendezvous(
    config: &ResolvedDistributedConfig,
    timeout: Duration,
) -> Box<dyn Rendezvous> {
    match &config.rendezvous {
        RendezvousTransport::File => Box::new(FileRendezvous::new(
            config.rendezvous_dir.clone(),
            config.run_id.clone(),
            timeout,
        )),
        RendezvousTransport::Tcp { endpoint } => Box::new(TcpRendezvous::new(
            endpoint.clone(),
            config.world_size,
            config.rank,
            timeout,
        )),
    }
}

/// Active distributed training context for a worker process.
pub struct DistributedContext {
    config: ResolvedDistributedConfig,
    device: candle_core::Device,
    #[cfg(feature = "cuda")]
    nccl: NcclGradientSync,
}

impl DistributedContext {
    /// Initialize a distributed context on the selected CUDA device.
    pub fn init(config: ResolvedDistributedConfig, device: &candle_core::Device) -> Result<Self> {
        Self::init_impl(config, device)
    }

    /// Return true when this worker is the global rank 0.
    pub fn is_rank0(&self) -> bool {
        self.config.is_rank0()
    }

    /// Return this worker's global rank.
    pub fn rank(&self) -> usize {
        self.config.rank
    }

    /// Return the total worker count.
    pub fn world_size(&self) -> usize {
        self.config.world_size
    }

    /// Return true when this is a multi-node run (Phase 44).
    pub fn is_multi_node(&self) -> bool {
        self.config.is_multi_node()
    }

    /// Average gradients across all ranks in place.
    pub fn all_reduce_gradients(&self, grads: &mut GradMap) -> Result<()> {
        if self.world_size() <= 1 {
            return Ok(());
        }
        self.all_reduce_gradients_impl(grads)
    }

    /// Synchronize all participating ranks.
    pub fn barrier(&self) -> Result<()> {
        if self.world_size() <= 1 {
            return Ok(());
        }
        let mut marker = GradMap::new();
        marker.insert(
            "barrier".into(),
            candle_core::Tensor::zeros((1,), candle_core::DType::F32, &self.device)?,
        );
        self.all_reduce_gradients(&mut marker)
    }

    /// Return whether any participating rank reported a local failure.
    pub fn any_rank_failed(&self, local_failed: bool) -> Result<bool> {
        if self.world_size() <= 1 {
            return Ok(local_failed);
        }
        let mut marker = GradMap::new();
        marker.insert(
            "observer_failure".into(),
            candle_core::Tensor::new(&[if local_failed { 1.0f32 } else { 0.0 }], &self.device)?,
        );
        self.all_reduce_gradients(&mut marker)?;
        let value = marker
            .remove("observer_failure")
            .ok_or_else(|| AarambhError::Config("distributed failure marker disappeared".into()))?
            .to_vec1::<f32>()?[0];
        Ok(value > 0.0)
    }

    #[cfg(feature = "cuda")]
    fn init_impl(config: ResolvedDistributedConfig, device: &candle_core::Device) -> Result<Self> {
        let nccl = NcclGradientSync::new(&config, device)?;
        Ok(Self {
            config,
            device: device.clone(),
            nccl,
        })
    }

    #[cfg(not(feature = "cuda"))]
    fn init_impl(
        _config: ResolvedDistributedConfig,
        _device: &candle_core::Device,
    ) -> Result<Self> {
        Err(AarambhError::Unsupported(
            "distributed training requires the cuda feature".into(),
        ))
    }

    #[cfg(feature = "cuda")]
    fn all_reduce_gradients_impl(&self, grads: &mut GradMap) -> Result<()> {
        let policy = RetryPolicy::with_retries(self.config.retry_attempts);
        policy.run(|| self.nccl.all_reduce_gradients(grads))
    }

    #[cfg(not(feature = "cuda"))]
    fn all_reduce_gradients_impl(&self, _grads: &mut GradMap) -> Result<()> {
        Err(AarambhError::Unsupported(
            "distributed gradient sync requires the cuda feature".into(),
        ))
    }
}

#[cfg(feature = "cuda")]
struct NcclGradientSync {
    comm: candle_core::cuda::cudarc::nccl::safe::Comm,
    world_size: usize,
    bucket_bytes: usize,
}

#[cfg(feature = "cuda")]
impl NcclGradientSync {
    fn new(config: &ResolvedDistributedConfig, device: &candle_core::Device) -> Result<Self> {
        use candle_core::cuda::cudarc::nccl::safe::{Comm, Id};

        let cuda = device.as_cuda_device().map_err(|err| {
            AarambhError::Config(format!("distributed training requires CUDA: {err}"))
        })?;
        let timeout = Duration::from_secs(config.init_timeout_secs);
        let policy = RetryPolicy::with_retries(config.retry_attempts);
        let rendezvous = build_rendezvous(config, timeout);
        let rank = config.rank;
        let world_size = config.world_size;
        let stream = cuda.cuda_stream();
        let comm = policy.run(|| -> Result<Comm> {
            if rank == 0 {
                let id = Id::new().map_err(|err| {
                    AarambhError::Config(format!("failed to create NCCL id: {err:?}"))
                })?;
                let bytes = id_to_bytes(&id);
                rendezvous.broadcast(&bytes)?;
                Comm::from_rank(stream, rank, world_size, id).map_err(|err| {
                    AarambhError::Config(format!("failed to initialize NCCL: {err:?}"))
                })
            } else {
                let bytes = rendezvous.receive()?;
                let id = bytes_to_id(&bytes)?;
                Comm::from_rank(stream, rank, world_size, id).map_err(|err| {
                    AarambhError::Config(format!("failed to initialize NCCL: {err:?}"))
                })
            }
        })?;
        Ok(Self {
            comm,
            world_size,
            bucket_bytes: config.bucket_bytes,
        })
    }

    fn all_reduce_gradients(&self, grads: &mut GradMap) -> Result<()> {
        if grads.is_empty() {
            return Ok(());
        }

        let mut names = grads.keys().cloned().collect::<Vec<_>>();
        names.sort();

        let mut flat_grads = Vec::with_capacity(names.len());
        for name in names {
            let grad = grads
                .get(&name)
                .ok_or_else(|| AarambhError::Config(format!("missing gradient {name}")))?;
            let shape = grad.shape().dims().to_vec();
            let tensor = grad.to_dtype(DType::F32)?.flatten_all()?.contiguous()?;
            flat_grads.push(FlatGrad {
                name,
                shape,
                elem_count: tensor.elem_count(),
                tensor,
            });
        }

        let bucket_elem_limit = (self.bucket_bytes / std::mem::size_of::<f32>()).max(1);
        let mut synced = Vec::with_capacity(flat_grads.len());
        let mut start = 0usize;
        while start < flat_grads.len() {
            let mut end = start;
            let mut elems = 0usize;
            while end < flat_grads.len() {
                let next = flat_grads[end].elem_count;
                if end > start && elems + next > bucket_elem_limit {
                    break;
                }
                elems += next;
                end += 1;
            }
            self.sync_bucket(&flat_grads[start..end], &mut synced)?;
            start = end;
        }

        for (name, tensor) in synced {
            grads.insert(name, tensor.detach());
        }
        Ok(())
    }

    fn sync_bucket(&self, bucket: &[FlatGrad], synced: &mut Vec<(String, Tensor)>) -> Result<()> {
        let bucket_tensor = if bucket.len() == 1 {
            bucket[0].tensor.clone()
        } else {
            let refs = bucket.iter().map(|grad| &grad.tensor).collect::<Vec<_>>();
            Tensor::cat(&refs, 0)?.contiguous()?
        };

        let reduced = self.all_reduce_flat(&bucket_tensor)?;
        let averaged = reduced.affine(1.0 / self.world_size as f64, 0.0)?;
        let mut offset = 0usize;
        for grad in bucket {
            let slice = averaged.narrow(0, offset, grad.elem_count)?;
            let restored = slice.reshape(grad.shape.as_slice())?;
            synced.push((grad.name.clone(), restored.detach()));
            offset += grad.elem_count;
        }
        Ok(())
    }

    fn all_reduce_flat(&self, tensor: &Tensor) -> Result<Tensor> {
        use candle_core::cuda::cudarc::nccl::safe::ReduceOp;
        use candle_core::op::BackpropOp;
        use candle_core::{CudaStorage, Storage};

        let tensor = tensor.to_dtype(DType::F32)?.flatten_all()?.contiguous()?;
        let shape = tensor.shape().clone();
        let elem_count = tensor.elem_count();
        let (storage, layout) = tensor.storage_and_layout();
        if !layout.is_contiguous() {
            return Err(AarambhError::Config(
                "NCCL all-reduce requires contiguous gradient buckets".into(),
            ));
        }
        let Storage::Cuda(cuda_storage) = &*storage else {
            return Err(AarambhError::Config(
                "NCCL all-reduce requires CUDA gradient tensors".into(),
            ));
        };
        let send = cuda_storage.as_cuda_slice::<f32>()?;
        let mut recv = cuda_storage
            .device
            .cuda_stream()
            .alloc_zeros::<f32>(elem_count)
            .map_err(|err| {
                AarambhError::Config(format!("failed to allocate NCCL receive bucket: {err:?}"))
            })?;
        self.comm
            .all_reduce(send, &mut recv, &ReduceOp::Sum)
            .map_err(|err| AarambhError::Config(format!("NCCL all-reduce failed: {err:?}")))?;
        let storage = Storage::Cuda(CudaStorage::wrap_cuda_slice(
            recv,
            cuda_storage.device.clone(),
        ));
        Ok(Tensor::from_storage(
            storage,
            shape,
            BackpropOp::none(),
            false,
        ))
    }
}

#[cfg(feature = "cuda")]
struct FlatGrad {
    name: String,
    shape: Vec<usize>,
    elem_count: usize,
    tensor: Tensor,
}

#[cfg(feature = "cuda")]
fn id_to_bytes(id: &candle_core::cuda::cudarc::nccl::safe::Id) -> Vec<u8> {
    id.internal().iter().map(|byte| *byte as u8).collect()
}

#[cfg(feature = "cuda")]
fn bytes_to_id(bytes: &[u8]) -> Result<candle_core::cuda::cudarc::nccl::safe::Id> {
    use candle_core::cuda::cudarc::nccl::safe::Id;
    if bytes.len() != NCCL_ID_BYTES {
        return Err(AarambhError::Config(format!(
            "invalid NCCL id length: {}",
            bytes.len()
        )));
    }
    let mut internal = [0 as std::ffi::c_char; 128];
    for (dst, src) in internal.iter_mut().zip(bytes) {
        *dst = *src as std::ffi::c_char;
    }
    Ok(Id::uninit(internal))
}

/// Average a set of per-rank gradient maps to their elementwise mean.
///
/// This is the reference implementation of the data-parallel all-reduce
/// math (sum then divide by the number of ranks) used by the unit tests
/// to verify gradient correctness across a simulated multi-node topology
/// without needing real NCCL hardware.
#[cfg(test)]
fn average_grad_maps_for_test(ranks: &[GradMap]) -> Result<Vec<GradMap>> {
    if ranks.is_empty() {
        return Ok(Vec::new());
    }
    let mut names = ranks[0].keys().cloned().collect::<Vec<_>>();
    names.sort();
    let mut averaged = GradMap::new();
    for name in names {
        let mut sum = None::<Tensor>;
        for rank in ranks {
            let grad = rank
                .get(&name)
                .ok_or_else(|| AarambhError::Config(format!("missing gradient {name}")))?;
            let grad = grad.to_dtype(DType::F32)?;
            sum = Some(match sum {
                Some(existing) => (existing + grad)?,
                None => grad,
            });
        }
        let mean = sum
            .expect("rank list is non-empty")
            .affine(1.0 / ranks.len() as f64, 0.0)?;
        averaged.insert(name, mean.detach());
    }
    Ok(ranks.iter().map(|_| averaged.clone()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use aarambh_studio_core::{Device as CoreDevice, TokenizerLike};
    use aarambh_studio_data::{DataLoader, DataShard, PlaintextDataset};
    use candle_core::{Device, Tensor};
    use std::collections::HashMap;
    use std::sync::{Arc, Barrier as StdBarrier};
    use std::thread;

    /// Minimal tokenizer mapping single characters to ids, used only to drive
    /// the data loader in unit tests without training a real BPE tokenizer.
    struct DummyTokenizer {
        vocab: HashMap<String, u32>,
    }

    impl TokenizerLike for DummyTokenizer {
        fn encode(&self, text: &str) -> Result<Vec<u32>> {
            Ok(text
                .chars()
                .filter_map(|c| self.vocab.get(&c.to_string()).copied())
                .collect())
        }

        fn decode(&self, ids: &[u32]) -> Result<String> {
            let rev: HashMap<u32, String> =
                self.vocab.iter().map(|(k, v)| (*v, k.clone())).collect();
            Ok(ids
                .iter()
                .filter_map(|id| rev.get(id).map(|s| s.as_str()))
                .collect())
        }

        fn vocab_size(&self) -> usize {
            self.vocab.len()
        }

        fn eos_token_id(&self) -> u32 {
            0
        }

        fn bos_token_id(&self) -> Option<u32> {
            None
        }
    }

    #[test]
    fn config_env_overrides_world_rank_and_local_rank() {
        // This test avoids mutating process-global env because Rust 2024 makes
        // env mutation unsafe. Direct validation still covers the resolved shape.
        let config = DistributedConfig {
            enabled: true,
            world_size: 2,
            rank: 1,
            local_rank: 1,
            ..DistributedConfig::default()
        };
        config.validate().unwrap();
    }

    #[test]
    fn gradient_average_matches_two_rank_mean() {
        let device = Device::Cpu;
        let mut rank0 = GradMap::new();
        rank0.insert(
            "w".into(),
            Tensor::from_vec(vec![1f32, 3f32], (2,), &device).unwrap(),
        );
        let mut rank1 = GradMap::new();
        rank1.insert(
            "w".into(),
            Tensor::from_vec(vec![5f32, 7f32], (2,), &device).unwrap(),
        );
        let averaged = average_grad_maps_for_test(&[rank0, rank1]).unwrap();
        for rank in averaged {
            let values = rank.get("w").unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(values, vec![3.0, 5.0]);
        }
    }

    #[test]
    fn invalid_rank_is_rejected() {
        let config = DistributedConfig {
            enabled: true,
            world_size: 2,
            rank: 2,
            ..DistributedConfig::default()
        };
        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains("less than world_size"), "{err}");
    }

    // ---- Phase 44 tests (CPU, no cuda) ----

    #[test]
    fn world_size_one_node_reproduces_v2_single_node_behaviour_exactly() {
        // A single-node config (num_nodes defaults to 1) resolves with the
        // exact same world_size/rank/local_rank as v2 — no topology derived,
        // no multi-node fields touched.
        let mut config = DistributedConfig {
            enabled: true,
            world_size: 2,
            rank: 0,
            local_rank: 0,
            ..DistributedConfig::default()
        };
        resolve_multi_node_topology(&mut config).unwrap();
        assert_eq!(config.num_nodes, 1);
        assert_eq!(config.world_size, 2);
        assert_eq!(config.rank, 0);
        // Single-node: a worker needs the full world_size locally (v2).
        let resolved = ResolvedDistributedConfig {
            backend: DistributedBackend::Nccl,
            world_size: 2,
            rank: 0,
            local_rank: 0,
            run_id: "x".into(),
            rendezvous_dir: PathBuf::new(),
            init_timeout_secs: 120,
            bucket_bytes: DEFAULT_BUCKET_BYTES,
            topology: None,
            rendezvous: RendezvousTransport::File,
            retry_attempts: 1,
        };
        assert_eq!(resolved.local_device_requirement(), 2);
        let runtime = resolve_runtime(Some(&DistributedConfig {
            enabled: true,
            world_size: 1,
            rank: 0,
            local_rank: 0,
            ..DistributedConfig::default()
        }))
        .unwrap();
        assert!(matches!(runtime, DistributedRuntime::Disabled));
    }

    #[test]
    fn gradient_all_reduce_correctness_across_simulated_multi_node_topology() {
        // Simulate a 2-node x 2-GPU topology (world_size = 4). Each rank
        // contributes a different gradient; the data-parallel all-reduce
        // math must produce the elementwise mean across all four ranks,
        // regardless of which node each rank lives on.
        let device = Device::Cpu;
        let mut ranks = Vec::new();
        for value in [1.0f32, 2.0, 3.0, 4.0] {
            let mut map = GradMap::new();
            map.insert(
                "w".into(),
                Tensor::from_vec(vec![value, value + 1.0], (2,), &device).unwrap(),
            );
            ranks.push(map);
        }
        let averaged = average_grad_maps_for_test(&ranks).unwrap();
        let values = averaged[0].get("w").unwrap().to_vec1::<f32>().unwrap();
        // mean of [1,2],[2,3],[3,4],[4,5] = [2.5, 3.5]
        assert!((values[0] - 2.5).abs() < 1e-6, "{values:?}");
        assert!((values[1] - 3.5).abs() < 1e-6, "{values:?}");
        // every rank receives the same averaged copy
        for rank in &averaged[1..] {
            let v = rank.get("w").unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(v, values);
        }
    }

    #[test]
    fn rank_zero_checkpoint_writes_from_exactly_one_process_globally() {
        // A 2-node x 2-GPU topology (world_size=4): only (node_rank=0,
        // local_rank=0) is global rank 0 and therefore the sole rank that
        // logs and checkpoints. Every node's own local_rank=0 must NOT be
        // rank 0 — that was the v2 multi-node duplicate-checkpoint bug this
        // phase fixes.
        let mut rank0_count = 0;
        for node_rank in 0..2 {
            for local_rank in 0..2 {
                let topology = MultiNodeTopology::new(2, 2, node_rank, local_rank);
                let resolved = ResolvedDistributedConfig {
                    backend: DistributedBackend::Nccl,
                    world_size: topology.global_world_size(),
                    rank: topology.global_rank(),
                    local_rank,
                    run_id: "x".into(),
                    rendezvous_dir: PathBuf::new(),
                    init_timeout_secs: 120,
                    bucket_bytes: DEFAULT_BUCKET_BYTES,
                    topology: Some(topology),
                    rendezvous: RendezvousTransport::Tcp {
                        endpoint: "127.0.0.1:0".into(),
                    },
                    retry_attempts: 1,
                };
                assert_eq!(resolved.is_rank0(), topology.is_global_rank0());
                if resolved.is_rank0() {
                    rank0_count += 1;
                }
            }
        }
        assert_eq!(rank0_count, 1, "exactly one global rank 0 across the world");
    }

    #[test]
    fn transient_nccl_timeout_triggers_single_retry_then_fails_loudly() {
        // A policy with max_retries=1 retries once on a transient (timeout)
        // error, then fails loudly on the second consecutive transient
        // error. A non-transient error fails immediately with no retry.
        let mut attempts = 0usize;
        let policy = RetryPolicy::with_retries(1);
        let err = policy
            .run(|| {
                attempts += 1;
                Err::<(), _>(AarambhError::Config(
                    "NCCL rendezvous timed out waiting for peers".into(),
                ))
            })
            .unwrap_err();
        assert_eq!(attempts, 2, "one retry after the first attempt");
        assert!(err.to_string().contains("timed out"), "{}", err);

        // Non-transient error: no retry.
        let mut attempts = 0usize;
        let err = policy
            .run(|| {
                attempts += 1;
                Err::<(), _>(AarambhError::Shape("mismatch".into()))
            })
            .unwrap_err();
        assert_eq!(attempts, 1, "non-transient errors are not retried");
        assert!(err.to_string().contains("mismatch"));

        // Transient-then-success: one retry then success.
        let mut attempts = 0usize;
        let value: u32 = policy
            .run(|| {
                attempts += 1;
                if attempts == 1 {
                    Err(AarambhError::Config("connection refused".into()))
                } else {
                    Ok(7)
                }
            })
            .unwrap();
        assert_eq!(attempts, 2);
        assert_eq!(value, 7);
    }

    #[test]
    fn multi_node_topology_derives_global_rank_and_world_size() {
        let topo = MultiNodeTopology::new(2, 2, 0, 0);
        assert_eq!(topo.global_world_size(), 4);
        assert_eq!(topo.global_rank(), 0);
        assert!(topo.is_global_rank0());
        assert!(topo.is_multi_node());

        let topo = MultiNodeTopology::new(2, 2, 1, 1);
        assert_eq!(topo.global_world_size(), 4);
        assert_eq!(topo.global_rank(), 3);
        assert!(!topo.is_global_rank0());

        let topo = MultiNodeTopology::new(3, 4, 2, 3);
        assert_eq!(topo.global_world_size(), 12);
        assert_eq!(topo.global_rank(), 11);

        // A single-node topology (num_nodes=1) is not multi-node.
        let topo = MultiNodeTopology::new(1, 4, 0, 2);
        assert_eq!(topo.global_world_size(), 4);
        assert_eq!(topo.global_rank(), 2);
        assert!(!topo.is_multi_node());
    }

    #[test]
    fn invalid_multi_node_topology_rejected() {
        let topo = MultiNodeTopology::new(2, 0, 0, 0);
        assert!(topo.validate().is_err());
        let topo = MultiNodeTopology::new(2, 2, 2, 0);
        assert!(topo.validate().is_err());
        let topo = MultiNodeTopology::new(2, 2, 0, 2);
        assert!(topo.validate().is_err());
        let topo = MultiNodeTopology::new(0, 2, 0, 0);
        assert!(topo.validate().is_err());
    }

    #[test]
    fn multi_node_config_requires_gpus_per_node_devices_not_world_size() {
        // A multi-node run only needs gpus_per_node devices locally, not
        // the full global world_size (the rest live on other nodes). This
        // is the Phase 44 fix to the v2 device-count check.
        let config = ResolvedDistributedConfig {
            backend: DistributedBackend::Nccl,
            world_size: 4,
            rank: 0,
            local_rank: 0,
            run_id: "x".into(),
            rendezvous_dir: PathBuf::new(),
            init_timeout_secs: 120,
            bucket_bytes: DEFAULT_BUCKET_BYTES,
            topology: Some(MultiNodeTopology::new(2, 2, 0, 0)),
            rendezvous: RendezvousTransport::Tcp {
                endpoint: "127.0.0.1:0".into(),
            },
            retry_attempts: 1,
        };
        assert_eq!(config.local_device_requirement(), 2);

        // Single-node: needs the full world_size locally.
        let config = ResolvedDistributedConfig {
            backend: DistributedBackend::Nccl,
            world_size: 2,
            rank: 0,
            local_rank: 0,
            run_id: "x".into(),
            rendezvous_dir: PathBuf::new(),
            init_timeout_secs: 120,
            bucket_bytes: DEFAULT_BUCKET_BYTES,
            topology: None,
            rendezvous: RendezvousTransport::File,
            retry_attempts: 1,
        };
        assert_eq!(config.local_device_requirement(), 2);
    }

    #[test]
    fn file_rendezvous_round_trips_id_bytes() {
        let dir = std::env::temp_dir().join(format!(
            "aarambh_file_rendezvous_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let rendezvous = FileRendezvous::new(&dir, "test-run", Duration::from_secs(5));
        let id: Vec<u8> = (0..NCCL_ID_BYTES as u8).collect();
        rendezvous.broadcast(&id).unwrap();
        let received = rendezvous.receive().unwrap();
        assert_eq!(received, id);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn file_rendezvous_receive_times_out_when_rank0_never_publishes() {
        let dir = std::env::temp_dir().join(format!(
            "aarambh_file_rendezvous_timeout_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let rendezvous = FileRendezvous::new(&dir, "missing-run", Duration::from_millis(200));
        let err = rendezvous.receive().unwrap_err();
        assert!(err.to_string().contains("timed out"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tcp_rendezvous_broadcasts_id_bytes_across_loopback() {
        // Rank 0 binds an ephemeral loopback port, broadcasts 128 id bytes,
        // three other ranks connect and each receives an identical copy.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let endpoint = addr.to_string();
        drop(listener);

        let id: Vec<u8> = (0..NCCL_ID_BYTES as u8)
            .map(|b| b.wrapping_mul(3).wrapping_add(7))
            .collect();
        let id = Arc::new(id);
        let world_size = 4;
        let barrier = Arc::new(StdBarrier::new(world_size));

        let mut handles = Vec::new();
        for rank in 0..world_size {
            let endpoint = endpoint.clone();
            let id = Arc::clone(&id);
            let barrier = Arc::clone(&barrier);
            handles.push(thread::spawn(move || -> Result<Vec<u8>> {
                let rendezvous =
                    TcpRendezvous::new(endpoint, world_size, rank, Duration::from_secs(5));
                barrier.wait();
                if rank == 0 {
                    rendezvous.broadcast(&id)?;
                    Ok((*id).clone())
                } else {
                    rendezvous.receive()
                }
            }));
        }
        let results: Vec<Vec<u8>> = handles
            .into_iter()
            .map(|h| h.join().unwrap().unwrap())
            .collect();
        for received in &results {
            assert_eq!(received.len(), NCCL_ID_BYTES);
            assert_eq!(received, &*id);
        }
    }

    #[test]
    fn sharded_data_loader_partitions_across_global_world_size_not_local_gpus() {
        // A 2-node x 2-GPU topology (global world_size=4) must shard the
        // dataset across all 4 global ranks, not just the 2 local GPUs of
        // any single node. Each of the 4 global-rank loaders gets an equal
        // non-empty slice, and a single-node (count=2) loader gets a larger
        // slice — proving the global world_size, not the local GPU count,
        // drives the partition. Disjointness itself is verified by the data
        // crate's own `sharded_dataloader_produces_equal_disjoint_batches`.
        let tokenizer = DummyTokenizer {
            vocab: HashMap::from([
                ("a".into(), 0),
                ("b".into(), 1),
                ("c".into(), 2),
                ("d".into(), 3),
            ]),
        };
        let lines: Vec<String> = std::iter::repeat_n("abcd".to_string(), 16).collect();
        let dataset = PlaintextDataset::from_lines(lines);
        let device = CoreDevice::Cpu;

        let mut global_shards = Vec::new();
        for global_rank in 0..4 {
            global_shards.push(DataLoader::new_sharded(
                &dataset,
                &tokenizer,
                1,
                4,
                false,
                device.clone(),
                DataShard {
                    rank: global_rank,
                    count: 4,
                    seed: 0,
                },
            ));
        }
        let per_global_rank = global_shards[0].len();
        assert!(per_global_rank > 0, "each global rank receives data");
        for shard in &global_shards {
            assert_eq!(shard.len(), per_global_rank, "global ranks split evenly");
        }

        // A single-node run (count=2) would give each rank a larger slice.
        let local_shard = DataLoader::new_sharded(
            &dataset,
            &tokenizer,
            1,
            4,
            false,
            device,
            DataShard {
                rank: 0,
                count: 2,
                seed: 0,
            },
        );
        assert!(
            local_shard.len() > per_global_rank,
            "count=2 ({} batches) > count=4 ({} batches): global world_size drives the partition",
            local_shard.len(),
            per_global_rank
        );
    }

    #[test]
    fn multi_node_topology_validate_requires_tcp_endpoint_when_configured() {
        let mut config = DistributedConfig {
            enabled: true,
            num_nodes: 2,
            gpus_per_node: 2,
            node_rank: 0,
            local_rank: 0,
            rendezvous: RendezvousTransport::Tcp {
                endpoint: "   ".into(),
            },
            ..DistributedConfig::default()
        };
        resolve_multi_node_topology(&mut config).unwrap();
        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains("endpoint"), "{err}");
    }
}
