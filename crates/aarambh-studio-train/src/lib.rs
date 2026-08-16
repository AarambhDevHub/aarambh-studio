//! Training configuration, trainer loop, optimizer, schedules, checkpoints, and loss helpers.
#![deny(missing_docs)]

/// Checkpoint save/load helpers.
pub mod checkpoint;
/// TOML-backed training run configuration.
pub mod config;
/// Single- and multi-node data-parallel training helpers.
pub mod distributed;
/// Language-model loss functions.
pub mod loss;
/// Multi-token prediction loss alignment and aggregation.
pub mod mtp_loss;
/// Read-only live training observer API.
pub mod observer;
/// Optimizer and gradient utilities.
pub mod optim;
/// Learning-rate schedules.
pub mod schedule;
/// Main training loop.
pub mod trainer;
/// Vision-projector-only pretraining loop.
pub mod vision_projector;

pub use checkpoint::{CheckpointManager, TrainState};
pub use config::{
    DsaTrainingConfig, ForgettingTrainingConfig, MoeRetrofitConfig, TrainingObserverFactory,
    TrainingRunConfig, run_training_from_config, run_training_from_config_with_observer,
};
pub use distributed::{
    DistributedBackend, DistributedConfig, DistributedContext, DistributedRuntime, FileRendezvous,
    MultiNodeTopology, NCCL_ID_BYTES, Rendezvous, RendezvousTransport, ResolvedDistributedConfig,
    RetryPolicy, TcpRendezvous, build_rendezvous,
};
pub use loss::cross_entropy_loss;
pub use mtp_loss::{MtpHeadLoss, MtpLossOutput, combine_mtp_losses, mtp_head_loss};
pub use observer::{TrainingObserver, TrainingObserverEvent, TrainingObserverSnapshot};
pub use optim::{AdamW, AdamWConfig, GradMap, TrainableParameter};
pub use schedule::CosineScheduleWithWarmup;
pub use trainer::{MtpHeadMetric, Trainer, TrainingMetrics};
pub use vision_projector::{
    AudioTrainingConfig, DocumentTrainingConfig, VideoTrainingConfig, VisionTrainingConfig,
    run_projector_pretrain,
};
