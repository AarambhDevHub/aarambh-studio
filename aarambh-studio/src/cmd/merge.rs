//! The `merge` subcommand — Phase 50 model merging / weight averaging.
//!
//! `aarambh-studio merge slerp --inputs a.safetensors,b.safetensors
//! --weights 0.5,0.5 --output merged.safetensors` combines two or more
//! architecturally-compatible checkpoints into one SafeTensors file using one
//! of five standard algorithms. See `docs/phase50_model_merging.md` for the
//! runbook and `ARCHITECTURE_V4.md` §64 for the design.
//!
//! This is distinct from `aarambh-studio finetune merge`, which folds a single
//! LoRA/DoRA *adapter* back into its base checkpoint. The top-level `merge`
//! command operates on full models (or full-model task vectors) only.

use std::path::PathBuf;

use aarambh_studio_core::ModelConfig;
use aarambh_studio_weights::{MergeConfig, MergeMethod, merge_models_from_paths};
use clap::{Args, Subcommand};

/// Merge compatible model checkpoints (Phase 50).
#[derive(Debug, Args)]
pub struct MergeArgs {
    #[command(subcommand)]
    pub command: MergeCommand,
}

/// Subcommands of `aarambh-studio merge`, one per merge algorithm.
#[derive(Debug, Subcommand)]
pub enum MergeCommand {
    /// Weighted linear averaging of N checkpoints (Model Soups).
    Linear(LinearArgs),
    /// Spherical linear interpolation between checkpoints.
    Slerp(SlerpArgs),
    /// Task-vector arithmetic: `out = base + Σ scaleᵢ·(Mᵢ − base)`.
    TaskArithmetic(TaskArithmeticArgs),
    /// TIES-Merging: trim, elect sign, disjoint merge of task vectors.
    Ties(TiesArgs),
    /// DARE: drop-and-rescale task vectors before linear combination.
    Dare(DareArgs),
}

/// Arguments shared by every interpolation-family method (linear, slerp).
#[derive(Debug, Args)]
pub struct InterpolationArgs {
    /// Comma-separated list of two or more input SafeTensors checkpoints to
    /// interpolate. All inputs must share an identical tensor-name set and
    /// per-tensor shape/dtype.
    #[arg(long, value_delimiter = ',')]
    pub inputs: Vec<PathBuf>,
    /// Comma-separated per-input weights. Need not sum to one — they are
    /// normalized internally. Must be non-negative and not all zero.
    #[arg(long, value_delimiter = ',')]
    pub weights: Vec<f64>,
    /// Output SafeTensors path. Parent directories are created as needed.
    #[arg(long)]
    pub output: PathBuf,
}

/// Arguments shared by every task-vector-family method
/// (task-arithmetic, ties, dare).
#[derive(Debug, Args)]
pub struct TaskVectorArgs {
    /// Shared base SafeTensors checkpoint that every delta is computed against.
    #[arg(long)]
    pub base: PathBuf,
    /// Comma-separated list of one or more independently-tuned SafeTensors
    /// checkpoints whose task vector is `delta = M − base`.
    #[arg(long, value_delimiter = ',')]
    pub deltas: Vec<PathBuf>,
    /// Comma-separated per-delta scaling factors. Length must match `--deltas`.
    #[arg(long, value_delimiter = ',')]
    pub scales: Vec<f64>,
    /// Output SafeTensors path. Parent directories are created as needed.
    #[arg(long)]
    pub output: PathBuf,
}

/// `aarambh-studio merge linear` — Model Soups weighted averaging.
#[derive(Debug, Args)]
pub struct LinearArgs {
    #[command(flatten)]
    pub common: InterpolationArgs,
}

/// `aarambh-studio merge slerp` — spherical linear interpolation.
#[derive(Debug, Args)]
pub struct SlerpArgs {
    #[command(flatten)]
    pub common: InterpolationArgs,
}

/// `aarambh-studio merge task-arithmetic` — combine task vectors onto a base.
#[derive(Debug, Args)]
pub struct TaskArithmeticArgs {
    #[command(flatten)]
    pub common: TaskVectorArgs,
}

/// `aarambh-studio merge ties` — TIES-Merging of task vectors.
#[derive(Debug, Args)]
pub struct TiesArgs {
    #[command(flatten)]
    pub common: TaskVectorArgs,
    /// Fraction of each delta retained by magnitude trimming, in `(0.0, 1.0]`.
    /// `0.5` keeps the largest-magnitude half. Default `0.5`.
    #[arg(long, default_value_t = aarambh_studio_weights::DEFAULT_DENSITY)]
    pub density: f64,
    /// Rescale surviving entries by `1/density` to preserve expected magnitude.
    /// Default `true` (recommended).
    #[arg(long, default_value_t = true)]
    pub normalize: bool,
}

/// `aarambh-studio merge dare` — DARE drop-and-rescale of task vectors.
#[derive(Debug, Args)]
pub struct DareArgs {
    #[command(flatten)]
    pub common: TaskVectorArgs,
    /// Keep probability for each parameter, in `(0.0, 1.0]`. `0.5` drops half.
    /// Default `0.5`.
    #[arg(long, default_value_t = aarambh_studio_weights::DEFAULT_DENSITY)]
    pub density: f64,
    /// Fixed seed for the deterministic drop mask (DARE is fully reproducible
    /// and never touches system randomness). Default `0`.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
}

impl MergeArgs {
    /// Dispatch to the selected merge algorithm.
    pub fn run(self) -> anyhow::Result<()> {
        match self.command {
            MergeCommand::Linear(args) => run_interpolation(args.common, MergeMethod::Linear),
            MergeCommand::Slerp(args) => run_interpolation(args.common, MergeMethod::Slerp),
            MergeCommand::TaskArithmetic(args) => {
                run_task_vector(args.common, MergeMethod::TaskArithmetic, None, true, 0)
            }
            MergeCommand::Ties(args) => run_task_vector(
                args.common,
                MergeMethod::Ties,
                Some(args.density),
                args.normalize,
                0,
            ),
            MergeCommand::Dare(args) => run_task_vector(
                args.common,
                MergeMethod::Dare,
                Some(args.density),
                true,
                args.seed,
            ),
        }
    }
}

/// Convenience entry point matching the `run(args)` convention of the other
/// command modules.
pub fn run(args: MergeArgs) -> anyhow::Result<()> {
    args.run()
}

/// Run an interpolation-family merge (linear or slerp).
fn run_interpolation(args: InterpolationArgs, method: MergeMethod) -> anyhow::Result<()> {
    eprintln!(
        "[merge] method={:?} inputs={} weights={:?} output={}",
        method,
        args.inputs.len(),
        args.weights,
        args.output.display()
    );
    let config = MergeConfig {
        method,
        weights: args.weights,
        scales: Vec::new(),
        density: aarambh_studio_weights::DEFAULT_DENSITY,
        normalize: true,
        seed: 0,
    };
    let model_config = ModelConfig::tiny();
    let report = merge_models_from_paths(
        &model_config,
        &args.inputs,
        None,
        &[],
        &args.output,
        &config,
    )?;
    eprintln!(
        "[merge] wrote {} tensors from {} inputs → {} (slerp_linear_fallbacks={})",
        report.tensor_count,
        report.input_count,
        report.output_path.display(),
        report.slerp_linear_fallback_count
    );
    Ok(())
}

/// Run a task-vector-family merge (task-arithmetic, ties, or dare).
#[allow(clippy::too_many_arguments)]
fn run_task_vector(
    args: TaskVectorArgs,
    method: MergeMethod,
    density: Option<f64>,
    normalize: bool,
    seed: u64,
) -> anyhow::Result<()> {
    eprintln!(
        "[merge] method={:?} base={} deltas={} scales={:?} output={}",
        method,
        args.base.display(),
        args.deltas.len(),
        args.scales,
        args.output.display()
    );
    let config = MergeConfig {
        method,
        weights: Vec::new(),
        scales: args.scales,
        density: density.unwrap_or(aarambh_studio_weights::DEFAULT_DENSITY),
        normalize,
        seed,
    };
    let model_config = ModelConfig::tiny();
    let report = merge_models_from_paths(
        &model_config,
        &[],
        Some(&args.base),
        &args.deltas,
        &args.output,
        &config,
    )?;
    eprintln!(
        "[merge] wrote {} tensors from {} deltas onto base → {} (ties_resolved={} dare_dropped_fraction={:.4})",
        report.tensor_count,
        report.input_count,
        report.output_path.display(),
        report.ties_resolved_tensors,
        report.dare_dropped_fraction
    );
    Ok(())
}
