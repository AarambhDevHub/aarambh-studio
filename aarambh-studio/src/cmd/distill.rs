use std::path::PathBuf;
use std::str::FromStr;

use aarambh_studio_distill::{
    DistillConfig, DistillEvalConfig, DistillObjective, DistillRunConfig, DistillThinkingMode,
    OfflinePrepareConfig, OfflineRunConfig, TeacherSourceConfig, evaluate_distillation,
    prepare_offline_dataset, run_distill_from_config, run_offline_distill_from_config,
};
use aarambh_studio_train::TrainingRunConfig;
use clap::{Args, Subcommand};

#[derive(Debug, Args)]
/// Distil a smaller student model from a frozen local or dataset teacher.
pub struct DistillArgs {
    #[command(subcommand)]
    pub command: DistillCommand,
}

#[derive(Debug, Subcommand)]
pub enum DistillCommand {
    /// Train a full student on fresh rollouts scored by a frozen teacher.
    Train(TrainArgs),
    /// Generate one static local-teacher completion per prompt.
    PrepareOffline(PrepareOfflineArgs),
    /// Train the matched static-completion offline control.
    TrainOffline(TrainOfflineArgs),
    /// Score fresh student rollouts and write alignment reports.
    Evaluate(EvaluateArgs),
}

#[derive(Debug, Args)]
/// `distill train` — train a full student on fresh rollouts scored by a frozen teacher.
pub struct TrainArgs {
    /// Student distillation TOML configuration (architecture + train settings).
    #[arg(long, default_value = "configs/distill_smoke.toml")]
    pub config: PathBuf,
    /// Student checkpoint path to fine-tune.
    #[arg(long)]
    pub student: PathBuf,
    /// Optional tokenizer JSON path; falls back to the config.
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,
    /// Prompts JSONL path; defaults to the config dataset path.
    #[arg(long)]
    pub prompts: Option<PathBuf>,
    /// Output adapter directory; defaults to the config checkpoint dir.
    #[arg(long)]
    pub output: Option<PathBuf>,
    /// Teacher backend: local (frozen checkpoint) or dataset (logged completions).
    #[arg(long, default_value = "local")]
    pub teacher: String,
    /// Frozen teacher checkpoint path (required for --teacher local).
    #[arg(long)]
    pub teacher_model: Option<PathBuf>,
    /// Frozen teacher TOML config path (required for --teacher local).
    #[arg(long)]
    pub teacher_config: Option<PathBuf>,
    /// Teacher completions JSONL path (required for --teacher dataset).
    #[arg(long)]
    pub teacher_data: Option<PathBuf>,
    /// Override the teacher device (e.g. cpu, cuda:0, metal).
    #[arg(long)]
    pub teacher_device: Option<String>,
    /// Override the teacher dtype (e.g. f16, bf16, f32).
    #[arg(long)]
    pub teacher_dtype: Option<String>,
    /// Distillation objective: soft-kl (forward KL) or reward (GRPO-style).
    #[arg(long)]
    pub objective: Option<String>,
    /// Number of student rollouts sampled per prompt.
    #[arg(long, default_value_t = 4)]
    pub rollouts_per_prompt: usize,
    /// Maximum new tokens generated per student rollout.
    #[arg(long, default_value_t = 128)]
    pub max_new_tokens: usize,
    /// Sampling temperature for student rollouts.
    #[arg(long, default_value_t = 0.8)]
    pub temperature: f32,
    /// Nucleus sampling probability mass for student rollouts.
    #[arg(long, default_value_t = 0.95)]
    pub top_p: f32,
    /// Top-k sampling width for student rollouts.
    #[arg(long, default_value_t = 50)]
    pub top_k: usize,
    /// Sampling temperature applied to the teacher distribution.
    #[arg(long, default_value_t = 1.0)]
    pub teacher_temperature: f64,
    /// Maximum absolute advantage used to clip reward-weighted rollouts.
    #[arg(long, default_value_t = 5.0)]
    pub advantage_clip: f64,
    /// Thinking budget: none, low, medium, high, or max.
    #[arg(long, default_value = "none")]
    pub thinking: String,
    /// Training batch size.
    #[arg(long)]
    pub batch_size: Option<usize>,
    /// Maximum optimiser steps (overrides max-epochs when set).
    #[arg(long)]
    pub max_steps: Option<usize>,
    /// Maximum training epochs.
    #[arg(long)]
    pub max_epochs: Option<usize>,
    /// Learning rate.
    #[arg(long)]
    pub lr: Option<f64>,
    /// Gradient accumulation steps before an optimiser update.
    #[arg(long)]
    pub grad_accum_steps: Option<usize>,
    /// Linear warmup steps before the cosine schedule.
    #[arg(long)]
    pub warmup_steps: Option<usize>,
    /// Save an adapter every N optimiser steps.
    #[arg(long)]
    pub save_every_n_steps: Option<usize>,
    /// Log training metrics every N optimiser steps.
    #[arg(long)]
    pub log_every_n_steps: Option<usize>,
    /// Resume training from the latest adapter in the output directory.
    #[arg(long)]
    pub resume: bool,
    /// Disable dataset shuffling between epochs.
    #[arg(long)]
    pub no_shuffle: bool,
}

#[derive(Debug, Args)]
/// `distill prepare-offline` — generate one static local-teacher completion per prompt.
pub struct PrepareOfflineArgs {
    /// Frozen teacher TOML config path.
    #[arg(long)]
    pub teacher_config: PathBuf,
    /// Frozen teacher checkpoint path.
    #[arg(long)]
    pub teacher_model: PathBuf,
    /// Optional tokenizer JSON path; falls back to the teacher config.
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,
    /// Input prompts JSONL path.
    #[arg(long)]
    pub prompts: PathBuf,
    /// Output JSONL path with one static teacher completion per prompt.
    #[arg(long)]
    pub output: PathBuf,
    /// Override the device used to generate completions.
    #[arg(long)]
    pub device: Option<String>,
    /// Override the dtype used to generate completions.
    #[arg(long)]
    pub dtype: Option<String>,
    /// Maximum new tokens generated per prompt.
    #[arg(long, default_value_t = 128)]
    pub max_new_tokens: usize,
    /// Sampling temperature for completion generation.
    #[arg(long, default_value_t = 0.8)]
    pub temperature: f32,
    /// Nucleus sampling probability mass for completion generation.
    #[arg(long, default_value_t = 0.95)]
    pub top_p: f32,
    /// Top-k sampling width for completion generation.
    #[arg(long, default_value_t = 50)]
    pub top_k: usize,
    /// RNG seed for deterministic completion sampling.
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
}

#[derive(Debug, Args)]
/// `distill train-offline` — train the matched static-completion offline control.
pub struct TrainOfflineArgs {
    /// Student distillation TOML configuration (architecture + train settings).
    #[arg(long, default_value = "configs/distill_smoke.toml")]
    pub config: PathBuf,
    /// Student checkpoint path to fine-tune.
    #[arg(long)]
    pub student: PathBuf,
    /// Optional tokenizer JSON path; falls back to the config.
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,
    /// Static-completion JSONL produced by `distill prepare-offline`.
    #[arg(long)]
    pub data: PathBuf,
    /// Output adapter directory; defaults to the config checkpoint dir.
    #[arg(long)]
    pub output: Option<PathBuf>,
    /// Training batch size.
    #[arg(long)]
    pub batch_size: Option<usize>,
    /// Maximum optimiser steps (overrides max-epochs when set).
    #[arg(long)]
    pub max_steps: Option<usize>,
    /// Maximum training epochs.
    #[arg(long)]
    pub max_epochs: Option<usize>,
    /// Learning rate.
    #[arg(long)]
    pub lr: Option<f64>,
    /// Gradient accumulation steps before an optimiser update.
    #[arg(long)]
    pub grad_accum_steps: Option<usize>,
    /// Save an adapter every N optimiser steps.
    #[arg(long)]
    pub save_every_n_steps: Option<usize>,
    /// Log training metrics every N optimiser steps.
    #[arg(long)]
    pub log_every_n_steps: Option<usize>,
    /// Resume training from the latest adapter in the output directory.
    #[arg(long)]
    pub resume: bool,
    /// Disable dataset shuffling between epochs.
    #[arg(long)]
    pub no_shuffle: bool,
}

#[derive(Debug, Args)]
/// `distill evaluate` — score fresh student rollouts and write alignment reports.
pub struct EvaluateArgs {
    /// Student distillation TOML configuration (architecture + train settings).
    #[arg(long, default_value = "configs/distill_smoke.toml")]
    pub config: PathBuf,
    /// Student checkpoint path to evaluate.
    #[arg(long)]
    pub student: PathBuf,
    /// Optional tokenizer JSON path; falls back to the config.
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,
    /// Prompts JSONL path; defaults to the config dataset path.
    #[arg(long)]
    pub prompts: Option<PathBuf>,
    /// Teacher backend: local (frozen checkpoint) or dataset (logged completions).
    #[arg(long, default_value = "local")]
    pub teacher: String,
    /// Frozen teacher checkpoint path (required for --teacher local).
    #[arg(long)]
    pub teacher_model: Option<PathBuf>,
    /// Frozen teacher TOML config path (required for --teacher local).
    #[arg(long)]
    pub teacher_config: Option<PathBuf>,
    /// Teacher completions JSONL path (required for --teacher dataset).
    #[arg(long)]
    pub teacher_data: Option<PathBuf>,
    /// Override the teacher device (e.g. cpu, cuda:0, metal).
    #[arg(long)]
    pub teacher_device: Option<String>,
    /// Override the teacher dtype (e.g. f16, bf16, f32).
    #[arg(long)]
    pub teacher_dtype: Option<String>,
    /// Distillation objective: soft-kl (forward KL) or reward (GRPO-style).
    #[arg(long)]
    pub objective: Option<String>,
    /// Number of student rollouts sampled per prompt.
    #[arg(long, default_value_t = 2)]
    pub rollouts_per_prompt: usize,
    /// Maximum new tokens generated per student rollout.
    #[arg(long, default_value_t = 128)]
    pub max_new_tokens: usize,
    /// Sampling temperature for student rollouts.
    #[arg(long, default_value_t = 0.8)]
    pub temperature: f32,
    /// Nucleus sampling probability mass for student rollouts.
    #[arg(long, default_value_t = 0.95)]
    pub top_p: f32,
    /// Top-k sampling width for student rollouts.
    #[arg(long, default_value_t = 50)]
    pub top_k: usize,
    /// Sampling temperature applied to the teacher distribution.
    #[arg(long, default_value_t = 1.0)]
    pub teacher_temperature: f64,
    /// Optional cap on the number of prompts evaluated.
    #[arg(long)]
    pub max_prompts: Option<usize>,
    /// Optional JSON output path for the alignment report.
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Optional Markdown output path for the alignment report.
    #[arg(long)]
    pub markdown: Option<PathBuf>,
    /// RNG seed for deterministic rollout sampling.
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
}

pub fn run(args: DistillArgs) -> anyhow::Result<()> {
    match args.command {
        DistillCommand::Train(args) => run_train(args),
        DistillCommand::PrepareOffline(args) => run_prepare_offline(args),
        DistillCommand::TrainOffline(args) => run_train_offline(args),
        DistillCommand::Evaluate(args) => run_evaluate(args),
    }
}

fn run_train(args: TrainArgs) -> anyhow::Result<()> {
    let mut run = TrainingRunConfig::from_toml(&args.config)?;
    apply_train_overrides(
        &mut run,
        args.batch_size,
        args.max_steps,
        args.max_epochs,
        args.lr,
        args.grad_accum_steps,
        args.warmup_steps,
        args.save_every_n_steps,
        args.log_every_n_steps,
    );
    let tokenizer = tokenizer_path(args.tokenizer, &run)?;
    let prompts = args.prompts.unwrap_or_else(|| run.dataset_path.clone());
    let output = args
        .output
        .unwrap_or_else(|| run.train.checkpoint_dir.clone());
    let teacher = teacher_source(
        &args.teacher,
        args.teacher_model,
        args.teacher_config,
        args.teacher_data,
        args.teacher_device,
        args.teacher_dtype,
        &run,
    )?;
    let default_objective = if matches!(&teacher, TeacherSourceConfig::Dataset { .. }) {
        DistillObjective::Reward
    } else {
        DistillObjective::SoftKl
    };
    let objective = args
        .objective
        .map(|value| DistillObjective::from_str(&value))
        .transpose()
        .map_err(anyhow::Error::msg)?
        .unwrap_or(default_objective);
    let distill_config = DistillConfig {
        rollouts_per_prompt: args.rollouts_per_prompt,
        max_new_tokens: args.max_new_tokens,
        temperature: args.temperature,
        top_p: probability_option(args.top_p)?,
        top_k: (args.top_k > 0).then_some(args.top_k),
        teacher_temperature: args.teacher_temperature,
        advantage_clip: args.advantage_clip,
        thinking: DistillThinkingMode::from_str(&args.thinking).map_err(anyhow::Error::msg)?,
        objective,
    };
    let device = run.device()?;
    let dtype = run.dtype_for_device(&device)?;
    run_distill_from_config(DistillRunConfig {
        model_config: run.model,
        train_config: run.train,
        dsa_training_config: run.dsa_training,
        distill_config,
        student_model_path: args.student,
        tokenizer_path: tokenizer,
        prompt_path: prompts,
        output_dir: output,
        device,
        dtype,
        teacher,
        shuffle: run.shuffle && !args.no_shuffle,
        resume: run.resume || args.resume,
    })?;
    Ok(())
}

fn run_prepare_offline(args: PrepareOfflineArgs) -> anyhow::Result<()> {
    let mut run = TrainingRunConfig::from_toml(&args.teacher_config)?;
    if let Some(device) = args.device {
        run.device = device;
    }
    if let Some(dtype) = args.dtype {
        run.dtype = dtype;
    }
    let device = run.device()?;
    let dtype = run.dtype_for_device(&device)?;
    let tokenizer = tokenizer_path(args.tokenizer, &run)?;
    prepare_offline_dataset(OfflinePrepareConfig {
        teacher_model_config: run.model,
        teacher_model_path: args.teacher_model,
        tokenizer_path: tokenizer,
        prompt_path: args.prompts,
        output_path: args.output,
        device,
        dtype,
        generation: DistillConfig {
            rollouts_per_prompt: 1,
            max_new_tokens: args.max_new_tokens,
            temperature: args.temperature,
            top_p: probability_option(args.top_p)?,
            top_k: (args.top_k > 0).then_some(args.top_k),
            ..DistillConfig::default()
        },
        seed: args.seed,
    })?;
    Ok(())
}

fn run_train_offline(args: TrainOfflineArgs) -> anyhow::Result<()> {
    let mut run = TrainingRunConfig::from_toml(&args.config)?;
    apply_train_overrides(
        &mut run,
        args.batch_size,
        args.max_steps,
        args.max_epochs,
        args.lr,
        args.grad_accum_steps,
        None,
        args.save_every_n_steps,
        args.log_every_n_steps,
    );
    let tokenizer = tokenizer_path(args.tokenizer, &run)?;
    let output = args
        .output
        .unwrap_or_else(|| run.train.checkpoint_dir.clone());
    let device = run.device()?;
    let dtype = run.dtype_for_device(&device)?;
    run_offline_distill_from_config(OfflineRunConfig {
        model_config: run.model,
        train_config: run.train,
        dsa_training_config: run.dsa_training,
        student_model_path: args.student,
        tokenizer_path: tokenizer,
        data_path: args.data,
        output_dir: output,
        device,
        dtype,
        shuffle: run.shuffle && !args.no_shuffle,
        resume: run.resume || args.resume,
    })?;
    Ok(())
}

fn run_evaluate(args: EvaluateArgs) -> anyhow::Result<()> {
    let run = TrainingRunConfig::from_toml(&args.config)?;
    let tokenizer = tokenizer_path(args.tokenizer, &run)?;
    let prompts = args.prompts.unwrap_or_else(|| run.dataset_path.clone());
    let teacher = teacher_source(
        &args.teacher,
        args.teacher_model,
        args.teacher_config,
        args.teacher_data,
        args.teacher_device,
        args.teacher_dtype,
        &run,
    )?;
    let default_objective = if matches!(&teacher, TeacherSourceConfig::Dataset { .. }) {
        DistillObjective::Reward
    } else {
        DistillObjective::SoftKl
    };
    let objective = args
        .objective
        .map(|value| DistillObjective::from_str(&value))
        .transpose()
        .map_err(anyhow::Error::msg)?
        .unwrap_or(default_objective);
    let device = run.device()?;
    let dtype = run.dtype_for_device(&device)?;
    let report = evaluate_distillation(DistillEvalConfig {
        model_config: run.model,
        student_model_path: args.student,
        tokenizer_path: tokenizer,
        prompt_path: prompts,
        teacher,
        distill_config: DistillConfig {
            rollouts_per_prompt: args.rollouts_per_prompt,
            max_new_tokens: args.max_new_tokens,
            temperature: args.temperature,
            top_p: probability_option(args.top_p)?,
            top_k: (args.top_k > 0).then_some(args.top_k),
            teacher_temperature: args.teacher_temperature,
            objective,
            ..DistillConfig::default()
        },
        device,
        dtype,
        max_prompts: args.max_prompts,
        output_json: args.out,
        output_markdown: args.markdown,
        seed: args.seed,
    })?;
    println!(
        "distill_eval backend={} prompts={} rollouts={} reward={:.6} kl={:.6} tok/s={:.2}",
        report.teacher_backend,
        report.prompts,
        report.rollouts,
        report.reward_mean,
        report.teacher_student_kl.unwrap_or(0.0),
        report.tokens_per_second
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn teacher_source(
    backend: &str,
    teacher_model: Option<PathBuf>,
    teacher_config: Option<PathBuf>,
    teacher_data: Option<PathBuf>,
    teacher_device: Option<String>,
    teacher_dtype: Option<String>,
    student_run: &TrainingRunConfig,
) -> anyhow::Result<TeacherSourceConfig> {
    match backend.trim().to_ascii_lowercase().as_str() {
        "local" => {
            if teacher_data.is_some() {
                anyhow::bail!("--teacher-data cannot be used with --teacher local");
            }
            let model_path = teacher_model
                .ok_or_else(|| anyhow::anyhow!("--teacher-model is required for local teachers"))?;
            let config_path = teacher_config.ok_or_else(|| {
                anyhow::anyhow!("--teacher-config is required for local teachers")
            })?;
            let mut teacher_run = TrainingRunConfig::from_toml(config_path)?;
            teacher_run.device = teacher_device.unwrap_or_else(|| student_run.device.clone());
            teacher_run.dtype = teacher_dtype.unwrap_or_else(|| student_run.dtype.clone());
            let device = teacher_run.device()?;
            let dtype = teacher_run.dtype_for_device(&device)?;
            Ok(TeacherSourceConfig::Local {
                model_path,
                model_config: Box::new(teacher_run.model),
                device,
                dtype,
            })
        }
        "dataset" => {
            if teacher_model.is_some() || teacher_config.is_some() {
                anyhow::bail!(
                    "--teacher-model and --teacher-config cannot be used with --teacher dataset"
                );
            }
            Ok(TeacherSourceConfig::Dataset {
                data_path: teacher_data.ok_or_else(|| {
                    anyhow::anyhow!("--teacher-data is required for dataset teachers")
                })?,
            })
        }
        other => anyhow::bail!("unsupported teacher backend '{other}', expected local or dataset"),
    }
}

fn tokenizer_path(
    override_path: Option<PathBuf>,
    run: &TrainingRunConfig,
) -> anyhow::Result<PathBuf> {
    override_path
        .or_else(|| run.tokenizer_path.clone())
        .ok_or_else(|| anyhow::anyhow!("tokenizer path is required"))
}

#[allow(clippy::too_many_arguments)]
fn apply_train_overrides(
    run: &mut TrainingRunConfig,
    batch_size: Option<usize>,
    max_steps: Option<usize>,
    max_epochs: Option<usize>,
    lr: Option<f64>,
    grad_accum_steps: Option<usize>,
    warmup_steps: Option<usize>,
    save_every_n_steps: Option<usize>,
    log_every_n_steps: Option<usize>,
) {
    if let Some(value) = batch_size {
        run.train.batch_size = value;
    }
    if let Some(value) = max_steps {
        run.train.max_steps = value;
    }
    if let Some(value) = max_epochs {
        run.train.max_epochs = value;
    }
    if let Some(value) = lr {
        run.train.lr = value;
    }
    if let Some(value) = grad_accum_steps {
        run.train.grad_accum_steps = value;
    }
    if let Some(value) = warmup_steps {
        run.train.warmup_steps = value;
    }
    if let Some(value) = save_every_n_steps {
        run.train.save_every_n_steps = value;
    }
    if let Some(value) = log_every_n_steps {
        run.train.log_every_n_steps = value;
    }
}

fn probability_option(value: f32) -> anyhow::Result<Option<f32>> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        anyhow::bail!("--top-p must be finite and in [0, 1]");
    }
    Ok((value > 0.0 && value < 1.0).then_some(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_p_cli_bounds_are_validated() {
        assert_eq!(probability_option(0.0).unwrap(), None);
        assert_eq!(probability_option(1.0).unwrap(), None);
        assert_eq!(probability_option(0.9).unwrap(), Some(0.9));
        assert!(probability_option(-0.1).is_err());
        assert!(probability_option(1.1).is_err());
    }
}
