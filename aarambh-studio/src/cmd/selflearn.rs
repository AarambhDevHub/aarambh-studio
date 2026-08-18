use std::path::PathBuf;

use aarambh_studio_core::TokenizerLike;
use aarambh_studio_selflearn::{
    ReplayBuffer, SelfLearnBuildConfig, SelfLearnConfig, SelfLearnForgettingConfig, SelfLearnLoop,
    SelfLearnMode, VisionCache,
};
use aarambh_studio_tokenizer::BpeTokenizer;
use aarambh_studio_train::TrainingRunConfig;
use clap::{Args, Subcommand};

use crate::cmd::infer::{self, InferArgs};

#[derive(Debug, Args)]
/// Self-learning loop operator: start, flush, replay, stats, or reset.
pub struct SelflearnArgs {
    #[command(subcommand)]
    pub command: SelflearnCommand,
}

#[derive(Debug, Subcommand)]
pub enum SelflearnCommand {
    /// Run a single self-learning inference (text or vision) step.
    Start(Box<StartArgs>),
    /// Flush pending self-learning gradients through the loop optimiser.
    FlushGradients(SelflearnRunArgs),
    /// Run a self-learning replay fine-tune over the replay buffer.
    Replay(SelflearnRunArgs),
    /// Print replay buffer, metrics, and forgetting-curve statistics.
    Stats(StatsArgs),
    /// Print the stored forgetting-curve report and exit.
    ForgettingReport(ForgettingReportArgs),
    /// Delete the replay buffer and self-learning state directory.
    Reset(ResetArgs),
}

#[derive(Debug, Args, Clone)]
/// Shared forgetting-probe arguments for `selflearn` subcommands.
pub struct ForgettingArgs {
    /// Capability probe manifest path (enables forgetting analysis).
    #[arg(long)]
    pub forgetting_manifest: Option<PathBuf>,
    /// Forgetting curves store path (defaults to `<state-dir>/forgetting_curves.json`).
    #[arg(long)]
    pub forgetting_store: Option<PathBuf>,
    /// Optional JSONL export path for forgetting deltas.
    #[arg(long)]
    pub forgetting_jsonl: Option<PathBuf>,
    /// Absolute capability-score delta treated as significant forgetting.
    #[arg(long, default_value_t = 0.02)]
    pub forgetting_threshold: f64,
    /// Maximum examples evaluated per capability probe.
    #[arg(long, default_value_t = 8)]
    pub forgetting_max_examples: usize,
    /// Allow code execution in capability probes.
    #[arg(long)]
    pub forgetting_allow_code_exec: bool,
    /// Require every manifest probe to run; otherwise missing probes are skipped.
    #[arg(long)]
    pub require_all_probes: bool,
    /// Baseline checkpoint or session id used to compute forgetting deltas.
    #[arg(long)]
    pub forgetting_baseline_id: Option<String>,
}

#[derive(Debug, Args)]
/// `selflearn start` — run a single self-learning inference step.
pub struct StartArgs {
    /// Self-learning runtime mode: text or vision.
    #[arg(long, default_value = "text")]
    pub mode: String,
    /// Training/model TOML configuration (provides architecture + device).
    #[arg(long, default_value = "configs/tiny_shakespeare.toml")]
    pub config: PathBuf,
    /// Model checkpoint path; falls back to best.json/latest.json pointer.
    #[arg(long)]
    pub model: Option<PathBuf>,
    /// Optional tokenizer JSON path; falls back to the config.
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,
    /// Image file path for vision self-learning (required when --mode vision).
    #[arg(long)]
    pub image: Option<PathBuf>,
    /// Prompt text fed to the self-learning loop.
    #[arg(long)]
    pub prompt: String,
    /// Maximum number of new tokens generated.
    #[arg(long, default_value_t = 256)]
    pub max_tokens: usize,
    /// Sampling temperature for stochastic decoding.
    #[arg(long, default_value_t = 0.7)]
    pub temperature: f32,
    /// Nucleus sampling probability mass.
    #[arg(long, default_value_t = 0.9)]
    pub top_p: f32,
    /// Top-k sampling width.
    #[arg(long, default_value_t = 50)]
    pub top_k: usize,
    /// Deterministic sampler seed.
    #[arg(long)]
    pub seed: Option<u64>,
    /// Stream tokens to stdout as they are generated.
    #[arg(long)]
    pub stream: bool,
    /// Use greedy (argmax) decoding instead of stochastic sampling.
    #[arg(long)]
    pub greedy: bool,
    /// Thinking budget: none, low, medium, high, or max (Phase 39).
    #[arg(long, default_value = "none")]
    pub thinking: String,
    /// Safety policy: strict, permissive, research, or none.
    #[arg(long, default_value = "strict")]
    pub safety: String,
    /// JSONL safety audit log path.
    #[arg(long, default_value = "safety_audit.jsonl")]
    pub safety_audit_log: PathBuf,
    /// Self-learning replay JSONL path (defaults to data/replay.jsonl).
    #[arg(long)]
    pub replay_path: Option<PathBuf>,
    /// Self-learning state directory for adapters and metrics.
    #[arg(long, default_value = "adapters/selflearn")]
    pub self_learn_state_dir: PathBuf,
    /// Reference (frozen) checkpoint used for KL during self-learning.
    #[arg(long)]
    pub self_learn_reference: Option<PathBuf>,
    /// Built-in verifier kind: none, math, format, or math-format.
    #[arg(long, default_value = "none")]
    pub self_learn_verifier: String,
    /// Grounded vision verifier: none, auto, count, color, presence, or exact.
    #[arg(long, default_value = "auto")]
    pub self_learn_vision_verifier: String,
    /// Ground-truth answer required when a verifier other than none is used.
    #[arg(long)]
    pub self_learn_ground_truth: Option<String>,
    #[command(flatten)]
    pub forgetting: ForgettingArgs,
}

#[derive(Debug, Args)]
/// `selflearn flush-gradients` / `selflearn replay` shared arguments.
pub struct SelflearnRunArgs {
    /// Training/model TOML configuration (provides architecture + device).
    #[arg(long, default_value = "configs/tiny_shakespeare.toml")]
    pub config: PathBuf,
    /// Base (policy) checkpoint path.
    #[arg(long)]
    pub base: PathBuf,
    /// Reference (frozen) checkpoint path; defaults to the base.
    #[arg(long)]
    pub reference: Option<PathBuf>,
    /// Optional tokenizer JSON path; falls back to the config.
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,
    /// Self-learning runtime mode: cpu or gpu.
    #[arg(long, default_value = "cpu")]
    pub mode: String,
    /// Self-learning replay JSONL path.
    #[arg(long, default_value = "data/replay.jsonl")]
    pub replay_path: PathBuf,
    /// Self-learning state directory for adapters and metrics.
    #[arg(long, default_value = "adapters/selflearn")]
    pub self_learn_state_dir: PathBuf,
    #[command(flatten)]
    pub forgetting: ForgettingArgs,
}

#[derive(Debug, Args)]
/// `selflearn stats` — print replay, metrics, and forgetting statistics.
pub struct StatsArgs {
    /// Self-learning runtime mode: text or vision.
    #[arg(long, default_value = "text")]
    pub mode: String,
    /// Override the replay JSONL path (defaults by mode).
    #[arg(long)]
    pub replay_path: Option<PathBuf>,
    /// Self-learning state directory for adapters and metrics.
    #[arg(long, default_value = "adapters/selflearn")]
    pub self_learn_state_dir: PathBuf,
}

#[derive(Debug, Args)]
/// `selflearn reset` — delete the replay buffer and self-learning state.
pub struct ResetArgs {
    /// Replay JSONL path to delete.
    #[arg(long, default_value = "data/replay.jsonl")]
    pub replay_path: PathBuf,
    /// Self-learning state directory to delete recursively.
    #[arg(long, default_value = "adapters/selflearn")]
    pub self_learn_state_dir: PathBuf,
    /// Confirm the destructive reset (required to proceed).
    #[arg(long)]
    pub yes: bool,
}

#[derive(Debug, Args)]
/// `selflearn forgetting-report` — print the stored forgetting-curve report.
pub struct ForgettingReportArgs {
    /// Forgetting curves store JSON path to print.
    #[arg(long, default_value = "adapters/selflearn/forgetting_curves.json")]
    pub forgetting_store: PathBuf,
}

pub fn run(args: SelflearnArgs) -> anyhow::Result<()> {
    match args.command {
        SelflearnCommand::Start(args) => run_start(*args),
        SelflearnCommand::FlushGradients(args) => run_flush(args),
        SelflearnCommand::Replay(args) => run_replay(args),
        SelflearnCommand::Stats(args) => run_stats(args),
        SelflearnCommand::ForgettingReport(args) => run_forgetting_report(args),
        SelflearnCommand::Reset(args) => run_reset(args),
    }
}

fn run_start(args: StartArgs) -> anyhow::Result<()> {
    let mode = args.mode.trim().to_ascii_lowercase();
    let self_learn = match mode.as_str() {
        "text" => "cpu".to_string(),
        "vision" => {
            if args.image.is_none() {
                return Err(anyhow::anyhow!(
                    "selflearn start --mode vision requires --image"
                ));
            }
            "gpu".to_string()
        }
        other => {
            return Err(anyhow::anyhow!(
                "unsupported selflearn start mode '{other}', expected text|vision"
            ));
        }
    };
    let replay_path = args
        .replay_path
        .or_else(|| (mode == "vision").then(|| PathBuf::from("data/replay_buffer_v2.jsonl")));
    infer::run(InferArgs {
        config: args.config,
        model: args.model,
        tokenizer: args.tokenizer,
        image: args.image,
        video: None,
        document: None,
        audio: None,
        pages: None,
        document_dpi: None,
        max_document_pages: None,
        frames: None,
        frame_sampling: None,
        prompt: args.prompt,
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        seed: args.seed,
        thinking: args.thinking,
        predict_view: false,
        stream: args.stream,
        greedy: args.greedy,
        speculative: false,
        draft_model: None,
        draft_config: None,
        draft_tokenizer: None,
        draft_tokens: None,
        stats: false,
        tools: None,
        tool_choice: "auto".into(),
        safety: args.safety,
        safety_audit_log: args.safety_audit_log,
        self_learn,
        replay_path,
        self_learn_state_dir: args.self_learn_state_dir,
        self_learn_reference: args.self_learn_reference,
        self_learn_verifier: args.self_learn_verifier,
        self_learn_vision_verifier: args.self_learn_vision_verifier,
        self_learn_ground_truth: args.self_learn_ground_truth,
        forgetting_manifest: args.forgetting.forgetting_manifest,
        forgetting_store: args.forgetting.forgetting_store,
        forgetting_jsonl: args.forgetting.forgetting_jsonl,
        forgetting_threshold: args.forgetting.forgetting_threshold,
        forgetting_max_examples: args.forgetting.forgetting_max_examples,
        forgetting_allow_code_exec: args.forgetting.forgetting_allow_code_exec,
        forgetting_require_all_probes: args.forgetting.require_all_probes,
        forgetting_baseline_id: args.forgetting.forgetting_baseline_id,
        best_of_n: None,
        selection: "self-consistency".into(),
        ground_truth: None,
        rag: false,
        index: None,
        rag_top_k: 4,
    })
}

fn run_flush(args: SelflearnRunArgs) -> anyhow::Result<()> {
    let mut loop_ = build_loop(args)?;
    match loop_.flush_pending_gradients()? {
        Some(norm) => println!("flushed pending self-learning gradients grad_norm={norm:.4}"),
        None => println!("no pending self-learning gradients to flush"),
    }
    Ok(())
}

fn run_replay(args: SelflearnRunArgs) -> anyhow::Result<()> {
    let mut loop_ = build_loop(args)?;
    match loop_.replay_finetune()? {
        Some(norm) => println!("self-learning replay fine-tune completed grad_norm={norm:.4}"),
        None => println!("no replay entries available for self-learning replay"),
    }
    Ok(())
}

fn run_stats(args: StatsArgs) -> anyhow::Result<()> {
    let vision_mode = matches!(args.mode.trim().to_ascii_lowercase().as_str(), "vision");
    let replay_path = args.replay_path.unwrap_or_else(|| {
        if vision_mode {
            PathBuf::from("data/replay_buffer_v2.jsonl")
        } else {
            PathBuf::from("data/replay.jsonl")
        }
    });
    let config = if vision_mode {
        SelfLearnConfig::for_gpu()
    } else {
        SelfLearnConfig::for_cpu()
    }
    .with_replay_path(replay_path.clone());
    let replay = ReplayBuffer::load_jsonl(&replay_path, config.replay)?;
    let stats = replay.stats();
    let vision_entries = replay
        .entries()
        .iter()
        .filter(|entry| entry.is_vision())
        .count();
    println!(
        "Replay buffer: {} / {} entries  avg score: {:.2}",
        stats.len, stats.capacity, stats.avg_score
    );
    if vision_mode {
        let cache = VisionCache::new(&args.self_learn_state_dir);
        println!(
            "Vision entries: {}  cached image-token files: {}",
            vision_entries,
            cache.cached_entry_count()
        );
    }
    let mut topics = stats.topics.into_iter().collect::<Vec<_>>();
    topics.sort_by(|a, b| a.0.cmp(&b.0));
    for (topic, count) in topics {
        println!("{topic}: {count}");
    }
    let metrics = aarambh_studio_selflearn::LearningMetrics::load_jsonl(
        args.self_learn_state_dir.join("metrics.jsonl"),
    )?;
    println!("{}", metrics.summary());
    let forgetting_path = args.self_learn_state_dir.join("forgetting_curves.json");
    if forgetting_path.exists() {
        print_forgetting_store(&forgetting_path)?;
    }
    Ok(())
}

fn run_forgetting_report(args: ForgettingReportArgs) -> anyhow::Result<()> {
    print_forgetting_store(&args.forgetting_store)
}

fn print_forgetting_store(path: &std::path::Path) -> anyhow::Result<()> {
    let store = aarambh_studio_eval::ForgettingStore::load(path)?;
    println!(
        "Forgetting curves: {} capabilities  threshold: {:.4}",
        store.curves.len(),
        store.significance_threshold
    );
    for curve in store.curves.values() {
        let latest = curve.points.last();
        println!(
            "{}: points={} latest={}",
            curve.capability,
            curve.points.len(),
            latest
                .map(|point| format!(
                    "{} score={:.4}",
                    point.checkpoint_or_session_id, point.score
                ))
                .unwrap_or_else(|| "none".into())
        );
    }
    Ok(())
}

fn run_reset(args: ResetArgs) -> anyhow::Result<()> {
    if !args.yes {
        return Err(anyhow::anyhow!(
            "reset requires --yes because it deletes replay and self-learning state"
        ));
    }
    if args.replay_path.exists() {
        std::fs::remove_file(&args.replay_path)?;
    }
    if args.self_learn_state_dir.exists() {
        std::fs::remove_dir_all(&args.self_learn_state_dir)?;
    }
    println!("self-learning state reset");
    Ok(())
}

fn build_loop(args: SelflearnRunArgs) -> anyhow::Result<SelfLearnLoop> {
    let run_config = TrainingRunConfig::from_toml(&args.config)?;
    let run_device = run_config.device()?;
    let dtype = run_config.dtype_for_device(&run_device)?.to_candle();
    let device = run_device.to_candle()?;
    let mode = args
        .mode
        .parse::<SelfLearnMode>()
        .map_err(anyhow::Error::msg)?;
    let tokenizer_path = args
        .tokenizer
        .clone()
        .or_else(|| run_config.tokenizer_path.clone())
        .or_else(|| run_config.tokenizer_save_path.clone())
        .unwrap_or_else(|| run_config.train.checkpoint_dir.join("tokenizer.json"));
    let tokenizer = BpeTokenizer::from_pretrained(&tokenizer_path)?;
    let mut model_config = run_config.model.clone();
    model_config.vocab_size = tokenizer.vocab_size();
    let mut config = SelfLearnConfig::for_mode(mode)
        .with_replay_path(args.replay_path)
        .with_state_dir(args.self_learn_state_dir);
    if let Some(forgetting) = forgetting_config(&args.forgetting, &config.state_dir, &args.config) {
        config = config.with_forgetting(forgetting);
    }
    SelfLearnLoop::from_paths(SelfLearnBuildConfig {
        model_config,
        base_model_path: args.base.clone(),
        reference_model_path: args.reference.unwrap_or(args.base),
        tokenizer_path,
        config,
        device,
        dtype,
        seed: run_config.train.seed,
    })
    .map_err(anyhow::Error::from)
}

fn forgetting_config(
    args: &ForgettingArgs,
    state_dir: &std::path::Path,
    config_path: &std::path::Path,
) -> Option<SelfLearnForgettingConfig> {
    let manifest = args.forgetting_manifest.clone()?;
    Some(SelfLearnForgettingConfig {
        enabled: true,
        manifest,
        config_path: Some(config_path.to_path_buf()),
        store: args
            .forgetting_store
            .clone()
            .unwrap_or_else(|| state_dir.join("forgetting_curves.json")),
        jsonl: args.forgetting_jsonl.clone(),
        max_examples: Some(args.forgetting_max_examples),
        significance_threshold: args.forgetting_threshold,
        allow_code_exec: args.forgetting_allow_code_exec,
        require_all_probes: args.require_all_probes,
        baseline_id: args.forgetting_baseline_id.clone(),
        ..SelfLearnForgettingConfig::default()
    })
}
