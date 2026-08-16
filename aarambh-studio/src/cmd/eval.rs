use std::fs;
use std::path::{Path, PathBuf};

use aarambh_studio_core::{AttentionKind, TokenizerLike};
use aarambh_studio_eval::{
    DEFAULT_SIGNIFICANCE_THRESHOLD, EvalConfig, EvalContext, ForgettingReport, ForgettingStore,
    ProbeManifest, QatRobustnessReport, Scorecard, ScorecardComparison, run_all,
    run_capability_probes, tokenizer_fingerprint,
};
use aarambh_studio_inference::{SelectionStrategy, ThinkingMode};
use aarambh_studio_model::{KvCacheLayerReport, kv_cache_report};
use aarambh_studio_quant::GgufFormat;
use aarambh_studio_tokenizer::BpeTokenizer;
use aarambh_studio_train::TrainingRunConfig;
use clap::Args;
use serde::Deserialize;
use std::str::FromStr;

#[derive(Debug, Args)]
pub struct EvalArgs {
    #[arg(long)]
    pub config: Option<PathBuf>,
    #[arg(long)]
    pub model: Option<PathBuf>,
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,
    #[arg(long, default_value = "ppl")]
    pub tasks: String,
    #[arg(long, default_value = "data/eval")]
    pub data_dir: PathBuf,
    #[arg(long)]
    pub max_examples: Option<usize>,
    #[arg(long, default_value_t = 128)]
    pub max_new_tokens: usize,
    /// Thinking budget: none, low, medium, high, or max (Phase 39).
    #[arg(long, default_value = "none")]
    pub thinking: String,
    #[arg(long, default_value_t = 8)]
    pub agent_max_steps: usize,
    #[arg(long)]
    pub allow_code_exec: bool,
    #[arg(long)]
    pub out: Option<PathBuf>,
    #[arg(long)]
    pub markdown: Option<PathBuf>,
    #[arg(long, num_args = 2)]
    pub compare: Vec<PathBuf>,
    #[arg(long)]
    pub qat_compare: bool,
    #[arg(long, requires = "qat_compare")]
    pub baseline_model: Option<PathBuf>,
    #[arg(long)]
    pub forgetting_manifest: Option<PathBuf>,
    #[arg(long, default_value = "checkpoints/forgetting/curves.json")]
    pub forgetting_store: PathBuf,
    #[arg(long, requires = "forgetting_manifest")]
    pub checkpoint_id: Option<String>,
    #[arg(long, requires = "forgetting_manifest")]
    pub baseline_id: Option<String>,
    #[arg(long, default_value_t = DEFAULT_SIGNIFICANCE_THRESHOLD)]
    pub significance_threshold: f64,
    #[arg(long, requires_all = ["forgetting_manifest", "baseline_id"])]
    pub forgetting_jsonl: Option<PathBuf>,
    #[arg(long, requires = "forgetting_manifest")]
    pub require_all_probes: bool,
    /// Print per-layer KV-cache bytes/token by attention kind and exit (v4 Phase 41).
    #[arg(long)]
    pub kv_cache_report: bool,
    /// Generate N independent candidate completions per generative task and
    /// record single-sample vs best-of-N accuracy in the scorecard details
    /// (Phase 45). When set, gsm8k and humaneval tasks compute both.
    #[arg(long)]
    pub best_of_n: Option<usize>,
    /// Selection strategy for best-of-N evaluation: verifier, self-consistency,
    /// majority, or process-reward (Phase 45). Defaults to self-consistency.
    #[arg(long, default_value = "self-consistency")]
    pub best_of_n_selection: String,
    /// Base RNG seed for best-of-N candidate sampling (Phase 45).
    #[arg(long, default_value_t = 0)]
    pub best_of_n_seed: u64,
}

#[derive(Debug, Deserialize)]
struct CheckpointPointer {
    path: PathBuf,
}

pub fn run(args: EvalArgs) -> anyhow::Result<()> {
    if !args.compare.is_empty() {
        return run_compare(&args);
    }
    if args.qat_compare {
        return run_qat_compare(&args);
    }

    let config_path = args
        .config
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--config is required unless --compare is used"))?;
    let run_config = TrainingRunConfig::from_toml(config_path)?;
    let run_device = run_config.device()?;
    let dtype = run_config.dtype_for_device(&run_device)?.to_candle();
    if args.kv_cache_report {
        return run_kv_cache_report(&run_config.model, dtype);
    }
    let device = run_device.to_candle()?;
    let tokenizer_path = tokenizer_path(&args, &run_config);
    let model_path = match args.model.clone() {
        Some(path) => path,
        None => default_model_path(&run_config.train.checkpoint_dir)?,
    };

    let tokenizer = BpeTokenizer::from_pretrained(&tokenizer_path)?;
    tokenizer.validate_special_tokens()?;
    let tokenizer_sha256 = tokenizer_fingerprint(&tokenizer)?;
    let mut model_config = run_config.model.clone();
    model_config.vocab_size = tokenizer.vocab_size();
    let model = aarambh_studio_weights::load_any_model_with_dtype(
        &model_path,
        &model_config,
        &device,
        dtype,
    )?;
    let context = EvalContext::new(model, tokenizer, device, dtype);
    let thinking_mode = ThinkingMode::from_str(&args.thinking).map_err(anyhow::Error::msg)?;
    let best_of_n_selection = SelectionStrategy::from_str(&args.best_of_n_selection)
        .map_err(|err| anyhow::anyhow!("{err}"))?;
    let eval_config = EvalConfig {
        tasks: parse_tasks(&args.tasks),
        data_dir: args.data_dir.clone(),
        max_examples: args.max_examples,
        max_new_tokens: args.max_new_tokens,
        agent_max_steps: args.agent_max_steps,
        allow_code_exec: args.allow_code_exec,
        thinking_mode,
        best_of_n: args.best_of_n,
        best_of_n_selection,
        best_of_n_seed: args.best_of_n_seed,
        model_path: Some(model_path.display().to_string()),
        tokenizer_path: Some(tokenizer_path.display().to_string()),
        config_path: Some(config_path.display().to_string()),
    };

    let scorecard = if let Some(manifest_path) = &args.forgetting_manifest {
        run_forgetting(
            &context,
            &eval_config,
            &tokenizer_sha256,
            manifest_path,
            &args,
        )?
    } else {
        run_all(&context, &eval_config)?
    };
    write_outputs(
        &scorecard.to_json()?,
        &scorecard.to_markdown(),
        args.out.as_deref(),
        args.markdown.as_deref(),
    )
}

fn run_forgetting(
    context: &EvalContext,
    eval_config: &EvalConfig,
    tokenizer_sha256: &str,
    manifest_path: &Path,
    args: &EvalArgs,
) -> anyhow::Result<Scorecard> {
    let checkpoint_id = args
        .checkpoint_id
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--checkpoint-id is required with --forgetting-manifest"))?;
    let manifest = ProbeManifest::from_path(manifest_path)?;
    let run = run_capability_probes(
        context,
        eval_config,
        &manifest,
        checkpoint_id,
        Some(tokenizer_sha256.to_string()),
        args.require_all_probes,
    )?;
    let mut store = ForgettingStore::load_or_new(
        &args.forgetting_store,
        &manifest,
        Some(tokenizer_sha256.to_string()),
        args.significance_threshold,
    )?;
    store.record(&run)?;

    let (deltas, routing_drift) = match args.baseline_id.as_deref() {
        Some(baseline_id) => {
            let deltas = store.deltas(baseline_id, checkpoint_id)?;
            (deltas, store.routing_drift(baseline_id, checkpoint_id))
        }
        None => (Vec::new(), Vec::new()),
    };
    store.save_atomic(&args.forgetting_store)?;
    if let (Some(path), Some(baseline_id)) = (&args.forgetting_jsonl, args.baseline_id.as_deref()) {
        store.export_jsonl(path, baseline_id, checkpoint_id)?;
    }

    let tasks = run
        .scores
        .iter()
        .flat_map(|score| score.tasks.iter().cloned())
        .collect();
    Ok(Scorecard::new(
        tasks,
        context.context_len_used(),
        eval_config.max_new_tokens,
        eval_config.model_path.clone(),
        eval_config.tokenizer_path.clone(),
        eval_config.config_path.clone(),
    )
    .with_forgetting(ForgettingReport {
        baseline_checkpoint_or_session: args.baseline_id.clone(),
        current_checkpoint_or_session: checkpoint_id.to_string(),
        significance_threshold: args.significance_threshold,
        deltas,
        routing_drift,
        skipped: run.skipped,
    }))
}

fn run_qat_compare(args: &EvalArgs) -> anyhow::Result<()> {
    let config_path = args
        .config
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--config is required for --qat-compare"))?;
    let qat_path = args
        .model
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--model is required for --qat-compare"))?;
    let baseline_path = args
        .baseline_model
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--baseline-model is required for --qat-compare"))?;
    let run_config = TrainingRunConfig::from_toml(config_path)?;
    let qat = run_config.model.qat.as_ref().ok_or_else(|| {
        anyhow::anyhow!("--qat-compare requires [model.qat] in the training config")
    })?;
    let format = match qat.bits.bits() {
        4 => GgufFormat::Q4KM,
        8 => GgufFormat::Q80,
        bits => return Err(anyhow::anyhow!("unsupported QAT comparison width {bits}")),
    };
    let run_device = run_config.device()?;
    let dtype = run_config.dtype_for_device(&run_device)?.to_candle();
    let device = run_device.to_candle()?;
    let tokenizer_path = tokenizer_path(args, &run_config);
    let tokenizer = BpeTokenizer::from_pretrained(&tokenizer_path)?;
    tokenizer.validate_special_tokens()?;
    let mut model_config = run_config.model.clone();
    model_config.vocab_size = tokenizer.vocab_size();

    let temporary = TemporaryQatExports::new(format);
    export_for_comparison(
        baseline_path,
        &temporary.baseline,
        &model_config,
        &device,
        dtype,
        format,
    )?;
    export_for_comparison(
        qat_path,
        &temporary.qat,
        &model_config,
        &device,
        dtype,
        format,
    )?;

    let baseline_fp = evaluate_checkpoint(
        baseline_path,
        &model_config,
        &tokenizer,
        &tokenizer_path,
        config_path,
        &device,
        dtype,
        args,
    )?;
    let baseline_quantized = evaluate_checkpoint(
        &temporary.baseline,
        &model_config,
        &tokenizer,
        &tokenizer_path,
        config_path,
        &device,
        dtype,
        args,
    )?;
    let qat_fp = evaluate_checkpoint(
        qat_path,
        &model_config,
        &tokenizer,
        &tokenizer_path,
        config_path,
        &device,
        dtype,
        args,
    )?;
    let qat_quantized = evaluate_checkpoint(
        &temporary.qat,
        &model_config,
        &tokenizer,
        &tokenizer_path,
        config_path,
        &device,
        dtype,
        args,
    )?;
    let report =
        QatRobustnessReport::compare(baseline_fp, baseline_quantized, qat_fp, qat_quantized);
    write_outputs(
        &report.to_json()?,
        &report.to_markdown(),
        args.out.as_deref(),
        args.markdown.as_deref(),
    )
}

#[allow(clippy::too_many_arguments)]
fn evaluate_checkpoint(
    model_path: &Path,
    model_config: &aarambh_studio_core::ModelConfig,
    tokenizer: &BpeTokenizer,
    tokenizer_path: &Path,
    config_path: &Path,
    device: &candle_core::Device,
    dtype: candle_core::DType,
    args: &EvalArgs,
) -> anyhow::Result<Scorecard> {
    let model =
        aarambh_studio_weights::load_any_model_with_dtype(model_path, model_config, device, dtype)?;
    let context = EvalContext::new(model, tokenizer.clone(), device.clone(), dtype);
    let best_of_n_selection = SelectionStrategy::from_str(&args.best_of_n_selection)
        .map_err(|err| anyhow::anyhow!("{err}"))?;
    run_all(
        &context,
        &EvalConfig {
            tasks: parse_tasks(&args.tasks),
            data_dir: args.data_dir.clone(),
            max_examples: args.max_examples,
            max_new_tokens: args.max_new_tokens,
            agent_max_steps: args.agent_max_steps,
            allow_code_exec: args.allow_code_exec,
            thinking_mode: ThinkingMode::from_str(&args.thinking).map_err(anyhow::Error::msg)?,
            best_of_n: args.best_of_n,
            best_of_n_selection,
            best_of_n_seed: args.best_of_n_seed,
            model_path: Some(model_path.display().to_string()),
            tokenizer_path: Some(tokenizer_path.display().to_string()),
            config_path: Some(config_path.display().to_string()),
        },
    )
    .map_err(Into::into)
}

fn export_for_comparison(
    source: &Path,
    output: &Path,
    model_config: &aarambh_studio_core::ModelConfig,
    device: &candle_core::Device,
    dtype: candle_core::DType,
    format: GgufFormat,
) -> anyhow::Result<()> {
    let model =
        aarambh_studio_weights::load_any_model_with_dtype(source, model_config, device, dtype)?;
    aarambh_studio_weights::save_gguf(&model, format, output)?;
    Ok(())
}

struct TemporaryQatExports {
    baseline: PathBuf,
    qat: PathBuf,
}

impl TemporaryQatExports {
    fn new(format: GgufFormat) -> Self {
        let suffix = match format {
            GgufFormat::Q80 => "q8",
            _ => "q4",
        };
        let nonce = format!("{}_{}", std::process::id(), suffix);
        Self {
            baseline: std::env::temp_dir().join(format!("aarambh_qat_baseline_{nonce}.gguf")),
            qat: std::env::temp_dir().join(format!("aarambh_qat_model_{nonce}.gguf")),
        }
    }
}

impl Drop for TemporaryQatExports {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.baseline);
        let _ = fs::remove_file(&self.qat);
    }
}

fn run_compare(args: &EvalArgs) -> anyhow::Result<()> {
    if args.compare.len() != 2 {
        return Err(anyhow::anyhow!(
            "--compare expects exactly two scorecard paths"
        ));
    }
    let before = read_scorecard(&args.compare[0])?;
    let after = read_scorecard(&args.compare[1])?;
    let comparison = ScorecardComparison::compare(&before, &after);
    write_outputs(
        &comparison.to_json()?,
        &comparison.to_markdown(),
        args.out.as_deref(),
        args.markdown.as_deref(),
    )
}

fn tokenizer_path(args: &EvalArgs, run_config: &TrainingRunConfig) -> PathBuf {
    args.tokenizer
        .clone()
        .or_else(|| run_config.tokenizer_path.clone())
        .or_else(|| run_config.tokenizer_save_path.clone())
        .unwrap_or_else(|| run_config.train.checkpoint_dir.join("tokenizer.json"))
}

fn default_model_path(checkpoint_dir: &Path) -> anyhow::Result<PathBuf> {
    for pointer_name in ["best.json", "latest.json"] {
        let pointer_path = checkpoint_dir.join(pointer_name);
        if pointer_path.exists() {
            let file = fs::File::open(&pointer_path)?;
            let pointer: CheckpointPointer = serde_json::from_reader(file)?;
            return Ok(pointer.path.join("model.safetensors"));
        }
    }
    Err(anyhow::anyhow!(
        "no model provided and no best.json or latest.json found in {}",
        checkpoint_dir.display()
    ))
}

fn parse_tasks(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|task| !task.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn read_scorecard(path: &Path) -> anyhow::Result<Scorecard> {
    let file = fs::File::open(path)?;
    Ok(serde_json::from_reader(file)?)
}

fn write_outputs(
    json: &str,
    markdown: &str,
    out: Option<&Path>,
    markdown_out: Option<&Path>,
) -> anyhow::Result<()> {
    if let Some(path) = out {
        fs::write(path, json)?;
    }
    if let Some(path) = markdown_out {
        fs::write(path, markdown)?;
    }
    if out.is_none() && markdown_out.is_none() {
        println!("{markdown}");
    }
    Ok(())
}

fn dtype_bytes(dtype: candle_core::DType) -> usize {
    use candle_core::DType;
    match dtype {
        DType::F64 => 8,
        DType::F32 => 4,
        DType::F16 | DType::BF16 => 2,
        DType::U8 => 1,
        _ => 4,
    }
}

fn kind_name(kind: AttentionKind) -> &'static str {
    match kind {
        AttentionKind::Full => "full",
        AttentionKind::Sparse => "sparse_dsa",
        AttentionKind::GatedDeltaNet => "gated_deltanet",
        AttentionKind::LatentMLA => "latent_mla",
    }
}

/// Print a per-layer KV-cache bytes/token breakdown and exit (Phase 41).
fn run_kv_cache_report(
    cfg: &aarambh_studio_core::ModelConfig,
    dtype: candle_core::DType,
) -> anyhow::Result<()> {
    let bytes = dtype_bytes(dtype);
    let report = kv_cache_report(cfg, bytes);
    let full_baseline: usize = 2 * cfg.n_kv_heads * cfg.head_dim() * bytes;
    let total: usize = report.iter().map(|r| r.bytes_per_token).sum();

    println!(
        "KV-cache bytes/token (dtype={:?}, {} bytes/element, {} layers)",
        dtype,
        bytes,
        report.len()
    );
    println!("{:<6} {:<16} {:>12}  note", "layer", "kind", "bytes/tok");
    for KvCacheLayerReport {
        layer,
        kind,
        bytes_per_token,
        note,
    } in &report
    {
        println!(
            "{:<6} {:<16} {:>12}  {}",
            layer,
            kind_name(*kind),
            bytes_per_token,
            note
        );
    }
    println!("-------------------------------------------------------------");
    println!("total bytes/token across all layers: {total}");
    println!(
        "all-full baseline ({} layers): {}",
        cfg.n_layers,
        full_baseline * cfg.n_layers
    );
    if full_baseline * cfg.n_layers > 0 {
        let ratio = total as f64 / (full_baseline * cfg.n_layers) as f64;
        println!(
            "hybrid/all-full ratio: {:.3} ({:.1}% of all-full cache)",
            ratio,
            100.0 * ratio
        );
    }
    Ok(())
}
