use std::path::PathBuf;
use std::str::FromStr;

use aarambh_studio_finetune::{
    AdapterMethod, AudioVlmDoraRunConfig, DocumentVlmDoraRunConfig, DpoConfig, DpoRunConfig,
    GrpoConfig, GrpoRunConfig, GrpoThinkingMode, LoraConfig, SftRunConfig, VerifierKind,
    VideoVlmDoraRunConfig, VlmDoraRunConfig, merge_adapter_from_paths,
    run_audio_vlm_dora_from_config, run_document_vlm_dora_from_config, run_dora_from_config,
    run_dpo_from_config, run_grpo_from_config, run_sft_from_config, run_tool_sft_from_config,
    run_video_vlm_dora_from_config, run_vlm_dora_from_config,
};
use aarambh_studio_train::TrainingRunConfig;
use aarambh_studio_vision::{FrameSamplingStrategy, LayoutEncodingKind, TemporalEncodingKind};
use clap::{Args, Subcommand};

#[derive(Debug, Args)]
pub struct FinetuneArgs {
    #[command(subcommand)]
    pub command: FinetuneCommand,
}

#[derive(Debug, Subcommand)]
pub enum FinetuneCommand {
    Sft(FinetuneRunArgs),
    Qlora(FinetuneRunArgs),
    ToolSft(FinetuneRunArgs),
    ToolQlora(FinetuneRunArgs),
    Dora(FinetuneRunArgs),
    Qdora(FinetuneRunArgs),
    VlmDora(VlmFinetuneArgs),
    VlmQdora(VlmFinetuneArgs),
    VideoDora(VlmFinetuneArgs),
    VideoQdora(VlmFinetuneArgs),
    DocumentDora(VlmFinetuneArgs),
    DocumentQdora(VlmFinetuneArgs),
    AudioDora(VlmFinetuneArgs),
    AudioQdora(VlmFinetuneArgs),
    Grpo(GrpoArgs),
    Dpo(DpoArgs),
    Qdpo(DpoArgs),
    Merge(MergeArgs),
}

#[derive(Debug, Args)]
pub struct FinetuneRunArgs {
    #[arg(long, default_value = "configs/tiny_shakespeare.toml")]
    pub config: PathBuf,
    #[arg(long)]
    pub base: PathBuf,
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,
    #[arg(long)]
    pub data: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long, default_value_t = 16)]
    pub lora_rank: usize,
    #[arg(long)]
    pub lora_alpha: Option<f64>,
    #[arg(long, default_value_t = 0.05)]
    pub lora_dropout: f32,
    #[arg(long, default_value = "attn.wq,attn.wk,attn.wv,attn.wo")]
    pub target_modules: String,
    #[arg(long)]
    pub batch_size: Option<usize>,
    #[arg(long)]
    pub max_steps: Option<usize>,
    #[arg(long)]
    pub max_epochs: Option<usize>,
    #[arg(long)]
    pub lr: Option<f64>,
    #[arg(long)]
    pub grad_accum_steps: Option<usize>,
    #[arg(long)]
    pub warmup_steps: Option<usize>,
    #[arg(long)]
    pub save_every_n_steps: Option<usize>,
    #[arg(long)]
    pub log_every_n_steps: Option<usize>,
    #[arg(long)]
    pub no_shuffle: bool,
}

#[derive(Debug, Args)]
pub struct MergeArgs {
    #[arg(long, default_value = "configs/tiny_shakespeare.toml")]
    pub config: PathBuf,
    #[arg(long)]
    pub base: PathBuf,
    #[arg(long)]
    pub adapter: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long, default_value = "auto")]
    pub method: String,
}

#[derive(Debug, Args)]
pub struct GrpoArgs {
    #[arg(long, default_value = "configs/tiny_shakespeare.toml")]
    pub config: PathBuf,
    #[arg(long)]
    pub base: PathBuf,
    #[arg(long)]
    pub reference: PathBuf,
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,
    #[arg(long)]
    pub data: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long, default_value = "math-format")]
    pub verifier: String,
    #[arg(long, default_value_t = 8)]
    pub group_size: usize,
    #[arg(long, default_value_t = 128)]
    pub max_new_tokens: usize,
    #[arg(long, default_value_t = 0.8)]
    pub temperature: f32,
    #[arg(long, default_value_t = 0.95)]
    pub top_p: f32,
    #[arg(long, default_value_t = 50)]
    pub top_k: usize,
    #[arg(long, default_value = "low")]
    pub thinking: String,
    #[arg(long, default_value_t = 16)]
    pub lora_rank: usize,
    #[arg(long)]
    pub lora_alpha: Option<f64>,
    #[arg(long, default_value_t = 0.05)]
    pub lora_dropout: f32,
    #[arg(long, default_value = "attn.wq,attn.wk,attn.wv,attn.wo")]
    pub target_modules: String,
    #[arg(long, alias = "max-steps")]
    pub steps: Option<usize>,
    #[arg(long)]
    pub max_epochs: Option<usize>,
    #[arg(long)]
    pub lr: Option<f64>,
    #[arg(long, default_value_t = 0.01)]
    pub kl_coeff: f64,
    #[arg(long)]
    pub grad_accum_steps: Option<usize>,
    #[arg(long)]
    pub warmup_steps: Option<usize>,
    #[arg(long)]
    pub save_every_n_steps: Option<usize>,
    #[arg(long)]
    pub log_every_n_steps: Option<usize>,
    #[arg(long)]
    pub no_shuffle: bool,
}

#[derive(Debug, Args)]
pub struct DpoArgs {
    #[arg(long, default_value = "configs/tiny_shakespeare.toml")]
    pub config: PathBuf,
    #[arg(long)]
    pub base: PathBuf,
    #[arg(long, conflicts_with = "reference_free")]
    pub reference: Option<PathBuf>,
    #[arg(long, conflicts_with = "reference")]
    pub reference_free: bool,
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,
    #[arg(long)]
    pub data: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long, default_value_t = 0.1)]
    pub beta: f64,
    #[arg(long)]
    pub max_prompt_tokens: Option<usize>,
    #[arg(long)]
    pub max_completion_tokens: Option<usize>,
    #[arg(long, default_value_t = 16)]
    pub lora_rank: usize,
    #[arg(long)]
    pub lora_alpha: Option<f64>,
    #[arg(long, default_value_t = 0.0)]
    pub lora_dropout: f32,
    #[arg(long, default_value = "attn.wq,attn.wk,attn.wv,attn.wo")]
    pub target_modules: String,
    #[arg(long)]
    pub batch_size: Option<usize>,
    #[arg(long)]
    pub max_steps: Option<usize>,
    #[arg(long)]
    pub max_epochs: Option<usize>,
    #[arg(long)]
    pub lr: Option<f64>,
    #[arg(long)]
    pub grad_accum_steps: Option<usize>,
    #[arg(long)]
    pub warmup_steps: Option<usize>,
    #[arg(long)]
    pub save_every_n_steps: Option<usize>,
    #[arg(long)]
    pub log_every_n_steps: Option<usize>,
    #[arg(long)]
    pub no_shuffle: bool,
}

#[derive(Debug, Args)]
pub struct VlmFinetuneArgs {
    #[arg(long, default_value = "configs/vision_vqa_instruct.toml")]
    pub config: PathBuf,
    #[arg(long)]
    pub base: PathBuf,
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,
    #[arg(long)]
    pub data: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long)]
    pub projector: Option<PathBuf>,
    #[arg(long)]
    pub clip_config: Option<PathBuf>,
    #[arg(long)]
    pub clip_weights: Option<PathBuf>,
    #[arg(long)]
    pub image_root: Option<PathBuf>,
    #[arg(long)]
    pub video_root: Option<PathBuf>,
    #[arg(long)]
    pub frames: Option<usize>,
    #[arg(long)]
    pub frame_sampling: Option<String>,
    #[arg(long)]
    pub temporal_encoding: Option<String>,
    #[arg(long)]
    pub temporal: Option<PathBuf>,
    #[arg(long)]
    pub document_root: Option<PathBuf>,
    #[arg(long)]
    pub document_dpi: Option<u32>,
    #[arg(long)]
    pub max_document_pages: Option<usize>,
    #[arg(long)]
    pub layout_encoding: Option<String>,
    #[arg(long)]
    pub layout: Option<PathBuf>,
    #[arg(long)]
    pub freeze_projector: bool,
    #[arg(long, default_value_t = 16)]
    pub lora_rank: usize,
    #[arg(long)]
    pub lora_alpha: Option<f64>,
    #[arg(long, default_value_t = 0.05)]
    pub lora_dropout: f32,
    #[arg(
        long,
        default_value = "attn.wq,attn.wk,attn.wv,attn.wo,ffn.w_gate,ffn.w_up,ffn.w_down"
    )]
    pub target_modules: String,
    #[arg(long)]
    pub batch_size: Option<usize>,
    #[arg(long)]
    pub max_steps: Option<usize>,
    #[arg(long)]
    pub max_epochs: Option<usize>,
    #[arg(long)]
    pub lr: Option<f64>,
    #[arg(long)]
    pub grad_accum_steps: Option<usize>,
    #[arg(long)]
    pub warmup_steps: Option<usize>,
    #[arg(long)]
    pub save_every_n_steps: Option<usize>,
    #[arg(long)]
    pub log_every_n_steps: Option<usize>,
    #[arg(long)]
    pub no_shuffle: bool,
}

pub fn run(args: FinetuneArgs) -> anyhow::Result<()> {
    match args.command {
        FinetuneCommand::Sft(args) => run_lora_finetune(args, false),
        FinetuneCommand::Qlora(args) => run_lora_finetune(args, true),
        FinetuneCommand::ToolSft(args) => run_tool_finetune(args, false),
        FinetuneCommand::ToolQlora(args) => run_tool_finetune(args, true),
        FinetuneCommand::Dora(args) => run_dora_finetune(args, false),
        FinetuneCommand::Qdora(args) => run_dora_finetune(args, true),
        FinetuneCommand::VlmDora(args) => run_vlm_dora_finetune(args, false),
        FinetuneCommand::VlmQdora(args) => run_vlm_dora_finetune(args, true),
        FinetuneCommand::VideoDora(args) => run_video_vlm_dora_finetune(args, false),
        FinetuneCommand::VideoQdora(args) => run_video_vlm_dora_finetune(args, true),
        FinetuneCommand::DocumentDora(args) => run_document_vlm_dora_finetune(args, false),
        FinetuneCommand::DocumentQdora(args) => run_document_vlm_dora_finetune(args, true),
        FinetuneCommand::AudioDora(args) => run_audio_vlm_dora_finetune(args, false),
        FinetuneCommand::AudioQdora(args) => run_audio_vlm_dora_finetune(args, true),
        FinetuneCommand::Grpo(args) => run_grpo(args),
        FinetuneCommand::Dpo(args) => run_dpo(args, false),
        FinetuneCommand::Qdpo(args) => run_dpo(args, true),
        FinetuneCommand::Merge(args) => run_merge(args),
    }
}

fn run_tool_finetune(args: FinetuneRunArgs, qlora: bool) -> anyhow::Result<()> {
    let run_config = TrainingRunConfig::from_toml(&args.config)?;
    let device = run_config.device()?;
    let tokenizer_path = tokenizer_path(args.tokenizer.as_ref(), &run_config);
    let mut train_config = run_config.train.clone();
    apply_train_overrides(&mut train_config, &args);
    train_config.checkpoint_dir = args.output.clone();
    let lora_config = LoraConfig {
        rank: args.lora_rank,
        alpha: args.lora_alpha.unwrap_or(args.lora_rank as f64 * 2.0),
        dropout: args.lora_dropout,
        target_modules: LoraConfig::from_target_csv(&args.target_modules),
        ..Default::default()
    };
    run_tool_sft_from_config(SftRunConfig {
        model_config: run_config.model,
        train_config,
        base_model_path: args.base,
        tokenizer_path,
        data_path: args.data,
        output_dir: args.output,
        lora_config,
        device,
        qlora,
        shuffle: !args.no_shuffle && run_config.shuffle,
    })?;
    Ok(())
}

fn run_lora_finetune(args: FinetuneRunArgs, qlora: bool) -> anyhow::Result<()> {
    let run_config = TrainingRunConfig::from_toml(&args.config)?;
    let device = run_config.device()?;
    let tokenizer_path = tokenizer_path(args.tokenizer.as_ref(), &run_config);
    let mut train_config = run_config.train.clone();
    apply_train_overrides(&mut train_config, &args);
    train_config.checkpoint_dir = args.output.clone();

    let lora_config = LoraConfig {
        rank: args.lora_rank,
        alpha: args.lora_alpha.unwrap_or(args.lora_rank as f64 * 2.0),
        dropout: args.lora_dropout,
        target_modules: LoraConfig::from_target_csv(&args.target_modules),
        ..Default::default()
    };

    let config = SftRunConfig {
        model_config: run_config.model.clone(),
        train_config,
        base_model_path: args.base,
        tokenizer_path,
        data_path: args.data,
        output_dir: args.output,
        lora_config,
        device,
        qlora,
        shuffle: !args.no_shuffle && run_config.shuffle,
    };
    run_sft_from_config(config)?;
    Ok(())
}

fn run_dora_finetune(args: FinetuneRunArgs, qdora: bool) -> anyhow::Result<()> {
    let run_config = TrainingRunConfig::from_toml(&args.config)?;
    let device = run_config.device()?;
    let tokenizer_path = tokenizer_path(args.tokenizer.as_ref(), &run_config);
    let mut train_config = run_config.train.clone();
    apply_train_overrides(&mut train_config, &args);
    train_config.checkpoint_dir = args.output.clone();

    let lora_config = LoraConfig {
        rank: args.lora_rank,
        alpha: args.lora_alpha.unwrap_or(args.lora_rank as f64 * 2.0),
        dropout: args.lora_dropout,
        target_modules: LoraConfig::from_target_csv(&args.target_modules),
        ..Default::default()
    };

    let config = SftRunConfig {
        model_config: run_config.model.clone(),
        train_config,
        base_model_path: args.base,
        tokenizer_path,
        data_path: args.data,
        output_dir: args.output,
        lora_config,
        device,
        qlora: qdora,
        shuffle: !args.no_shuffle && run_config.shuffle,
    };
    run_dora_from_config(config)?;
    Ok(())
}

fn run_vlm_dora_finetune(args: VlmFinetuneArgs, qdora: bool) -> anyhow::Result<()> {
    let config = build_vlm_dora_config(args, qdora, false, false, false)?;
    run_vlm_dora_from_config(config)?;
    Ok(())
}

fn run_video_vlm_dora_finetune(args: VlmFinetuneArgs, qdora: bool) -> anyhow::Result<()> {
    let config = build_vlm_dora_config(args, qdora, true, false, false)?;
    run_video_vlm_dora_from_config(VideoVlmDoraRunConfig { vlm: config })?;
    Ok(())
}

fn run_document_vlm_dora_finetune(args: VlmFinetuneArgs, qdora: bool) -> anyhow::Result<()> {
    let config = build_vlm_dora_config(args, qdora, false, true, false)?;
    run_document_vlm_dora_from_config(DocumentVlmDoraRunConfig { vlm: config })?;
    Ok(())
}

fn run_audio_vlm_dora_finetune(args: VlmFinetuneArgs, qdora: bool) -> anyhow::Result<()> {
    let config = build_vlm_dora_config(args, qdora, false, false, true)?;
    run_audio_vlm_dora_from_config(AudioVlmDoraRunConfig { vlm: config })?;
    Ok(())
}

fn build_vlm_dora_config(
    args: VlmFinetuneArgs,
    qdora: bool,
    video_mode: bool,
    document_mode: bool,
    audio_mode: bool,
) -> anyhow::Result<VlmDoraRunConfig> {
    let run_config = TrainingRunConfig::from_toml(&args.config)?;
    let device = run_config.device()?;
    let dtype = run_config.dtype_for_device(&device)?.to_candle();
    let tokenizer_path = tokenizer_path(args.tokenizer.as_ref(), &run_config);
    let mut train_config = run_config.train.clone();
    apply_vlm_train_overrides(&mut train_config, &args);
    train_config.checkpoint_dir = args.output.clone();

    let mut vision = run_config
        .vision
        .clone()
        .ok_or_else(|| anyhow::anyhow!("VLM fine-tuning requires a [vision] config block"))?;
    if let Some(path) = args.projector.clone() {
        vision.projector_path = Some(path);
    }
    if let Some(path) = args.clip_config.clone() {
        vision.clip_config_path = path;
    }
    if let Some(path) = args.clip_weights.clone() {
        vision.clip_weights_path = path;
    }
    if let Some(path) = args.image_root.clone() {
        vision.image_root = path;
    }
    if video_mode {
        let video = vision.video.as_mut().ok_or_else(|| {
            anyhow::anyhow!("video VLM fine-tuning requires a [vision.video] config block")
        })?;
        if let Some(path) = args.video_root.clone() {
            video.video_root = path;
        }
        if let Some(frames) = args.frames {
            video.frame_count = frames;
        }
        if let Some(value) = args.frame_sampling.as_deref() {
            video.sampling = parse_frame_sampling(value)?;
        }
        if let Some(value) = args.temporal_encoding.as_deref() {
            video.temporal_encoding = parse_temporal_encoding(value)?;
        }
        if let Some(path) = args.temporal.clone() {
            video.temporal_path = Some(path);
        }
        video.validate()?;
    }
    if document_mode {
        let document = vision.document.as_mut().ok_or_else(|| {
            anyhow::anyhow!("document VLM fine-tuning requires a [vision.document] config block")
        })?;
        if let Some(path) = args.document_root.clone() {
            document.document_root = path;
        }
        if let Some(dpi) = args.document_dpi {
            document.target_dpi = dpi;
        }
        if let Some(max_pages) = args.max_document_pages {
            document.max_pages_per_document = max_pages;
        }
        if let Some(value) = args.layout_encoding.as_deref() {
            document.layout_encoding = parse_layout_encoding(value)?;
        }
        if let Some(path) = args.layout.clone() {
            document.layout_path = Some(path);
        }
        document.validate()?;
    }
    if audio_mode {
        let audio = vision.audio.as_mut().ok_or_else(|| {
            anyhow::anyhow!("audio VLM fine-tuning requires a [vision.audio] config block")
        })?;
        audio.validate()?;
    }
    let projector_path = vision.projector_path.clone().ok_or_else(|| {
        anyhow::anyhow!("VLM fine-tuning requires --projector or vision.projector_path")
    })?;

    let lora_config = LoraConfig {
        rank: args.lora_rank,
        alpha: args.lora_alpha.unwrap_or(args.lora_rank as f64 * 2.0),
        dropout: args.lora_dropout,
        target_modules: LoraConfig::from_target_csv(&args.target_modules),
        ..Default::default()
    };

    Ok(VlmDoraRunConfig {
        model_config: run_config.model,
        train_config,
        base_model_path: args.base,
        tokenizer_path,
        data_path: args.data,
        output_dir: args.output,
        lora_config,
        device,
        dtype,
        qdora,
        shuffle: !args.no_shuffle && run_config.shuffle,
        vision,
        projector_path,
        train_projector: !args.freeze_projector,
    })
}

fn parse_frame_sampling(value: &str) -> anyhow::Result<FrameSamplingStrategy> {
    match value.trim().to_ascii_lowercase().as_str() {
        "uniform" => Ok(FrameSamplingStrategy::Uniform),
        "scene" | "scene-aware" | "scene_aware" => Ok(FrameSamplingStrategy::SceneAware),
        other => Err(anyhow::anyhow!(
            "unsupported frame sampling '{other}', expected uniform|scene-aware"
        )),
    }
}

fn parse_temporal_encoding(value: &str) -> anyhow::Result<TemporalEncodingKind> {
    match value.trim().to_ascii_lowercase().as_str() {
        "learned" => Ok(TemporalEncodingKind::Learned),
        "sinusoidal" => Ok(TemporalEncodingKind::Sinusoidal),
        other => Err(anyhow::anyhow!(
            "unsupported temporal encoding '{other}', expected learned|sinusoidal"
        )),
    }
}

fn parse_layout_encoding(value: &str) -> anyhow::Result<LayoutEncodingKind> {
    match value.trim().to_ascii_lowercase().as_str() {
        "learned" => Ok(LayoutEncodingKind::Learned),
        "sinusoidal" => Ok(LayoutEncodingKind::Sinusoidal),
        other => Err(anyhow::anyhow!(
            "unsupported layout encoding '{other}', expected learned|sinusoidal"
        )),
    }
}

fn run_grpo(args: GrpoArgs) -> anyhow::Result<()> {
    let run_config = TrainingRunConfig::from_toml(&args.config)?;
    let device = run_config.device()?;
    let tokenizer_path = tokenizer_path(args.tokenizer.as_ref(), &run_config);
    let verifier = VerifierKind::from_str(&args.verifier).map_err(anyhow::Error::msg)?;
    let thinking = GrpoThinkingMode::from_str(&args.thinking).map_err(anyhow::Error::msg)?;

    let mut train_config = run_config.train.clone();
    train_config.checkpoint_dir = args.output.clone();
    train_config.batch_size = 1;
    train_config.lr = args.lr.unwrap_or(1e-5);
    train_config.max_epochs = args.max_epochs.unwrap_or(usize::MAX);
    if let Some(value) = args.steps {
        train_config.max_steps = value;
    }
    if let Some(value) = args.grad_accum_steps {
        train_config.grad_accum_steps = value;
    }
    if let Some(value) = args.warmup_steps {
        train_config.warmup_steps = value;
    }
    if let Some(value) = args.save_every_n_steps {
        train_config.save_every_n_steps = value;
    }
    if let Some(value) = args.log_every_n_steps {
        train_config.log_every_n_steps = value;
    }

    let lora_config = LoraConfig {
        rank: args.lora_rank,
        alpha: args.lora_alpha.unwrap_or(args.lora_rank as f64 * 2.0),
        dropout: args.lora_dropout,
        target_modules: LoraConfig::from_target_csv(&args.target_modules),
        ..Default::default()
    };
    let grpo_config = GrpoConfig {
        group_size: args.group_size,
        kl_coeff: args.kl_coeff,
        max_new_tokens: args.max_new_tokens,
        temperature: args.temperature,
        top_p: (args.top_p > 0.0 && args.top_p < 1.0).then_some(args.top_p),
        top_k: (args.top_k > 0).then_some(args.top_k),
        thinking,
    };

    let config = GrpoRunConfig {
        model_config: run_config.model,
        train_config,
        grpo_config,
        base_model_path: args.base,
        reference_model_path: args.reference,
        tokenizer_path,
        data_path: args.data,
        output_dir: args.output,
        lora_config,
        verifier,
        device,
        shuffle: !args.no_shuffle && run_config.shuffle,
    };
    run_grpo_from_config(config)?;
    Ok(())
}

fn run_dpo(args: DpoArgs, qdpo: bool) -> anyhow::Result<()> {
    let run_config = TrainingRunConfig::from_toml(&args.config)?;
    let device = run_config.device()?;
    let tokenizer_path = tokenizer_path(args.tokenizer.as_ref(), &run_config);
    let mut train_config = run_config.train.clone();
    train_config.checkpoint_dir = args.output.clone();
    train_config.lr = args.lr.unwrap_or(1e-5);
    if let Some(value) = args.batch_size {
        train_config.batch_size = value;
    }
    if let Some(value) = args.max_steps {
        train_config.max_steps = value;
    }
    if let Some(value) = args.max_epochs {
        train_config.max_epochs = value;
    }
    if let Some(value) = args.grad_accum_steps {
        train_config.grad_accum_steps = value;
    }
    if let Some(value) = args.warmup_steps {
        train_config.warmup_steps = value;
    }
    if let Some(value) = args.save_every_n_steps {
        train_config.save_every_n_steps = value;
    }
    if let Some(value) = args.log_every_n_steps {
        train_config.log_every_n_steps = value;
    }

    let dora_config = LoraConfig {
        rank: args.lora_rank,
        alpha: args.lora_alpha.unwrap_or(args.lora_rank as f64 * 2.0),
        dropout: args.lora_dropout,
        target_modules: LoraConfig::from_target_csv(&args.target_modules),
        ..Default::default()
    };
    let dpo_config = DpoConfig {
        beta: args.beta,
        reference_free: args.reference_free,
        max_prompt_tokens: args.max_prompt_tokens,
        max_completion_tokens: args.max_completion_tokens,
    };
    let config = DpoRunConfig {
        model_config: run_config.model,
        train_config,
        dpo_config,
        base_model_path: args.base,
        reference_model_path: args.reference,
        tokenizer_path,
        data_path: args.data,
        output_dir: args.output,
        dora_config,
        device,
        qdpo,
        shuffle: !args.no_shuffle && run_config.shuffle,
    };
    run_dpo_from_config(config)?;
    Ok(())
}

fn run_merge(args: MergeArgs) -> anyhow::Result<()> {
    let run_config = TrainingRunConfig::from_toml(&args.config)?;
    let device = run_config.device()?.to_candle()?;
    let method = parse_merge_method(&args.method)?;
    let output = merge_adapter_from_paths(
        &run_config.model,
        args.base,
        args.adapter,
        args.output,
        &device,
        method,
    )?;
    eprintln!("merged adapter written to {}", output.display());
    Ok(())
}

fn parse_merge_method(value: &str) -> anyhow::Result<Option<AdapterMethod>> {
    match value.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(None),
        "lora" | "qlora" => Ok(Some(AdapterMethod::Lora)),
        "dora" | "qdora" => Ok(Some(AdapterMethod::Dora)),
        other => Err(anyhow::anyhow!(
            "unsupported merge method '{other}', expected auto|lora|dora"
        )),
    }
}

fn tokenizer_path(tokenizer: Option<&PathBuf>, run_config: &TrainingRunConfig) -> PathBuf {
    tokenizer
        .cloned()
        .or_else(|| run_config.tokenizer_path.clone())
        .or_else(|| run_config.tokenizer_save_path.clone())
        .unwrap_or_else(|| run_config.train.checkpoint_dir.join("tokenizer.json"))
}

fn apply_train_overrides(
    train_config: &mut aarambh_studio_core::TrainConfig,
    args: &FinetuneRunArgs,
) {
    if let Some(value) = args.batch_size {
        train_config.batch_size = value;
    }
    if let Some(value) = args.max_steps {
        train_config.max_steps = value;
    }
    if let Some(value) = args.max_epochs {
        train_config.max_epochs = value;
    }
    if let Some(value) = args.lr {
        train_config.lr = value;
    }
    if let Some(value) = args.grad_accum_steps {
        train_config.grad_accum_steps = value;
    }
    if let Some(value) = args.warmup_steps {
        train_config.warmup_steps = value;
    }
    if let Some(value) = args.save_every_n_steps {
        train_config.save_every_n_steps = value;
    }
    if let Some(value) = args.log_every_n_steps {
        train_config.log_every_n_steps = value;
    }
}

fn apply_vlm_train_overrides(
    train_config: &mut aarambh_studio_core::TrainConfig,
    args: &VlmFinetuneArgs,
) {
    if let Some(value) = args.batch_size {
        train_config.batch_size = value;
    }
    if let Some(value) = args.max_steps {
        train_config.max_steps = value;
    }
    if let Some(value) = args.max_epochs {
        train_config.max_epochs = value;
    }
    if let Some(value) = args.lr {
        train_config.lr = value;
    }
    if let Some(value) = args.grad_accum_steps {
        train_config.grad_accum_steps = value;
    }
    if let Some(value) = args.warmup_steps {
        train_config.warmup_steps = value;
    }
    if let Some(value) = args.save_every_n_steps {
        train_config.save_every_n_steps = value;
    }
    if let Some(value) = args.log_every_n_steps {
        train_config.log_every_n_steps = value;
    }
}
