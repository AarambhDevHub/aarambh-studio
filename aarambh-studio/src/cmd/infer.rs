use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use aarambh_studio_audio::{
    AudioEncoderConfig, AudioModel, AudioPreprocessor, AudioProjector, AudioProjectorConfig,
    FrozenAudioEncoder, interleave_audio_tokens,
};
use aarambh_studio_core::{AarambhError, TokenizerLike};
use aarambh_studio_finetune::{MathVerifier, Verifier, VerifierKind};
use aarambh_studio_inference::{
    BestOfNConfig, BestOfNEngine, CompletionVerifier, GenerationConfig, GenerationOutput,
    GenerationPhase, GenerationStep, HeuristicProcessRewardScorer, InferenceEngine,
    MtpSpeculativeEngine, Sampler, SelectionStrategy, SpeculativeConfig, SpeculativeEngine,
    ThinkingMode, ToolCallingConfig, ToolChoice, ToolDefinition,
};
use aarambh_studio_safety::{
    SafeResponse, SafeStreamEvent, SafetyGenerator, SafetyGuard, SafetyMode, SafetyPolicy,
    SafetyVerdict,
};
use aarambh_studio_selflearn::{
    SelfLearnBuildConfig, SelfLearnConfig, SelfLearnForgettingConfig, SelfLearnLoop, SelfLearnMode,
    VisionCache, VisionVerifierKind, require_vision_hardware,
};
use aarambh_studio_tokenizer::{
    ASSISTANT, AUDIO, AUDIO_END, AUDIO_ID, BpeTokenizer, DOCUMENT, DOCUMENT_END, DOCUMENT_ID,
    FRAME_SEP, FRAME_SEP_ID, IMAGE, IMAGE_END, IMAGE_ID, PAGE_SEP, PAGE_SEP_ID, SYSTEM,
    THINK_END_ID, THINK_START_ID, USER, VIDEO, VIDEO_END, VIDEO_ID,
};
use aarambh_studio_train::TrainingRunConfig;
use aarambh_studio_vision::{
    ClipVisionEncoder, DocumentSource, FrameSamplingStrategy, ImagePreprocessor,
    LayoutAwareProjector, LayoutEncodingKind, LayoutProjectorConfig, PageRasterizer,
    PageRasterizerConfig, ProjectorConfig, TemporalEncoder, TemporalEncodingConfig,
    TemporalEncodingKind, VideoSamplingConfig, VisionEncoderConfig, VisionModel,
    VisionPreprocessConfig, VisionProjector, decode_sampled_video, interleave_document_tokens,
    interleave_image_tokens, interleave_video_tokens,
};
use candle_core::Tensor;
use candle_nn::{VarBuilder, VarMap};
use clap::Args;
use serde::Deserialize;

use crate::ui::predict_view;

const ANSI_DIM: &str = "\x1b[2m";
const ANSI_RESET: &str = "\x1b[0m";

#[derive(Debug, Args)]
/// Generate text, multimodal, tool-use, speculative, best-of-N, or
/// self-learning completions from a trained checkpoint.
pub struct InferArgs {
    /// Training/model TOML configuration (provides architecture + device).
    #[arg(long, default_value = "configs/tiny_shakespeare.toml")]
    pub config: PathBuf,
    /// Model checkpoint path; falls back to best.json/latest.json pointer.
    #[arg(long)]
    pub model: Option<PathBuf>,
    /// Optional tokenizer JSON path; falls back to the configured tokenizer.
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,
    /// Image file path for vision (VQA) inference.
    #[arg(long)]
    pub image: Option<PathBuf>,
    /// Video file path for temporal video inference (mutually exclusive with --image).
    #[arg(long, conflicts_with = "image")]
    pub video: Option<PathBuf>,
    /// Document file path for layout-aware document inference.
    #[arg(long, conflicts_with_all = ["image", "video"])]
    pub document: Option<PathBuf>,
    /// Audio file path for audio-language inference.
    #[arg(long, conflicts_with_all = ["image", "video", "document"])]
    pub audio: Option<PathBuf>,
    /// Comma-separated 1-based page numbers to rasterise from the document.
    #[arg(long, requires = "document")]
    pub pages: Option<String>,
    /// DPI used when rasterising document pages to images.
    #[arg(long, requires = "document")]
    pub document_dpi: Option<u32>,
    /// Maximum number of document pages to rasterise.
    #[arg(long, requires = "document")]
    pub max_document_pages: Option<usize>,
    /// Number of frames to sample from a --video input.
    #[arg(long)]
    pub frames: Option<usize>,
    /// Frame sampling strategy for --video: uniform or scene-aware.
    #[arg(long)]
    pub frame_sampling: Option<String>,
    /// Prompt text (or chat-template user message) fed to the model.
    #[arg(long)]
    pub prompt: String,
    /// Operator-set system instructions prepended as a single leading system
    /// turn (Phase 52). Omitted by default — a session with no system turn
    /// reproduces the v1.0.0 prompt format exactly.
    #[arg(long)]
    pub system: Option<String>,
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
    #[arg(long, default_value = "none")]
    /// Thinking budget: none, low, medium, high, or max.
    pub thinking: String,
    /// Render the live next-token prediction view to stdout.
    #[arg(long)]
    pub predict_view: bool,
    /// Stream tokens to stdout as they are generated.
    #[arg(long)]
    pub stream: bool,
    /// Use greedy (argmax) decoding instead of stochastic sampling.
    #[arg(long)]
    pub greedy: bool,
    /// Enable speculative decoding with an external draft model or MTP head.
    #[arg(long)]
    pub speculative: bool,
    /// External draft model checkpoint path (requires --draft-config).
    #[arg(long)]
    pub draft_model: Option<PathBuf>,
    /// External draft model TOML config path.
    #[arg(long)]
    pub draft_config: Option<PathBuf>,
    /// External draft model tokenizer JSON path (defaults to the target).
    #[arg(long)]
    pub draft_tokenizer: Option<PathBuf>,
    /// Number of tokens the draft model proposes per target forward (MTP width).
    #[arg(long)]
    pub draft_tokens: Option<usize>,
    /// Print generation, DSA, and MoE statistics to stderr after decoding.
    #[arg(long)]
    pub stats: bool,
    /// JSON tool definitions file (native or OpenAI-compatible) for tool calling.
    #[arg(long)]
    pub tools: Option<PathBuf>,
    /// Tool-choice policy: auto, none, required, or a named tool.
    #[arg(long, default_value = "auto")]
    pub tool_choice: String,
    /// Safety policy: strict, permissive, research, or none.
    #[arg(long, default_value = "strict")]
    pub safety: String,
    /// JSONL safety audit log path.
    #[arg(long, default_value = "safety_audit.jsonl")]
    pub safety_audit_log: PathBuf,
    /// Self-learning mode: disabled, cpu, or gpu.
    #[arg(long, default_value = "disabled")]
    pub self_learn: String,
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
    #[arg(long, default_value = "none")]
    pub self_learn_vision_verifier: String,
    /// Ground-truth answer required when a verifier other than none is used.
    #[arg(long)]
    pub self_learn_ground_truth: Option<String>,
    /// Capability probe manifest path (enables self-learning forgetting analysis).
    #[arg(long)]
    pub forgetting_manifest: Option<PathBuf>,
    /// Forgetting curves store path (defaults to `<state-dir>/forgetting_curves.json`).
    #[arg(long)]
    pub forgetting_store: Option<PathBuf>,
    /// Optional JSONL export path for self-learning forgetting deltas.
    #[arg(long)]
    pub forgetting_jsonl: Option<PathBuf>,
    /// Absolute capability-score delta treated as significant forgetting.
    #[arg(long, default_value_t = 0.02)]
    pub forgetting_threshold: f64,
    /// Maximum examples evaluated per capability probe during forgetting checks.
    #[arg(long, default_value_t = 8)]
    pub forgetting_max_examples: usize,
    /// Allow code execution in capability probes during forgetting checks.
    #[arg(long)]
    pub forgetting_allow_code_exec: bool,
    /// Require every manifest probe to run; otherwise missing probes are skipped.
    #[arg(long)]
    pub forgetting_require_all_probes: bool,
    /// Baseline checkpoint or session id used to compute forgetting deltas.
    #[arg(long)]
    pub forgetting_baseline_id: Option<String>,
    /// Generate N independent candidate completions and select the best one
    /// (Phase 45). When set, requires a stochastic sampler (use --temperature
    /// > 0 or --top-k/--top-p) for N > 1; greedy best-of-N is degenerate.
    #[arg(long)]
    pub best_of_n: Option<usize>,
    /// Selection strategy for best-of-N: verifier, self-consistency, majority,
    /// or process-reward (Phase 45). Defaults to self-consistency.
    #[arg(long, default_value = "self-consistency")]
    pub selection: String,
    /// Ground-truth answer required when --selection verifier is used (Phase 45).
    #[arg(long)]
    pub ground_truth: Option<String>,
    /// Enable retrieval-augmented generation (Phase 49). When set, requires
    /// `--index <PATH>` pointing at an index built by `retrieve build-index`.
    /// Retrieved chunks are spliced into the prompt ahead of the user's
    /// question before generation; the decoder is unchanged.
    #[arg(long)]
    pub rag: bool,
    /// Path to a retrieval index directory (built by `retrieve build-index`).
    /// Required when `--rag` is set.
    #[arg(long, requires = "rag")]
    pub index: Option<PathBuf>,
    /// Number of chunks to retrieve per RAG query (default 4). Only used with
    /// `--rag`.
    #[arg(long, default_value_t = 4, requires = "rag")]
    pub rag_top_k: usize,
}

#[derive(Debug, Deserialize)]
struct CheckpointPointer {
    path: PathBuf,
}

pub fn run(args: InferArgs) -> anyhow::Result<()> {
    if args.video.is_none() && (args.frames.is_some() || args.frame_sampling.is_some()) {
        return Err(anyhow::anyhow!(
            "--frames and --frame-sampling require --video"
        ));
    }
    if args.rag {
        // RAG augments the text prompt ahead of the user's question. It does
        // not touch the decoder, so it is text-only by design — mirroring the
        // best-of-N discipline. Multimodal paths construct their own prompts
        // (image/video/document/audio tokens spliced at the embedding level),
        // which is a different fusion mechanism than prompt augmentation.
        if args.image.is_some()
            || args.video.is_some()
            || args.document.is_some()
            || args.audio.is_some()
        {
            return Err(AarambhError::Unsupported(
                "RAG (--rag) is text-only; --image/--video/--document/--audio are not supported with --rag".into(),
            )
            .into());
        }
        if args.index.is_none() {
            return Err(anyhow::anyhow!("--rag requires --index <PATH>"));
        }
    }
    let run_config = TrainingRunConfig::from_toml(&args.config)?;
    let run_device = run_config.device()?;
    let dtype = run_config.dtype_for_device(&run_device)?.to_candle();
    let device = run_device.to_candle()?;
    let tokenizer_path = tokenizer_path(&args, &run_config);
    let model_path = match args.model.clone() {
        Some(path) => path,
        None => default_model_path(&run_config.train.checkpoint_dir)?,
    };
    let sampler = if args.greedy {
        Sampler::greedy()
    } else {
        Sampler::top_k_top_p(
            args.temperature,
            Some(args.top_k),
            Some(args.top_p),
            args.seed,
        )?
    };
    let thinking_mode = parse_thinking_mode(&args.thinking)?;
    let safety_mode = parse_safety_mode(&args.safety)?;
    let self_learn_mode = parse_self_learn_mode(&args.self_learn)?;
    validate_speculative_args(&args, self_learn_mode, &run_config.model)?;
    let tool_calling = load_tool_calling_config(&args, self_learn_mode)?;
    let config = GenerationConfig {
        max_new_tokens: args.max_tokens,
        sampler,
        thinking_mode,
        top_candidates: 5,
        tool_calling,
        stop_sequences: Vec::new(),
        capture_steps: true,
    };

    let prompt = if config.tool_calling.is_some() {
        args.prompt.clone()
    } else {
        prompt_for_mode(&args.prompt, thinking_mode)
    };
    // Phase 52: an optional operator-set system turn is prepended as a single
    // leading ` IMS` turn. Omitting it reproduces the v1.0.0 prompt format
    // exactly; the system turn is purely additive.
    let prompt = args
        .system
        .as_deref()
        .map(|system| format!("{SYSTEM}\n{system}\n{prompt}"))
        .unwrap_or(prompt);
    // Phase 49: retrieval-augmented generation splices retrieved chunks into
    // the prompt ahead of the user's question *before* any generation path
    // runs. The decoder is unchanged — this is pure prompt augmentation.
    let prompt = if args.rag {
        let index_dir = args
            .index
            .as_ref()
            .expect("validated: --rag requires --index");
        let pipeline =
            aarambh_studio_retrieve::RetrievalPipeline::load_hashing(index_dir, args.rag_top_k)?;
        let retrieved = pipeline.query(&args.prompt)?;
        eprintln!(
            "[rag] retrieved {} chunks (top_k={}) from {}",
            retrieved.len(),
            args.rag_top_k,
            index_dir.display()
        );
        for (i, chunk) in retrieved.iter().enumerate() {
            eprintln!(
                "  [{}] score={:.4} source={} offset={}",
                i + 1,
                chunk.score,
                chunk.source.display(),
                chunk.offset
            );
        }
        aarambh_studio_retrieve::augment_prompt(&prompt, &retrieved)
    } else {
        prompt
    };
    if args.best_of_n.is_some() {
        return run_best_of_n_infer(
            &args,
            &run_config,
            model_path,
            tokenizer_path,
            device,
            dtype,
            config,
            prompt,
            thinking_mode,
        );
    }
    if args.speculative {
        return run_speculative_infer(
            &args,
            &run_config,
            model_path,
            tokenizer_path,
            device,
            dtype,
            config,
            prompt,
            safety_mode,
            thinking_mode,
        );
    }
    if self_learn_mode.is_enabled() {
        if args.video.is_some() || args.document.is_some() || args.audio.is_some() {
            return Err(AarambhError::Unsupported(
                "video/document/audio self-learning is not supported; use text/image self-learning or disable --self-learn"
                    .into(),
            )
            .into());
        }
        if let Some(image_path) = args.image.clone() {
            return run_vision_self_learn_infer(
                &args,
                run_config,
                run_device,
                model_path,
                tokenizer_path,
                image_path,
                device,
                dtype,
                config,
                prompt,
                safety_mode,
                self_learn_mode,
                thinking_mode,
            );
        }
        return run_self_learn_infer(
            &args,
            run_config,
            model_path,
            tokenizer_path,
            device,
            dtype,
            config,
            prompt,
            safety_mode,
            self_learn_mode,
            thinking_mode,
        );
    }

    let mut engine = InferenceEngine::from_paths_with_dtype(
        model_path,
        &run_config.model,
        tokenizer_path,
        device,
        dtype,
    )?;
    let tokenizer_for_view = engine.tokenizer().clone();
    if let Some(document_path) = args.document.clone() {
        return run_document_infer(
            &args,
            &run_config,
            engine,
            document_path,
            dtype,
            config,
            prompt,
            safety_mode,
            thinking_mode,
            tokenizer_for_view,
        );
    }
    if let Some(video_path) = args.video.clone() {
        return run_video_infer(
            &args,
            &run_config,
            engine,
            video_path,
            dtype,
            config,
            prompt,
            safety_mode,
            thinking_mode,
            tokenizer_for_view,
        );
    }
    if let Some(image_path) = args.image.clone() {
        return run_vision_infer(
            &args,
            &run_config,
            engine,
            image_path,
            dtype,
            config,
            prompt,
            safety_mode,
            thinking_mode,
            tokenizer_for_view,
        );
    }
    if let Some(audio_path) = args.audio.clone() {
        return run_audio_infer(
            &args,
            &run_config,
            engine,
            audio_path,
            dtype,
            config,
            prompt,
            safety_mode,
            thinking_mode,
            tokenizer_for_view,
        );
    }
    if let Some(policy) = SafetyPolicy::for_mode(safety_mode)
        .map(|policy| policy.with_audit_path(&args.safety_audit_log))
    {
        let mut guard = SafetyGuard::new(engine, policy);
        let mut stream_state = StreamState::default();
        let started = Instant::now();
        let response = if args.stream {
            guard.generate_streaming_with_callback(&prompt, config, print_safe_stream_event)?
        } else {
            guard.generate_with_callback(&prompt, config, |step| {
                if args.predict_view {
                    print!(
                        "{}",
                        predict_view::render(
                            step,
                            &tokenizer_for_view,
                            args.temperature,
                            args.top_p,
                        )
                    );
                    io::stdout().flush()?;
                }
                Ok(())
            })?
        };
        let elapsed = started.elapsed();
        print_safe_response(&response, thinking_mode, args.stream, &mut stream_state)?;
        io::stdout().flush()?;
        if let Some(output) = &response.output {
            eprintln!("finish_reason={:?}", output.finish_reason);
            if args.stats {
                print_generation_stats("target", output, elapsed, &run_config);
            }
        } else {
            eprintln!("finish_reason=SafetyBlocked");
        }
        return Ok(());
    }

    let mut stream_state = StreamState::default();
    let started = Instant::now();
    let output = engine.generate_with_callback(&prompt, config, |step| {
        if args.predict_view {
            print!(
                "{}",
                predict_view::render(step, &tokenizer_for_view, args.temperature, args.top_p)
            );
        }
        if args.stream {
            stream_step(step, thinking_mode, &mut stream_state)?;
        }
        if args.predict_view || args.stream {
            io::stdout().flush()?;
        }
        Ok(())
    })?;
    let elapsed = started.elapsed();

    if args.stream {
        finish_stream(&mut stream_state);
    } else {
        print_generation_output(&output, thinking_mode)?;
    }
    io::stdout().flush()?;
    eprintln!("finish_reason={:?}", output.finish_reason);
    if args.stats {
        print_generation_stats("target", &output, elapsed, &run_config);
    }
    Ok(())
}

/// Adapter wrapping `aarambh_studio_finetune::MathVerifier` into the
/// inference crate's [`CompletionVerifier`] trait, so the inference crate
/// does not depend on the finetune crate (Phase 45).
#[derive(Debug, Clone, Copy, Default)]
struct MathVerifierAdapter {
    verifier: MathVerifier,
}

impl CompletionVerifier for MathVerifierAdapter {
    fn extract_answer(&self, completion: &str) -> Option<String> {
        aarambh_studio_inference::extract_final_number(completion).map(|n| n.to_string())
    }
    fn verify(&self, completion: &str, ground_truth: &str) -> f32 {
        self.verifier.score(completion, ground_truth)
    }
}

fn parse_selection_strategy(value: &str) -> anyhow::Result<SelectionStrategy> {
    use std::str::FromStr;
    SelectionStrategy::from_str(value).map_err(anyhow::Error::msg)
}

#[allow(clippy::too_many_arguments)]
fn run_best_of_n_infer(
    args: &InferArgs,
    run_config: &TrainingRunConfig,
    model_path: PathBuf,
    tokenizer_path: PathBuf,
    device: candle_core::Device,
    dtype: candle_core::DType,
    generation_config: GenerationConfig,
    prompt: String,
    thinking_mode: ThinkingMode,
) -> anyhow::Result<()> {
    if args.image.is_some()
        || args.video.is_some()
        || args.document.is_some()
        || args.audio.is_some()
    {
        return Err(AarambhError::Unsupported(
            "best-of-N is text-only; --image/--video/--document/--audio are not supported with --best-of-n".into(),
        )
        .into());
    }
    if args.tools.is_some() {
        return Err(AarambhError::Unsupported(
            "best-of-N does not support tool-calling prompts".into(),
        )
        .into());
    }
    let n = args.best_of_n.expect("validated: --best-of-n is set");
    let strategy = parse_selection_strategy(&args.selection)?;
    let base_seed = args.seed.unwrap_or(0);
    let mut best_of_n_config = BestOfNConfig::new(n, strategy)?.with_base_seed(base_seed);
    match strategy {
        SelectionStrategy::Verifier => {
            let ground_truth = args.ground_truth.clone().ok_or_else(|| {
                AarambhError::Config("--selection verifier requires --ground-truth <answer>".into())
            })?;
            best_of_n_config = best_of_n_config
                .with_verifier(Box::new(MathVerifierAdapter::default()))
                .with_ground_truth(ground_truth);
        }
        SelectionStrategy::ProcessReward => {
            best_of_n_config =
                best_of_n_config.with_process_reward(Box::new(HeuristicProcessRewardScorer::new()));
        }
        SelectionStrategy::SelfConsistency | SelectionStrategy::Majority => {}
    }
    if !generation_config.sampler.is_deterministic() && n > 1 {
        eprintln!(
            "best-of-N with N={n} stochastic sampler (seed={base_seed}); candidate i uses seed {base_seed}+i"
        );
    } else if generation_config.sampler.is_deterministic() && n > 1 {
        eprintln!(
            "warning: best-of-N with a greedy sampler produces N identical candidates; \
             use --temperature > 0 for diverse candidates"
        );
    }

    let target = InferenceEngine::from_paths_with_dtype(
        model_path,
        &run_config.model,
        tokenizer_path,
        device,
        dtype,
    )?;
    let mut engine = BestOfNEngine::new(target, best_of_n_config)?;
    let started = Instant::now();
    let output = engine.generate(&prompt, generation_config)?;
    let elapsed = started.elapsed();

    print_generation_output(&output.chosen, thinking_mode)?;
    io::stdout().flush()?;
    eprintln!("finish_reason={:?}", output.chosen.finish_reason);
    eprintln!(
        "selection={strategy} chosen_index={} candidates={}",
        output.chosen_index,
        output.candidates.len()
    );
    if args.stats {
        print_generation_stats("best-of-n-chosen", &output.chosen, elapsed, run_config);
        for (index, candidate) in output.candidates.iter().enumerate() {
            eprintln!(
                "  candidate[{index}] tokens={} finish={:?}",
                candidate.token_ids.len(),
                candidate.finish_reason
            );
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_speculative_infer(
    args: &InferArgs,
    target_config: &TrainingRunConfig,
    target_model: PathBuf,
    target_tokenizer: PathBuf,
    device: candle_core::Device,
    dtype: candle_core::DType,
    generation_config: GenerationConfig,
    prompt: String,
    safety_mode: SafetyMode,
    thinking_mode: ThinkingMode,
) -> anyhow::Result<()> {
    if let Some(draft_model) = args.draft_model.clone() {
        let draft_config_path = args.draft_config.as_ref().expect("validated draft config");
        let draft_config = TrainingRunConfig::from_toml(draft_config_path)?;
        let draft_tokenizer = args
            .draft_tokenizer
            .clone()
            .unwrap_or_else(|| target_tokenizer.clone());
        let speculative_config = SpeculativeConfig::new(args.draft_tokens.unwrap_or(4))?;
        let engine = SpeculativeEngine::from_paths_with_dtype(
            target_model,
            &target_config.model,
            &target_tokenizer,
            draft_model,
            &draft_config.model,
            draft_tokenizer,
            device,
            dtype,
            speculative_config,
        )?;
        let tokenizer_for_view = engine.tokenizer().clone();
        return run_speculative_engine(
            args,
            target_config,
            engine,
            tokenizer_for_view,
            generation_config,
            prompt,
            safety_mode,
            thinking_mode,
        );
    }

    let mtp = target_config
        .model
        .mtp
        .as_ref()
        .expect("validated MTP configuration");
    let speculative_config =
        SpeculativeConfig::new(args.draft_tokens.unwrap_or(mtp.num_future_tokens))?;
    let engine = MtpSpeculativeEngine::from_paths_with_dtype(
        target_model,
        &target_config.model,
        &target_tokenizer,
        device,
        dtype,
        speculative_config,
    )?;
    let tokenizer_for_view = engine.tokenizer().clone();
    run_speculative_engine(
        args,
        target_config,
        engine,
        tokenizer_for_view,
        generation_config,
        prompt,
        safety_mode,
        thinking_mode,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_speculative_engine<G: SafetyGenerator>(
    args: &InferArgs,
    target_config: &TrainingRunConfig,
    mut engine: G,
    tokenizer_for_view: BpeTokenizer,
    generation_config: GenerationConfig,
    prompt: String,
    safety_mode: SafetyMode,
    thinking_mode: ThinkingMode,
) -> anyhow::Result<()> {
    if let Some(policy) = SafetyPolicy::for_mode(safety_mode)
        .map(|policy| policy.with_audit_path(&args.safety_audit_log))
    {
        let mut guard = SafetyGuard::new(engine, policy);
        let mut stream_state = StreamState::default();
        let started = Instant::now();
        let response = if args.stream {
            guard.generate_streaming_with_callback(
                &prompt,
                generation_config,
                print_safe_stream_event,
            )?
        } else {
            guard.generate_with_callback(&prompt, generation_config, |step| {
                render_text_step(
                    args,
                    step,
                    thinking_mode,
                    &tokenizer_for_view,
                    &mut stream_state,
                )
            })?
        };
        let elapsed = started.elapsed();
        print_safe_response(&response, thinking_mode, args.stream, &mut stream_state)?;
        io::stdout().flush()?;
        if let Some(output) = &response.output {
            eprintln!("finish_reason={:?}", output.finish_reason);
            if args.stats {
                print_generation_stats("speculative", output, elapsed, target_config);
            }
        } else {
            eprintln!("finish_reason=SafetyBlocked");
        }
        return Ok(());
    }

    let mut stream_state = StreamState::default();
    let started = Instant::now();
    let output = engine.generate_with_callback(&prompt, generation_config, |step| {
        render_text_step(
            args,
            step,
            thinking_mode,
            &tokenizer_for_view,
            &mut stream_state,
        )
    })?;
    let elapsed = started.elapsed();
    if args.stream {
        finish_stream(&mut stream_state);
    } else {
        print_generation_output(&output, thinking_mode)?;
    }
    io::stdout().flush()?;
    eprintln!("finish_reason={:?}", output.finish_reason);
    if args.stats {
        print_generation_stats("speculative", &output, elapsed, target_config);
    }
    Ok(())
}

fn render_text_step(
    args: &InferArgs,
    step: &GenerationStep,
    thinking_mode: ThinkingMode,
    tokenizer: &BpeTokenizer,
    stream_state: &mut StreamState,
) -> aarambh_studio_core::Result<()> {
    if args.predict_view {
        print!(
            "{}",
            predict_view::render(step, tokenizer, args.temperature, args.top_p)
        );
    }
    if args.stream {
        stream_step(step, thinking_mode, stream_state)?;
    }
    if args.predict_view || args.stream {
        io::stdout().flush()?;
    }
    Ok(())
}

fn validate_speculative_args(
    args: &InferArgs,
    self_learn_mode: SelfLearnMode,
    model_config: &aarambh_studio_core::ModelConfig,
) -> anyhow::Result<()> {
    if !args.speculative {
        if args.draft_model.is_some()
            || args.draft_config.is_some()
            || args.draft_tokenizer.is_some()
            || args.draft_tokens.is_some()
        {
            return Err(
                AarambhError::Config("draft model options require --speculative".into()).into(),
            );
        }
        return Ok(());
    }
    if args.image.is_some()
        || args.video.is_some()
        || args.document.is_some()
        || args.audio.is_some()
    {
        return Err(AarambhError::Unsupported(
            "speculative decoding supports text inference only; --image/--video/--document/--audio are not supported"
                .into(),
        )
        .into());
    }
    if self_learn_mode.is_enabled() {
        return Err(AarambhError::Unsupported(
            "speculative decoding cannot be combined with --self-learn".into(),
        )
        .into());
    }
    let external_draft =
        args.draft_model.is_some() || args.draft_config.is_some() || args.draft_tokenizer.is_some();
    if external_draft {
        if args.draft_model.is_none() {
            return Err(AarambhError::Config(
                "--draft-model is required when external draft options are used".into(),
            )
            .into());
        }
        if args.draft_config.is_none() {
            return Err(AarambhError::Config(
                "--draft-config is required when --draft-model is used".into(),
            )
            .into());
        }
        SpeculativeConfig::new(args.draft_tokens.unwrap_or(4))?;
        return Ok(());
    }

    let mtp = model_config.mtp.as_ref().ok_or_else(|| {
        AarambhError::Config(
            "--speculative without --draft-model requires an MTP-enabled checkpoint config".into(),
        )
    })?;
    let proposal_width = args.draft_tokens.unwrap_or(mtp.num_future_tokens);
    SpeculativeConfig::new(proposal_width)?;
    if proposal_width < 2 || proposal_width > mtp.num_future_tokens {
        return Err(AarambhError::Config(format!(
            "MTP --draft-tokens must be in 2..={}, got {proposal_width}",
            mtp.num_future_tokens
        ))
        .into());
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ToolFile {
    Array(Vec<ToolEntry>),
    Object { tools: Vec<ToolEntry> },
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ToolEntry {
    Native(ToolDefinition),
    OpenAi {
        r#type: String,
        function: ToolDefinition,
    },
}

fn load_tool_calling_config(
    args: &InferArgs,
    self_learn_mode: SelfLearnMode,
) -> anyhow::Result<Option<ToolCallingConfig>> {
    let Some(path) = &args.tools else {
        if !args.tool_choice.eq_ignore_ascii_case("auto") {
            return Err(AarambhError::Config("--tool-choice requires --tools".into()).into());
        }
        return Ok(None);
    };
    if args.image.is_some()
        || args.video.is_some()
        || args.document.is_some()
        || args.audio.is_some()
    {
        return Err(AarambhError::Unsupported(
            "Phase 26 tool calling supports text inference only; multimodal inputs are not supported"
                .into(),
        )
        .into());
    }
    if self_learn_mode.is_enabled() {
        return Err(AarambhError::Unsupported(
            "Phase 26 tool calling cannot be combined with --self-learn".into(),
        )
        .into());
    }
    let definitions = load_tool_definitions(path)?;
    let choice = match args.tool_choice.trim() {
        value if value.eq_ignore_ascii_case("auto") => ToolChoice::Auto,
        value if value.eq_ignore_ascii_case("none") => ToolChoice::None,
        value if value.eq_ignore_ascii_case("required") => ToolChoice::Required,
        value => ToolChoice::Named(value.to_string()),
    };
    Ok(Some(ToolCallingConfig::new(definitions, choice)?))
}

pub(super) fn load_tool_definitions(path: &Path) -> anyhow::Result<Vec<ToolDefinition>> {
    let metadata = fs::metadata(path)?;
    if metadata.len() > 1024 * 1024 {
        return Err(AarambhError::Config(
            "tool definition file exceeds the 1 MiB request limit".into(),
        )
        .into());
    }
    let file = fs::File::open(path)?;
    let parsed: ToolFile = serde_json::from_reader(file)?;
    let entries = match parsed {
        ToolFile::Array(entries) | ToolFile::Object { tools: entries } => entries,
    };
    let definitions = entries
        .into_iter()
        .map(|entry| match entry {
            ToolEntry::Native(definition) => Ok(definition),
            ToolEntry::OpenAi { r#type, function } if r#type == "function" => Ok(function),
            ToolEntry::OpenAi {
                r#type: tool_type, ..
            } => Err(AarambhError::Config(format!(
                "unsupported OpenAI tool type {tool_type:?}; expected \"function\""
            ))),
        })
        .collect::<aarambh_studio_core::Result<Vec<_>>>()?;
    Ok(definitions)
}

fn print_generation_stats(
    mode: &str,
    output: &GenerationOutput,
    elapsed: Duration,
    run_config: &TrainingRunConfig,
) {
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    let tokens_per_second = if elapsed.is_zero() {
        0.0
    } else {
        output.token_ids.len() as f64 / elapsed.as_secs_f64()
    };
    if let Some(stats) = &output.speculative_stats {
        eprintln!(
            "generation_stats mode={mode} source={:?} tokens={} elapsed_ms={elapsed_ms:.3} tok_s={tokens_per_second:.3} target_decode_forwards={} draft_decode_forwards={} mtp_head_forwards={} proposed={} accepted={} rejected={} acceptance_rate={:.4} accepted_per_target_forward={:.3}",
            stats.proposal_source,
            output.token_ids.len(),
            stats.target_decode_forwards,
            stats.draft_decode_forwards,
            stats.mtp_head_forwards,
            stats.draft_tokens_proposed,
            stats.draft_tokens_accepted,
            stats.draft_tokens_rejected,
            stats.acceptance_rate(),
            stats.accepted_tokens_per_target_forward(),
        );
    } else {
        eprintln!(
            "generation_stats mode={mode} tokens={} elapsed_ms={elapsed_ms:.3} tok_s={tokens_per_second:.3}",
            output.token_ids.len(),
        );
    }
    if let Some(dsa) = &run_config.model.dsa_config {
        let sparse_layers = (0..run_config.model.n_layers)
            .filter(|layer| {
                run_config.model.attention_kind_for_layer(*layer)
                    == aarambh_studio_core::AttentionKind::Sparse
            })
            .count();
        let seq_len = output.usage.total_tokens;
        let dtype_bytes = match run_config.dtype.trim().to_ascii_lowercase().as_str() {
            "f16" | "fp16" | "bf16" => 2usize,
            _ => 4usize,
        };
        let kv_row_elements = 2 * run_config.model.n_kv_heads * run_config.model.head_dim();
        let stored_kv_bytes = sparse_layers * seq_len * kv_row_elements * dtype_bytes;
        let index_bytes = sparse_layers
            * seq_len.div_ceil(dsa.block_size)
            * run_config.model.head_dim()
            * std::mem::size_of::<f32>();
        let selected_tokens = seq_len.min(dsa.top_k_blocks * dsa.block_size);
        let selected_working_set_bytes =
            sparse_layers * selected_tokens * kv_row_elements * dtype_bytes;
        eprintln!(
            "dsa_cache_stats sparse_layers={sparse_layers} stored_cache_bytes={} selected_working_set_bytes={selected_working_set_bytes} selected_token_limit={selected_tokens}",
            stored_kv_bytes + index_bytes,
        );
    }
    if let Some(moe) = &run_config.model.moe
        && let (Ok(routed_experts), Ok(fine_dim), Ok(active_width)) = (
            moe.routed_expert_count(),
            moe.fine_grained_expert_dim(),
            moe.active_routed_width(),
        )
    {
        let moe_layers = (0..run_config.model.n_layers)
            .filter(|layer| moe.applies_to_layer(*layer))
            .count();
        let expert_params = 3u128 * run_config.model.hidden_dim as u128 * fine_dim as u128;
        let router_params = run_config.model.hidden_dim as u128 * routed_experts as u128;
        let total_params_per_layer =
            (routed_experts + moe.num_shared_experts) as u128 * expert_params + router_params;
        let active_params_per_token = (moe.top_k + moe.num_shared_experts) as u128 * expert_params;
        eprintln!(
            "moe_stats layers={moe_layers} routed_experts={routed_experts} active_routed={} shared_experts={} fine_dim={fine_dim} active_width={active_width} params_per_moe_layer={total_params_per_layer} active_expert_params_per_token={active_params_per_token}",
            moe.top_k, moe.num_shared_experts,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn run_vision_infer(
    args: &InferArgs,
    run_config: &TrainingRunConfig,
    mut engine: InferenceEngine,
    image_path: PathBuf,
    dtype: candle_core::DType,
    config: GenerationConfig,
    prompt: String,
    safety_mode: SafetyMode,
    thinking_mode: ThinkingMode,
    tokenizer_for_view: BpeTokenizer,
) -> anyhow::Result<()> {
    let runtime = load_vision_runtime(run_config, engine.device(), dtype)?;
    let prompt = ensure_image_prompt(&prompt);

    if let Some(policy) = SafetyPolicy::for_mode(safety_mode)
        .map(|policy| policy.with_audit_path(&args.safety_audit_log))
    {
        let adapter = VisionSafetyAdapter {
            engine,
            runtime,
            image_path,
        };
        let mut guard = SafetyGuard::new(adapter, policy);
        let mut stream_state = StreamState::default();
        let response = if args.stream {
            guard.generate_streaming_with_callback(&prompt, config, print_safe_stream_event)?
        } else {
            guard.generate_with_callback(&prompt, config, |step| {
                if args.predict_view {
                    print!(
                        "{}",
                        predict_view::render(
                            step,
                            &tokenizer_for_view,
                            args.temperature,
                            args.top_p,
                        )
                    );
                    io::stdout().flush()?;
                }
                Ok(())
            })?
        };
        print_safe_response(&response, thinking_mode, args.stream, &mut stream_state)?;
        io::stdout().flush()?;
        if let Some(output) = &response.output {
            eprintln!("finish_reason={:?}", output.finish_reason);
        } else {
            eprintln!("finish_reason=SafetyBlocked");
        }
        return Ok(());
    }

    let embeddings = build_vision_prompt_embeddings(&engine, &runtime, &image_path, &prompt)?;
    let mut stream_state = StreamState::default();
    let output = engine.generate_with_embeddings_callback(&embeddings, config, |step| {
        if args.predict_view {
            print!(
                "{}",
                predict_view::render(step, &tokenizer_for_view, args.temperature, args.top_p)
            );
        }
        if args.stream {
            stream_step(step, thinking_mode, &mut stream_state)?;
        }
        if args.predict_view || args.stream {
            io::stdout().flush()?;
        }
        Ok(())
    })?;
    if args.stream {
        finish_stream(&mut stream_state);
    } else {
        print_generation_output(&output, thinking_mode)?;
    }
    io::stdout().flush()?;
    eprintln!("finish_reason={:?}", output.finish_reason);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_video_infer(
    args: &InferArgs,
    run_config: &TrainingRunConfig,
    mut engine: InferenceEngine,
    video_path: PathBuf,
    dtype: candle_core::DType,
    config: GenerationConfig,
    prompt: String,
    safety_mode: SafetyMode,
    thinking_mode: ThinkingMode,
    tokenizer_for_view: BpeTokenizer,
) -> anyhow::Result<()> {
    let runtime = load_vision_runtime(run_config, engine.device(), dtype)?;
    let video_config = run_config
        .vision
        .as_ref()
        .and_then(|vision| vision.video.as_ref())
        .ok_or_else(|| anyhow::anyhow!("--video requires a [vision.video] config block"))?;
    let sampling = VideoSamplingConfig {
        frame_count: args.frames.unwrap_or(video_config.frame_count),
        max_frame_count: video_config.max_frame_count,
        strategy: args
            .frame_sampling
            .as_deref()
            .map(parse_frame_sampling)
            .transpose()?
            .unwrap_or(video_config.sampling),
        scene_min_gap: video_config.scene_min_gap,
    };
    sampling.validate()?;
    let prompt = ensure_video_prompt(&prompt, sampling.frame_count);

    if let Some(policy) = SafetyPolicy::for_mode(safety_mode)
        .map(|policy| policy.with_audit_path(&args.safety_audit_log))
    {
        let adapter = VideoSafetyAdapter {
            engine,
            runtime,
            video_path,
            sampling,
            encoder_batch_size: video_config.encoder_frame_batch_size,
        };
        let mut guard = SafetyGuard::new(adapter, policy);
        let mut stream_state = StreamState::default();
        let response = if args.stream {
            guard.generate_streaming_with_callback(&prompt, config, print_safe_stream_event)?
        } else {
            guard.generate_with_callback(&prompt, config, |step| {
                if args.predict_view {
                    print!(
                        "{}",
                        predict_view::render(
                            step,
                            &tokenizer_for_view,
                            args.temperature,
                            args.top_p,
                        )
                    );
                    io::stdout().flush()?;
                }
                Ok(())
            })?
        };
        print_safe_response(&response, thinking_mode, args.stream, &mut stream_state)?;
        io::stdout().flush()?;
        eprintln!(
            "finish_reason={}",
            response
                .output
                .as_ref()
                .map(|output| format!("{:?}", output.finish_reason))
                .unwrap_or_else(|| "SafetyBlocked".to_string())
        );
        return Ok(());
    }

    let embeddings = build_video_prompt_embeddings(
        &engine,
        &runtime,
        &video_path,
        &prompt,
        &sampling,
        video_config.encoder_frame_batch_size,
    )?;
    let mut stream_state = StreamState::default();
    let output = engine.generate_with_embeddings_callback(&embeddings, config, |step| {
        if args.predict_view {
            print!(
                "{}",
                predict_view::render(step, &tokenizer_for_view, args.temperature, args.top_p)
            );
        }
        if args.stream {
            stream_step(step, thinking_mode, &mut stream_state)?;
        }
        if args.predict_view || args.stream {
            io::stdout().flush()?;
        }
        Ok(())
    })?;
    if args.stream {
        finish_stream(&mut stream_state);
    } else {
        print_generation_output(&output, thinking_mode)?;
    }
    io::stdout().flush()?;
    eprintln!("finish_reason={:?}", output.finish_reason);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_document_infer(
    args: &InferArgs,
    run_config: &TrainingRunConfig,
    mut engine: InferenceEngine,
    document_path: PathBuf,
    dtype: candle_core::DType,
    config: GenerationConfig,
    prompt: String,
    safety_mode: SafetyMode,
    thinking_mode: ThinkingMode,
    tokenizer_for_view: BpeTokenizer,
) -> anyhow::Result<()> {
    let runtime = load_document_runtime(run_config, engine.device(), dtype)?;
    let selected_pages = args
        .pages
        .as_deref()
        .map(parse_page_selection)
        .transpose()?;
    let (page_tokens, page_count) = project_document_tokens(
        &runtime,
        &document_path,
        selected_pages.as_deref(),
        engine.device(),
        args.document_dpi,
        args.max_document_pages,
    )?;
    let prompt = ensure_document_prompt(&prompt, page_count);

    if let Some(policy) = SafetyPolicy::for_mode(safety_mode)
        .map(|policy| policy.with_audit_path(&args.safety_audit_log))
    {
        let adapter = DocumentSafetyAdapter {
            engine,
            page_tokens,
        };
        let mut guard = SafetyGuard::new(adapter, policy);
        let mut stream_state = StreamState::default();
        let response = if args.stream {
            guard.generate_streaming_with_callback(&prompt, config, print_safe_stream_event)?
        } else {
            guard.generate_with_callback(&prompt, config, |step| {
                if args.predict_view {
                    print!(
                        "{}",
                        predict_view::render(
                            step,
                            &tokenizer_for_view,
                            args.temperature,
                            args.top_p,
                        )
                    );
                    io::stdout().flush()?;
                }
                Ok(())
            })?
        };
        print_safe_response(&response, thinking_mode, args.stream, &mut stream_state)?;
        io::stdout().flush()?;
        eprintln!(
            "finish_reason={}",
            response
                .output
                .as_ref()
                .map(|output| format!("{:?}", output.finish_reason))
                .unwrap_or_else(|| "SafetyBlocked".to_string())
        );
        return Ok(());
    }

    let embeddings = build_document_prompt_embeddings(&engine, &page_tokens, &prompt)?;
    let mut stream_state = StreamState::default();
    let output = engine.generate_with_embeddings_callback(&embeddings, config, |step| {
        if args.predict_view {
            print!(
                "{}",
                predict_view::render(step, &tokenizer_for_view, args.temperature, args.top_p)
            );
        }
        if args.stream {
            stream_step(step, thinking_mode, &mut stream_state)?;
        }
        if args.predict_view || args.stream {
            io::stdout().flush()?;
        }
        Ok(())
    })?;
    if args.stream {
        finish_stream(&mut stream_state);
    } else {
        print_generation_output(&output, thinking_mode)?;
    }
    io::stdout().flush()?;
    eprintln!("finish_reason={:?}", output.finish_reason);
    Ok(())
}

fn ensure_video_prompt(prompt: &str, frame_count: usize) -> String {
    if prompt.contains(VIDEO) {
        return prompt.to_string();
    }
    let mut marker = String::from(VIDEO);
    for _ in 1..frame_count {
        marker.push_str(FRAME_SEP);
    }
    marker.push_str(VIDEO_END);
    format!("{marker}\n{prompt}")
}

fn ensure_document_prompt(prompt: &str, page_count: usize) -> String {
    if prompt.contains(DOCUMENT) {
        return prompt.to_string();
    }
    let mut marker = String::from(DOCUMENT);
    for _ in 1..page_count {
        marker.push_str(PAGE_SEP);
    }
    marker.push_str(DOCUMENT_END);
    format!("{marker}\n{prompt}")
}

fn parse_page_selection(value: &str) -> anyhow::Result<Vec<usize>> {
    let pages = value
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(|item| {
            item.parse::<usize>()
                .map_err(|error| anyhow::anyhow!("invalid 1-based document page {item:?}: {error}"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    if pages.is_empty() || pages.contains(&0) {
        return Err(anyhow::anyhow!(
            "--pages requires non-zero 1-based page numbers"
        ));
    }
    Ok(pages)
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

pub(super) struct VisionRuntime {
    model: VisionModel,
    preprocess: ImagePreprocessor,
    temporal: Option<TemporalEncoder>,
    cache_salt: String,
}

pub(super) struct DocumentRuntime {
    vision: VisionRuntime,
    layout: LayoutAwareProjector,
    rasterizer_config: PageRasterizerConfig,
    encoder_page_batch_size: usize,
}

pub(super) fn load_vision_runtime(
    run_config: &TrainingRunConfig,
    device: &candle_core::Device,
    dtype: candle_core::DType,
) -> anyhow::Result<VisionRuntime> {
    let vision = run_config
        .vision
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--image/--video requires a [vision] config block"))?;
    let encoder_config = VisionEncoderConfig::from_json(&vision.clip_config_path)?;
    let encoder = ClipVisionEncoder::load_pretrained(
        &vision.clip_weights_path,
        encoder_config.clone(),
        device,
        dtype,
    )?;
    let projector_path = match &vision.projector_path {
        Some(path) => path.clone(),
        None => default_model_path(&run_config.train.checkpoint_dir)?,
    };
    let projector_config = ProjectorConfig {
        vit_d_model: encoder_config.vit_d_model,
        llm_d_model: run_config.model.hidden_dim,
        hidden_mult: vision.projector_hidden_mult,
    };
    let projector =
        VisionProjector::load_safetensors(&projector_path, projector_config, device, dtype)?;
    let preprocess = ImagePreprocessor::new(VisionPreprocessConfig {
        image_size: encoder_config.image_size,
        ..VisionPreprocessConfig::default()
    })?;
    let cache_salt = format!(
        "clip_config={};clip_weights={};projector={};hidden_mult={};llm_hidden={}",
        vision.clip_config_path.display(),
        vision.clip_weights_path.display(),
        projector_path.display(),
        vision.projector_hidden_mult,
        run_config.model.hidden_dim
    );
    let temporal = if let Some(video) = &vision.video {
        video.validate()?;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, device);
        let temporal = TemporalEncoder::new(
            TemporalEncodingConfig {
                max_frames: video.max_frame_count,
                hidden_dim: encoder_config.vit_d_model,
                kind: video.temporal_encoding,
            },
            (video.temporal_encoding == TemporalEncodingKind::Learned).then_some(vb),
        )?;
        if video.temporal_encoding == TemporalEncodingKind::Learned {
            let path = video.temporal_path.as_ref().ok_or_else(|| {
                AarambhError::Config(
                    "learned video temporal encoding requires vision.video.temporal_path".into(),
                )
            })?;
            let mut varmap = varmap;
            varmap.load(path)?;
        }
        Some(temporal)
    } else {
        None
    };
    Ok(VisionRuntime {
        model: VisionModel::new(encoder, projector),
        preprocess,
        temporal,
        cache_salt,
    })
}

pub(super) fn load_document_runtime(
    run_config: &TrainingRunConfig,
    device: &candle_core::Device,
    dtype: candle_core::DType,
) -> anyhow::Result<DocumentRuntime> {
    let vision_config = run_config
        .vision
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--document requires a [vision] config block"))?;
    let document = vision_config
        .document
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--document requires a [vision.document] config block"))?;
    document.validate()?;
    let encoder_config = VisionEncoderConfig::from_json(&vision_config.clip_config_path)?;
    let vision = load_vision_runtime(run_config, device, dtype)?;
    let patch_side = encoder_config.image_size / encoder_config.patch_size;
    let layout_varmap = VarMap::new();
    let layout_vb = VarBuilder::from_varmap(&layout_varmap, dtype, device);
    let layout = LayoutAwareProjector::new(
        vision.model.projector().clone(),
        LayoutProjectorConfig {
            patch_rows: patch_side,
            patch_cols: patch_side,
            hidden_dim: run_config.model.hidden_dim,
            encoding: document.layout_encoding,
        },
        (document.layout_encoding == LayoutEncodingKind::Learned).then_some(layout_vb),
    )?;
    if document.layout_encoding == LayoutEncodingKind::Learned {
        let path = document.layout_path.as_ref().ok_or_else(|| {
            AarambhError::Config(
                "learned document layout encoding requires vision.document.layout_path".into(),
            )
        })?;
        let mut layout_varmap = layout_varmap;
        layout_varmap.load(path)?;
    }
    Ok(DocumentRuntime {
        vision,
        layout,
        rasterizer_config: PageRasterizerConfig {
            target_dpi: document.target_dpi,
            max_pages_per_document: document.max_pages_per_document,
            max_page_pixels: document.max_page_pixels,
        },
        encoder_page_batch_size: document.encoder_page_batch_size,
    })
}

fn ensure_image_prompt(prompt: &str) -> String {
    if prompt.contains(IMAGE) {
        prompt.to_string()
    } else {
        format!("{IMAGE}{IMAGE_END}\n{prompt}")
    }
}

fn build_vision_prompt_embeddings(
    engine: &InferenceEngine,
    runtime: &VisionRuntime,
    image_path: &Path,
    prompt: &str,
) -> aarambh_studio_core::Result<Tensor> {
    engine.tokenizer().validate_vision_special_tokens()?;
    let mut prompt_ids = engine.tokenizer().encode(prompt)?;
    if prompt_ids.is_empty() {
        if let Some(bos) = engine.tokenizer().bos_token_id() {
            prompt_ids.push(bos);
        } else {
            return Err(AarambhError::Config(
                "prompt produced no tokens and tokenizer has no BOS token".into(),
            ));
        }
    }
    let text = Tensor::from_vec(prompt_ids.clone(), (1, prompt_ids.len()), engine.device())?;
    let text_embeddings = engine.model().embed_tokens(&text)?;
    let image_tokens = project_image_tokens(runtime, image_path, engine.device())?;
    interleave_image_tokens(&prompt_ids, &text_embeddings, &image_tokens, IMAGE_ID)
}

pub(super) fn project_image_tokens(
    runtime: &VisionRuntime,
    image_path: &Path,
    device: &candle_core::Device,
) -> aarambh_studio_core::Result<Tensor> {
    let image = runtime
        .preprocess
        .preprocess_path(image_path, device)?
        .unsqueeze(0)?;
    runtime.model.forward(&image)
}

fn build_document_prompt_embeddings(
    engine: &InferenceEngine,
    page_tokens: &Tensor,
    prompt: &str,
) -> aarambh_studio_core::Result<Tensor> {
    engine.tokenizer().validate_document_special_tokens()?;
    let prompt_ids = engine.tokenizer().encode(prompt)?;
    if prompt_ids.is_empty() {
        return Err(AarambhError::Config(
            "document prompt produced no tokens".into(),
        ));
    }
    let text = Tensor::from_vec(prompt_ids.clone(), (1, prompt_ids.len()), engine.device())?;
    let text_embeddings = engine.model().embed_tokens(&text)?;
    interleave_document_tokens(
        &prompt_ids,
        &text_embeddings,
        page_tokens,
        DOCUMENT_ID,
        PAGE_SEP_ID,
    )
}

pub(super) fn project_document_tokens(
    runtime: &DocumentRuntime,
    document_path: &Path,
    selected_pages: Option<&[usize]>,
    device: &candle_core::Device,
    target_dpi: Option<u32>,
    max_pages: Option<usize>,
) -> aarambh_studio_core::Result<(Tensor, usize)> {
    let mut rasterizer_config = runtime.rasterizer_config.clone();
    if let Some(dpi) = target_dpi {
        rasterizer_config.target_dpi = dpi;
    }
    if let Some(max_pages) = max_pages {
        rasterizer_config.max_pages_per_document = max_pages;
    }
    let rasterizer = PageRasterizer::new(rasterizer_config)?;
    let rendered = rasterizer.rasterize(
        &DocumentSource::File(document_path.to_path_buf()),
        selected_pages,
    )?;
    if rendered.truncated {
        eprintln!(
            "warning: document has {} pages; using the first {}",
            rendered.source_page_count,
            rendered.pages.len()
        );
    }
    let pages = rendered
        .pages
        .into_iter()
        .map(|page| page.image)
        .collect::<Vec<_>>();
    let pixels = runtime
        .vision
        .preprocess
        .preprocess_document_pages(&pages, device)?;
    let mut chunks = Vec::new();
    for start in (0..pages.len()).step_by(runtime.encoder_page_batch_size) {
        let len = runtime.encoder_page_batch_size.min(pages.len() - start);
        chunks.push(
            runtime
                .vision
                .model
                .encoder()
                .forward(&pixels.narrow(0, start, len)?)?,
        );
    }
    let references = chunks.iter().collect::<Vec<_>>();
    let patch_tokens = Tensor::cat(&references, 0)?;
    let projected = runtime.layout.forward(
        &patch_tokens,
        (
            runtime.layout.config().patch_rows,
            runtime.layout.config().patch_cols,
        ),
    )?;
    Ok((projected, pages.len()))
}

pub(super) struct AudioRuntime {
    model: AudioModel,
    preprocess: AudioPreprocessor,
}

pub(super) fn load_audio_runtime(
    run_config: &TrainingRunConfig,
    device: &candle_core::Device,
    dtype: candle_core::DType,
) -> anyhow::Result<AudioRuntime> {
    let vision = run_config
        .vision
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--audio requires a [vision] config block"))?;
    let audio = vision
        .audio
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--audio requires a [vision.audio] config block"))?;
    audio.validate()?;
    let encoder_config = AudioEncoderConfig::from_json(&audio.encoder_config_path)?;
    let encoder = FrozenAudioEncoder::load_pretrained(
        &audio.encoder_weights_path,
        encoder_config.clone(),
        device,
        dtype,
    )?;
    let projector_path = match &vision.projector_path {
        Some(path) => path.clone(),
        None => default_model_path(&run_config.train.checkpoint_dir)?,
    };
    let projector_config = AudioProjectorConfig {
        audio_d_model: encoder_config.audio_d_model,
        llm_d_model: run_config.model.hidden_dim,
        hidden_mult: vision.projector_hidden_mult,
    };
    let projector =
        AudioProjector::load_safetensors(&projector_path, projector_config, device, dtype)?;
    let preprocess = AudioPreprocessor::new(audio.mel.clone())?;
    Ok(AudioRuntime {
        model: AudioModel::new(encoder, projector),
        preprocess,
    })
}

fn ensure_audio_prompt(prompt: &str) -> String {
    if prompt.contains(AUDIO) {
        prompt.to_string()
    } else {
        format!("{AUDIO}{AUDIO_END}\n{prompt}")
    }
}

fn build_audio_prompt_embeddings(
    engine: &InferenceEngine,
    runtime: &AudioRuntime,
    audio_path: &Path,
    prompt: &str,
) -> aarambh_studio_core::Result<Tensor> {
    engine.tokenizer().validate_audio_special_tokens()?;
    let mut prompt_ids = engine.tokenizer().encode(prompt)?;
    if prompt_ids.is_empty() {
        if let Some(bos) = engine.tokenizer().bos_token_id() {
            prompt_ids.push(bos);
        } else {
            return Err(AarambhError::Config(
                "prompt produced no tokens and tokenizer has no BOS token".into(),
            ));
        }
    }
    let text = Tensor::from_vec(prompt_ids.clone(), (1, prompt_ids.len()), engine.device())?;
    let text_embeddings = engine.model().embed_tokens(&text)?;
    let audio_tokens = project_audio_tokens(runtime, audio_path, engine.device())?;
    interleave_audio_tokens(&prompt_ids, &text_embeddings, &audio_tokens, AUDIO_ID)
}

pub(super) fn project_audio_tokens(
    runtime: &AudioRuntime,
    audio_path: &Path,
    device: &candle_core::Device,
) -> aarambh_studio_core::Result<Tensor> {
    let spectrogram = runtime
        .preprocess
        .preprocess_path(audio_path, device)?
        .unsqueeze(0)?;
    runtime.model.forward(&spectrogram)
}

#[allow(clippy::too_many_arguments)]
fn run_audio_infer(
    args: &InferArgs,
    run_config: &TrainingRunConfig,
    mut engine: InferenceEngine,
    audio_path: PathBuf,
    dtype: candle_core::DType,
    config: GenerationConfig,
    prompt: String,
    safety_mode: SafetyMode,
    thinking_mode: ThinkingMode,
    tokenizer_for_view: BpeTokenizer,
) -> anyhow::Result<()> {
    let runtime = load_audio_runtime(run_config, engine.device(), dtype)?;
    let prompt = ensure_audio_prompt(&prompt);

    if let Some(policy) = SafetyPolicy::for_mode(safety_mode)
        .map(|policy| policy.with_audit_path(&args.safety_audit_log))
    {
        let adapter = AudioSafetyAdapter {
            engine,
            runtime,
            audio_path,
        };
        let mut guard = SafetyGuard::new(adapter, policy);
        let mut stream_state = StreamState::default();
        let response = if args.stream {
            guard.generate_streaming_with_callback(&prompt, config, print_safe_stream_event)?
        } else {
            guard.generate_with_callback(&prompt, config, |step| {
                if args.predict_view {
                    print!(
                        "{}",
                        predict_view::render(
                            step,
                            &tokenizer_for_view,
                            args.temperature,
                            args.top_p,
                        )
                    );
                    io::stdout().flush()?;
                }
                Ok(())
            })?
        };
        print_safe_response(&response, thinking_mode, args.stream, &mut stream_state)?;
        io::stdout().flush()?;
        if let Some(output) = &response.output {
            eprintln!("finish_reason={:?}", output.finish_reason);
        } else {
            eprintln!("finish_reason=SafetyBlocked");
        }
        return Ok(());
    }

    let embeddings = build_audio_prompt_embeddings(&engine, &runtime, &audio_path, &prompt)?;
    let mut stream_state = StreamState::default();
    let output = engine.generate_with_embeddings_callback(&embeddings, config, |step| {
        if args.predict_view {
            print!(
                "{}",
                predict_view::render(step, &tokenizer_for_view, args.temperature, args.top_p)
            );
        }
        if args.stream {
            stream_step(step, thinking_mode, &mut stream_state)?;
        }
        if args.predict_view || args.stream {
            io::stdout().flush()?;
        }
        Ok(())
    })?;
    if args.stream {
        finish_stream(&mut stream_state);
    } else {
        print_generation_output(&output, thinking_mode)?;
    }
    io::stdout().flush()?;
    eprintln!("finish_reason={:?}", output.finish_reason);
    Ok(())
}

struct AudioSafetyAdapter {
    engine: InferenceEngine,
    runtime: AudioRuntime,
    audio_path: PathBuf,
}

impl SafetyGenerator for AudioSafetyAdapter {
    fn generate(
        &mut self,
        prompt: &str,
        config: GenerationConfig,
    ) -> aarambh_studio_core::Result<GenerationOutput> {
        self.generate_with_callback(prompt, config, |_| Ok(()))
    }

    fn generate_with_callback<F>(
        &mut self,
        prompt: &str,
        config: GenerationConfig,
        on_step: F,
    ) -> aarambh_studio_core::Result<GenerationOutput>
    where
        F: FnMut(&GenerationStep) -> aarambh_studio_core::Result<()>,
    {
        let embeddings =
            build_audio_prompt_embeddings(&self.engine, &self.runtime, &self.audio_path, prompt)?;
        self.engine
            .generate_with_embeddings_callback(&embeddings, config, on_step)
    }
}

fn build_video_prompt_embeddings(
    engine: &InferenceEngine,
    runtime: &VisionRuntime,
    video_path: &Path,
    prompt: &str,
    sampling: &VideoSamplingConfig,
    encoder_batch_size: usize,
) -> aarambh_studio_core::Result<Tensor> {
    engine.tokenizer().validate_video_special_tokens()?;
    let prompt_ids = engine.tokenizer().encode(prompt)?;
    if prompt_ids.is_empty() {
        return Err(AarambhError::Config(
            "video prompt produced no tokens".into(),
        ));
    }
    let text = Tensor::from_vec(prompt_ids.clone(), (1, prompt_ids.len()), engine.device())?;
    let text_embeddings = engine.model().embed_tokens(&text)?;
    let frame_tokens = project_video_tokens(
        runtime,
        video_path,
        engine.device(),
        sampling,
        encoder_batch_size,
    )?;
    interleave_video_tokens(
        &prompt_ids,
        &text_embeddings,
        &frame_tokens,
        VIDEO_ID,
        FRAME_SEP_ID,
    )
}

pub(super) fn project_video_tokens(
    runtime: &VisionRuntime,
    video_path: &Path,
    device: &candle_core::Device,
    sampling: &VideoSamplingConfig,
    encoder_batch_size: usize,
) -> aarambh_studio_core::Result<Tensor> {
    if encoder_batch_size == 0 {
        return Err(AarambhError::Config(
            "vision.video.encoder_frame_batch_size must be non-zero".into(),
        ));
    }
    let sampled = decode_sampled_video(video_path, sampling)?;
    let pixels = runtime
        .preprocess
        .preprocess_rgb_batch(&sampled.frames, device)?;
    let mut encoded = Vec::new();
    for start in (0..sampled.frames.len()).step_by(encoder_batch_size) {
        let len = encoder_batch_size.min(sampled.frames.len() - start);
        encoded.push(
            runtime
                .model
                .encoder()
                .forward(&pixels.narrow(0, start, len)?)?,
        );
    }
    let references = encoded.iter().collect::<Vec<_>>();
    let patch_tokens = Tensor::cat(&references, 0)?;
    let temporal = runtime.temporal.as_ref().ok_or_else(|| {
        AarambhError::Config("video inference requires a temporal encoder".into())
    })?;
    runtime
        .model
        .projector()
        .forward(&temporal.forward(&patch_tokens)?)
}

struct VisionSafetyAdapter {
    engine: InferenceEngine,
    runtime: VisionRuntime,
    image_path: PathBuf,
}

struct VideoSafetyAdapter {
    engine: InferenceEngine,
    runtime: VisionRuntime,
    video_path: PathBuf,
    sampling: VideoSamplingConfig,
    encoder_batch_size: usize,
}

struct DocumentSafetyAdapter {
    engine: InferenceEngine,
    page_tokens: Tensor,
}

impl SafetyGenerator for DocumentSafetyAdapter {
    fn generate(
        &mut self,
        prompt: &str,
        config: GenerationConfig,
    ) -> aarambh_studio_core::Result<GenerationOutput> {
        self.generate_with_callback(prompt, config, |_| Ok(()))
    }

    fn generate_with_callback<F>(
        &mut self,
        prompt: &str,
        config: GenerationConfig,
        on_step: F,
    ) -> aarambh_studio_core::Result<GenerationOutput>
    where
        F: FnMut(&GenerationStep) -> aarambh_studio_core::Result<()>,
    {
        let embeddings = build_document_prompt_embeddings(&self.engine, &self.page_tokens, prompt)?;
        self.engine
            .generate_with_embeddings_callback(&embeddings, config, on_step)
    }
}

impl SafetyGenerator for VideoSafetyAdapter {
    fn generate(
        &mut self,
        prompt: &str,
        config: GenerationConfig,
    ) -> aarambh_studio_core::Result<GenerationOutput> {
        self.generate_with_callback(prompt, config, |_| Ok(()))
    }

    fn generate_with_callback<F>(
        &mut self,
        prompt: &str,
        config: GenerationConfig,
        on_step: F,
    ) -> aarambh_studio_core::Result<GenerationOutput>
    where
        F: FnMut(&GenerationStep) -> aarambh_studio_core::Result<()>,
    {
        let embeddings = build_video_prompt_embeddings(
            &self.engine,
            &self.runtime,
            &self.video_path,
            prompt,
            &self.sampling,
            self.encoder_batch_size,
        )?;
        self.engine
            .generate_with_embeddings_callback(&embeddings, config, on_step)
    }
}

impl SafetyGenerator for VisionSafetyAdapter {
    fn generate(
        &mut self,
        prompt: &str,
        config: GenerationConfig,
    ) -> aarambh_studio_core::Result<GenerationOutput> {
        self.generate_with_callback(prompt, config, |_| Ok(()))
    }

    fn generate_with_callback<F>(
        &mut self,
        prompt: &str,
        config: GenerationConfig,
        on_step: F,
    ) -> aarambh_studio_core::Result<GenerationOutput>
    where
        F: FnMut(&GenerationStep) -> aarambh_studio_core::Result<()>,
    {
        let embeddings =
            build_vision_prompt_embeddings(&self.engine, &self.runtime, &self.image_path, prompt)?;
        self.engine
            .generate_with_embeddings_callback(&embeddings, config, on_step)
    }
}

fn tokenizer_path(args: &InferArgs, run_config: &TrainingRunConfig) -> PathBuf {
    args.tokenizer
        .clone()
        .or_else(|| run_config.tokenizer_path.clone())
        .or_else(|| run_config.tokenizer_save_path.clone())
        .unwrap_or_else(|| run_config.train.checkpoint_dir.join("tokenizer.json"))
}

pub(super) fn default_model_path(checkpoint_dir: &Path) -> anyhow::Result<PathBuf> {
    for pointer_name in ["latest.json", "best.json"] {
        let pointer_path = checkpoint_dir.join(pointer_name);
        if pointer_path.exists() {
            let file = fs::File::open(&pointer_path)?;
            let pointer: CheckpointPointer = serde_json::from_reader(file)?;
            return Ok(pointer.path.join("model.safetensors"));
        }
    }
    Err(anyhow::anyhow!(
        "no model provided and no latest.json or best.json found in {}",
        checkpoint_dir.display()
    ))
}

pub(super) fn parse_thinking_mode(value: &str) -> anyhow::Result<ThinkingMode> {
    use std::str::FromStr;
    ThinkingMode::from_str(value).map_err(anyhow::Error::msg)
}

pub(super) fn parse_safety_mode(value: &str) -> anyhow::Result<SafetyMode> {
    value.parse::<SafetyMode>().map_err(anyhow::Error::msg)
}

fn parse_self_learn_mode(value: &str) -> anyhow::Result<SelfLearnMode> {
    value.parse::<SelfLearnMode>().map_err(anyhow::Error::msg)
}

fn parse_self_learn_verifier(value: &str) -> anyhow::Result<Option<VerifierKind>> {
    match value.trim().to_ascii_lowercase().as_str() {
        "none" | "disabled" | "off" => Ok(None),
        other => other
            .parse::<VerifierKind>()
            .map(Some)
            .map_err(anyhow::Error::msg),
    }
}

fn parse_vision_verifier(value: &str) -> anyhow::Result<VisionVerifierKind> {
    value
        .parse::<VisionVerifierKind>()
        .map_err(anyhow::Error::msg)
}

fn prompt_for_mode(prompt: &str, thinking_mode: ThinkingMode) -> String {
    if thinking_mode.is_enabled() {
        format!("{USER}\n{prompt}\n{ASSISTANT}\n")
    } else {
        prompt.to_string()
    }
}

fn attach_forgetting_config(args: &InferArgs, config: SelfLearnConfig) -> SelfLearnConfig {
    let Some(manifest) = args.forgetting_manifest.clone() else {
        return config;
    };
    let forgetting = SelfLearnForgettingConfig {
        enabled: true,
        manifest,
        config_path: Some(args.config.clone()),
        store: args
            .forgetting_store
            .clone()
            .unwrap_or_else(|| args.self_learn_state_dir.join("forgetting_curves.json")),
        jsonl: args.forgetting_jsonl.clone(),
        max_examples: Some(args.forgetting_max_examples),
        significance_threshold: args.forgetting_threshold,
        allow_code_exec: args.forgetting_allow_code_exec,
        require_all_probes: args.forgetting_require_all_probes,
        baseline_id: args.forgetting_baseline_id.clone(),
        ..SelfLearnForgettingConfig::default()
    };
    config.with_forgetting(forgetting)
}

#[allow(clippy::too_many_arguments)]
fn run_self_learn_infer(
    args: &InferArgs,
    mut run_config: TrainingRunConfig,
    model_path: PathBuf,
    tokenizer_path: PathBuf,
    device: candle_core::Device,
    dtype: candle_core::DType,
    config: GenerationConfig,
    prompt: String,
    safety_mode: SafetyMode,
    self_learn_mode: SelfLearnMode,
    thinking_mode: ThinkingMode,
) -> anyhow::Result<()> {
    let replay_path = args
        .replay_path
        .clone()
        .unwrap_or_else(|| PathBuf::from("data/replay.jsonl"));
    let mut self_config = SelfLearnConfig::for_mode(self_learn_mode)
        .with_replay_path(replay_path)
        .with_state_dir(args.self_learn_state_dir.clone());
    self_config.grpo.max_new_tokens = args.max_tokens;
    self_config.critique.rewrite_max_tokens = self_config
        .critique
        .rewrite_max_tokens
        .min(args.max_tokens)
        .max(1);
    self_config = attach_forgetting_config(args, self_config);
    let reference_path = args
        .self_learn_reference
        .clone()
        .unwrap_or_else(|| model_path.clone());
    let verifier = parse_self_learn_verifier(&args.self_learn_verifier)?.map(VerifierKind::build);
    if verifier.is_some() && args.self_learn_ground_truth.is_none() {
        eprintln!(
            "[self-learn] deterministic verifier requested without --self-learn-ground-truth; online GRPO will be skipped"
        );
    }

    let tokenizer_for_view = BpeTokenizer::from_pretrained(&tokenizer_path)?;
    let loop_ = SelfLearnLoop::from_paths(SelfLearnBuildConfig {
        model_config: {
            run_config.model.vocab_size = tokenizer_for_view.vocab_size();
            run_config.model.clone()
        },
        base_model_path: model_path,
        reference_model_path: reference_path,
        tokenizer_path,
        config: self_config,
        device,
        dtype,
        seed: run_config.train.seed,
    })?;
    let mut adapter = SelfLearnSafetyAdapter {
        loop_,
        verifier,
        ground_truth: args.self_learn_ground_truth.clone(),
    };

    if let Some(policy) = SafetyPolicy::for_mode(safety_mode)
        .map(|policy| policy.with_audit_path(&args.safety_audit_log))
    {
        let mut guard = SafetyGuard::new(adapter, policy);
        let mut stream_state = StreamState::default();
        let response = if args.stream {
            guard.generate_streaming_with_callback(&prompt, config, print_safe_stream_event)?
        } else {
            guard.generate_with_callback(&prompt, config, |step| {
                if args.predict_view {
                    print!(
                        "{}",
                        predict_view::render(
                            step,
                            &tokenizer_for_view,
                            args.temperature,
                            args.top_p,
                        )
                    );
                    io::stdout().flush()?;
                }
                Ok(())
            })?
        };
        print_safe_response(&response, thinking_mode, args.stream, &mut stream_state)?;
        let mut adapter = guard.into_inner();
        if response.is_blocked() {
            adapter.loop_.discard_last_draft();
        } else {
            let learned = adapter
                .loop_
                .commit_last_draft(Some(response.text.clone()))?;
            print_self_learn_summary(&learned);
        }
        io::stdout().flush()?;
        if let Some(output) = &response.output {
            eprintln!("finish_reason={:?}", output.finish_reason);
        } else {
            eprintln!("finish_reason=SafetyBlocked");
        }
        return Ok(());
    }

    let mut stream_state = StreamState::default();
    let output = adapter.generate_with_callback(&prompt, config, |step| {
        if args.predict_view {
            print!(
                "{}",
                predict_view::render(step, &tokenizer_for_view, args.temperature, args.top_p)
            );
        }
        if args.stream {
            stream_step(step, thinking_mode, &mut stream_state)?;
        }
        if args.predict_view || args.stream {
            io::stdout().flush()?;
        }
        Ok(())
    })?;
    if args.stream {
        finish_stream(&mut stream_state);
    } else {
        print_generation_output(&output, thinking_mode)?;
    }
    let learned = adapter.loop_.commit_last_draft(Some(output.text.clone()))?;
    print_self_learn_summary(&learned);
    io::stdout().flush()?;
    eprintln!("finish_reason={:?}", output.finish_reason);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_vision_self_learn_infer(
    args: &InferArgs,
    mut run_config: TrainingRunConfig,
    run_device: aarambh_studio_core::Device,
    model_path: PathBuf,
    tokenizer_path: PathBuf,
    image_path: PathBuf,
    device: candle_core::Device,
    dtype: candle_core::DType,
    config: GenerationConfig,
    prompt: String,
    safety_mode: SafetyMode,
    self_learn_mode: SelfLearnMode,
    thinking_mode: ThinkingMode,
) -> anyhow::Result<()> {
    require_vision_hardware(&run_device)?;
    if self_learn_mode != SelfLearnMode::Gpu {
        return Err(anyhow::anyhow!(
            "vision self-learning requires --self-learn gpu"
        ));
    }
    let replay_path = args
        .replay_path
        .clone()
        .unwrap_or_else(|| PathBuf::from("data/replay_buffer_v2.jsonl"));
    let mut self_config = SelfLearnConfig::for_mode(self_learn_mode)
        .with_replay_path(replay_path)
        .with_state_dir(args.self_learn_state_dir.clone());
    self_config.grpo.max_new_tokens = args.max_tokens;
    self_config.critique.rewrite_max_tokens = self_config
        .critique
        .rewrite_max_tokens
        .min(args.max_tokens)
        .max(1);
    self_config = attach_forgetting_config(args, self_config);

    let runtime = load_vision_runtime(&run_config, &device, dtype)?;
    let prompt = ensure_image_prompt(&prompt);
    let cache = VisionCache::new(&args.self_learn_state_dir);
    let image_ref = cache.image_ref(&image_path, &runtime.cache_salt)?;
    let image_tokens = match cache.load_projected_tokens(&image_ref, &device)? {
        Some(tokens) => tokens,
        None => {
            let tokens = project_image_tokens(&runtime, &image_path, &device)?;
            cache.save_projected_tokens(&image_ref, &tokens)?;
            tokens
        }
    };

    let reference_path = args
        .self_learn_reference
        .clone()
        .unwrap_or_else(|| model_path.clone());
    let vision_verifier_kind =
        parse_vision_verifier(&args.self_learn_vision_verifier)?.resolve_for_prompt(&prompt);
    let verifier = if vision_verifier_kind == VisionVerifierKind::None {
        None
    } else {
        if args.self_learn_ground_truth.is_none() {
            eprintln!(
                "[self-learn] vision verifier requested without --self-learn-ground-truth; grounded vision GRPO will be skipped"
            );
        }
        vision_verifier_kind
            .build()
            .map(|verifier| Box::new(verifier) as Box<dyn Verifier>)
    };

    let tokenizer_for_view = BpeTokenizer::from_pretrained(&tokenizer_path)?;
    let loop_ = SelfLearnLoop::from_paths(SelfLearnBuildConfig {
        model_config: {
            run_config.model.vocab_size = tokenizer_for_view.vocab_size();
            run_config.model.clone()
        },
        base_model_path: model_path,
        reference_model_path: reference_path,
        tokenizer_path,
        config: self_config,
        device,
        dtype,
        seed: run_config.train.seed,
    })?;
    let mut adapter = VisionSelfLearnSafetyAdapter {
        loop_,
        image_tokens,
        image_ref,
        verifier,
        ground_truth: args.self_learn_ground_truth.clone(),
    };

    if let Some(policy) = SafetyPolicy::for_mode(safety_mode)
        .map(|policy| policy.with_audit_path(&args.safety_audit_log))
    {
        let mut guard = SafetyGuard::new(adapter, policy);
        let mut stream_state = StreamState::default();
        let response = if args.stream {
            guard.generate_streaming_with_callback(&prompt, config, print_safe_stream_event)?
        } else {
            guard.generate_with_callback(&prompt, config, |step| {
                if args.predict_view {
                    print!(
                        "{}",
                        predict_view::render(
                            step,
                            &tokenizer_for_view,
                            args.temperature,
                            args.top_p,
                        )
                    );
                    io::stdout().flush()?;
                }
                Ok(())
            })?
        };
        print_safe_response(&response, thinking_mode, args.stream, &mut stream_state)?;
        let mut adapter = guard.into_inner();
        if response.is_blocked() {
            adapter.loop_.discard_last_draft();
        } else {
            let learned = adapter
                .loop_
                .commit_last_draft(Some(response.text.clone()))?;
            print_self_learn_summary(&learned);
        }
        io::stdout().flush()?;
        if let Some(output) = &response.output {
            eprintln!("finish_reason={:?}", output.finish_reason);
        } else {
            eprintln!("finish_reason=SafetyBlocked");
        }
        return Ok(());
    }

    let mut stream_state = StreamState::default();
    let output = adapter.generate_with_callback(&prompt, config, |step| {
        if args.predict_view {
            print!(
                "{}",
                predict_view::render(step, &tokenizer_for_view, args.temperature, args.top_p)
            );
        }
        if args.stream {
            stream_step(step, thinking_mode, &mut stream_state)?;
        }
        if args.predict_view || args.stream {
            io::stdout().flush()?;
        }
        Ok(())
    })?;
    if args.stream {
        finish_stream(&mut stream_state);
    } else {
        print_generation_output(&output, thinking_mode)?;
    }
    let learned = adapter.loop_.commit_last_draft(Some(output.text.clone()))?;
    print_self_learn_summary(&learned);
    io::stdout().flush()?;
    eprintln!("finish_reason={:?}", output.finish_reason);
    Ok(())
}

struct SelfLearnSafetyAdapter {
    loop_: SelfLearnLoop,
    verifier: Option<Box<dyn Verifier>>,
    ground_truth: Option<String>,
}

impl SafetyGenerator for SelfLearnSafetyAdapter {
    fn generate(
        &mut self,
        prompt: &str,
        config: GenerationConfig,
    ) -> aarambh_studio_core::Result<GenerationOutput> {
        self.generate_with_callback(prompt, config, |_| Ok(()))
    }

    fn generate_with_callback<F>(
        &mut self,
        prompt: &str,
        config: GenerationConfig,
        on_step: F,
    ) -> aarambh_studio_core::Result<GenerationOutput>
    where
        F: FnMut(&GenerationStep) -> aarambh_studio_core::Result<()>,
    {
        self.loop_.generate_draft_with_callback(
            prompt,
            config,
            self.verifier.as_deref(),
            self.ground_truth.as_deref(),
            on_step,
        )
    }
}

struct VisionSelfLearnSafetyAdapter {
    loop_: SelfLearnLoop,
    image_tokens: Tensor,
    image_ref: PathBuf,
    verifier: Option<Box<dyn Verifier>>,
    ground_truth: Option<String>,
}

impl SafetyGenerator for VisionSelfLearnSafetyAdapter {
    fn generate(
        &mut self,
        prompt: &str,
        config: GenerationConfig,
    ) -> aarambh_studio_core::Result<GenerationOutput> {
        self.generate_with_callback(prompt, config, |_| Ok(()))
    }

    fn generate_with_callback<F>(
        &mut self,
        prompt: &str,
        config: GenerationConfig,
        on_step: F,
    ) -> aarambh_studio_core::Result<GenerationOutput>
    where
        F: FnMut(&GenerationStep) -> aarambh_studio_core::Result<()>,
    {
        self.loop_.generate_vision_draft_with_callback(
            prompt,
            &self.image_tokens,
            self.image_ref.clone(),
            config,
            self.verifier.as_deref(),
            self.ground_truth.as_deref(),
            on_step,
        )
    }
}

fn print_self_learn_summary(response: &aarambh_studio_selflearn::SelfLearnResponse) {
    eprintln!(
        "[self-learn] critique_score={:.2} stored={} rewritten={} grpo={} image_ref={} metrics={}",
        response.critique_score,
        response.stored_in_replay,
        response.was_rewritten,
        response.used_grpo,
        response
            .image_ref
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "-".into()),
        response.metrics_summary
    );
    if let Some(forgetting) = &response.forgetting {
        eprintln!(
            "[forgetting] baseline={} current={} capabilities={} skipped={}",
            forgetting.baseline_id,
            forgetting.current_id,
            forgetting.deltas.len(),
            forgetting.skipped.len()
        );
    }
}

#[derive(Default)]
struct StreamState {
    dim_active: bool,
    header_printed: bool,
    thinking_tokens: usize,
    tool_buffer: String,
}

fn stream_step(
    step: &GenerationStep,
    _thinking_mode: ThinkingMode,
    state: &mut StreamState,
) -> io::Result<()> {
    match step.phase {
        GenerationPhase::Thinking => {
            if !state.header_printed {
                print!("[thinking]\n{ANSI_DIM}");
                state.header_printed = true;
                state.dim_active = true;
            }
            if !is_thinking_marker(step.token_id) {
                state.thinking_tokens += 1;
                print!("{}", step.token_text);
            }
        }
        GenerationPhase::Answer => {
            if state.dim_active {
                println!("{ANSI_RESET}");
                println!("[thinking: {} tokens]", state.thinking_tokens);
                state.dim_active = false;
            }
            print!("{}", step.token_text);
        }
        GenerationPhase::ToolCall => state.tool_buffer.push_str(&step.token_text),
        GenerationPhase::Control => {}
    }
    Ok(())
}

fn finish_stream(state: &mut StreamState) {
    if state.dim_active {
        println!("{ANSI_RESET}");
        println!("[thinking: {} tokens]", state.thinking_tokens);
        state.dim_active = false;
    }
    if !state.tool_buffer.is_empty() {
        print!("{}", state.tool_buffer);
        state.tool_buffer.clear();
    }
    println!();
}

fn print_safe_stream_event(event: SafeStreamEvent) -> aarambh_studio_core::Result<()> {
    if let SafeStreamEvent::Text(text) = event {
        print!("{text}");
        io::stdout().flush()?;
    }
    Ok(())
}

fn print_safe_response(
    response: &SafeResponse,
    thinking_mode: ThinkingMode,
    stream: bool,
    stream_state: &mut StreamState,
) -> io::Result<()> {
    if stream {
        finish_stream(stream_state);
        if let SafetyVerdict::Block(reason) = &response.verdict {
            println!("blocked by safety: {reason}");
        }
        return Ok(());
    }
    if let SafetyVerdict::Block(reason) = &response.verdict {
        println!("blocked by safety: {reason}");
        return Ok(());
    }

    let Some(output) = &response.output else {
        println!("blocked by safety");
        return Ok(());
    };

    print_generation_output(output, thinking_mode)?;
    Ok(())
}

fn print_generation_output(
    output: &GenerationOutput,
    thinking_mode: ThinkingMode,
) -> io::Result<()> {
    if !thinking_mode.is_enabled() {
        println!("{}", output.text);
        return Ok(());
    }

    println!("[thinking: {} tokens]", output.thinking_tokens);
    if !output.thinking_text.is_empty() {
        println!("{ANSI_DIM}{}{ANSI_RESET}", output.thinking_text);
    }
    println!("{}", output.text);
    Ok(())
}

fn is_thinking_marker(token_id: u32) -> bool {
    token_id == THINK_START_ID || token_id == THINK_END_ID
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> InferArgs {
        InferArgs {
            config: "target.toml".into(),
            model: Some("target.safetensors".into()),
            tokenizer: Some("tokenizer.json".into()),
            image: None,
            video: None,
            document: None,
            audio: None,
            pages: None,
            document_dpi: None,
            max_document_pages: None,
            frames: None,
            frame_sampling: None,
            prompt: "test".into(),
            system: None,
            max_tokens: 8,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            seed: Some(42),
            thinking: "none".into(),
            predict_view: false,
            stream: false,
            greedy: true,
            speculative: true,
            draft_model: Some("draft.safetensors".into()),
            draft_config: Some("draft.toml".into()),
            draft_tokenizer: None,
            draft_tokens: Some(4),
            stats: false,
            tools: None,
            tool_choice: "auto".into(),
            safety: "none".into(),
            safety_audit_log: "safety.jsonl".into(),
            self_learn: "disabled".into(),
            replay_path: None,
            self_learn_state_dir: "adapters/selflearn".into(),
            self_learn_reference: None,
            self_learn_verifier: "none".into(),
            self_learn_vision_verifier: "none".into(),
            self_learn_ground_truth: None,
            forgetting_manifest: None,
            forgetting_store: None,
            forgetting_jsonl: None,
            forgetting_threshold: 0.02,
            forgetting_max_examples: 8,
            forgetting_allow_code_exec: false,
            forgetting_require_all_probes: false,
            forgetting_baseline_id: None,
            best_of_n: None,
            selection: "self-consistency".into(),
            ground_truth: None,
            rag: false,
            index: None,
            rag_top_k: 4,
        }
    }

    #[test]
    fn speculative_cli_requires_draft_model_and_config() {
        let mut args = args();
        let model = aarambh_studio_core::ModelConfig::tiny();
        args.draft_model = None;
        assert!(validate_speculative_args(&args, SelfLearnMode::Disabled, &model).is_err());
        args.draft_model = Some("draft.safetensors".into());
        args.draft_config = None;
        assert!(validate_speculative_args(&args, SelfLearnMode::Disabled, &model).is_err());
    }

    #[test]
    fn speculative_cli_rejects_unsupported_modes() {
        let mut args = args();
        let model = aarambh_studio_core::ModelConfig::tiny();
        args.image = Some("image.png".into());
        assert!(validate_speculative_args(&args, SelfLearnMode::Disabled, &model).is_err());
        args.image = None;
        assert!(validate_speculative_args(&args, SelfLearnMode::Cpu, &model).is_err());
    }

    #[test]
    fn document_page_selection_is_one_based() {
        assert_eq!(parse_page_selection("3, 1").unwrap(), vec![3, 1]);
        assert!(parse_page_selection("0").is_err());
        assert!(parse_page_selection("").is_err());
    }

    #[test]
    fn draft_options_require_speculative_flag() {
        let mut args = args();
        let model = aarambh_studio_core::ModelConfig::tiny();
        args.speculative = false;
        assert!(validate_speculative_args(&args, SelfLearnMode::Disabled, &model).is_err());
        args.draft_model = None;
        args.draft_config = None;
        args.draft_tokens = None;
        assert!(validate_speculative_args(&args, SelfLearnMode::Disabled, &model).is_ok());
    }

    #[test]
    fn internal_speculation_requires_mtp_and_valid_width() {
        let mut args = args();
        args.draft_model = None;
        args.draft_config = None;
        args.draft_tokens = None;
        let mut model = aarambh_studio_core::ModelConfig::tiny();
        assert!(validate_speculative_args(&args, SelfLearnMode::Disabled, &model).is_err());

        model.mtp = Some(aarambh_studio_core::MtpConfig {
            num_future_tokens: 3,
            aux_loss_weight: 0.3,
        });
        assert!(validate_speculative_args(&args, SelfLearnMode::Disabled, &model).is_ok());
        args.draft_tokens = Some(4);
        assert!(validate_speculative_args(&args, SelfLearnMode::Disabled, &model).is_err());
    }
}
