use std::fs;
use std::io::{self, BufReader};
use std::path::PathBuf;

use aarambh_studio_agent::{
    AgentError, AgentResult, AuthorizationScope, ChainDecoder, ChainEvent, EvictionPolicy,
    ReadFileInWorkdir, ReplayResultProvider, SandboxLimits, SandboxedToolProvider,
    StdinResultProvider, ToolChain, ToolChainConfig, ToolExchange, ToolResult, ToolResultContent,
    ToolResultProvider, ToolResultRequest, ToolSandbox,
};
use aarambh_studio_core::{AarambhError, Configurable, TokenizerLike};
use aarambh_studio_inference::{
    GenerationConfig, GenerationOutput, InferenceEngine, Sampler, ThinkingMode, ToolCallingConfig,
    ToolChoice, ToolDefinition,
};
use aarambh_studio_safety::{SafetyInspector, SafetyPolicy, SafetyVerdict};
use aarambh_studio_tokenizer::{
    ASSISTANT, DOCUMENT, DOCUMENT_END, DOCUMENT_ID, FRAME_SEP, FRAME_SEP_ID, IMAGE, IMAGE_END,
    IMAGE_ID, PAGE_SEP, PAGE_SEP_ID, USER, VIDEO, VIDEO_END, VIDEO_ID,
};
use aarambh_studio_train::TrainingRunConfig;
use aarambh_studio_vision::{
    VideoSamplingConfig, interleave_document_tokens, interleave_image_tokens,
    interleave_video_tokens,
};
use candle_core::{DType, Tensor};
use clap::Args;
use serde_json::json;

use super::infer::{
    DocumentRuntime, VisionRuntime, default_model_path, load_document_runtime,
    load_tool_definitions, load_vision_runtime, parse_safety_mode, parse_thinking_mode,
    project_document_tokens, project_image_tokens, project_video_tokens,
};

#[derive(Debug, Args)]
/// Run a bounded caller-executed long-horizon tool-use chain.
pub struct AgentArgs {
    /// Training/model configuration.
    #[arg(long, default_value = "configs/tiny_shakespeare.toml")]
    pub config: PathBuf,
    /// Model checkpoint; defaults to the configured latest/best pointer.
    #[arg(long)]
    pub model: Option<PathBuf>,
    /// Tokenizer JSON; defaults to the configured tokenizer path.
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,
    /// Native or OpenAI-compatible JSON tool definitions.
    #[arg(long)]
    pub tools: PathBuf,
    /// Initial user request.
    #[arg(long)]
    pub prompt: String,
    /// Maximum caller-executed tool calls.
    #[arg(long, default_value_t = 8)]
    pub max_steps: usize,
    /// Maximum generated tokens per model decision.
    #[arg(long, default_value_t = 256)]
    pub max_tokens: usize,
    /// Sampling temperature.
    #[arg(long, default_value_t = 0.7)]
    pub temperature: f32,
    /// Nucleus sampling probability.
    #[arg(long, default_value_t = 0.9)]
    pub top_p: f32,
    /// Top-k sampling width.
    #[arg(long, default_value_t = 50)]
    pub top_k: usize,
    /// Deterministic sampler seed.
    #[arg(long)]
    pub seed: Option<u64>,
    /// Use greedy decoding.
    #[arg(long)]
    pub greedy: bool,
    /// Thinking budget: none, low, medium, high, or max.
    #[arg(long, default_value = "none")]
    pub thinking: String,
    /// Scripted JSONL tool results; stdin JSONL is used when omitted.
    #[arg(long)]
    pub results: Option<PathBuf>,
    /// Root that all image, video, and document result paths must stay under.
    #[arg(long, default_value = ".")]
    pub result_root: PathBuf,
    /// Context policy: drop-oldest or summarise.
    #[arg(long, default_value = "drop-oldest")]
    pub eviction: String,
    /// Recent completed exchanges protected from eviction.
    #[arg(long, default_value_t = 4)]
    pub keep_recent: usize,
    /// Maximum model-generated summary tokens per eviction.
    #[arg(long, default_value_t = 128)]
    pub summary_tokens: usize,
    /// Emit machine-readable lifecycle events on stdout.
    #[arg(long)]
    pub jsonl: bool,
    /// Safety mode applied to prompts, results, calls, and final output.
    #[arg(long, default_value = "strict")]
    pub safety: String,
    /// JSONL safety audit path.
    #[arg(long, default_value = "safety_audit.jsonl")]
    pub safety_audit_log: PathBuf,
    /// Execute tool calls inside the sandbox instead of reading caller
    /// results from stdin/replay (Phase 47). Only tools listed via
    /// `--allow-tool` are executable; everything else is a hard refusal.
    #[arg(long)]
    pub execute_tools: bool,
    /// Operator-authorized tool name. Repeat to enable multiple tools.
    /// Only these names may ever execute when `--execute-tools` is set.
    #[arg(long = "allow-tool", value_name = "NAME")]
    pub allow_tool: Vec<String>,
    /// Per-call wall-clock ceiling for sandboxed execution, in milliseconds.
    #[arg(long, default_value_t = aarambh_studio_agent::DEFAULT_TIMEOUT_MS)]
    pub exec_timeout_ms: u64,
    /// Maximum output payload bytes a sandboxed executor may return.
    #[arg(long, default_value_t = aarambh_studio_agent::DEFAULT_MAX_OUTPUT_BYTES)]
    pub exec_max_output_bytes: usize,
    /// Working directory for the `read_file_in_workdir` executor. The
    /// executor is registered only when this is set; it can never read
    /// outside this directory.
    #[arg(long)]
    pub exec_workdir: Option<PathBuf>,
}

enum CliResultProvider {
    Replay(ReplayResultProvider),
    Stdin(StdinResultProvider<BufReader<io::Stdin>>),
    Sandbox(SandboxedToolProvider),
}

impl ToolResultProvider for CliResultProvider {
    fn next_result(&mut self, request: &ToolResultRequest) -> AgentResult<ToolResult> {
        match self {
            Self::Replay(provider) => provider.next_result(request),
            Self::Stdin(provider) => provider.next_result(request),
            Self::Sandbox(provider) => provider.next_result(request),
        }
    }

    fn finish(&self) -> AgentResult<()> {
        match self {
            Self::Replay(provider) => provider.finish(),
            Self::Stdin(provider) => provider.finish(),
            Self::Sandbox(provider) => provider.finish(),
        }
    }
}

enum ProjectedMedia {
    Image(Tensor),
    Video(Tensor),
    Document(Tensor),
}

struct CliChainDecoder {
    engine: InferenceEngine,
    run_config: TrainingRunConfig,
    dtype: DType,
    tool_calling: ToolCallingConfig,
    sampler: Sampler,
    thinking: ThinkingMode,
    safety: Option<SafetyInspector>,
    result_root: PathBuf,
    vision_runtime: Option<VisionRuntime>,
    document_runtime: Option<DocumentRuntime>,
    projected_media: Option<ProjectedMedia>,
}

impl CliChainDecoder {
    fn inspect_input(&self, text: &str) -> AgentResult<String> {
        let Some(safety) = &self.safety else {
            return Ok(text.to_string());
        };
        let checked = safety.inspect_input(text)?;
        match checked.verdict {
            SafetyVerdict::Block(reason) | SafetyVerdict::Regenerate(reason) => {
                Err(AgentError::Config(format!("blocked by safety: {reason}")))
            }
            SafetyVerdict::Allow | SafetyVerdict::Redact(_) => Ok(checked.text),
        }
    }

    fn generation_config(&self, max_new_tokens: usize, tools: bool) -> GenerationConfig {
        GenerationConfig {
            max_new_tokens,
            sampler: self.sampler.clone(),
            thinking_mode: self.thinking,
            top_candidates: 5,
            tool_calling: tools.then(|| self.tool_calling.clone()),
            stop_sequences: Vec::new(),
            capture_steps: false,
        }
    }

    fn checked_generation<F>(
        &mut self,
        prompt_for_audit: &str,
        mut generate: F,
    ) -> AgentResult<GenerationOutput>
    where
        F: FnMut(&mut Self) -> AgentResult<GenerationOutput>,
    {
        let attempts = self
            .safety
            .as_ref()
            .map_or(1, |safety| safety.policy().max_regenerations + 1);
        for attempt in 0..attempts {
            let output = generate(self)?;
            let Some(safety) = &self.safety else {
                return Ok(output);
            };
            let checked = safety.inspect_output(prompt_for_audit, output)?;
            match checked.verdict {
                SafetyVerdict::Allow | SafetyVerdict::Redact(_) => {
                    return checked.output.ok_or_else(|| {
                        AgentError::Config(
                            "safety allowed generation without returning output".into(),
                        )
                    });
                }
                SafetyVerdict::Regenerate(reason) if attempt + 1 < attempts => {
                    eprintln!("agent safety regeneration: {reason}");
                }
                SafetyVerdict::Regenerate(reason) | SafetyVerdict::Block(reason) => {
                    return Err(AgentError::Config(format!(
                        "generated decision blocked by safety: {reason}"
                    )));
                }
            }
        }
        Err(AgentError::Config(
            "safety generation attempts were exhausted".into(),
        ))
    }

    fn canonical_media_path(&self, path: &str) -> AgentResult<PathBuf> {
        let path = fs::canonicalize(path).map_err(|error| {
            AgentError::ResultProtocol(format!("cannot resolve media path {path:?}: {error}"))
        })?;
        if !path.starts_with(&self.result_root) {
            return Err(AgentError::ResultProtocol(format!(
                "media path {} escapes result root {}",
                path.display(),
                self.result_root.display()
            )));
        }
        Ok(path)
    }

    fn ensure_vision_runtime(&mut self) -> AgentResult<&VisionRuntime> {
        if self.vision_runtime.is_none() {
            let runtime = load_vision_runtime(&self.run_config, self.engine.device(), self.dtype)
                .map_err(|error| AgentError::Config(error.to_string()))?;
            self.vision_runtime = Some(runtime);
        }
        Ok(self
            .vision_runtime
            .as_ref()
            .expect("vision runtime was initialized"))
    }

    fn ensure_document_runtime(&mut self) -> AgentResult<&DocumentRuntime> {
        if self.document_runtime.is_none() {
            let runtime = load_document_runtime(&self.run_config, self.engine.device(), self.dtype)
                .map_err(|error| AgentError::Config(error.to_string()))?;
            self.document_runtime = Some(runtime);
        }
        Ok(self
            .document_runtime
            .as_ref()
            .expect("document runtime was initialized"))
    }

    fn encode_result_turn(
        &self,
        result: &ToolResult,
        marker: Option<&str>,
    ) -> AgentResult<Vec<u32>> {
        let safe_text = self.inspect_input(&result.transcript_text())?;
        let marker = marker.unwrap_or_default();
        let turn = format!(
            "{USER}\nTool result {}: {safe_text}\n{marker}{ASSISTANT}\n",
            result.call_id
        );
        Ok(self.engine.tokenizer().encode(&turn)?)
    }

    fn embeddings_for(
        &self,
        transcript_ids: &[u32],
        projected: ProjectedMedia,
    ) -> AgentResult<Tensor> {
        let text = Tensor::from_vec(
            transcript_ids.to_vec(),
            (1, transcript_ids.len()),
            self.engine.device(),
        )
        .map_err(AarambhError::from)?;
        let text_embeddings = self.engine.model().embed_tokens(&text)?;
        match projected {
            ProjectedMedia::Image(tokens) => Ok(interleave_image_tokens(
                transcript_ids,
                &text_embeddings,
                &tokens,
                IMAGE_ID,
            )?),
            ProjectedMedia::Video(tokens) => Ok(interleave_video_tokens(
                transcript_ids,
                &text_embeddings,
                &tokens,
                VIDEO_ID,
                FRAME_SEP_ID,
            )?),
            ProjectedMedia::Document(tokens) => Ok(interleave_document_tokens(
                transcript_ids,
                &text_embeddings,
                &tokens,
                DOCUMENT_ID,
                PAGE_SEP_ID,
            )?),
        }
    }
}

impl ChainDecoder for CliChainDecoder {
    fn context_limit(&self) -> usize {
        self.engine.model().config().max_seq_len
    }

    fn encode_prefix(
        &mut self,
        prompt: &str,
        _tools: &[ToolDefinition],
        summary: Option<&str>,
    ) -> AgentResult<Vec<u32>> {
        let prompt = self.inspect_input(prompt)?;
        let request = match summary {
            Some(summary) => {
                format!("Prior tool-chain summary:\n{summary}\n\nCurrent request:\n{prompt}")
            }
            None => prompt,
        };
        let rendered = self.tool_calling.render_prompt(&request)?;
        Ok(self.engine.encode_prompt(&rendered)?)
    }

    fn encode_result(&mut self, result: &ToolResult) -> AgentResult<Vec<u32>> {
        self.projected_media = None;
        let marker = match result.content.as_ref() {
            Some(ToolResultContent::Image { path, .. }) => {
                let path = self.canonical_media_path(path)?;
                let device = self.engine.device().clone();
                let tokens = project_image_tokens(self.ensure_vision_runtime()?, &path, &device)?;
                self.projected_media = Some(ProjectedMedia::Image(tokens));
                Some(format!("{IMAGE}{IMAGE_END}\n"))
            }
            Some(ToolResultContent::Video { path, .. }) => {
                let path = self.canonical_media_path(path)?;
                let video = self
                    .run_config
                    .vision
                    .as_ref()
                    .and_then(|vision| vision.video.as_ref())
                    .ok_or_else(|| {
                        AgentError::Config(
                            "video tool results require a [vision.video] config block".into(),
                        )
                    })?
                    .clone();
                let sampling = VideoSamplingConfig {
                    frame_count: video.frame_count,
                    max_frame_count: video.max_frame_count,
                    strategy: video.sampling,
                    scene_min_gap: video.scene_min_gap,
                };
                sampling.validate()?;
                let device = self.engine.device().clone();
                let tokens = project_video_tokens(
                    self.ensure_vision_runtime()?,
                    &path,
                    &device,
                    &sampling,
                    video.encoder_frame_batch_size,
                )?;
                let frame_count = tokens.dim(0).map_err(AarambhError::from)?;
                self.projected_media = Some(ProjectedMedia::Video(tokens));
                Some(format!(
                    "{VIDEO}{}{VIDEO_END}\n",
                    FRAME_SEP.repeat(frame_count.saturating_sub(1))
                ))
            }
            Some(ToolResultContent::Document { path, pages, .. }) => {
                let path = self.canonical_media_path(path)?;
                let selected = (!pages.is_empty())
                    .then(|| pages.iter().map(|page| *page as usize).collect::<Vec<_>>());
                let device = self.engine.device().clone();
                let (tokens, page_count) = project_document_tokens(
                    self.ensure_document_runtime()?,
                    &path,
                    selected.as_deref(),
                    &device,
                    None,
                    None,
                )?;
                self.projected_media = Some(ProjectedMedia::Document(tokens));
                Some(format!(
                    "{DOCUMENT}{}{DOCUMENT_END}\n",
                    PAGE_SEP.repeat(page_count.saturating_sub(1))
                ))
            }
            Some(ToolResultContent::Text { .. }) | None => None,
        };
        self.encode_result_turn(result, marker.as_deref())
    }

    fn encode_result_metadata(&mut self, result: &ToolResult) -> AgentResult<Vec<u32>> {
        self.encode_result_turn(result, None)
    }

    fn generate(
        &mut self,
        transcript_ids: &[u32],
        pending_media: Option<&ToolResultContent>,
        max_new_tokens: usize,
    ) -> AgentResult<GenerationOutput> {
        let config = self.generation_config(max_new_tokens, true);
        let audit = format!("agent transcript tokens={}", transcript_ids.len());
        let projected = self.projected_media.take();
        if pending_media.is_some() != projected.is_some() {
            return Err(AgentError::Config(
                "native media result and projected embeddings are out of sync".into(),
            ));
        }
        self.checked_generation(&audit, |decoder| {
            if let Some(projected) = projected.as_ref() {
                let projected = match projected {
                    ProjectedMedia::Image(tensor) => ProjectedMedia::Image(tensor.clone()),
                    ProjectedMedia::Video(tensor) => ProjectedMedia::Video(tensor.clone()),
                    ProjectedMedia::Document(tensor) => ProjectedMedia::Document(tensor.clone()),
                };
                let embeddings = decoder.embeddings_for(transcript_ids, projected)?;
                Ok(decoder.engine.generate_with_embeddings_callback(
                    &embeddings,
                    config.clone(),
                    |_| Ok(()),
                )?)
            } else {
                Ok(decoder
                    .engine
                    .generate_from_token_ids(transcript_ids.to_vec(), config.clone())?)
            }
        })
    }

    fn summarise(
        &mut self,
        previous_summary: Option<&str>,
        evicted: &[ToolExchange],
        max_tokens: usize,
    ) -> AgentResult<String> {
        let mut history = previous_summary.unwrap_or_default().to_string();
        for exchange in evicted {
            history.push_str(&format!(
                "\n{} {}({}) -> {}",
                exchange.request.call_id,
                exchange.request.call.name,
                exchange.request.call.arguments,
                exchange.result.transcript_text()
            ));
        }
        let prompt = format!(
            "Summarise these completed tool interactions as compact factual state. Preserve identifiers, arguments, outputs, and errors. Do not add facts.\n{history}"
        );
        let config = self.generation_config(max_tokens, false);
        let output = self.checked_generation(&prompt, |decoder| {
            Ok(decoder.engine.generate(&prompt, config.clone())?)
        })?;
        Ok(output.text)
    }
}

/// Execute the agent CLI command.
pub fn run(args: AgentArgs) -> anyhow::Result<()> {
    // Validate sandboxed-execution config first so operator errors surface
    // before any model is loaded or checkpoint is resolved.
    if args.execute_tools {
        validate_sandbox_config(&args)?;
    }
    let run_config = TrainingRunConfig::from_toml(&args.config)?;
    let run_device = run_config.device()?;
    let dtype = run_config.dtype_for_device(&run_device)?.to_candle();
    let device = run_device.to_candle()?;
    let model_path = match args.model.clone() {
        Some(path) => path,
        None => default_model_path(&run_config.train.checkpoint_dir)?,
    };
    let tokenizer_path = args
        .tokenizer
        .clone()
        .or_else(|| run_config.tokenizer_path.clone())
        .or_else(|| run_config.tokenizer_save_path.clone())
        .unwrap_or_else(|| run_config.train.checkpoint_dir.join("tokenizer.json"));
    let definitions = load_tool_definitions(&args.tools)?;
    let tool_calling = ToolCallingConfig::new(definitions.clone(), ToolChoice::Auto)?;
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
    let thinking = parse_thinking_mode(&args.thinking)?;
    let safety_mode = parse_safety_mode(&args.safety)?;
    let safety = SafetyPolicy::for_mode(safety_mode)
        .map(|policy| SafetyInspector::new(policy.with_audit_path(&args.safety_audit_log)));
    let result_root = fs::canonicalize(&args.result_root).map_err(|error| {
        anyhow::anyhow!(
            "cannot resolve --result-root {}: {error}",
            args.result_root.display()
        )
    })?;
    let engine = InferenceEngine::from_paths_with_dtype(
        model_path,
        &run_config.model,
        tokenizer_path,
        device,
        dtype,
    )?;
    let decoder = CliChainDecoder {
        engine,
        run_config,
        dtype,
        tool_calling,
        sampler,
        thinking,
        safety,
        result_root,
        vision_runtime: None,
        document_runtime: None,
        projected_media: None,
    };
    let provider = if args.execute_tools {
        CliResultProvider::Sandbox(build_sandbox_provider(&args, &definitions)?)
    } else {
        match args.results {
            Some(path) => CliResultProvider::Replay(ReplayResultProvider::from_jsonl(path)?),
            None => CliResultProvider::Stdin(StdinResultProvider::new(BufReader::new(io::stdin()))),
        }
    };
    let eviction_policy = parse_eviction(&args.eviction)?;
    let chain_config = ToolChainConfig {
        max_steps: args.max_steps,
        max_tokens_per_step: args.max_tokens,
        context_reserve: 32,
        keep_recent: args.keep_recent,
        summary_tokens: args.summary_tokens,
        eviction_policy,
    };
    let mut chain = ToolChain::new(decoder, provider, chain_config)?;
    let jsonl = args.jsonl;
    let output = chain.run_with_callback(args.prompt, definitions, |event| {
        print_event(event, jsonl)?;
        Ok(())
    })?;
    if !jsonl {
        println!("{}", output.final_output.text);
        eprintln!(
            "agent_complete model_turns={} tool_calls={} prompt_tokens={} completion_tokens={} evictions={} summaries={} media_turns={}",
            output.metrics.model_turns,
            output.metrics.tool_calls,
            output.metrics.prompt_tokens,
            output.metrics.completion_tokens,
            output.metrics.evictions,
            output.metrics.summaries,
            output.metrics.media_turns,
        );
    } else {
        println!(
            "{}",
            serde_json::to_string(&json!({
                "type": "metrics",
                "metrics": output.metrics,
            }))?
        );
    }
    Ok(())
}

fn print_event(event: &ChainEvent, jsonl: bool) -> AgentResult<()> {
    if jsonl {
        println!(
            "{}",
            serde_json::to_string(event).map_err(AarambhError::from)?
        );
        return Ok(());
    }
    match event {
        ChainEvent::ToolCall { request } => {
            eprintln!(
                "[agent] {} tool_call {}",
                request.call_id,
                serde_json::to_string(&request.call).map_err(AarambhError::from)?
            );
            eprintln!(
                "[agent] waiting for one ToolResult JSON line on stdin (caller executes the tool)"
            );
        }
        ChainEvent::ToolResult { result } => {
            eprintln!("[agent] {} result={:?}", result.call_id, result.status);
        }
        ChainEvent::Eviction {
            call_id,
            summarised,
        } => {
            eprintln!("[agent] evicted {call_id} summarised={summarised}");
        }
        ChainEvent::Final { .. } => {}
    }
    Ok(())
}

fn parse_eviction(value: &str) -> anyhow::Result<EvictionPolicy> {
    match value.trim().to_ascii_lowercase().as_str() {
        "drop-oldest" | "drop_oldest" | "drop" => Ok(EvictionPolicy::DropOldest),
        "summarise" | "summarize" | "summary" => Ok(EvictionPolicy::Summarise),
        other => Err(anyhow::anyhow!(
            "invalid eviction policy {other:?}, expected drop-oldest|summarise"
        )),
    }
}

/// Build the sandboxed-execution provider from the operator's CLI flags.
///
/// Authorization is an operator decision: only `--allow-tool` names may
/// execute. The closed-world allowlist is the set of registered executors
/// (currently `read_file_in_workdir`, bound to `--exec-workdir`). A name
/// that is authorized but has no registered executor is still a hard
/// refusal at execution time (`ExecError::UnknownTool`).
fn build_sandbox_provider(
    args: &AgentArgs,
    definitions: &[ToolDefinition],
) -> anyhow::Result<SandboxedToolProvider> {
    // Defense-in-depth: validate_sandbox_config already ran at the top of
    // run(), so these checks are redundant here but kept so a programmatic
    // caller cannot bypass them.
    validate_sandbox_config(args)?;
    let mut authorization = AuthorizationScope::empty();
    for name in &args.allow_tool {
        authorization
            .enable(name)
            .map_err(|error| anyhow::anyhow!("invalid --allow-tool value: {error}"))?;
    }
    let limits = SandboxLimits {
        timeout_ms: args.exec_timeout_ms,
        max_output_bytes: args.exec_max_output_bytes,
        max_args_bytes: aarambh_studio_agent::DEFAULT_MAX_ARGS_BYTES,
    };
    let mut sandbox = ToolSandbox::new(authorization, limits)
        .map_err(|error| anyhow::anyhow!("sandbox configuration error: {error}"))?;
    sandbox
        .register_definitions(definitions)
        .map_err(|error| anyhow::anyhow!("sandbox definition error: {error}"))?;
    if let Some(workdir) = &args.exec_workdir {
        let executor = ReadFileInWorkdir::new(workdir)
            .map_err(|error| anyhow::anyhow!("--exec-workdir error: {error}"))?;
        sandbox
            .register_executor(Box::new(executor))
            .map_err(|error| anyhow::anyhow!("executor registration error: {error}"))?;
    }
    Ok(SandboxedToolProvider::new(sandbox))
}

/// Validate the sandboxed-execution config before any model is loaded, so
/// operator errors (missing `--allow-tool`, unauthorized `--exec-workdir`)
/// surface immediately rather than after a checkpoint is resolved.
fn validate_sandbox_config(args: &AgentArgs) -> anyhow::Result<()> {
    if args.allow_tool.is_empty() {
        return Err(anyhow::anyhow!(
            "--execute-tools requires at least one --allow-tool <NAME> to authorize execution"
        ));
    }
    if args.exec_workdir.is_some()
        && !args
            .allow_tool
            .iter()
            .any(|name| name == ReadFileInWorkdir::NAME)
    {
        return Err(anyhow::anyhow!(
            "--exec-workdir registers the {:?} executor but it was not authorized via --allow-tool",
            ReadFileInWorkdir::NAME
        ));
    }
    Ok(())
}
