use std::fs;
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::time::Duration;

use aarambh_studio_inference::{InferenceEngine, ThinkingMode, ToolDefinition};
use aarambh_studio_safety::{SafetyMode, SafetyPolicy};
use aarambh_studio_serve::{BatcherConfig, ServeConfig, run_server};
use aarambh_studio_train::TrainingRunConfig;
use clap::Args;
use serde::Deserialize;

#[derive(Debug, Args)]
/// Start the local OpenAI-compatible inference server.
pub struct ServeArgs {
    /// Training/model TOML configuration (provides architecture + device).
    #[arg(long, default_value = "configs/tiny_shakespeare.toml")]
    config: PathBuf,
    /// Model checkpoint to serve.
    #[arg(long)]
    model: PathBuf,
    /// Optional tokenizer JSON path; falls back to the configured tokenizer.
    #[arg(long)]
    tokenizer: Option<PathBuf>,
    /// Public model id advertised by the OpenAI-compatible API.
    #[arg(long, default_value = "aarambh-studio-local")]
    model_id: String,
    /// Bind host IP address.
    #[arg(long, default_value = "127.0.0.1")]
    host: IpAddr,
    /// Bind TCP port.
    #[arg(long, default_value_t = 8080)]
    port: u16,
    /// Maximum requests processed in a single continuous batch.
    #[arg(long, default_value_t = 8)]
    max_batch_size: usize,
    /// Maximum pending requests queued before backpressure applies.
    #[arg(long, default_value_t = 128)]
    queue_capacity: usize,
    /// Milliseconds to wait for additional requests before flushing a batch.
    #[arg(long, default_value_t = 2)]
    batch_wait_ms: u64,
    /// Maximum tokens prefilled per chunked prefill pass.
    #[arg(long, default_value_t = 128)]
    prefill_chunk_size: usize,
    /// Maximum total tokens (prompt + completion) accepted per request.
    #[arg(long, default_value_t = 2048)]
    max_request_tokens: usize,
    #[arg(long, default_value = "none")]
    /// Default thinking budget: none, low, medium, high, or max.
    thinking: String,
    /// Optional JSON tool definitions file advertised to clients.
    #[arg(long)]
    tools: Option<PathBuf>,
    /// Safety policy: strict, permissive, research, or none.
    #[arg(long, default_value = "strict")]
    safety: String,
    /// JSONL safety audit log path.
    #[arg(long, default_value = "safety_audit.jsonl")]
    safety_audit_log: PathBuf,
    /// Environment variable name holding the optional bearer API key.
    #[arg(long, default_value = "AARAMBH_STUDIO_STUDIO_API_KEY")]
    api_key_env: String,
    /// Allowed CORS origin(s); repeat to enable multiple origins.
    #[arg(long)]
    cors_origin: Vec<String>,
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

pub fn run(args: ServeArgs) -> anyhow::Result<()> {
    let run_config = TrainingRunConfig::from_toml(&args.config)?;
    let run_device = run_config.device()?;
    let dtype = run_config.dtype_for_device(&run_device)?.to_candle();
    let device = run_device.to_candle()?;
    let tokenizer = args
        .tokenizer
        .clone()
        .or(run_config.tokenizer_path.clone())
        .unwrap_or_else(|| run_config.train.checkpoint_dir.join("tokenizer.json"));
    let engine = InferenceEngine::from_paths_with_dtype(
        &args.model,
        &run_config.model,
        tokenizer,
        device,
        dtype,
    )?;
    let safety_mode = args
        .safety
        .parse::<SafetyMode>()
        .map_err(anyhow::Error::msg)?;
    let default_thinking = parse_thinking(&args.thinking)?;
    let default_tools = args
        .tools
        .as_ref()
        .map(load_tools)
        .transpose()?
        .unwrap_or_default();
    let api_key = std::env::var(&args.api_key_env)
        .ok()
        .filter(|key| !key.is_empty());
    let safety_policy = SafetyPolicy::for_mode(safety_mode)
        .map(|policy| policy.with_audit_path(&args.safety_audit_log));
    let config = ServeConfig {
        bind: SocketAddr::new(args.host, args.port),
        model_id: args.model_id,
        max_request_tokens: args.max_request_tokens,
        default_thinking,
        safety_policy,
        api_key,
        cors_origins: args.cors_origin,
        default_tools,
        batcher: BatcherConfig {
            max_batch_size: args.max_batch_size,
            queue_capacity: args.queue_capacity,
            batch_wait: Duration::from_millis(args.batch_wait_ms),
            prefill_chunk_size: args.prefill_chunk_size,
        },
    };
    config.validate()?;

    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                tracing_subscriber::EnvFilter::new("aarambh_studio_serve=info")
            }),
        )
        .try_init();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(run_server(config, engine))?;
    Ok(())
}

fn parse_thinking(value: &str) -> anyhow::Result<ThinkingMode> {
    use std::str::FromStr;
    ThinkingMode::from_str(value).map_err(anyhow::Error::msg)
}

fn load_tools(path: &PathBuf) -> anyhow::Result<Vec<ToolDefinition>> {
    let metadata = fs::metadata(path)?;
    if metadata.len() > 1024 * 1024 {
        return Err(anyhow::anyhow!("tool definition file exceeds 1 MiB"));
    }
    let parsed: ToolFile = serde_json::from_slice(&fs::read(path)?)?;
    let entries = match parsed {
        ToolFile::Array(entries) | ToolFile::Object { tools: entries } => entries,
    };
    entries
        .into_iter()
        .map(|entry| match entry {
            ToolEntry::Native(definition) => Ok(definition),
            ToolEntry::OpenAi { r#type, function } if r#type == "function" => Ok(function),
            ToolEntry::OpenAi { r#type, .. } => {
                Err(anyhow::anyhow!("unsupported tool type '{type}'", type = r#type))
            }
        })
        .collect()
}
