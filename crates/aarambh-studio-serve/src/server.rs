use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use aarambh_studio_core::AarambhError;
use aarambh_studio_inference::{
    FinishReason, GenerationConfig, GenerationOutput, InferenceEngine, Sampler, ThinkingMode,
    ToolCallingConfig, ToolChoice, ToolDefinition,
};
use aarambh_studio_safety::{
    PiiPolicy, SafetyEvent, SafetyPolicy, SafetyStage, ViolationAction, detect_injection,
    detect_jailbreak, detect_pii, hash_prompt, log_event, redact_pii,
};
use axum::Json;
use axum::Router;
use axum::extract::{DefaultBodyLimit, State, rejection::JsonRejection};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use futures_util::Stream;
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tower_http::request_id::{MakeRequestUuid, PropagateRequestIdLayer, SetRequestIdLayer};
use tower_http::trace::TraceLayer;

use crate::api::{
    ApiToolChoice, AssistantMessage, ChatChoice, ChatCompletionRequest, ChatCompletionResponse,
    CompletionChoice, CompletionRequest, CompletionResponse, ErrorBody, ErrorResponse,
    FunctionCallResponse, ModelList, ModelObject, ToolCallResponse, Usage,
};
use crate::auth::{
    ApiKeyStore, AuthConfig, AuthGate, AuthOutcome, AuthRejection, RateLimit, RateLimiter, TenantId,
};
use crate::batching::{
    BatcherConfig, BatcherHandle, GenerationEvent, GenerationRequest, SubmitError,
};
use crate::metrics::ServerMetrics;
use crate::prefix_cache::{KvFootprint, PrefixCache, PrefixCacheConfig};
use crate::tenant_isolation::{TenantBusy, TenantIsolationConfig, TenantLimiter, TenantPermit};

#[derive(Clone)]
/// Complete local inference-server configuration.
pub struct ServeConfig {
    /// TCP address used by Axum.
    pub bind: SocketAddr,
    /// Public model identifier accepted by API requests.
    pub model_id: String,
    /// Maximum output tokens accepted per request.
    pub max_request_tokens: usize,
    /// Default thinking mode when the request omits `reasoning_effort`.
    pub default_thinking: ThinkingMode,
    /// Optional server-wide safety policy.
    pub safety_policy: Option<SafetyPolicy>,
    /// Optional bearer API key (single-key legacy mode).
    pub api_key: Option<String>,
    /// Optional multi-tenant API-key file (Phase 51).
    pub auth: Option<AuthConfig>,
    /// Prompt-prefix cache configuration (Phase 51).
    pub prefix_cache: PrefixCacheConfig,
    /// Per-tenant concurrent-in-flight ceiling (Phase 51).
    pub tenant_isolation: TenantIsolationConfig,
    /// Explicit CORS origins.
    pub cors_origins: Vec<String>,
    /// Optional server-provided function catalog used when a request omits tools.
    pub default_tools: Vec<ToolDefinition>,
    /// Continuous batching controls.
    pub batcher: BatcherConfig,
}

impl Default for ServeConfig {
    fn default() -> Self {
        Self {
            bind: SocketAddr::from(([127, 0, 0, 1], 8080)),
            model_id: "aarambh-studio-local".to_string(),
            max_request_tokens: 2048,
            default_thinking: ThinkingMode::None,
            safety_policy: Some(SafetyPolicy::strict()),
            api_key: None,
            auth: None,
            prefix_cache: PrefixCacheConfig::DISABLED,
            tenant_isolation: TenantIsolationConfig::UNLIMITED,
            cors_origins: Vec::new(),
            default_tools: Vec::new(),
            batcher: BatcherConfig::default(),
        }
    }
}

impl ServeConfig {
    /// Validate network, authentication, and capacity settings.
    pub fn validate(&self) -> Result<(), AarambhError> {
        if self.model_id.trim().is_empty() {
            return Err(AarambhError::Config("model id cannot be empty".into()));
        }
        if self.max_request_tokens == 0 {
            return Err(AarambhError::Config(
                "max request tokens must be greater than zero".into(),
            ));
        }
        let has_auth =
            self.api_key.as_deref().is_some_and(|key| !key.is_empty()) || self.auth.is_some();
        if !self.bind.ip().is_loopback() && !has_auth {
            return Err(AarambhError::Config(
                "an API key or key file is required for non-loopback binds".into(),
            ));
        }
        // Eagerly parse and validate the key file so a malformed file fails
        // fast at startup rather than on the first request.
        self.auth_store()?;
        if self.tenant_isolation.is_bounded() && self.auth.is_none() {
            return Err(AarambhError::Config(
                "tenant isolation requires an API-key file (multi-tenant auth)".into(),
            ));
        }
        for origin in &self.cors_origins {
            origin.parse::<HeaderValue>().map_err(|error| {
                AarambhError::Config(format!("invalid CORS origin {origin:?}: {error}"))
            })?;
        }
        Ok(())
    }

    /// Build the in-memory key store when multi-tenant auth is configured.
    pub fn auth_store(&self) -> Result<Option<ApiKeyStore>, AarambhError> {
        let Some(config) = self.auth.as_ref() else {
            return Ok(None);
        };
        let store = ApiKeyStore::from_config(config)?;
        Ok(Some(store))
    }
}

#[derive(Clone)]
struct AppState {
    config: ServeConfig,
    batcher: BatcherHandle,
    metrics: Arc<ServerMetrics>,
    auth_gate: Arc<AuthGate>,
    rate_limiter: Arc<RateLimiter>,
    tenant_limiter: Arc<TenantLimiter>,
    model_created: u64,
}

/// Build the complete Axum router around an existing inference worker.
pub fn build_router(
    config: ServeConfig,
    batcher: BatcherHandle,
    metrics_store: Arc<ServerMetrics>,
) -> std::result::Result<Router, AarambhError> {
    let store = config.auth_store()?;
    let auth_gate = Arc::new(AuthGate::new(store));
    let rate_limiter = Arc::new(RateLimiter::new());
    let tenant_limiter = Arc::new(TenantLimiter::new(config.tenant_isolation));
    let state = AppState {
        config: config.clone(),
        batcher,
        metrics: metrics_store,
        auth_gate,
        rate_limiter,
        tenant_limiter,
        model_created: unix_seconds(),
    };
    let mut router = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/models", get(models))
        .route("/healthz", get(health))
        .route("/readyz", get(health))
        .route("/metrics", get(metrics))
        .layer(DefaultBodyLimit::max(1024 * 1024))
        .layer(TraceLayer::new_for_http())
        .layer(PropagateRequestIdLayer::x_request_id())
        .layer(SetRequestIdLayer::x_request_id(MakeRequestUuid))
        .with_state(state);
    if !config.cors_origins.is_empty() {
        let origins = config
            .cors_origins
            .iter()
            .map(|origin| {
                origin
                    .parse::<HeaderValue>()
                    .expect("CORS origins must be validated before building the router")
            })
            .collect::<Vec<_>>();
        router = router.layer(
            CorsLayer::new()
                .allow_origin(AllowOrigin::list(origins))
                .allow_methods([Method::GET, Method::POST])
                .allow_headers([header::AUTHORIZATION, header::CONTENT_TYPE]),
        );
    }
    Ok(router)
}

/// Bind, serve requests, and shut down cleanly after SIGINT or SIGTERM.
pub async fn run_server(config: ServeConfig, engine: InferenceEngine) -> std::io::Result<()> {
    config
        .validate()
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidInput, error))?;
    let metrics = Arc::new(ServerMetrics::default());
    let prefix_cache = if config.prefix_cache.is_enabled() {
        let footprint = KvFootprint::from_model_config(engine.model_config(), 4);
        Some(Arc::new(PrefixCache::new(config.prefix_cache, footprint)))
    } else {
        None
    };
    let batcher = BatcherHandle::start_with_prefix_cache(
        engine,
        config.batcher.clone(),
        config.safety_policy.clone(),
        metrics.clone(),
        prefix_cache,
    )
    .map_err(std::io::Error::other)?;
    let router =
        build_router(config.clone(), batcher.clone(), metrics).map_err(std::io::Error::other)?;
    let listener = tokio::net::TcpListener::bind(config.bind).await?;
    tracing::info!(address = %config.bind, model = %config.model_id, "inference server ready");
    let result = axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await;
    batcher.shutdown();
    result
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    payload: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Response {
    let authorisation = match authorize(&headers, &state) {
        Authorised::Ok { tenant, limits } => Some((tenant, limits)),
        Authorised::Reject(failure) => {
            state.metrics.auth_rejection();
            return failure.into_response();
        }
    };
    let Json(request) = match payload {
        Ok(payload) => payload,
        Err(error) => return ApiFailure::json(error.body_text()).into_response(),
    };
    let stream = request.stream;
    let include_usage = request.stream_options.include_usage;
    let prepared = match prepare_chat_request(request, &state) {
        Ok(prepared) => prepared,
        Err(error) => return error.into_response(),
    };
    let permit = match admit_request(&state, authorisation, &prepared) {
        Ok(permit) => permit,
        Err(failure) => return failure.into_response(),
    };
    let receiver = match state.batcher.submit(prepared) {
        Ok(receiver) => receiver,
        Err(error) => return submit_failure(error).into_response(),
    };
    let id = next_id("chatcmpl");
    let created = unix_seconds();
    if stream {
        chat_stream(
            receiver,
            id,
            created,
            state.config.model_id.clone(),
            include_usage,
            permit,
        )
        .into_response()
    } else {
        let _permit = permit; // hold the admission permit for the response lifetime
        match await_generation(receiver).await {
            GenerationResult::Completed(output) => Json(chat_response(
                id,
                created,
                state.config.model_id.clone(),
                *output,
            ))
            .into_response(),
            GenerationResult::Blocked => Json(filtered_chat_response(
                id,
                created,
                state.config.model_id.clone(),
            ))
            .into_response(),
            GenerationResult::Failed(message) => ApiFailure::internal(message).into_response(),
        }
    }
}

async fn completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    payload: Result<Json<CompletionRequest>, JsonRejection>,
) -> Response {
    let authorisation = match authorize(&headers, &state) {
        Authorised::Ok { tenant, limits } => Some((tenant, limits)),
        Authorised::Reject(failure) => {
            state.metrics.auth_rejection();
            return failure.into_response();
        }
    };
    let Json(request) = match payload {
        Ok(payload) => payload,
        Err(error) => return ApiFailure::json(error.body_text()).into_response(),
    };
    let stream = request.stream;
    let prepared = match prepare_completion_request(request, &state) {
        Ok(prepared) => prepared,
        Err(error) => return error.into_response(),
    };
    let permit = match admit_request(&state, authorisation, &prepared) {
        Ok(permit) => permit,
        Err(failure) => return failure.into_response(),
    };
    let receiver = match state.batcher.submit(prepared) {
        Ok(receiver) => receiver,
        Err(error) => return submit_failure(error).into_response(),
    };
    let id = next_id("cmpl");
    let created = unix_seconds();
    if stream {
        completion_stream(receiver, id, created, state.config.model_id.clone(), permit)
            .into_response()
    } else {
        let _permit = permit; // hold the admission permit for the response lifetime
        match await_generation(receiver).await {
            GenerationResult::Completed(output) => Json(completion_response(
                id,
                created,
                state.config.model_id.clone(),
                *output,
            ))
            .into_response(),
            GenerationResult::Blocked => Json(CompletionResponse {
                id,
                object: "text_completion",
                created,
                model: state.config.model_id.clone(),
                choices: vec![CompletionChoice {
                    text: String::new(),
                    index: 0,
                    logprobs: None,
                    finish_reason: "content_filter".to_string(),
                }],
                usage: zero_usage(),
            })
            .into_response(),
            GenerationResult::Failed(message) => ApiFailure::internal(message).into_response(),
        }
    }
}

async fn models(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Authorised::Reject(failure) = authorize(&headers, &state) {
        state.metrics.auth_rejection();
        return failure.into_response();
    }
    Json(ModelList {
        object: "list",
        data: vec![ModelObject {
            id: state.config.model_id.clone(),
            object: "model",
            created: state.model_created,
            owned_by: "aarambh-studio",
        }],
    })
    .into_response()
}

async fn health() -> impl IntoResponse {
    (StatusCode::OK, Json(json!({"status": "ok"})))
}

async fn metrics(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Authorised::Reject(failure) = authorize(&headers, &state) {
        state.metrics.auth_rejection();
        return failure.into_response();
    }
    Json(state.metrics.snapshot()).into_response()
}

fn prepare_chat_request(
    request: ChatCompletionRequest,
    state: &AppState,
) -> Result<GenerationRequest, ApiFailure> {
    validate_common(
        &request.model,
        request.n,
        request.temperature,
        request.top_p,
        state,
    )?;
    if request.parallel_tool_calls == Some(true) {
        return Err(ApiFailure::param(
            "parallel_tool_calls",
            "parallel tool calls are not supported",
        ));
    }
    for (name, value) in [
        ("frequency_penalty", request.frequency_penalty),
        ("presence_penalty", request.presence_penalty),
    ] {
        if value.is_some_and(|value| value != 0.0) {
            return Err(ApiFailure::param(
                name,
                "non-zero penalties are not supported",
            ));
        }
    }
    if request.logprobs == Some(true) {
        return Err(ApiFailure::param(
            "logprobs",
            "log probabilities are not supported",
        ));
    }
    let mut prompt = String::new();
    if request.messages.is_empty() {
        return Err(ApiFailure::param("messages", "messages cannot be empty"));
    }
    for message in request.messages {
        let text = message
            .content
            .into_text()
            .map_err(|message| ApiFailure::param("messages", message))?;
        match message.role.as_str() {
            "system" | "developer" => prompt.push_str(&format!("System: {text}\n")),
            "user" => prompt.push_str(&format!("<|user|>\n{text}\n")),
            "assistant" => prompt.push_str(&format!("<|assistant|>\n{text}\n")),
            _ => {
                return Err(ApiFailure::param(
                    "messages",
                    "only developer, system, user, and assistant roles are supported",
                ));
            }
        }
    }
    prompt.push_str("<|assistant|>\n");
    let prompt = screen_prompt(&prompt, state.config.safety_policy.as_ref())?;
    let tool_calling = if request.tools.is_none() && !state.config.default_tools.is_empty() {
        if request.tool_choice.is_some() {
            return Err(ApiFailure::param(
                "tool_choice",
                "request tool_choice requires request tools when server defaults are active",
            ));
        }
        Some(
            ToolCallingConfig::new(state.config.default_tools.clone(), ToolChoice::Auto)
                .map_err(|error| ApiFailure::param("tools", error.to_string()))?,
        )
    } else {
        build_tools(request.tools, request.tool_choice)?
    };
    let max_tokens = request
        .max_completion_tokens
        .or(request.max_tokens)
        .unwrap_or(256);
    let config = generation_config(
        max_tokens,
        request.temperature,
        request.top_p,
        request.seed,
        request.stop.map(|stop| stop.into_vec()).unwrap_or_default(),
        request.reasoning_effort.as_deref(),
        tool_calling,
        state,
    )?;
    Ok(GenerationRequest {
        prompt,
        config,
        stream: request.stream,
    })
}

fn prepare_completion_request(
    request: CompletionRequest,
    state: &AppState,
) -> Result<GenerationRequest, ApiFailure> {
    validate_common(
        &request.model,
        request.n,
        request.temperature,
        request.top_p,
        state,
    )?;
    let prompt = request
        .prompt
        .as_str()
        .ok_or_else(|| ApiFailure::param("prompt", "only a string prompt is supported"))?;
    let prompt = screen_prompt(prompt, state.config.safety_policy.as_ref())?;
    let config = generation_config(
        request.max_tokens.unwrap_or(256),
        request.temperature,
        request.top_p,
        request.seed,
        request.stop.map(|stop| stop.into_vec()).unwrap_or_default(),
        None,
        None,
        state,
    )?;
    Ok(GenerationRequest {
        prompt,
        config,
        stream: request.stream,
    })
}

#[allow(clippy::too_many_arguments)]
fn generation_config(
    max_tokens: usize,
    temperature: Option<f32>,
    top_p: Option<f32>,
    seed: Option<u64>,
    stop_sequences: Vec<String>,
    reasoning_effort: Option<&str>,
    tool_calling: Option<ToolCallingConfig>,
    state: &AppState,
) -> Result<GenerationConfig, ApiFailure> {
    if max_tokens == 0 || max_tokens > state.config.max_request_tokens {
        return Err(ApiFailure::param(
            "max_tokens",
            format!(
                "max_tokens must be in 1..={}",
                state.config.max_request_tokens
            ),
        ));
    }
    // Resolve the thinking mode first so its per-mode sampling defaults
    // (ARCHITECTURE_V3.md §48.3) can fill in any sampling parameters the
    // caller left unspecified. Explicit user parameters are never overridden.
    let thinking_mode = reasoning_effort
        .map(parse_thinking)
        .transpose()?
        .unwrap_or(state.config.default_thinking);
    let (default_temp, default_top_p) = thinking_mode.default_sampler();
    let temperature = temperature.unwrap_or(default_temp);
    let top_p = top_p.unwrap_or(default_top_p);
    let sampler = if temperature == 0.0 {
        Sampler::greedy()
    } else {
        Sampler::top_k_top_p(temperature, None, Some(top_p), seed)
            .map_err(|error| ApiFailure::param("temperature", error.to_string()))?
    };
    let config = GenerationConfig {
        max_new_tokens: max_tokens,
        sampler,
        thinking_mode,
        top_candidates: 0,
        tool_calling,
        stop_sequences,
        capture_steps: false,
    };
    config
        .validate()
        .map_err(|error| ApiFailure::param("stop", error.to_string()))?;
    Ok(config)
}

fn validate_common(
    model: &str,
    n: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    state: &AppState,
) -> Result<(), ApiFailure> {
    if model != state.config.model_id {
        return Err(ApiFailure::model(model));
    }
    if n.unwrap_or(1) != 1 {
        return Err(ApiFailure::param("n", "only n=1 is supported"));
    }
    if temperature.is_some_and(|value| !value.is_finite() || !(0.0..=2.0).contains(&value)) {
        return Err(ApiFailure::param(
            "temperature",
            "temperature must be finite and in [0, 2]",
        ));
    }
    if top_p.is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value)) {
        return Err(ApiFailure::param(
            "top_p",
            "top_p must be finite and in [0, 1]",
        ));
    }
    Ok(())
}

fn build_tools(
    tools: Option<Vec<crate::api::FunctionTool>>,
    choice: Option<ApiToolChoice>,
) -> Result<Option<ToolCallingConfig>, ApiFailure> {
    let Some(tools) = tools else {
        if choice.is_some() {
            return Err(ApiFailure::param(
                "tool_choice",
                "tool_choice requires tools",
            ));
        }
        return Ok(None);
    };
    if tools.iter().any(|tool| tool.r#type != "function") {
        return Err(ApiFailure::param(
            "tools",
            "only function tools are supported",
        ));
    }
    let choice = match choice {
        None => ToolChoice::Auto,
        Some(ApiToolChoice::Mode(mode)) if mode == "auto" => ToolChoice::Auto,
        Some(ApiToolChoice::Mode(mode)) if mode == "none" => ToolChoice::None,
        Some(ApiToolChoice::Mode(mode)) if mode == "required" => ToolChoice::Required,
        Some(ApiToolChoice::Named { r#type, function }) if r#type == "function" => {
            ToolChoice::Named(function.name)
        }
        _ => return Err(ApiFailure::param("tool_choice", "invalid tool choice")),
    };
    ToolCallingConfig::new(
        tools.into_iter().map(|tool| tool.function).collect(),
        choice,
    )
    .map(Some)
    .map_err(|error| ApiFailure::param("tools", error.to_string()))
}

fn parse_thinking(value: &str) -> Result<ThinkingMode, ApiFailure> {
    use std::str::FromStr;
    ThinkingMode::from_str(value).map_err(|err| ApiFailure::param("reasoning_effort", err))
}

fn screen_prompt(prompt: &str, policy: Option<&SafetyPolicy>) -> Result<String, ApiFailure> {
    let Some(policy) = policy else {
        return Ok(prompt.to_string());
    };
    if policy
        .max_prompt_chars
        .is_some_and(|limit| prompt.chars().count() > limit)
    {
        return Err(ApiFailure::policy("prompt exceeds the safety length limit"));
    }
    let injection = detect_injection(prompt);
    if policy.check_prompt_injection
        && injection.is_triggered(policy.injection_threshold)
        && matches!(policy.on_input_violation, ViolationAction::Block)
    {
        audit_input(policy, prompt, "block", injection.matched_rules);
        return Err(ApiFailure::policy("prompt injection detected"));
    }
    let jailbreak = detect_jailbreak(prompt);
    if policy.check_jailbreak
        && jailbreak.is_triggered(policy.jailbreak_threshold)
        && matches!(policy.on_input_violation, ViolationAction::Block)
    {
        audit_input(policy, prompt, "block", jailbreak.matched_rules);
        return Err(ApiFailure::policy("jailbreak attempt detected"));
    }
    let findings = detect_pii(prompt);
    match policy.input_pii {
        PiiPolicy::Block if !findings.is_empty() => {
            audit_input(policy, prompt, "block", findings.rules("input"));
            Err(ApiFailure::policy("input PII detected"))
        }
        PiiPolicy::Redact if !findings.is_empty() => {
            audit_input(policy, prompt, "redact", findings.rules("input"));
            Ok(redact_pii(prompt, &findings))
        }
        _ => Ok(prompt.to_string()),
    }
}

fn audit_input(policy: &SafetyPolicy, prompt: &str, verdict: &str, rules: Vec<String>) {
    if !policy.audit_enabled {
        return;
    }
    if let Some(path) = &policy.audit_path {
        let event = SafetyEvent::new(hash_prompt(prompt), SafetyStage::Input, verdict, rules, 0);
        let _ = log_event(&event, path);
    }
}

enum GenerationResult {
    Completed(Box<GenerationOutput>),
    Blocked,
    Failed(String),
}

async fn await_generation(
    mut receiver: tokio::sync::mpsc::UnboundedReceiver<GenerationEvent>,
) -> GenerationResult {
    while let Some(event) = receiver.recv().await {
        match event {
            GenerationEvent::Delta(_) => {}
            GenerationEvent::Completed(output) => return GenerationResult::Completed(output),
            GenerationEvent::SafetyBlocked(_) => return GenerationResult::Blocked,
            GenerationEvent::Failed(message) => return GenerationResult::Failed(message),
        }
    }
    GenerationResult::Failed("inference worker closed the response channel".to_string())
}

fn chat_stream(
    mut receiver: tokio::sync::mpsc::UnboundedReceiver<GenerationEvent>,
    id: String,
    created: u64,
    model: String,
    include_usage: bool,
    permit: TenantPermit,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::channel(32);
    tokio::spawn(async move {
        // Hold the admission permit for the streaming-response lifetime so the
        // tenant's concurrency slot is released only when the stream ends.
        let _permit = permit;
        let first = json!({"id": id, "object":"chat.completion.chunk", "created":created,
            "model":model, "choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null,"logprobs":null}]});
        let _ = tx.send(Ok(Event::default().data(first.to_string()))).await;
        while let Some(event) = receiver.recv().await {
            match event {
                GenerationEvent::Delta(text) => {
                    let chunk = json!({"id": id, "object":"chat.completion.chunk", "created":created,
                        "model":model, "choices":[{"index":0,"delta":{"content":text},"finish_reason":null,"logprobs":null}]});
                    if tx
                        .send(Ok(Event::default().data(chunk.to_string())))
                        .await
                        .is_err()
                    {
                        return;
                    }
                }
                GenerationEvent::Completed(output) => {
                    if let Some(call) = &output.tool_call {
                        let chunk = json!({"id": id, "object":"chat.completion.chunk", "created":created,
                            "model":model, "choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":next_id("call"),"type":"function","function":{"name":call.name,"arguments":call.arguments.to_string()}}]},"finish_reason":null,"logprobs":null}]});
                        let _ = tx.send(Ok(Event::default().data(chunk.to_string()))).await;
                    }
                    let finish = json!({"id": id, "object":"chat.completion.chunk", "created":created,
                        "model":model, "choices":[{"index":0,"delta":{},"finish_reason":finish_reason(output.finish_reason),"logprobs":null}]});
                    let _ = tx.send(Ok(Event::default().data(finish.to_string()))).await;
                    if include_usage {
                        let usage = usage(&output);
                        let chunk = json!({"id": id, "object":"chat.completion.chunk", "created":created,
                            "model":model, "choices":[], "usage":usage});
                        let _ = tx.send(Ok(Event::default().data(chunk.to_string()))).await;
                    }
                    break;
                }
                GenerationEvent::SafetyBlocked(_) => {
                    let chunk = json!({"id": id, "object":"chat.completion.chunk", "created":created,
                        "model":model, "choices":[{"index":0,"delta":{},"finish_reason":"content_filter","logprobs":null}]});
                    let _ = tx.send(Ok(Event::default().data(chunk.to_string()))).await;
                    break;
                }
                GenerationEvent::Failed(message) => {
                    let chunk = json!({"error":{"message":message,"type":"server_error","param":null,"code":"inference_error"}});
                    let _ = tx.send(Ok(Event::default().data(chunk.to_string()))).await;
                    break;
                }
            }
        }
        let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
    });
    Sse::new(ReceiverStream::new(rx)).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
}

fn completion_stream(
    mut receiver: tokio::sync::mpsc::UnboundedReceiver<GenerationEvent>,
    id: String,
    created: u64,
    model: String,
    permit: TenantPermit,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::channel(32);
    tokio::spawn(async move {
        // Hold the admission permit for the streaming-response lifetime.
        let _permit = permit;
        while let Some(event) = receiver.recv().await {
            let (text, finish) = match event {
                GenerationEvent::Delta(text) => (text, None),
                GenerationEvent::Completed(output) => {
                    let chunk = (String::new(), Some(finish_reason(output.finish_reason)));
                    let payload = completion_chunk(&id, created, &model, chunk.0, chunk.1);
                    let _ = tx
                        .send(Ok(Event::default().data(payload.to_string())))
                        .await;
                    break;
                }
                GenerationEvent::SafetyBlocked(_) => {
                    let payload = completion_chunk(
                        &id,
                        created,
                        &model,
                        String::new(),
                        Some("content_filter"),
                    );
                    let _ = tx
                        .send(Ok(Event::default().data(payload.to_string())))
                        .await;
                    break;
                }
                GenerationEvent::Failed(message) => {
                    let payload = json!({"error":{"message":message,"type":"server_error","param":null,"code":"inference_error"}});
                    let _ = tx
                        .send(Ok(Event::default().data(payload.to_string())))
                        .await;
                    break;
                }
            };
            let payload = completion_chunk(&id, created, &model, text, finish);
            if tx
                .send(Ok(Event::default().data(payload.to_string())))
                .await
                .is_err()
            {
                return;
            }
        }
        let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
    });
    Sse::new(ReceiverStream::new(rx)).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
}

fn completion_chunk(
    id: &str,
    created: u64,
    model: &str,
    text: String,
    finish_reason: Option<&str>,
) -> serde_json::Value {
    json!({"id":id,"object":"text_completion","created":created,"model":model,
        "choices":[{"text":text,"index":0,"logprobs":null,"finish_reason":finish_reason}]})
}

fn chat_response(
    id: String,
    created: u64,
    model: String,
    output: GenerationOutput,
) -> ChatCompletionResponse {
    let tool_calls = output.tool_call.as_ref().map(|call| {
        vec![ToolCallResponse {
            id: next_id("call"),
            r#type: "function",
            function: FunctionCallResponse {
                name: call.name.clone(),
                arguments: call.arguments.to_string(),
            },
        }]
    });
    ChatCompletionResponse {
        id,
        object: "chat.completion",
        created,
        model,
        choices: vec![ChatChoice {
            index: 0,
            message: AssistantMessage {
                role: "assistant",
                content: output.tool_call.is_none().then_some(output.text.clone()),
                tool_calls,
            },
            finish_reason: finish_reason(output.finish_reason).to_string(),
            logprobs: None,
        }],
        usage: usage(&output),
    }
}

fn filtered_chat_response(id: String, created: u64, model: String) -> ChatCompletionResponse {
    ChatCompletionResponse {
        id,
        object: "chat.completion",
        created,
        model,
        choices: vec![ChatChoice {
            index: 0,
            message: AssistantMessage {
                role: "assistant",
                content: None,
                tool_calls: None,
            },
            finish_reason: "content_filter".to_string(),
            logprobs: None,
        }],
        usage: zero_usage(),
    }
}

fn completion_response(
    id: String,
    created: u64,
    model: String,
    output: GenerationOutput,
) -> CompletionResponse {
    CompletionResponse {
        id,
        object: "text_completion",
        created,
        model,
        choices: vec![CompletionChoice {
            text: output.text.clone(),
            index: 0,
            logprobs: None,
            finish_reason: finish_reason(output.finish_reason).to_string(),
        }],
        usage: usage(&output),
    }
}

fn usage(output: &GenerationOutput) -> Usage {
    Usage {
        prompt_tokens: output.usage.prompt_tokens,
        completion_tokens: output.usage.completion_tokens,
        total_tokens: output.usage.total_tokens,
    }
}

fn zero_usage() -> Usage {
    Usage {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
    }
}

fn finish_reason(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::EosToken | FinishReason::StopSequence => "stop",
        FinishReason::MaxTokens | FinishReason::ContextLimit => "length",
        FinishReason::ToolCall => "tool_calls",
    }
}

fn authorize(headers: &HeaderMap, state: &AppState) -> Authorised {
    // Multi-tenant auth takes precedence when configured.
    if !state.auth_gate.is_open() {
        return match state.auth_gate.authorize(headers) {
            AuthOutcome::Authenticated { key } => Authorised::Ok {
                tenant: key.tenant.clone(),
                limits: key.limits,
            },
            AuthOutcome::UnauthenticatedLocal => Authorised::Ok {
                tenant: TenantId::local(),
                limits: RateLimit::UNLIMITED,
            },
            AuthOutcome::Rejected(AuthRejection::MissingKey) => {
                Authorised::Reject(ApiFailure::unauthorized_with("missing bearer API key"))
            }
            AuthOutcome::Rejected(AuthRejection::InvalidKey) => {
                Authorised::Reject(ApiFailure::unauthorized_with("invalid bearer API key"))
            }
        };
    }
    // Legacy single-key auth path.
    match state.config.api_key.as_deref() {
        Some(expected) => {
            let supplied = headers
                .get("authorization")
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.strip_prefix("Bearer "))
                .map(str::trim);
            if supplied.is_some_and(|supplied| {
                crate::auth::constant_time_eq(supplied.as_bytes(), expected.as_bytes())
            }) {
                Authorised::Ok {
                    tenant: TenantId::local(),
                    limits: RateLimit::UNLIMITED,
                }
            } else {
                Authorised::Reject(ApiFailure::unauthorized())
            }
        }
        None => Authorised::Ok {
            tenant: TenantId::local(),
            limits: RateLimit::UNLIMITED,
        },
    }
}

/// Result of authorising one request.
enum Authorised {
    /// Request is authenticated as `tenant` with per-key `limits`.
    Ok {
        /// Resolved tenant id.
        tenant: TenantId,
        /// Per-key rate limits (UNLIMITED in loopback-open mode).
        limits: RateLimit,
    },
    /// Request was rejected; the failure carries the HTTP response.
    Reject(ApiFailure),
}

/// Estimate the prompt+completion token cost of a request for rate limiting.
fn estimated_tokens(request: &GenerationRequest) -> usize {
    // Honest about being an estimate: prompt chars / 4 (a common
    // approximation) plus the requested max_new_tokens. The actual admitted
    // token count is reported by the inference engine after generation.
    let prompt_estimate = request.prompt.chars().count() / 4;
    prompt_estimate + request.config.max_new_tokens
}

/// Run the per-tenant rate limiter and concurrency limiter before admission.
fn admit_request(
    state: &AppState,
    authorisation: Option<(TenantId, RateLimit)>,
    request: &GenerationRequest,
) -> Result<TenantPermit, ApiFailure> {
    let (tenant, limits) =
        authorisation.unwrap_or_else(|| (TenantId::local(), RateLimit::UNLIMITED));
    let tokens = estimated_tokens(request);
    if !state.rate_limiter.check(&tenant, limits, tokens) {
        state.metrics.rate_limited();
        return Err(ApiFailure::rate_limited());
    }
    match state.tenant_limiter.try_admit(&tenant) {
        Ok(permit) => Ok(permit),
        Err(TenantBusy) => {
            state.metrics.tenant_throttled();
            Err(ApiFailure::tenant_busy())
        }
    }
}

fn submit_failure(error: SubmitError) -> ApiFailure {
    match error {
        SubmitError::QueueFull => ApiFailure::new(
            StatusCode::TOO_MANY_REQUESTS,
            "server_error",
            "queue_full",
            "inference queue is full",
            None,
        ),
        SubmitError::WorkerStopped => ApiFailure::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "server_error",
            "worker_unavailable",
            "inference worker is unavailable",
            None,
        ),
    }
}

#[derive(Debug)]
struct ApiFailure {
    status: StatusCode,
    body: ErrorResponse,
}

impl ApiFailure {
    fn new(
        status: StatusCode,
        kind: &str,
        code: &str,
        message: impl Into<String>,
        param: Option<String>,
    ) -> Self {
        Self {
            status,
            body: ErrorResponse {
                error: ErrorBody {
                    message: message.into(),
                    r#type: kind.to_string(),
                    param,
                    code: code.to_string(),
                },
            },
        }
    }

    fn param(param: &str, message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "invalid_parameter",
            message,
            Some(param.to_string()),
        )
    }

    fn json(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "invalid_json",
            message,
            None,
        )
    }

    fn model(model: &str) -> Self {
        Self::new(
            StatusCode::NOT_FOUND,
            "invalid_request_error",
            "model_not_found",
            format!("model '{model}' is not loaded"),
            Some("model".to_string()),
        )
    }

    fn policy(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "content_policy_violation",
            message,
            Some("messages".to_string()),
        )
    }

    fn unauthorized() -> Self {
        Self::new(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "invalid_api_key",
            "missing or invalid bearer API key",
            None,
        )
    }

    /// Construct an authentication failure with a specific reason message.
    fn unauthorized_with(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "invalid_api_key",
            message,
            None,
        )
    }

    /// Construct a per-key rate-limit failure.
    fn rate_limited() -> Self {
        Self::new(
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_error",
            "rate_limited",
            "per-key rate limit exceeded",
            None,
        )
    }

    /// Construct a per-tenant concurrency-ceiling failure.
    fn tenant_busy() -> Self {
        Self::new(
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_error",
            "tenant_busy",
            "tenant concurrent request ceiling reached",
            None,
        )
    }

    fn internal(_message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            "inference_error",
            "inference failed",
            None,
        )
    }
}

impl IntoResponse for ApiFailure {
    fn into_response(self) -> Response {
        (self.status, Json(self.body)).into_response()
    }
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

fn next_id(prefix: &str) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    format!(
        "{prefix}-{:x}-{:x}",
        unix_seconds(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut signal) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            signal.recv().await;
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use aarambh_studio_core::ModelConfig;
    use aarambh_studio_model::AarambhModel;
    use aarambh_studio_tokenizer::{
        ASSISTANT, ASSISTANT_ID, BOS, BOS_ID, BpeTokenizer, ENDOFTEXT, ENDOFTEXT_ID, PAD, PAD_ID,
        THINK_END, THINK_END_ID, THINK_START, THINK_START_ID, USER, USER_ID, Vocab,
    };
    use axum::body::Body;
    use axum::http::Request;
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use super::*;

    fn test_engine() -> InferenceEngine {
        let pairs: [(&str, u32); 12] = [
            (ENDOFTEXT, ENDOFTEXT_ID),
            (PAD, PAD_ID),
            (BOS, BOS_ID),
            (THINK_START, THINK_START_ID),
            (THINK_END, THINK_END_ID),
            (USER, USER_ID),
            (ASSISTANT, ASSISTANT_ID),
            ("H", 7),
            ("e", 8),
            ("l", 9),
            ("o", 10),
            (" ", 11),
        ];
        let token_to_id = pairs
            .iter()
            .map(|(token, id)| ((*token).to_string(), *id))
            .collect::<HashMap<_, _>>();
        let mut id_to_token = vec![String::new(); pairs.len()];
        for (token, id) in pairs {
            id_to_token[id as usize] = token.to_string();
        }
        let tokenizer = BpeTokenizer {
            vocab: Vocab {
                token_to_id,
                id_to_token,
            },
            merges: Vec::new(),
            merge_rank: HashMap::new(),
        };
        let config = ModelConfig {
            vocab_size: 12,
            hidden_dim: 64,
            ffn_dim: 128,
            n_layers: 1,
            n_heads: 1,
            n_kv_heads: 1,
            max_seq_len: 32,
            rope_theta: 10000.0,
            rope_scaling: None,
            moe: None,
            attention_schedule: None,
            dsa_config: None,
            mtp: None,
            qat: None,
            norm_eps: 1e-5,
            tie_embeddings: true,
        };
        let device = Device::Cpu;
        let model = AarambhModel::new(&config, VarBuilder::zeros(DType::F32, &device)).unwrap();
        InferenceEngine::new(model, tokenizer, device).unwrap()
    }

    fn test_router() -> (Router, BatcherHandle) {
        let metrics = Arc::new(ServerMetrics::default());
        let batcher = BatcherHandle::start(
            test_engine(),
            BatcherConfig {
                max_batch_size: 2,
                queue_capacity: 8,
                batch_wait: Duration::from_millis(1),
                prefill_chunk_size: 8,
            },
            None,
            metrics.clone(),
        )
        .unwrap();
        let config = ServeConfig {
            safety_policy: None,
            max_request_tokens: 8,
            ..ServeConfig::default()
        };
        let router = build_router(config, batcher.clone(), metrics).expect("test router builds");
        (router, batcher)
    }

    #[test]
    fn external_bind_requires_key() {
        let config = ServeConfig {
            bind: SocketAddr::new(std::net::IpAddr::from([0, 0, 0, 0]), 8080),
            ..ServeConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn finish_reason_mapping_is_openai_compatible() {
        assert_eq!(finish_reason(FinishReason::EosToken), "stop");
        assert_eq!(finish_reason(FinishReason::StopSequence), "stop");
        assert_eq!(finish_reason(FinishReason::ToolCall), "tool_calls");
        assert_eq!(finish_reason(FinishReason::MaxTokens), "length");
    }

    #[test]
    fn api_key_compare_checks_full_value() {
        use crate::auth::constant_time_eq;
        assert!(constant_time_eq(b"secret", b"secret"));
        assert!(!constant_time_eq(b"secret", b"secrex"));
        assert!(!constant_time_eq(b"secret", b"short"));
    }

    #[test]
    fn parse_thinking_accepts_max_and_rejects_unknown_values() {
        assert_eq!(parse_thinking("max").unwrap(), ThinkingMode::Max);
        assert_eq!(parse_thinking("MAX").unwrap(), ThinkingMode::Max);
        assert_eq!(parse_thinking("high").unwrap(), ThinkingMode::High);
        assert!(parse_thinking("ultra").is_err());
        assert!(parse_thinking("").is_err());
    }

    fn test_state(max_request_tokens: usize, default_thinking: ThinkingMode) -> AppState {
        let metrics = Arc::new(ServerMetrics::default());
        let batcher = BatcherHandle::start(
            test_engine(),
            BatcherConfig {
                max_batch_size: 2,
                queue_capacity: 8,
                batch_wait: Duration::from_millis(1),
                prefill_chunk_size: 8,
            },
            None,
            metrics.clone(),
        )
        .unwrap();
        AppState {
            config: ServeConfig {
                max_request_tokens,
                default_thinking,
                safety_policy: None,
                ..ServeConfig::default()
            },
            batcher,
            metrics,
            auth_gate: Arc::new(AuthGate::new(None)),
            rate_limiter: Arc::new(RateLimiter::new()),
            tenant_limiter: Arc::new(TenantLimiter::new(TenantIsolationConfig::UNLIMITED)),
            model_created: 0,
        }
    }

    #[test]
    fn generation_config_accepts_max_reasoning_effort_and_applies_defaults() {
        let state = test_state(64, ThinkingMode::None);
        // No explicit temperature/top_p: the server must still build a valid
        // sampler using Max's per-mode defaults and report Max as the mode.
        let config =
            generation_config(16, None, None, None, Vec::new(), Some("max"), None, &state).unwrap();
        assert_eq!(config.thinking_mode, ThinkingMode::Max);
    }

    #[test]
    fn generation_config_explicit_temperature_is_not_overridden_by_mode_defaults() {
        let state = test_state(64, ThinkingMode::None);
        // temperature=0.0 forces greedy decoding regardless of the mode default.
        let config = generation_config(
            16,
            Some(0.0),
            None,
            None,
            Vec::new(),
            Some("max"),
            None,
            &state,
        )
        .unwrap();
        assert!(matches!(config.sampler, Sampler::Greedy));
        assert_eq!(config.thinking_mode, ThinkingMode::Max);
    }

    #[test]
    fn generation_config_falls_back_to_default_thinking_without_reasoning_effort() {
        let state = test_state(64, ThinkingMode::Max);
        let config =
            generation_config(16, None, None, None, Vec::new(), None, None, &state).unwrap();
        assert_eq!(config.thinking_mode, ThinkingMode::Max);
    }

    #[tokio::test]
    async fn models_endpoint_matches_openai_list_shape() {
        let (router, batcher) = test_router();
        let response = router
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["object"], "list");
        assert_eq!(value["data"][0]["object"], "model");
        batcher.shutdown();
    }

    #[tokio::test]
    async fn chat_completion_matches_openai_response_shape() {
        let (router, batcher) = test_router();
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"aarambh-studio-local","messages":[{"role":"user","content":"Hello"}],"max_tokens":2,"temperature":0}"#,
            ))
            .unwrap();
        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert!(response.headers().contains_key("x-request-id"));
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["object"], "chat.completion");
        assert_eq!(value["choices"][0]["message"]["role"], "assistant");
        assert!(value["usage"]["total_tokens"].as_u64().unwrap() > 0);
        batcher.shutdown();
    }

    #[tokio::test]
    async fn completion_stream_uses_sse_and_done_marker() {
        let (router, batcher) = test_router();
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"aarambh-studio-local","prompt":"Hello","max_tokens":2,"stream":true}"#,
            ))
            .unwrap();
        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(body.contains("text_completion"));
        assert!(body.contains("data: [DONE]"));
        batcher.shutdown();
    }
}
