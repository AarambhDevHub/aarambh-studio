use std::path::Path;

use aarambh_studio_core::{AarambhError, Configurable, Result, TokenizerLike};
use aarambh_studio_model::AarambhModel;
use aarambh_studio_tokenizer::{BpeTokenizer, THINK_END_ID, THINK_START_ID};
use candle_core::{DType, Tensor};

use crate::kvcache::KvCache;
use crate::sampler::{Sampler, TokenCandidate};
use crate::speculative::SpeculativeStats;
use crate::thinking::{ThinkingController, ThinkingMode};
use crate::tool_calling::{
    TokenConstraint, ToolCall, ToolCallController, ToolCallingConfig, ToolPhase,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Reason generation stopped.
pub enum FinishReason {
    /// Requested token budget was exhausted.
    MaxTokens,
    /// End-of-sequence token was sampled.
    EosToken,
    /// Model context window left no room for more tokens.
    ContextLimit,
    /// A complete schema-valid tool call was produced.
    ToolCall,
    /// A configured stop sequence was generated.
    StopSequence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Generation phase for a produced token.
pub enum GenerationPhase {
    /// Token belongs to the hidden thinking span.
    Thinking,
    /// Token belongs to the final answer span.
    Answer,
    /// Token belongs to grammar-constrained tool-call JSON.
    ToolCall,
    /// Token is an internal tool-protocol marker.
    Control,
}

#[derive(Debug, Clone)]
/// Configuration for one generation request.
pub struct GenerationConfig {
    /// Maximum number of new tokens to generate.
    pub max_new_tokens: usize,
    /// Sampling strategy.
    pub sampler: Sampler,
    /// Thinking budget mode.
    pub thinking_mode: ThinkingMode,
    /// Number of top candidates to capture per step.
    pub top_candidates: usize,
    /// Optional validated tool-calling configuration.
    pub tool_calling: Option<ToolCallingConfig>,
    /// Text sequences that terminate visible answer generation.
    pub stop_sequences: Vec<String>,
    /// Whether completed output retains per-token step metadata.
    pub capture_steps: bool,
}

impl GenerationConfig {
    /// Create a greedy generation configuration.
    pub fn greedy(max_new_tokens: usize) -> Self {
        Self {
            max_new_tokens,
            sampler: Sampler::greedy(),
            thinking_mode: ThinkingMode::None,
            top_candidates: 5,
            tool_calling: None,
            stop_sequences: Vec::new(),
            capture_steps: true,
        }
    }

    /// Validate request-level generation limits.
    pub fn validate(&self) -> Result<()> {
        if self.stop_sequences.len() > 4 {
            return Err(AarambhError::Config(
                "at most four stop sequences are supported".into(),
            ));
        }
        for stop in &self.stop_sequences {
            if stop.is_empty() || stop.len() > 256 {
                return Err(AarambhError::Config(
                    "stop sequences must contain 1..=256 UTF-8 bytes".into(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
/// Metadata for one generated token.
pub struct GenerationStep {
    /// One-based generated token index.
    pub step: usize,
    /// Generated token id.
    pub token_id: u32,
    /// Decoded token text.
    pub token_text: String,
    /// Top candidate tokens for this step.
    pub candidates: Vec<TokenCandidate>,
    /// Thinking or answer phase.
    pub phase: GenerationPhase,
    /// Whether the token was forced by the thinking controller.
    pub forced: bool,
}

#[derive(Debug, Clone)]
/// Complete generation result.
pub struct GenerationOutput {
    /// User-visible answer text.
    pub text: String,
    /// Raw generated text including thinking markers.
    pub raw_text: String,
    /// Extracted thinking text.
    pub thinking_text: String,
    /// Extracted answer text.
    pub answer_text: String,
    /// All generated token ids.
    pub token_ids: Vec<u32>,
    /// Generated thinking token ids excluding markers.
    pub thinking_token_ids: Vec<u32>,
    /// Generated answer token ids.
    pub answer_token_ids: Vec<u32>,
    /// Number of thinking content tokens.
    pub thinking_tokens: usize,
    /// Reason generation stopped.
    pub finish_reason: FinishReason,
    /// Per-token generation metadata.
    pub steps: Vec<GenerationStep>,
    /// Speculative-decoding counters when a draft model was used.
    pub speculative_stats: Option<SpeculativeStats>,
    /// Parsed function call when generation selected a tool.
    pub tool_call: Option<ToolCall>,
    /// Token usage for this generation.
    pub usage: GenerationUsage,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
/// Prompt and completion token accounting.
pub struct GenerationUsage {
    /// Number of prompt tokens consumed by the model.
    pub prompt_tokens: usize,
    /// Number of sampled completion tokens, including a matched stop sequence.
    pub completion_tokens: usize,
    /// Prompt plus completion tokens.
    pub total_tokens: usize,
}

impl GenerationUsage {
    fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
}

/// Stateful autoregressive inference engine.
pub struct InferenceEngine {
    model: AarambhModel,
    tokenizer: BpeTokenizer,
    device: candle_core::Device,
}

/// Resumable state for one autoregressive generation request.
pub struct GenerationSession {
    cache: KvCache,
    next_logits: Tensor,
    prompt_len: usize,
    max_new_tokens: usize,
    available: usize,
    config: GenerationConfig,
    thinking: ThinkingController,
    tools: Option<ToolCallController>,
    generated_ids: Vec<u32>,
    raw_text: String,
    thinking_text: String,
    answer_text: String,
    thinking_token_ids: Vec<u32>,
    answer_token_ids: Vec<u32>,
    tool_text: String,
    steps: Vec<GenerationStep>,
    finish_reason: Option<FinishReason>,
    pending_token: Option<u32>,
}

impl InferenceEngine {
    /// Create an inference engine from loaded model and tokenizer.
    pub fn new(
        model: AarambhModel,
        tokenizer: BpeTokenizer,
        device: candle_core::Device,
    ) -> Result<Self> {
        tokenizer.validate_special_tokens()?;
        if model.config().vocab_size != tokenizer.vocab_size() {
            return Err(AarambhError::Shape(format!(
                "model vocabulary size {} does not match tokenizer vocabulary size {}",
                model.config().vocab_size,
                tokenizer.vocab_size()
            )));
        }
        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    /// Load an f32 model and tokenizer from disk.
    pub fn from_paths(
        model_path: impl AsRef<Path>,
        model_config: &aarambh_studio_core::ModelConfig,
        tokenizer_path: impl AsRef<Path>,
        device: candle_core::Device,
    ) -> Result<Self> {
        Self::from_paths_with_dtype(model_path, model_config, tokenizer_path, device, DType::F32)
    }

    /// Load a model and tokenizer from disk using the requested dtype.
    pub fn from_paths_with_dtype(
        model_path: impl AsRef<Path>,
        model_config: &aarambh_studio_core::ModelConfig,
        tokenizer_path: impl AsRef<Path>,
        device: candle_core::Device,
        dtype: DType,
    ) -> Result<Self> {
        let tokenizer = BpeTokenizer::from_pretrained(tokenizer_path)?;
        tokenizer.validate_special_tokens()?;
        let mut model_config = model_config.clone();
        model_config.vocab_size = tokenizer.vocab_size();
        let model = aarambh_studio_weights::load_any_model_with_dtype(
            model_path,
            &model_config,
            &device,
            dtype,
        )?;
        Self::new(model, tokenizer, device)
    }

    /// Return the tokenizer.
    pub fn tokenizer(&self) -> &BpeTokenizer {
        &self.tokenizer
    }

    /// Return the model.
    pub fn model(&self) -> &AarambhModel {
        &self.model
    }

    /// Return the Candle device used by this engine.
    pub fn device(&self) -> &candle_core::Device {
        &self.device
    }

    /// Return the model configuration backing this engine.
    pub fn model_config(&self) -> &aarambh_studio_core::ModelConfig {
        self.model.config()
    }

    /// Prepare a resumable text-generation session and prefill its KV cache.
    pub fn prepare_session(
        &self,
        prompt: &str,
        config: GenerationConfig,
    ) -> Result<GenerationSession> {
        self.prepare_session_with_chunk_size(prompt, config, usize::MAX)
    }

    /// Prepare a session while limiting each prompt-prefill forward pass.
    pub fn prepare_session_with_chunk_size(
        &self,
        prompt: &str,
        config: GenerationConfig,
        chunk_size: usize,
    ) -> Result<GenerationSession> {
        self.prepare_session_with_prefix_cache(prompt, config, chunk_size, |_| None, |_, _| {})
    }

    /// Prepare a session, reusing a cached prefix KV when the lookup closure
    /// returns one.
    ///
    /// `lookup` is invoked with the encoded prompt token ids and may return a
    /// previously stored [`KvCache`] plus the matched prefix length. The
    /// matched length is capped at `prompt_ids.len() - 1` so at least one
    /// token remains to prefill. On a hit, the restored cache is truncated to
    /// the matched length and only `prompt_ids[matched_len..]` is prefilled.
    /// On a miss, a fresh cache is prefilled from token 0. After prefill,
    /// `store` is always invoked with the prompt ids and the prefilled cache
    /// (the caller is responsible for any caching policy).
    pub fn prepare_session_with_prefix_cache<L, S>(
        &self,
        prompt: &str,
        config: GenerationConfig,
        chunk_size: usize,
        lookup: L,
        store: S,
    ) -> Result<GenerationSession>
    where
        L: FnOnce(&[u32]) -> Option<(KvCache, usize)>,
        S: FnOnce(&[u32], &KvCache),
    {
        if chunk_size == 0 {
            return Err(AarambhError::Config(
                "prefill chunk size must be greater than zero".into(),
            ));
        }
        config.validate()?;
        let effective_prompt = match &config.tool_calling {
            Some(tools) => tools.render_prompt(prompt)?,
            None => prompt.to_string(),
        };
        let prompt_ids = self.encode_prompt(&effective_prompt)?;
        let max_seq_len = self.model.config().max_seq_len;
        if prompt_ids.len() >= max_seq_len {
            return Err(AarambhError::Shape(format!(
                "prompt has {} tokens but model max_seq_len is {max_seq_len}",
                prompt_ids.len()
            )));
        }
        let available = max_seq_len - prompt_ids.len();
        let max_new_tokens = config.max_new_tokens.min(available);
        let capacity = prompt_ids.len() + max_new_tokens;
        let (cache, _prefilled_len, next_logits) = match lookup(&prompt_ids) {
            Some((mut cached, matched_len)) => {
                let cap = prompt_ids.len().saturating_sub(1);
                let matched_len = matched_len.min(cap);
                cached.truncate(matched_len)?;
                let next_logits = if matched_len == prompt_ids.len() {
                    // Pure prefix hit: nothing new to prefill, so rerun the
                    // last token id to recover the next-token logits (the KV
                    // cache for that token is already in place).
                    let last = prompt_ids[matched_len - 1];
                    let input = Tensor::from_vec(vec![last], (1, 1), &self.device)?;
                    let logits = self.model.forward_with_cache(
                        &input,
                        matched_len - 1,
                        cached.layers_mut(),
                    )?;
                    last_logits(&logits)?
                } else {
                    let mut tail_logits: Option<Tensor> = None;
                    let mut offset = matched_len;
                    for chunk in prompt_ids[matched_len..].chunks(chunk_size) {
                        let input =
                            Tensor::from_vec(chunk.to_vec(), (1, chunk.len()), &self.device)?;
                        let logits =
                            self.model
                                .forward_with_cache(&input, offset, cached.layers_mut())?;
                        tail_logits = Some(last_logits(&logits)?);
                        offset += chunk.len();
                    }
                    tail_logits.expect("non-empty prompt tail must be prefilled")
                };
                (cached, matched_len, next_logits)
            }
            None => {
                let mut fresh = KvCache::for_model_with_capacity(&self.model, capacity);
                let mut tail_logits: Option<Tensor> = None;
                for (chunk_index, chunk) in prompt_ids.chunks(chunk_size).enumerate() {
                    let offset = chunk_index * chunk_size;
                    let input = Tensor::from_vec(chunk.to_vec(), (1, chunk.len()), &self.device)?;
                    let logits =
                        self.model
                            .forward_with_cache(&input, offset, fresh.layers_mut())?;
                    tail_logits = Some(last_logits(&logits)?);
                }
                (
                    fresh,
                    0,
                    tail_logits.expect("prompt ids are guaranteed non-empty"),
                )
            }
        };
        store(&prompt_ids, &cache);
        GenerationSession::new(
            cache,
            next_logits,
            prompt_ids.len(),
            max_new_tokens,
            available,
            config,
            &self.tokenizer,
        )
    }

    /// Advance all supplied sessions through one shared model decode pass.
    ///
    /// Each session must have produced one pending token through
    /// [`GenerationSession::advance`] before this method is called.
    pub fn decode_sessions(&self, sessions: &mut [&mut GenerationSession]) -> Result<()> {
        if sessions.is_empty() {
            return Ok(());
        }
        let mut token_ids = Vec::with_capacity(sessions.len());
        let mut offsets = Vec::with_capacity(sessions.len());
        for session in sessions.iter() {
            if session.is_finished() {
                return Err(AarambhError::Config(
                    "finished session cannot enter a decode batch".into(),
                ));
            }
            token_ids.push(session.pending_token.ok_or_else(|| {
                AarambhError::Config("session has no pending token for batched decode".into())
            })?);
            offsets.push(session.prompt_len + session.generated_ids.len() - 1);
        }
        let input = Tensor::from_vec(token_ids, (sessions.len(), 1), &self.device)?;
        let logits = {
            let mut caches = sessions
                .iter_mut()
                .map(|session| session.cache.layers_mut())
                .collect::<Vec<_>>();
            self.model
                .forward_decode_batch(&input, &offsets, &mut caches)?
        };
        let vocab = logits.dim(2)?;
        for (row, session) in sessions.iter_mut().enumerate() {
            session.next_logits = logits.narrow(0, row, 1)?.reshape((vocab,))?;
            session.pending_token = None;
        }
        Ok(())
    }

    /// Generate text without per-step callbacks.
    pub fn generate(&mut self, prompt: &str, config: GenerationConfig) -> Result<GenerationOutput> {
        self.generate_with_callback(prompt, config, |_| Ok(()))
    }

    /// Generate text and invoke `on_step` after every produced token.
    pub fn generate_with_callback<F>(
        &mut self,
        prompt: &str,
        config: GenerationConfig,
        on_step: F,
    ) -> Result<GenerationOutput>
    where
        F: FnMut(&GenerationStep) -> Result<()>,
    {
        config.validate()?;
        let effective_prompt = match &config.tool_calling {
            Some(tools) => tools.render_prompt(prompt)?,
            None => prompt.to_string(),
        };
        let prompt_ids = self.encode_prompt(&effective_prompt)?;
        self.generate_from_token_ids_with_callback(prompt_ids, config, on_step)
    }

    /// Generate from a multimodal or otherwise precomputed prompt embedding prefix.
    pub fn generate_with_embeddings_callback<F>(
        &mut self,
        prompt_embeddings: &Tensor,
        config: GenerationConfig,
        on_step: F,
    ) -> Result<GenerationOutput>
    where
        F: FnMut(&GenerationStep) -> Result<()>,
    {
        config.validate()?;
        let dims = prompt_embeddings.dims();
        if dims.len() != 3 || dims[0] != 1 {
            return Err(AarambhError::Shape(format!(
                "prompt_embeddings must have shape [1, seq, hidden_dim], got {dims:?}"
            )));
        }
        let prompt_len = dims[1];
        self.generate_from_embeddings_with_callback(prompt_embeddings, prompt_len, config, on_step)
    }

    /// Encode a prompt exactly as the generation path does.
    pub fn encode_prompt(&self, prompt: &str) -> Result<Vec<u32>> {
        let mut prompt_ids = self.tokenizer.encode(prompt)?;
        if prompt_ids.is_empty() {
            if let Some(bos) = self.tokenizer.bos_token_id() {
                prompt_ids.push(bos);
            } else {
                return Err(AarambhError::Config(
                    "prompt produced no tokens and tokenizer has no BOS token".into(),
                ));
            }
        }
        Ok(prompt_ids)
    }

    /// Generate directly from an exact token transcript.
    ///
    /// Unlike [`Self::generate_with_callback`], this method does not render
    /// tool definitions into the prompt. Callers must include the complete
    /// protocol prefix in `prompt_ids`. Generated ids can therefore be
    /// appended byte-for-byte across tool-use turns without text round trips.
    pub fn generate_from_token_ids_with_callback<F>(
        &mut self,
        prompt_ids: Vec<u32>,
        config: GenerationConfig,
        mut on_step: F,
    ) -> Result<GenerationOutput>
    where
        F: FnMut(&GenerationStep) -> Result<()>,
    {
        config.validate()?;
        let max_seq_len = self.model.config().max_seq_len;
        if prompt_ids.len() >= max_seq_len {
            return Err(AarambhError::Shape(format!(
                "prompt has {} tokens but model max_seq_len is {max_seq_len}",
                prompt_ids.len()
            )));
        }
        let available = max_seq_len - prompt_ids.len();
        let max_new_tokens = config.max_new_tokens.min(available);
        let mut cache =
            KvCache::for_model_with_capacity(&self.model, prompt_ids.len() + max_new_tokens);
        let input = Tensor::from_vec(prompt_ids.clone(), (1, prompt_ids.len()), &self.device)?;
        let logits = self
            .model
            .forward_with_cache(&input, 0, cache.layers_mut())?;
        let next_logits = last_logits(&logits)?;
        let mut session = GenerationSession::new(
            cache,
            next_logits,
            prompt_ids.len(),
            max_new_tokens,
            available,
            config,
            &self.tokenizer,
        )?;
        while !session.is_finished() {
            if let Some(step) = session.advance(&self.tokenizer)? {
                on_step(&step)?;
            }
            if !session.is_finished() {
                self.decode_sessions(&mut [&mut session])?;
            }
        }
        session.into_output()
    }

    /// Generate directly from an exact token transcript without callbacks.
    pub fn generate_from_token_ids(
        &mut self,
        prompt_ids: Vec<u32>,
        config: GenerationConfig,
    ) -> Result<GenerationOutput> {
        self.generate_from_token_ids_with_callback(prompt_ids, config, |_| Ok(()))
    }

    fn generate_from_embeddings_with_callback<F>(
        &mut self,
        prompt_embeddings: &Tensor,
        prompt_len: usize,
        config: GenerationConfig,
        on_step: F,
    ) -> Result<GenerationOutput>
    where
        F: FnMut(&GenerationStep) -> Result<()>,
    {
        let max_seq_len = self.model.config().max_seq_len;
        if prompt_len >= max_seq_len {
            return Err(AarambhError::Shape(format!(
                "prompt has {prompt_len} embeddings but model max_seq_len is {max_seq_len}"
            )));
        }
        let available = max_seq_len - prompt_len;
        let max_new_tokens = config.max_new_tokens.min(available);
        if max_new_tokens == 0 {
            return Ok(empty_output(FinishReason::ContextLimit));
        }

        let mut cache = KvCache::for_model(&self.model);
        let logits =
            self.model
                .forward_embeddings_with_cache(prompt_embeddings, 0, cache.layers_mut())?;
        let next_logits = last_logits(&logits)?;
        self.decode_from_next_logits(
            DecodeSeed {
                prompt_len,
                max_new_tokens,
                available,
                next_logits,
            },
            &mut cache,
            config,
            on_step,
        )
    }

    fn decode_from_next_logits<F>(
        &mut self,
        seed: DecodeSeed,
        cache: &mut KvCache,
        mut config: GenerationConfig,
        mut on_step: F,
    ) -> Result<GenerationOutput>
    where
        F: FnMut(&GenerationStep) -> Result<()>,
    {
        let mut next_logits = seed.next_logits;
        let mut thinking =
            ThinkingController::for_generation(config.thinking_mode, seed.max_new_tokens);
        let mut tools = config
            .tool_calling
            .clone()
            .map(|tool_config| {
                ToolCallController::new(
                    config.thinking_mode,
                    seed.max_new_tokens,
                    tool_config,
                    &self.tokenizer,
                )
            })
            .transpose()?;
        let mut generated_ids = Vec::with_capacity(seed.max_new_tokens);
        let mut raw_text = String::new();
        let mut thinking_text = String::new();
        let mut answer_text = String::new();
        let mut thinking_token_ids = Vec::new();
        let mut answer_token_ids = Vec::new();
        let mut tool_text = String::new();
        let mut steps = Vec::with_capacity(seed.max_new_tokens);
        let mut finish_reason = FinishReason::MaxTokens;

        for step in 0..seed.max_new_tokens {
            let logits_vec = next_logits.to_vec1::<f32>()?;
            let (mut token_id, candidates, mut forced, phase) = if let Some(controller) = &mut tools
            {
                let phase = tool_phase(controller.phase_for_next());
                match controller.constraint(&self.tokenizer)? {
                    TokenConstraint::Any => (
                        config.sampler.sample(&logits_vec)?,
                        config
                            .sampler
                            .top_candidates(&logits_vec, config.top_candidates)?,
                        false,
                        phase,
                    ),
                    TokenConstraint::Forced(token_id) => {
                        let mut probabilities = vec![0.0; logits_vec.len()];
                        let index = token_id as usize;
                        if index >= probabilities.len() {
                            return Err(AarambhError::Shape(format!(
                                "forced token {token_id} exceeds vocabulary size {}",
                                probabilities.len()
                            )));
                        }
                        probabilities[index] = 1.0;
                        (
                            token_id,
                            Sampler::top_candidates_from_probabilities(
                                &probabilities,
                                config.top_candidates,
                            )?,
                            true,
                            phase,
                        )
                    }
                    TokenConstraint::Allowed(allowed) => {
                        let probabilities = config
                            .sampler
                            .probabilities_allowed(&logits_vec, &allowed)?;
                        (
                            config.sampler.sample_probabilities(&probabilities)?,
                            Sampler::top_candidates_from_probabilities(
                                &probabilities,
                                config.top_candidates,
                            )?,
                            false,
                            phase,
                        )
                    }
                }
            } else {
                let candidates = config
                    .sampler
                    .top_candidates(&logits_vec, config.top_candidates)?;
                let forced_token = thinking.take_forced_token();
                (
                    match forced_token {
                        Some(force) => force.token_id(),
                        None => config.sampler.sample(&logits_vec)?,
                    },
                    candidates,
                    forced_token.is_some(),
                    phase_for_token(&thinking, config.thinking_mode, THINK_START_ID),
                )
            };

            if token_id == self.tokenizer.eos_token_id() && phase != GenerationPhase::ToolCall {
                let in_thinking = tools
                    .as_ref()
                    .map(|controller| controller.thinking().in_thinking_block())
                    .unwrap_or_else(|| thinking.in_thinking_block());
                if in_thinking {
                    token_id = THINK_END_ID;
                    forced = true;
                } else {
                    finish_reason = FinishReason::EosToken;
                    break;
                }
            }

            let phase = if tools.is_none() {
                phase_for_token(&thinking, config.thinking_mode, token_id)
            } else {
                phase
            };
            let token_text = match &tools {
                Some(controller) => controller.token_text(token_id, &self.tokenizer)?,
                None => self.tokenizer.decode(&[token_id])?,
            };
            let generation_step = GenerationStep {
                step: step + 1,
                token_id,
                token_text: token_text.clone(),
                candidates,
                phase,
                forced,
            };
            on_step(&generation_step)?;
            if let Some(controller) = &mut tools {
                controller.on_token(token_id, &token_text, &self.tokenizer)?;
            } else {
                let _ = thinking.on_token(token_id);
            }

            generated_ids.push(token_id);
            raw_text.push_str(&token_text);
            if phase == GenerationPhase::Thinking && !is_thinking_marker(token_id) {
                thinking_text.push_str(&token_text);
                thinking_token_ids.push(token_id);
            } else if phase == GenerationPhase::Answer {
                answer_text.push_str(&token_text);
                answer_token_ids.push(token_id);
            } else if phase == GenerationPhase::ToolCall {
                tool_text.push_str(&token_text);
            }
            steps.push(generation_step);

            if tools.as_ref().is_some_and(ToolCallController::is_complete) {
                finish_reason = FinishReason::ToolCall;
                break;
            }

            if step + 1 == seed.max_new_tokens {
                if generated_ids.len() == seed.available {
                    finish_reason = FinishReason::ContextLimit;
                }
                break;
            }

            let offset = seed.prompt_len + generated_ids.len() - 1;
            let input = Tensor::from_vec(vec![token_id], (1, 1), &self.device)?;
            let logits = self
                .model
                .forward_with_cache(&input, offset, cache.layers_mut())?;
            next_logits = last_logits(&logits)?;
        }

        if finish_reason != FinishReason::ToolCall
            && tools
                .as_ref()
                .is_some_and(|controller| !controller.action_is_resolved())
        {
            return Err(AarambhError::Config(
                "generation ended before the constrained tool action completed".into(),
            ));
        }
        let tool_call = tools
            .as_ref()
            .and_then(ToolCallController::tool_call)
            .cloned();
        let text = match &tool_call {
            Some(call) => serde_json::to_string(call)?,
            None => answer_text.clone(),
        };
        debug_assert!(tool_call.is_none() || text == tool_text);
        let thinking_tokens = tools
            .as_ref()
            .map(|controller| controller.thinking().tokens_used())
            .unwrap_or_else(|| thinking.tokens_used());
        let usage = GenerationUsage::new(seed.prompt_len, generated_ids.len());
        Ok(GenerationOutput {
            text,
            raw_text,
            thinking_text,
            answer_text,
            token_ids: generated_ids,
            thinking_token_ids,
            answer_token_ids,
            thinking_tokens,
            finish_reason,
            steps,
            speculative_stats: None,
            tool_call,
            usage,
        })
    }
}

impl GenerationSession {
    #[allow(clippy::too_many_arguments)]
    fn new(
        cache: KvCache,
        next_logits: Tensor,
        prompt_len: usize,
        max_new_tokens: usize,
        available: usize,
        config: GenerationConfig,
        tokenizer: &BpeTokenizer,
    ) -> Result<Self> {
        let thinking = ThinkingController::for_generation(config.thinking_mode, max_new_tokens);
        let tools = config
            .tool_calling
            .clone()
            .map(|tool_config| {
                ToolCallController::new(
                    config.thinking_mode,
                    max_new_tokens,
                    tool_config,
                    tokenizer,
                )
            })
            .transpose()?;
        Ok(Self {
            cache,
            next_logits,
            prompt_len,
            max_new_tokens,
            available,
            config,
            thinking,
            tools,
            generated_ids: Vec::with_capacity(max_new_tokens),
            raw_text: String::new(),
            thinking_text: String::new(),
            answer_text: String::new(),
            thinking_token_ids: Vec::new(),
            answer_token_ids: Vec::new(),
            tool_text: String::new(),
            steps: Vec::new(),
            finish_reason: (max_new_tokens == 0).then_some(FinishReason::ContextLimit),
            pending_token: None,
        })
    }

    /// Fork an untouched prefilled session with a new sampling configuration.
    ///
    /// This reuses the prompt KV/recurrent state while giving the fork an
    /// independent sampler. The new token budget cannot exceed the capacity
    /// reserved by the original prefill.
    pub fn fork_with_config(
        &self,
        config: GenerationConfig,
        tokenizer: &BpeTokenizer,
    ) -> Result<Self> {
        if !self.generated_ids.is_empty() || self.pending_token.is_some() {
            return Err(AarambhError::Config(
                "only an untouched prefilled session can be forked".into(),
            ));
        }
        if self.cache.seqlen() != self.prompt_len {
            return Err(AarambhError::Config(
                "prefilled session cache length does not match its prompt".into(),
            ));
        }
        if config.max_new_tokens > self.max_new_tokens {
            return Err(AarambhError::Config(format!(
                "fork token budget {} exceeds reserved capacity {}",
                config.max_new_tokens, self.max_new_tokens
            )));
        }
        if config.tool_calling.is_some() || self.config.tool_calling.is_some() {
            return Err(AarambhError::Config(
                "prefilled session forking does not support tool-calling prompts".into(),
            ));
        }
        GenerationSession::new(
            self.cache.snapshot(),
            self.next_logits.clone(),
            self.prompt_len,
            config.max_new_tokens.min(self.available),
            self.available,
            config,
            tokenizer,
        )
    }

    /// Snapshot the session's current KV cache for prefix-cache storage.
    ///
    /// The returned [`KvCache`] is a clone of the session's prefilled state
    /// (excluding any tokens generated after prefill). Callers should only
    /// invoke this on an untouched prefilled session.
    pub fn snapshot_prefix_cache(&self) -> KvCache {
        self.cache.snapshot()
    }

    /// Sample and commit the next token from this session's current logits.
    ///
    /// When the session remains active, call [`InferenceEngine::decode_sessions`]
    /// before advancing it again.
    pub fn advance(&mut self, tokenizer: &BpeTokenizer) -> Result<Option<GenerationStep>> {
        if self.is_finished() {
            return Ok(None);
        }
        if self.pending_token.is_some() {
            return Err(AarambhError::Config(
                "session must be decoded before sampling another token".into(),
            ));
        }

        let logits_vec = self.next_logits.to_vec1::<f32>()?;
        let (mut token_id, candidates, mut forced, phase) =
            if let Some(controller) = &mut self.tools {
                let phase = tool_phase(controller.phase_for_next());
                match controller.constraint(tokenizer)? {
                    TokenConstraint::Any => (
                        self.config.sampler.sample(&logits_vec)?,
                        self.config
                            .sampler
                            .top_candidates(&logits_vec, self.config.top_candidates)?,
                        false,
                        phase,
                    ),
                    TokenConstraint::Forced(token_id) => {
                        let mut probabilities = vec![0.0; logits_vec.len()];
                        let index = token_id as usize;
                        if index >= probabilities.len() {
                            return Err(AarambhError::Shape(format!(
                                "forced token {token_id} exceeds vocabulary size {}",
                                probabilities.len()
                            )));
                        }
                        probabilities[index] = 1.0;
                        (
                            token_id,
                            Sampler::top_candidates_from_probabilities(
                                &probabilities,
                                self.config.top_candidates,
                            )?,
                            true,
                            phase,
                        )
                    }
                    TokenConstraint::Allowed(allowed) => {
                        let probabilities = self
                            .config
                            .sampler
                            .probabilities_allowed(&logits_vec, &allowed)?;
                        (
                            self.config.sampler.sample_probabilities(&probabilities)?,
                            Sampler::top_candidates_from_probabilities(
                                &probabilities,
                                self.config.top_candidates,
                            )?,
                            false,
                            phase,
                        )
                    }
                }
            } else {
                let candidates = self
                    .config
                    .sampler
                    .top_candidates(&logits_vec, self.config.top_candidates)?;
                let forced_token = self.thinking.take_forced_token();
                (
                    match forced_token {
                        Some(force) => force.token_id(),
                        None => self.config.sampler.sample(&logits_vec)?,
                    },
                    candidates,
                    forced_token.is_some(),
                    phase_for_token(&self.thinking, self.config.thinking_mode, THINK_START_ID),
                )
            };

        if token_id == tokenizer.eos_token_id() && phase != GenerationPhase::ToolCall {
            let in_thinking = self
                .tools
                .as_ref()
                .map(|controller| controller.thinking().in_thinking_block())
                .unwrap_or_else(|| self.thinking.in_thinking_block());
            if in_thinking {
                token_id = THINK_END_ID;
                forced = true;
            } else {
                self.finish_reason = Some(FinishReason::EosToken);
                return Ok(None);
            }
        }

        let phase = if self.tools.is_none() {
            phase_for_token(&self.thinking, self.config.thinking_mode, token_id)
        } else {
            phase
        };
        let token_text = match &self.tools {
            Some(controller) => controller.token_text(token_id, tokenizer)?,
            None => tokenizer.decode(&[token_id])?,
        };
        let generation_step = GenerationStep {
            step: self.generated_ids.len() + 1,
            token_id,
            token_text: token_text.clone(),
            candidates,
            phase,
            forced,
        };

        if let Some(controller) = &mut self.tools {
            controller.on_token(token_id, &token_text, tokenizer)?;
        } else {
            let _ = self.thinking.on_token(token_id);
        }
        self.generated_ids.push(token_id);
        self.raw_text.push_str(&token_text);
        if phase == GenerationPhase::Thinking && !is_thinking_marker(token_id) {
            self.thinking_text.push_str(&token_text);
            self.thinking_token_ids.push(token_id);
        } else if phase == GenerationPhase::Answer {
            self.answer_text.push_str(&token_text);
            self.answer_token_ids.push(token_id);
        } else if phase == GenerationPhase::ToolCall {
            self.tool_text.push_str(&token_text);
        }
        if self.config.capture_steps {
            self.steps.push(generation_step.clone());
        }

        if self
            .tools
            .as_ref()
            .is_some_and(ToolCallController::is_complete)
        {
            self.finish_reason = Some(FinishReason::ToolCall);
        } else if phase == GenerationPhase::Answer
            && let Some(stop_len) =
                matching_stop_len(&self.answer_text, &self.config.stop_sequences)
        {
            self.answer_text.truncate(self.answer_text.len() - stop_len);
            self.raw_text.truncate(self.raw_text.len() - stop_len);
            self.finish_reason = Some(FinishReason::StopSequence);
        } else if self.generated_ids.len() == self.max_new_tokens {
            self.finish_reason = Some(if self.generated_ids.len() == self.available {
                FinishReason::ContextLimit
            } else {
                FinishReason::MaxTokens
            });
        }

        if !self.is_finished() {
            self.pending_token = Some(token_id);
        }
        Ok(Some(generation_step))
    }

    /// Return true after the session reaches any terminal condition.
    pub fn is_finished(&self) -> bool {
        self.finish_reason.is_some()
    }

    /// Return the terminal reason, when generation has finished.
    pub fn finish_reason(&self) -> Option<FinishReason> {
        self.finish_reason
    }

    /// Return the number of prompt tokens consumed by this session.
    pub fn prompt_tokens(&self) -> usize {
        self.prompt_len
    }

    /// Return the number of completion tokens sampled so far.
    pub fn completion_tokens(&self) -> usize {
        self.generated_ids.len()
    }

    /// Consume a finished session and produce its complete output.
    pub fn into_output(self) -> Result<GenerationOutput> {
        let finish_reason = self.finish_reason.ok_or_else(|| {
            AarambhError::Config("cannot finalize an unfinished generation session".into())
        })?;
        if finish_reason != FinishReason::ToolCall
            && self
                .tools
                .as_ref()
                .is_some_and(|controller| !controller.action_is_resolved())
        {
            return Err(AarambhError::Config(
                "generation ended before the constrained tool action completed".into(),
            ));
        }
        let tool_call = self
            .tools
            .as_ref()
            .and_then(ToolCallController::tool_call)
            .cloned();
        let text = match &tool_call {
            Some(call) => serde_json::to_string(call)?,
            None => self.answer_text.clone(),
        };
        debug_assert!(tool_call.is_none() || text == self.tool_text);
        let thinking_tokens = self
            .tools
            .as_ref()
            .map(|controller| controller.thinking().tokens_used())
            .unwrap_or_else(|| self.thinking.tokens_used());
        Ok(GenerationOutput {
            text,
            raw_text: self.raw_text,
            thinking_text: self.thinking_text,
            answer_text: self.answer_text,
            token_ids: self.generated_ids.clone(),
            thinking_token_ids: self.thinking_token_ids,
            answer_token_ids: self.answer_token_ids,
            thinking_tokens,
            finish_reason,
            steps: self.steps,
            speculative_stats: None,
            tool_call,
            usage: GenerationUsage::new(self.prompt_len, self.generated_ids.len()),
        })
    }
}

fn matching_stop_len(text: &str, stops: &[String]) -> Option<usize> {
    stops
        .iter()
        .filter(|stop| text.ends_with(stop.as_str()))
        .map(String::len)
        .max()
}

struct DecodeSeed {
    prompt_len: usize,
    max_new_tokens: usize,
    available: usize,
    next_logits: Tensor,
}

fn empty_output(finish_reason: FinishReason) -> GenerationOutput {
    GenerationOutput {
        text: String::new(),
        raw_text: String::new(),
        thinking_text: String::new(),
        answer_text: String::new(),
        token_ids: Vec::new(),
        thinking_token_ids: Vec::new(),
        answer_token_ids: Vec::new(),
        thinking_tokens: 0,
        finish_reason,
        steps: Vec::new(),
        speculative_stats: None,
        tool_call: None,
        usage: GenerationUsage::default(),
    }
}

fn tool_phase(phase: ToolPhase) -> GenerationPhase {
    match phase {
        ToolPhase::Thinking => GenerationPhase::Thinking,
        ToolPhase::Control => GenerationPhase::Control,
        ToolPhase::Answer => GenerationPhase::Answer,
        ToolPhase::ToolCall => GenerationPhase::ToolCall,
    }
}

fn phase_for_token(
    thinking: &ThinkingController,
    thinking_mode: ThinkingMode,
    token_id: u32,
) -> GenerationPhase {
    if !thinking_mode.is_enabled() {
        return GenerationPhase::Answer;
    }
    if thinking.in_thinking_block() || (!thinking.has_started() && token_id == THINK_START_ID) {
        GenerationPhase::Thinking
    } else {
        GenerationPhase::Answer
    }
}

fn is_thinking_marker(token_id: u32) -> bool {
    token_id == THINK_START_ID || token_id == THINK_END_ID
}

fn last_logits(logits: &Tensor) -> Result<Tensor> {
    let dims = logits.dims();
    if dims.len() != 3 || dims[0] != 1 {
        return Err(AarambhError::Shape(format!(
            "expected logits shape [1, seq, vocab], got {dims:?}"
        )));
    }
    let seq_len = dims[1];
    let vocab = dims[2];
    Ok(logits.narrow(1, seq_len - 1, 1)?.reshape((vocab,))?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aarambh_studio_core::ModelConfig;
    use aarambh_studio_tokenizer::{
        ASSISTANT, ASSISTANT_ID, BOS, BOS_ID, ENDOFTEXT, ENDOFTEXT_ID, PAD, PAD_ID, THINK_END,
        THINK_END_ID, THINK_START, THINK_START_ID, USER, USER_ID, Vocab,
    };
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;
    use std::collections::HashMap;

    fn test_tokenizer() -> BpeTokenizer {
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
        let mut id_to_token = vec![String::new(); 12];
        for (token, id) in pairs {
            id_to_token[id as usize] = token.to_string();
        }
        BpeTokenizer {
            vocab: Vocab {
                token_to_id,
                id_to_token,
            },
            merges: vec![],
            merge_rank: HashMap::new(),
        }
    }

    fn test_engine() -> InferenceEngine {
        let device = Device::Cpu;
        let config = ModelConfig {
            vocab_size: 12,
            hidden_dim: 64,
            ffn_dim: 128,
            n_layers: 1,
            n_heads: 1,
            n_kv_heads: 1,
            max_seq_len: 16,
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
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = AarambhModel::new(&config, vb).unwrap();
        InferenceEngine::new(model, test_tokenizer(), device).unwrap()
    }

    #[test]
    fn greedy_generation_is_deterministic() {
        let mut engine1 = test_engine();
        let mut engine2 = test_engine();
        let out1 = engine1
            .generate("Hello", GenerationConfig::greedy(4))
            .unwrap();
        let out2 = engine2
            .generate("Hello", GenerationConfig::greedy(4))
            .unwrap();
        assert_eq!(out1.text, out2.text);
        assert_eq!(out1.token_ids, out2.token_ids);
    }

    #[test]
    fn exact_token_generation_matches_text_generation() {
        let mut text_engine = test_engine();
        let mut token_engine = test_engine();
        let prompt_ids = token_engine.encode_prompt("Hello").unwrap();
        let expected = text_engine
            .generate("Hello", GenerationConfig::greedy(4))
            .unwrap();
        let actual = token_engine
            .generate_from_token_ids(prompt_ids, GenerationConfig::greedy(4))
            .unwrap();
        assert_eq!(actual.token_ids, expected.token_ids);
        assert_eq!(actual.text, expected.text);
        assert_eq!(actual.usage, expected.usage);
    }

    #[test]
    fn generate_respects_max_tokens() {
        let mut engine = test_engine();
        let out = engine
            .generate("Hello", GenerationConfig::greedy(5))
            .unwrap();
        assert!(out.token_ids.len() <= 5);
    }

    #[test]
    fn thinking_mode_forces_start_and_close_tokens() {
        let mut engine = test_engine();
        let mut cfg = GenerationConfig::greedy(4);
        cfg.thinking_mode = ThinkingMode::Low;
        let out = engine.generate("Hello", cfg).unwrap();

        assert!(out.token_ids.len() >= 2);
        assert_eq!(&out.token_ids[..2], &[THINK_START_ID, THINK_END_ID]);
        assert!(
            out.raw_text
                .starts_with(&format!("{THINK_START}{THINK_END}"))
        );
        assert_eq!(out.text, out.answer_text);
        assert_eq!(out.thinking_text, "");
        assert_eq!(out.thinking_tokens, 0);
        assert_eq!(out.steps[0].phase, GenerationPhase::Thinking);
        assert!(out.steps[0].forced);
        assert_eq!(out.steps[1].phase, GenerationPhase::Thinking);
        assert!(out.steps[1].forced);
    }

    #[test]
    fn generation_output_text_is_answer_only() {
        let mut engine = test_engine();
        let mut cfg = GenerationConfig::greedy(4);
        cfg.thinking_mode = ThinkingMode::Low;
        let out = engine.generate("Hello", cfg).unwrap();

        assert_eq!(out.text, out.answer_text);
        assert!(out.raw_text.contains(THINK_START));
        assert!(out.raw_text.contains(THINK_END));
    }

    #[test]
    fn invalid_tokenizer_special_ids_are_rejected() {
        let device = Device::Cpu;
        let config = ModelConfig {
            vocab_size: 8,
            hidden_dim: 64,
            ffn_dim: 128,
            n_layers: 1,
            n_heads: 1,
            n_kv_heads: 1,
            max_seq_len: 8,
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
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = AarambhModel::new(&config, vb).unwrap();
        let tokenizer = BpeTokenizer {
            vocab: Vocab {
                token_to_id: HashMap::from([("!".to_string(), 0)]),
                id_to_token: vec!["!".to_string()],
            },
            merges: vec![],
            merge_rank: HashMap::new(),
        };
        assert!(InferenceEngine::new(model, tokenizer, device).is_err());
    }

    #[test]
    fn batched_sessions_match_independent_greedy_generation() {
        let engine = test_engine();
        let mut first = engine
            .prepare_session("Hello", GenerationConfig::greedy(4))
            .unwrap();
        let mut second = engine
            .prepare_session("He", GenerationConfig::greedy(4))
            .unwrap();

        while !first.is_finished() || !second.is_finished() {
            let mut decoding = Vec::new();
            if !first.is_finished() {
                first.advance(engine.tokenizer()).unwrap();
                if !first.is_finished() {
                    decoding.push(&mut first);
                }
            }
            if !second.is_finished() {
                second.advance(engine.tokenizer()).unwrap();
                if !second.is_finished() {
                    decoding.push(&mut second);
                }
            }
            engine.decode_sessions(&mut decoding).unwrap();
        }
        let batched_first = first.into_output().unwrap();
        let batched_second = second.into_output().unwrap();

        let independent_first = test_engine()
            .generate("Hello", GenerationConfig::greedy(4))
            .unwrap();
        let independent_second = test_engine()
            .generate("He", GenerationConfig::greedy(4))
            .unwrap();
        assert_eq!(batched_first.token_ids, independent_first.token_ids);
        assert_eq!(batched_second.token_ids, independent_second.token_ids);
        assert_eq!(batched_first.text, independent_first.text);
        assert_eq!(batched_second.text, independent_second.text);
    }

    #[test]
    fn forked_prefill_matches_independent_generation() {
        let engine = test_engine();
        let base = engine
            .prepare_session("Hello", GenerationConfig::greedy(4))
            .unwrap();
        assert_eq!(
            base.prompt_tokens(),
            engine.encode_prompt("Hello").unwrap().len()
        );
        let mut first = base
            .fork_with_config(GenerationConfig::greedy(4), engine.tokenizer())
            .unwrap();
        let mut second = base
            .fork_with_config(GenerationConfig::greedy(4), engine.tokenizer())
            .unwrap();

        while !first.is_finished() || !second.is_finished() {
            let mut pending = Vec::new();
            for session in [&mut first, &mut second] {
                if !session.is_finished() {
                    session.advance(engine.tokenizer()).unwrap();
                    if !session.is_finished() {
                        pending.push(session);
                    }
                }
            }
            engine.decode_sessions(&mut pending).unwrap();
        }

        let expected = test_engine()
            .generate("Hello", GenerationConfig::greedy(4))
            .unwrap();
        assert_eq!(first.into_output().unwrap().token_ids, expected.token_ids);
        assert_eq!(second.into_output().unwrap().token_ids, expected.token_ids);
        assert_eq!(base.completion_tokens(), 0);
    }

    #[test]
    fn fork_rejects_more_tokens_than_prefill_reserved() {
        let engine = test_engine();
        let base = engine
            .prepare_session("Hello", GenerationConfig::greedy(2))
            .unwrap();
        assert!(
            base.fork_with_config(GenerationConfig::greedy(3), engine.tokenizer())
                .is_err()
        );
    }

    #[test]
    fn stop_sequence_is_removed_from_visible_output() {
        let mut config = GenerationConfig::greedy(4);
        config.stop_sequences = vec![" ".to_string()];
        let output = test_engine().generate("Hello", config).unwrap();
        assert_eq!(output.finish_reason, FinishReason::StopSequence);
        assert!(!output.text.ends_with(' '));
        assert!(output.usage.completion_tokens > 0);
    }
}
