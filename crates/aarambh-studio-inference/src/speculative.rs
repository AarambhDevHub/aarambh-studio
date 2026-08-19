use std::path::Path;

use aarambh_studio_core::{AarambhError, Configurable, Result, TokenizerLike};
use aarambh_studio_tokenizer::{BpeTokenizer, THINK_END_ID, THINK_START_ID};
use candle_core::{DType, Tensor};

use crate::tool_calling::{TokenConstraint, ToolCall, ToolCallController, ToolPhase};
use crate::{
    FinishReason, GenerationConfig, GenerationOutput, GenerationPhase, GenerationStep,
    GenerationUsage, InferenceEngine, KvCache, Sampler, ThinkingController, ThinkingMode,
    TokenCandidate,
};

/// Runtime controls for exact speculative decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeculativeConfig {
    /// Maximum draft tokens proposed in one verification round.
    pub num_draft_tokens: usize,
}

impl SpeculativeConfig {
    /// Create a configuration with the requested proposal width.
    pub fn new(num_draft_tokens: usize) -> Result<Self> {
        if num_draft_tokens == 0 {
            return Err(AarambhError::Config(
                "num_draft_tokens must be greater than zero".into(),
            ));
        }
        Ok(Self { num_draft_tokens })
    }
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            num_draft_tokens: 4,
        }
    }
}

/// Counters collected during one speculative generation request.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpeculativeStats {
    /// Proposal source used for this request.
    pub proposal_source: SpeculativeProposalSource,
    /// Number of target verification rounds.
    pub rounds: usize,
    /// Number of tokens proposed by the draft model.
    pub draft_tokens_proposed: usize,
    /// Number of draft proposals accepted by the target model.
    pub draft_tokens_accepted: usize,
    /// Number of proposals discarded after rejection or early termination.
    pub draft_tokens_rejected: usize,
    /// Number of target decode forwards, excluding prompt prefill.
    pub target_decode_forwards: usize,
    /// Number of draft decode forwards, excluding prompt prefill.
    pub draft_decode_forwards: usize,
    /// Number of auxiliary MTP-head forwards.
    pub mtp_head_forwards: usize,
}

/// Source used to propose speculative tokens.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SpeculativeProposalSource {
    /// A separately loaded draft model proposed tokens.
    #[default]
    ExternalDraft,
    /// Auxiliary heads in the target checkpoint proposed tokens.
    Mtp,
}

impl SpeculativeStats {
    /// Return the accepted fraction of proposed draft tokens.
    pub fn acceptance_rate(&self) -> f64 {
        if self.draft_tokens_proposed == 0 {
            0.0
        } else {
            self.draft_tokens_accepted as f64 / self.draft_tokens_proposed as f64
        }
    }

    /// Return committed draft tokens per target verification forward.
    pub fn accepted_tokens_per_target_forward(&self) -> f64 {
        if self.target_decode_forwards == 0 {
            0.0
        } else {
            self.draft_tokens_accepted as f64 / self.target_decode_forwards as f64
        }
    }
}

/// Two-model inference engine implementing exact speculative decoding.
pub struct SpeculativeEngine {
    target: InferenceEngine,
    draft: InferenceEngine,
    config: SpeculativeConfig,
}

impl SpeculativeEngine {
    /// Create an engine from loaded target and draft inference engines.
    pub fn new(
        target: InferenceEngine,
        draft: InferenceEngine,
        config: SpeculativeConfig,
    ) -> Result<Self> {
        if config.num_draft_tokens == 0 {
            return Err(AarambhError::Config(
                "num_draft_tokens must be greater than zero".into(),
            ));
        }
        if !target.device().same_device(draft.device()) {
            return Err(AarambhError::Config(
                "draft and target models must use the same device".into(),
            ));
        }
        target.tokenizer().validate_compatible(draft.tokenizer())?;
        Ok(Self {
            target,
            draft,
            config,
        })
    }

    /// Load target and draft models and tokenizers using a shared dtype and device.
    #[allow(clippy::too_many_arguments)]
    pub fn from_paths_with_dtype(
        target_model_path: impl AsRef<Path>,
        target_model_config: &aarambh_studio_core::ModelConfig,
        target_tokenizer_path: impl AsRef<Path>,
        draft_model_path: impl AsRef<Path>,
        draft_model_config: &aarambh_studio_core::ModelConfig,
        draft_tokenizer_path: impl AsRef<Path>,
        device: candle_core::Device,
        dtype: DType,
        config: SpeculativeConfig,
    ) -> Result<Self> {
        let target = InferenceEngine::from_paths_with_dtype(
            target_model_path,
            target_model_config,
            target_tokenizer_path,
            device.clone(),
            dtype,
        )?;
        let draft = InferenceEngine::from_paths_with_dtype(
            draft_model_path,
            draft_model_config,
            draft_tokenizer_path,
            device,
            dtype,
        )?;
        Self::new(target, draft, config)
    }

    /// Return the shared tokenizer used by draft and target models.
    pub fn tokenizer(&self) -> &BpeTokenizer {
        self.target.tokenizer()
    }

    /// Return the target inference engine.
    pub fn target(&self) -> &InferenceEngine {
        &self.target
    }

    /// Return the draft inference engine.
    pub fn draft(&self) -> &InferenceEngine {
        &self.draft
    }

    /// Generate text without per-token callbacks.
    pub fn generate(&mut self, prompt: &str, config: GenerationConfig) -> Result<GenerationOutput> {
        self.generate_with_callback(prompt, config, |_| Ok(()))
    }

    /// Generate text and invoke `on_step` for every committed token.
    pub fn generate_with_callback<F>(
        &mut self,
        prompt: &str,
        mut config: GenerationConfig,
        mut on_step: F,
    ) -> Result<GenerationOutput>
    where
        F: FnMut(&GenerationStep) -> Result<()>,
    {
        config.validate()?;
        if !config.stop_sequences.is_empty() {
            return Err(AarambhError::Unsupported(
                "stop sequences are not yet supported with speculative decoding".into(),
            ));
        }
        let effective_prompt = match &config.tool_calling {
            Some(tools) => tools.render_prompt(prompt)?,
            None => prompt.to_string(),
        };
        let mut prompt_ids = self.tokenizer().encode(&effective_prompt)?;
        if prompt_ids.is_empty() {
            prompt_ids.push(self.tokenizer().bos_token_id().ok_or_else(|| {
                AarambhError::Config(
                    "prompt produced no tokens and tokenizer has no BOS token".into(),
                )
            })?);
        }

        let target_limit = self.target.model().config().max_seq_len;
        let draft_limit = self.draft.model().config().max_seq_len;
        let max_seq_len = target_limit.min(draft_limit);
        if prompt_ids.len() >= max_seq_len {
            return Err(AarambhError::Shape(format!(
                "prompt has {} tokens but speculative context limit is {max_seq_len} (target={target_limit}, draft={draft_limit})",
                prompt_ids.len()
            )));
        }
        let available = max_seq_len - prompt_ids.len();
        let max_new_tokens = config.max_new_tokens.min(available);
        let mut stats = SpeculativeStats::default();
        let mut controller = match config.tool_calling.clone() {
            Some(tools) => DecodeController::Tools(ToolCallController::new(
                config.thinking_mode,
                max_new_tokens,
                tools,
                self.tokenizer(),
            )?),
            None => DecodeController::Thinking(ThinkingController::for_generation(
                config.thinking_mode,
                max_new_tokens,
            )),
        };
        if max_new_tokens == 0 {
            return OutputState::new(0, config.thinking_mode).finish(
                FinishReason::ContextLimit,
                Some(stats),
                &controller,
                prompt_ids.len(),
            );
        }

        let device = self.target.device();
        let prompt = Tensor::from_vec(prompt_ids.clone(), (1, prompt_ids.len()), device)?;
        let mut target_cache = KvCache::for_model(self.target.model());
        let mut draft_cache = KvCache::for_model(self.draft.model());
        let target_logits =
            self.target
                .model()
                .forward_with_cache(&prompt, 0, target_cache.layers_mut())?;
        let draft_logits =
            self.draft
                .model()
                .forward_with_cache(&prompt, 0, draft_cache.layers_mut())?;
        let mut target_next = Some(last_logits(&target_logits)?);
        let mut draft_next = last_logits(&draft_logits)?;
        let mut pending_target_token = None;
        let mut output = OutputState::new(max_new_tokens, config.thinking_mode);
        let eos = self.tokenizer().eos_token_id();
        let mut finish_reason = FinishReason::MaxTokens;

        'generation: while output.token_ids.len() < max_new_tokens {
            let remaining = max_new_tokens - output.token_ids.len();
            let proposal_count = self.config.num_draft_tokens.min(remaining);
            let committed_len = prompt_ids.len() + output.token_ids.len();
            let target_base_len = target_cache.seqlen();
            let mut target_snapshot = Some(target_cache.snapshot());
            let mut draft_snapshot = Some(draft_cache.snapshot());
            debug_assert_eq!(
                target_base_len + usize::from(pending_target_token.is_some()),
                committed_len
            );
            debug_assert_eq!(draft_cache.seqlen(), committed_len);

            let mut proposal_state = controller.clone();
            let mut proposals = Vec::with_capacity(proposal_count);
            for _ in 0..proposal_count {
                let logits = draft_next.to_vec1::<f32>()?;
                let (distribution, forced) = constrained_distribution(
                    &config.sampler,
                    &logits,
                    &mut proposal_state,
                    eos,
                    self.tokenizer(),
                )?;
                let token_id = config.sampler.sample_probabilities(&distribution)?;
                proposals.push(Proposal {
                    token_id,
                    distribution,
                    forced,
                });
                stats.draft_tokens_proposed += 1;
                if token_id == eos && proposal_state.eos_terminates() {
                    break;
                }
                let token_text = proposal_state.token_text(token_id, self.tokenizer())?;
                proposal_state.on_token(token_id, &token_text, self.tokenizer())?;
                let offset = draft_cache.seqlen();
                let input = Tensor::from_vec(vec![token_id], (1, 1), device)?;
                let logits = self.draft.model().forward_with_cache(
                    &input,
                    offset,
                    draft_cache.layers_mut(),
                )?;
                draft_next = last_logits(&logits)?;
                stats.draft_decode_forwards += 1;
                if proposal_state.tool_complete() {
                    break;
                }
            }

            let mut target_input_ids =
                Vec::with_capacity(proposals.len() + usize::from(pending_target_token.is_some()));
            if let Some(token_id) = pending_target_token {
                target_input_ids.push(token_id);
            }
            target_input_ids.extend(proposals.iter().map(|proposal| proposal.token_id));
            let target_input_len = target_input_ids.len();
            let target_input =
                Tensor::from_vec(target_input_ids.clone(), (1, target_input_len), device)?;
            let verified = self.target.model().forward_with_cache(
                &target_input,
                target_base_len,
                target_cache.layers_mut(),
            )?;
            stats.rounds += 1;
            stats.target_decode_forwards += 1;
            let verified_rows = verified.squeeze(0)?.to_vec2::<f32>()?;
            let has_pending = pending_target_token.is_some();
            pending_target_token = None;
            let mut rejected = false;

            for (index, proposal) in proposals.iter().enumerate() {
                let initial_target_logits;
                let target_logits = if has_pending || index > 0 {
                    &verified_rows[index - usize::from(!has_pending)]
                } else {
                    initial_target_logits = target_next
                        .as_ref()
                        .expect("initial target logits must be available")
                        .to_vec1::<f32>()?;
                    &initial_target_logits
                };
                let (target_distribution, target_forced) = constrained_distribution(
                    &config.sampler,
                    target_logits,
                    &mut controller,
                    eos,
                    self.tokenizer(),
                )?;
                let accepted = accept_proposal(
                    &mut config.sampler,
                    proposal.token_id,
                    &proposal.distribution,
                    &target_distribution,
                );
                let candidates = Sampler::top_candidates_from_probabilities(
                    &target_distribution,
                    config.top_candidates,
                )?;

                if accepted {
                    stats.draft_tokens_accepted += 1;
                    if proposal.token_id == eos && controller.eos_terminates() {
                        finish_reason = FinishReason::EosToken;
                        break 'generation;
                    }
                    output.commit(
                        proposal.token_id,
                        candidates,
                        target_forced || proposal.forced,
                        &mut controller,
                        self.tokenizer(),
                        &mut on_step,
                    )?;
                    if controller.tool_complete() {
                        finish_reason = FinishReason::ToolCall;
                        break 'generation;
                    }
                    if output.token_ids.len() == max_new_tokens {
                        break 'generation;
                    }
                    continue;
                }

                rejected = true;
                let replacement = if config.sampler.is_deterministic() {
                    config.sampler.sample_probabilities(&target_distribution)?
                } else {
                    let residual =
                        residual_distribution(&target_distribution, &proposal.distribution)?;
                    config.sampler.sample_probabilities(&residual)?
                };
                let accepted_before = index;
                target_cache.restore(
                    target_snapshot
                        .take()
                        .expect("speculative target snapshot is available"),
                );
                draft_cache.restore(
                    draft_snapshot
                        .take()
                        .expect("speculative draft snapshot is available"),
                );

                let target_replay_len = usize::from(has_pending) + accepted_before;
                if target_replay_len > 0 {
                    let replay = Tensor::from_vec(
                        target_input_ids[..target_replay_len].to_vec(),
                        (1, target_replay_len),
                        device,
                    )?;
                    let _ = self.target.model().forward_with_cache(
                        &replay,
                        target_base_len,
                        target_cache.layers_mut(),
                    )?;
                }
                if accepted_before > 0 {
                    let accepted_ids = proposals[..accepted_before]
                        .iter()
                        .map(|proposal| proposal.token_id)
                        .collect::<Vec<_>>();
                    let replay = Tensor::from_vec(accepted_ids, (1, accepted_before), device)?;
                    let _ = self.draft.model().forward_with_cache(
                        &replay,
                        committed_len,
                        draft_cache.layers_mut(),
                    )?;
                    stats.draft_decode_forwards += 1;
                }
                if replacement == eos && controller.eos_terminates() {
                    finish_reason = FinishReason::EosToken;
                    break 'generation;
                }
                output.commit(
                    replacement,
                    candidates,
                    target_forced,
                    &mut controller,
                    self.tokenizer(),
                    &mut on_step,
                )?;
                if controller.tool_complete() {
                    finish_reason = FinishReason::ToolCall;
                    break 'generation;
                }
                let input = Tensor::from_vec(vec![replacement], (1, 1), device)?;
                let offset = draft_cache.seqlen();
                let logits = self.draft.model().forward_with_cache(
                    &input,
                    offset,
                    draft_cache.layers_mut(),
                )?;
                draft_next = last_logits(&logits)?;
                stats.draft_decode_forwards += 1;
                pending_target_token = Some(replacement);
                break;
            }

            if rejected {
                target_next = None;
                continue;
            }
            if output.token_ids.len() == max_new_tokens {
                break;
            }

            let bonus_index = proposals.len() - usize::from(!has_pending);
            let bonus_logits = &verified_rows[bonus_index];
            let (bonus_distribution, forced) = constrained_distribution(
                &config.sampler,
                bonus_logits,
                &mut controller,
                eos,
                self.tokenizer(),
            )?;
            let candidates = Sampler::top_candidates_from_probabilities(
                &bonus_distribution,
                config.top_candidates,
            )?;
            let bonus = config.sampler.sample_probabilities(&bonus_distribution)?;
            if bonus == eos && controller.eos_terminates() {
                finish_reason = FinishReason::EosToken;
                break;
            }
            output.commit(
                bonus,
                candidates,
                forced,
                &mut controller,
                self.tokenizer(),
                &mut on_step,
            )?;
            if controller.tool_complete() {
                finish_reason = FinishReason::ToolCall;
                break;
            }
            let input = Tensor::from_vec(vec![bonus], (1, 1), device)?;
            let offset = draft_cache.seqlen();
            let logits =
                self.draft
                    .model()
                    .forward_with_cache(&input, offset, draft_cache.layers_mut())?;
            draft_next = last_logits(&logits)?;
            stats.draft_decode_forwards += 1;
            pending_target_token = Some(bonus);
            target_next = None;
        }

        if output.token_ids.len() == available && finish_reason == FinishReason::MaxTokens {
            finish_reason = FinishReason::ContextLimit;
        }
        stats.draft_tokens_rejected = stats
            .draft_tokens_proposed
            .saturating_sub(stats.draft_tokens_accepted);
        if finish_reason != FinishReason::ToolCall && !controller.action_is_resolved() {
            return Err(AarambhError::Config(
                "generation ended before the constrained tool action completed".into(),
            ));
        }
        output.finish(finish_reason, Some(stats), &controller, prompt_ids.len())
    }
}

pub(crate) struct Proposal {
    pub(crate) token_id: u32,
    pub(crate) distribution: Vec<f32>,
    pub(crate) forced: bool,
}

pub(crate) struct OutputState {
    pub(crate) token_ids: Vec<u32>,
    raw_text: String,
    thinking_text: String,
    answer_text: String,
    thinking_token_ids: Vec<u32>,
    answer_token_ids: Vec<u32>,
    tool_text: String,
    steps: Vec<GenerationStep>,
    thinking_mode: ThinkingMode,
}

impl OutputState {
    pub(crate) fn new(capacity: usize, thinking_mode: ThinkingMode) -> Self {
        Self {
            token_ids: Vec::with_capacity(capacity),
            raw_text: String::new(),
            thinking_text: String::new(),
            answer_text: String::new(),
            thinking_token_ids: Vec::new(),
            answer_token_ids: Vec::new(),
            tool_text: String::new(),
            steps: Vec::with_capacity(capacity),
            thinking_mode,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn commit<F>(
        &mut self,
        token_id: u32,
        candidates: Vec<TokenCandidate>,
        forced: bool,
        controller: &mut DecodeController,
        tokenizer: &BpeTokenizer,
        on_step: &mut F,
    ) -> Result<()>
    where
        F: FnMut(&GenerationStep) -> Result<()>,
    {
        let phase = controller.phase_for_next(self.thinking_mode, token_id);
        let token_text = controller.token_text(token_id, tokenizer)?;
        let step = GenerationStep {
            step: self.token_ids.len() + 1,
            token_id,
            token_text: token_text.clone(),
            candidates,
            phase,
            forced,
        };
        on_step(&step)?;
        controller.on_token(token_id, &token_text, tokenizer)?;
        self.token_ids.push(token_id);
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
        self.steps.push(step);
        Ok(())
    }

    pub(crate) fn finish(
        self,
        finish_reason: FinishReason,
        speculative_stats: Option<SpeculativeStats>,
        controller: &DecodeController,
        prompt_tokens: usize,
    ) -> Result<GenerationOutput> {
        let thinking_tokens = self.thinking_token_ids.len();
        let completion_tokens = self.token_ids.len();
        let tool_call = controller.tool_call().cloned();
        let text = match &tool_call {
            Some(call) => serde_json::to_string(call)?,
            None => self.answer_text.clone(),
        };
        debug_assert!(tool_call.is_none() || text == self.tool_text);
        Ok(GenerationOutput {
            text,
            raw_text: self.raw_text,
            thinking_text: self.thinking_text,
            answer_text: self.answer_text,
            token_ids: self.token_ids,
            thinking_token_ids: self.thinking_token_ids,
            answer_token_ids: self.answer_token_ids,
            thinking_tokens,
            finish_reason,
            steps: self.steps,
            speculative_stats,
            tool_call,
            usage: GenerationUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        })
    }
}

pub(crate) fn constrained_distribution(
    sampler: &Sampler,
    logits: &[f32],
    controller: &mut DecodeController,
    eos_token_id: u32,
    tokenizer: &BpeTokenizer,
) -> Result<(Vec<f32>, bool)> {
    let constraint = controller.constraint(tokenizer)?;
    let (mut probabilities, forced) = match constraint {
        TokenConstraint::Any => (sampler.probabilities(logits)?, false),
        TokenConstraint::Allowed(allowed) => {
            (sampler.probabilities_allowed(logits, &allowed)?, false)
        }
        TokenConstraint::Forced(token_id) => {
            let mut probabilities = vec![0.0; logits.len()];
            let index = token_id as usize;
            if index >= probabilities.len() {
                return Err(AarambhError::Shape(format!(
                    "forced token {token_id} exceeds vocabulary size {}",
                    probabilities.len()
                )));
            }
            probabilities[index] = 1.0;
            (probabilities, true)
        }
    };
    if controller.in_thinking_block() {
        let eos = eos_token_id as usize;
        let think_end = THINK_END_ID as usize;
        if eos < probabilities.len() && think_end < probabilities.len() && eos != think_end {
            probabilities[think_end] += probabilities[eos];
            probabilities[eos] = 0.0;
        }
    }
    Ok((probabilities, forced))
}

#[derive(Debug, Clone)]
pub(crate) enum DecodeController {
    Thinking(ThinkingController),
    Tools(ToolCallController),
}

impl DecodeController {
    pub(crate) fn constraint(&mut self, tokenizer: &BpeTokenizer) -> Result<TokenConstraint> {
        match self {
            Self::Thinking(thinking) => Ok(match thinking.take_forced_token() {
                Some(force) => TokenConstraint::Forced(force.token_id()),
                None => TokenConstraint::Any,
            }),
            Self::Tools(tools) => tools.constraint(tokenizer),
        }
    }

    pub(crate) fn phase_for_next(
        &self,
        thinking_mode: ThinkingMode,
        token_id: u32,
    ) -> GenerationPhase {
        match self {
            Self::Thinking(thinking) => phase_for_token(thinking, thinking_mode, token_id),
            Self::Tools(tools) => match tools.phase_for_next() {
                ToolPhase::Thinking => GenerationPhase::Thinking,
                ToolPhase::Control => GenerationPhase::Control,
                ToolPhase::Answer => GenerationPhase::Answer,
                ToolPhase::ToolCall => GenerationPhase::ToolCall,
            },
        }
    }

    pub(crate) fn on_token(
        &mut self,
        token_id: u32,
        token_text: &str,
        tokenizer: &BpeTokenizer,
    ) -> Result<()> {
        match self {
            Self::Thinking(thinking) => {
                thinking.on_token(token_id);
                Ok(())
            }
            Self::Tools(tools) => tools.on_token(token_id, token_text, tokenizer),
        }
    }

    pub(crate) fn in_thinking_block(&self) -> bool {
        match self {
            Self::Thinking(thinking) => thinking.in_thinking_block(),
            Self::Tools(tools) => tools.thinking().in_thinking_block(),
        }
    }

    pub(crate) fn tool_complete(&self) -> bool {
        matches!(self, Self::Tools(tools) if tools.is_complete())
    }

    pub(crate) fn action_is_resolved(&self) -> bool {
        match self {
            Self::Thinking(_) => true,
            Self::Tools(tools) => tools.action_is_resolved(),
        }
    }

    pub(crate) fn tool_call(&self) -> Option<&ToolCall> {
        match self {
            Self::Thinking(_) => None,
            Self::Tools(tools) => tools.tool_call(),
        }
    }

    pub(crate) fn eos_terminates(&self) -> bool {
        !self.in_thinking_block()
            && !matches!(
                self,
                Self::Tools(tools) if tools.phase_for_next() == ToolPhase::ToolCall
            )
    }

    pub(crate) fn token_text(&self, token_id: u32, tokenizer: &BpeTokenizer) -> Result<String> {
        match self {
            Self::Thinking(_) => tokenizer.decode(&[token_id]),
            Self::Tools(tools) => tools.token_text(token_id, tokenizer),
        }
    }
}

pub(crate) fn accept_proposal(
    sampler: &mut Sampler,
    token_id: u32,
    draft: &[f32],
    target: &[f32],
) -> bool {
    if sampler.is_deterministic() {
        return argmax(draft) == argmax(target);
    }
    let index = token_id as usize;
    let q = draft.get(index).copied().unwrap_or(0.0);
    let p = target.get(index).copied().unwrap_or(0.0);
    q > 0.0 && sampler.draw_uniform() < (p / q).min(1.0)
}

pub(crate) fn residual_distribution(target: &[f32], draft: &[f32]) -> Result<Vec<f32>> {
    if target.len() != draft.len() {
        return Err(AarambhError::Shape(format!(
            "target distribution has {} entries but draft has {}",
            target.len(),
            draft.len()
        )));
    }
    let mut residual = target
        .iter()
        .zip(draft)
        .map(|(p, q)| (p - q).max(0.0))
        .collect::<Vec<_>>();
    let sum = residual.iter().sum::<f32>();
    if !sum.is_finite() || sum <= f32::EPSILON {
        return Err(AarambhError::Config(
            "speculative residual distribution has zero probability mass".into(),
        ));
    }
    for probability in &mut residual {
        *probability /= sum;
    }
    Ok(residual)
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .unwrap_or(0)
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

pub(crate) fn last_logits(logits: &Tensor) -> Result<Tensor> {
    let dims = logits.dims();
    if dims.len() != 3 || dims[0] != 1 || dims[1] == 0 {
        return Err(AarambhError::Shape(format!(
            "expected logits shape [1, seq, vocab], got {dims:?}"
        )));
    }
    Ok(logits.narrow(1, dims[1] - 1, 1)?.reshape((dims[2],))?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aarambh_studio_core::ModelConfig;
    use aarambh_studio_tokenizer::{
        ASSISTANT, ASSISTANT_ID, BOS, BOS_ID, ENDOFTEXT, ENDOFTEXT_ID, PAD, PAD_ID, THINK_END,
        THINK_START, USER, USER_ID, Vocab,
    };
    use candle_core::Device;
    use candle_nn::VarBuilder;
    use std::collections::HashMap;

    fn tokenizer() -> BpeTokenizer {
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
            merges: Vec::new(),
            merge_rank: HashMap::new(),
            chat_template_version: None,
        }
    }

    fn engine(tokenizer: BpeTokenizer) -> InferenceEngine {
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
            chat_template_version: None,
        };
        let model = aarambh_studio_model::AarambhModel::new(
            &config,
            VarBuilder::zeros(DType::F32, &device),
        )
        .unwrap();
        InferenceEngine::new(model, tokenizer, device).unwrap()
    }

    #[test]
    fn residual_is_positive_difference_normalized() {
        let residual = residual_distribution(&[0.1, 0.2, 0.7], &[0.6, 0.3, 0.1]).unwrap();
        assert_eq!(residual[0], 0.0);
        assert_eq!(residual[1], 0.0);
        assert!((residual[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn deterministic_acceptance_requires_matching_argmax() {
        let mut sampler = Sampler::greedy();
        assert!(accept_proposal(&mut sampler, 1, &[0.0, 1.0], &[0.0, 1.0]));
        assert!(!accept_proposal(&mut sampler, 1, &[0.0, 1.0], &[1.0, 0.0]));
    }

    #[test]
    fn speculative_config_rejects_zero_width() {
        assert!(SpeculativeConfig::new(0).is_err());
    }

    #[test]
    fn sampled_speculation_matches_target_distribution() {
        let target = [0.1, 0.2, 0.7];
        let draft = [0.6, 0.3, 0.1];
        let mut sampler = Sampler::top_k_top_p(1.0, None, None, Some(7)).unwrap();
        let residual = residual_distribution(&target, &draft).unwrap();
        let mut counts = [0usize; 3];
        let samples = 50_000usize;
        for _ in 0..samples {
            let proposal = sampler.sample_probabilities(&draft).unwrap();
            let token = if accept_proposal(&mut sampler, proposal, &draft, &target) {
                proposal
            } else {
                sampler.sample_probabilities(&residual).unwrap()
            };
            counts[token as usize] += 1;
        }
        for (count, expected) in counts.into_iter().zip(target) {
            let observed = count as f32 / samples as f32;
            assert!(
                (observed - expected).abs() < 0.01,
                "{observed} != {expected}"
            );
        }
    }

    #[test]
    fn full_rejection_uses_target_residual() {
        let draft = [1.0, 0.0];
        let target = [0.0, 1.0];
        let residual = residual_distribution(&target, &draft).unwrap();
        let mut sampler = Sampler::top_k_top_p(1.0, None, None, Some(9)).unwrap();
        assert!(!accept_proposal(&mut sampler, 0, &draft, &target));
        assert_eq!(sampler.sample_probabilities(&residual).unwrap(), 1);
    }

    #[test]
    fn greedy_speculation_matches_target_only_generation() {
        let mut baseline = engine(tokenizer());
        let expected = baseline
            .generate("Hello", GenerationConfig::greedy(4))
            .unwrap();
        let mut speculative = SpeculativeEngine::new(
            engine(tokenizer()),
            engine(tokenizer()),
            SpeculativeConfig::new(4).unwrap(),
        )
        .unwrap();
        let actual = speculative
            .generate("Hello", GenerationConfig::greedy(4))
            .unwrap();
        assert_eq!(actual.token_ids, expected.token_ids);
        assert_eq!(actual.text, expected.text);
        let stats = actual.speculative_stats.unwrap();
        assert_eq!(stats.target_decode_forwards, 1);
        assert_eq!(stats.draft_tokens_accepted, 4);
    }

    #[test]
    fn mismatched_tokenizers_are_rejected() {
        let target = tokenizer();
        let mut draft = tokenizer();
        draft.merges.push(("H".into(), "e".into()));
        let result =
            SpeculativeEngine::new(engine(target), engine(draft), SpeculativeConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn speculative_thinking_matches_target_only_generation() {
        let mut config = GenerationConfig::greedy(4);
        config.thinking_mode = ThinkingMode::Low;
        let mut baseline = engine(tokenizer());
        let expected = baseline.generate("Hello", config.clone()).unwrap();
        let mut speculative = SpeculativeEngine::new(
            engine(tokenizer()),
            engine(tokenizer()),
            SpeculativeConfig::new(2).unwrap(),
        )
        .unwrap();
        let actual = speculative.generate("Hello", config).unwrap();
        assert_eq!(actual.token_ids, expected.token_ids);
        assert_eq!(actual.thinking_text, expected.thinking_text);
        assert_eq!(actual.answer_text, expected.answer_text);
    }

    #[test]
    fn callbacks_only_contain_committed_tokens() {
        let mut speculative = SpeculativeEngine::new(
            engine(tokenizer()),
            engine(tokenizer()),
            SpeculativeConfig::new(4).unwrap(),
        )
        .unwrap();
        let mut callback_ids = Vec::new();
        let output = speculative
            .generate_with_callback("Hello", GenerationConfig::greedy(6), |step| {
                callback_ids.push(step.token_id);
                Ok(())
            })
            .unwrap();
        assert_eq!(callback_ids, output.token_ids);
    }

    #[test]
    fn speculative_generation_respects_shared_context_limit() {
        let mut speculative = SpeculativeEngine::new(
            engine(tokenizer()),
            engine(tokenizer()),
            SpeculativeConfig::new(4).unwrap(),
        )
        .unwrap();
        let output = speculative
            .generate("Hello", GenerationConfig::greedy(20))
            .unwrap();
        assert_eq!(output.token_ids.len(), 11);
        assert_eq!(output.finish_reason, FinishReason::ContextLimit);
    }
}
