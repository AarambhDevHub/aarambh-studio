use std::path::Path;

use aarambh_studio_core::{AarambhError, Configurable, Result, TokenizerLike};
use aarambh_studio_tokenizer::BpeTokenizer;
use candle_core::{DType, Tensor};

use crate::speculative::{
    DecodeController, OutputState, Proposal, SpeculativeConfig, SpeculativeProposalSource,
    SpeculativeStats, accept_proposal, constrained_distribution, last_logits,
    residual_distribution,
};
use crate::tool_calling::ToolCallController;
use crate::{
    FinishReason, GenerationConfig, GenerationOutput, GenerationStep, InferenceEngine, KvCache,
    Sampler, ThinkingController,
};

/// One-checkpoint speculative decoder backed by auxiliary MTP heads.
pub struct MtpSpeculativeEngine {
    target: InferenceEngine,
    config: SpeculativeConfig,
}

impl MtpSpeculativeEngine {
    /// Create an MTP speculative engine from a loaded target model.
    pub fn new(target: InferenceEngine, config: SpeculativeConfig) -> Result<Self> {
        let mtp = target.model().config().mtp.as_ref().ok_or_else(|| {
            AarambhError::Config(
                "internal speculative decoding requires model.mtp configuration".into(),
            )
        })?;
        if target.model().mtp_heads().len() != mtp.auxiliary_head_count() {
            return Err(AarambhError::Config(
                "target model MTP heads do not match model.mtp configuration".into(),
            ));
        }
        if config.num_draft_tokens < 2 {
            return Err(AarambhError::Config(
                "MTP speculative decoding requires at least two proposal tokens".into(),
            ));
        }
        if config.num_draft_tokens > mtp.num_future_tokens {
            return Err(AarambhError::Config(format!(
                "requested {} MTP proposal tokens but checkpoint horizon is {}",
                config.num_draft_tokens, mtp.num_future_tokens
            )));
        }
        Ok(Self { target, config })
    }

    /// Load an MTP-enabled target checkpoint and tokenizer.
    pub fn from_paths_with_dtype(
        model_path: impl AsRef<Path>,
        model_config: &aarambh_studio_core::ModelConfig,
        tokenizer_path: impl AsRef<Path>,
        device: candle_core::Device,
        dtype: DType,
        config: SpeculativeConfig,
    ) -> Result<Self> {
        let target = InferenceEngine::from_paths_with_dtype(
            model_path,
            model_config,
            tokenizer_path,
            device,
            dtype,
        )?;
        Self::new(target, config)
    }

    /// Return the tokenizer used by the target model.
    pub fn tokenizer(&self) -> &BpeTokenizer {
        self.target.tokenizer()
    }

    /// Return the target inference engine.
    pub fn target(&self) -> &InferenceEngine {
        &self.target
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

        let max_seq_len = self.target.model().config().max_seq_len;
        if prompt_ids.len() >= max_seq_len {
            return Err(AarambhError::Shape(format!(
                "prompt has {} tokens but model context limit is {max_seq_len}",
                prompt_ids.len()
            )));
        }
        let available = max_seq_len - prompt_ids.len();
        let max_new_tokens = config.max_new_tokens.min(available);
        let mut stats = SpeculativeStats {
            proposal_source: SpeculativeProposalSource::Mtp,
            ..SpeculativeStats::default()
        };
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
        let prefill =
            self.target
                .model()
                .forward_with_cache_output(&prompt, 0, target_cache.layers_mut())?;
        let mut target_next = last_logits(&prefill.logits)?;
        let mut anchor_hidden = last_hidden(&prefill.final_hidden_states)?;
        let mut output = OutputState::new(max_new_tokens, config.thinking_mode);
        let eos = self.tokenizer().eos_token_id();
        let mut finish_reason = FinishReason::MaxTokens;

        'generation: while output.token_ids.len() < max_new_tokens {
            let remaining = max_new_tokens - output.token_ids.len();
            let proposal_count = self.config.num_draft_tokens.min(remaining);
            let target_base_len = target_cache.seqlen();
            let target_snapshot = target_cache.snapshot();
            let mut proposal_state = controller.clone();
            let mut proposals = Vec::with_capacity(proposal_count);

            for proposal_index in 0..proposal_count {
                let proposal_logits = if proposal_index == 0 {
                    target_next.to_vec1::<f32>()?
                } else {
                    let intervening_ids = proposals
                        .iter()
                        .map(|proposal: &Proposal| proposal.token_id)
                        .collect::<Vec<_>>();
                    let intervening =
                        Tensor::from_vec(intervening_ids, (1, proposal_index), device)?;
                    let prediction = self.target.model().forward_mtp_head(
                        proposal_index - 1,
                        &anchor_hidden,
                        &intervening,
                    )?;
                    stats.mtp_head_forwards += 1;
                    last_logits(&prediction.logits)?.to_vec1::<f32>()?
                };
                let (distribution, forced) = constrained_distribution(
                    &config.sampler,
                    &proposal_logits,
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
                if proposal_state.tool_complete() {
                    break;
                }
            }

            let proposal_ids = proposals
                .iter()
                .map(|proposal| proposal.token_id)
                .collect::<Vec<_>>();
            let verification =
                Tensor::from_vec(proposal_ids.clone(), (1, proposal_ids.len()), device)?;
            let verified = self.target.model().forward_with_cache_output(
                &verification,
                target_base_len,
                target_cache.layers_mut(),
            )?;
            stats.rounds += 1;
            stats.target_decode_forwards += 1;
            let verified_rows = verified.logits.squeeze(0)?.to_vec2::<f32>()?;
            let initial_target_logits = target_next.to_vec1::<f32>()?;
            let mut rejected = false;

            for (index, proposal) in proposals.iter().enumerate() {
                let target_logits = if index == 0 {
                    &initial_target_logits
                } else {
                    &verified_rows[index - 1]
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
                target_cache.restore(target_snapshot);
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

                let mut replay_ids = proposal_ids[..index].to_vec();
                replay_ids.push(replacement);
                let replay_len = replay_ids.len();
                let replay = Tensor::from_vec(replay_ids, (1, replay_len), device)?;
                let replayed = self.target.model().forward_with_cache_output(
                    &replay,
                    target_base_len,
                    target_cache.layers_mut(),
                )?;
                stats.target_decode_forwards += 1;
                target_next = last_logits(&replayed.logits)?;
                anchor_hidden = last_hidden(&replayed.final_hidden_states)?;
                break;
            }

            if rejected {
                continue;
            }
            target_next = last_logits(&verified.logits)?;
            anchor_hidden = last_hidden(&verified.final_hidden_states)?;
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

fn last_hidden(hidden: &Tensor) -> Result<Tensor> {
    let dims = hidden.dims();
    if dims.len() != 3 || dims[0] != 1 || dims[1] == 0 {
        return Err(AarambhError::Shape(format!(
            "expected hidden-state shape [1, seq, hidden], got {dims:?}"
        )));
    }
    Ok(hidden.narrow(1, dims[1] - 1, 1)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aarambh_studio_core::{ModelConfig, MtpConfig};
    use aarambh_studio_tokenizer::{
        ASSISTANT, ASSISTANT_ID, BOS, BOS_ID, ENDOFTEXT, ENDOFTEXT_ID, PAD, PAD_ID, THINK_END,
        THINK_END_ID, THINK_START, THINK_START_ID, USER, USER_ID, Vocab,
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

    fn engine(with_mtp: bool) -> InferenceEngine {
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
            mtp: with_mtp.then_some(MtpConfig {
                num_future_tokens: 3,
                aux_loss_weight: 0.3,
            }),
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
        InferenceEngine::new(model, tokenizer(), device).unwrap()
    }

    #[test]
    fn constructor_requires_mtp_and_respects_horizon() {
        assert!(
            MtpSpeculativeEngine::new(engine(false), SpeculativeConfig::new(2).unwrap()).is_err()
        );
        assert!(
            MtpSpeculativeEngine::new(engine(true), SpeculativeConfig::new(4).unwrap()).is_err()
        );
        assert!(
            MtpSpeculativeEngine::new(engine(true), SpeculativeConfig::new(3).unwrap()).is_ok()
        );
    }

    #[test]
    fn greedy_internal_speculation_matches_target_generation() {
        let mut baseline = engine(true);
        let expected = baseline
            .generate("Hello", GenerationConfig::greedy(6))
            .unwrap();
        let mut speculative =
            MtpSpeculativeEngine::new(engine(true), SpeculativeConfig::new(3).unwrap()).unwrap();
        let actual = speculative
            .generate("Hello", GenerationConfig::greedy(6))
            .unwrap();
        assert_eq!(actual.token_ids, expected.token_ids);
        assert_eq!(actual.text, expected.text);
        let stats = actual.speculative_stats.unwrap();
        assert_eq!(stats.proposal_source, SpeculativeProposalSource::Mtp);
        assert!(stats.mtp_head_forwards > 0);
    }

    #[test]
    fn callbacks_only_include_committed_tokens() {
        let mut speculative =
            MtpSpeculativeEngine::new(engine(true), SpeculativeConfig::new(3).unwrap()).unwrap();
        let mut callback_ids = Vec::new();
        let output = speculative
            .generate_with_callback("Hello", GenerationConfig::greedy(5), |step| {
                callback_ids.push(step.token_id);
                Ok(())
            })
            .unwrap();
        assert_eq!(callback_ids, output.token_ids);
    }
}
