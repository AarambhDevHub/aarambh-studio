use aarambh_studio_core::{AarambhError, Configurable, Result, TokenizerLike};
use aarambh_studio_inference::{
    FinishReason, GenerationConfig, GenerationSession, InferenceEngine, Sampler,
};
use serde::{Deserialize, Serialize};

use crate::config::DistillConfig;
use crate::dataset::PromptExample;

/// Serializable rollout stop reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RolloutFinish {
    /// End-of-sequence token was sampled.
    Eos,
    /// Requested token budget was exhausted.
    MaxTokens,
    /// Model context capacity was exhausted.
    ContextLimit,
    /// A stop sequence ended generation.
    StopSequence,
    /// Tool grammar completed generation.
    ToolCall,
}

impl From<FinishReason> for RolloutFinish {
    fn from(value: FinishReason) -> Self {
        match value {
            FinishReason::EosToken => Self::Eos,
            FinishReason::MaxTokens => Self::MaxTokens,
            FinishReason::ContextLimit => Self::ContextLimit,
            FinishReason::StopSequence => Self::StopSequence,
            FinishReason::ToolCall => Self::ToolCall,
        }
    }
}

/// One student-generated completion used for immediate policy replay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudentRollout {
    /// Stable prompt identifier.
    pub prompt_id: String,
    /// Exact prompt text.
    pub prompt: String,
    /// Prompt token IDs from the inference engine.
    pub prompt_token_ids: Vec<u32>,
    /// Generated completion token IDs, including terminal EOS when sampled.
    pub completion_token_ids: Vec<u32>,
    /// Decoded generated text.
    pub completion_text: String,
    /// Completion-token policy mask; forced controller tokens are false.
    pub loss_mask: Vec<bool>,
    /// Index within the prompt's sampled rollout group.
    pub rollout_index: usize,
    /// Reason generation stopped.
    pub finish_reason: RolloutFinish,
}

/// Generate student rollouts using prefill reuse and batched inference decode.
pub fn generate_student_rollouts(
    engine: &InferenceEngine,
    prompts: &[PromptExample],
    config: &DistillConfig,
    optimizer_step: usize,
    seed: u64,
) -> Result<Vec<StudentRollout>> {
    config.validate()?;
    if prompts.is_empty() {
        return Err(AarambhError::Config(
            "student rollout generation requires at least one prompt".into(),
        ));
    }

    let mut sessions =
        Vec::<GenerationSession>::with_capacity(prompts.len() * config.rollouts_per_prompt);
    let mut metadata = Vec::with_capacity(sessions.capacity());
    for prompt in prompts {
        let prompt_ids = engine.encode_prompt(&prompt.prompt)?;
        if prompt_ids.len() + config.max_new_tokens > engine.model().config().max_seq_len {
            return Err(AarambhError::Shape(format!(
                "prompt '{}' plus {} rollout tokens exceeds student max_seq_len {}",
                prompt.id,
                config.max_new_tokens,
                engine.model().config().max_seq_len
            )));
        }
        let base_config =
            generation_config(config, rollout_seed(seed, optimizer_step, &prompt.id, 0))?;
        let base = engine.prepare_session(&prompt.prompt, base_config)?;
        for rollout_index in 0..config.rollouts_per_prompt {
            let generation = generation_config(
                config,
                rollout_seed(seed, optimizer_step, &prompt.id, rollout_index),
            )?;
            sessions.push(base.fork_with_config(generation, engine.tokenizer())?);
            metadata.push((prompt.clone(), prompt_ids.clone(), rollout_index));
        }
    }

    while sessions.iter().any(|session| !session.is_finished()) {
        for session in sessions.iter_mut().filter(|session| !session.is_finished()) {
            let _ = session.advance(engine.tokenizer())?;
        }
        let mut pending = sessions
            .iter_mut()
            .filter(|session| !session.is_finished())
            .collect::<Vec<_>>();
        if !pending.is_empty() {
            engine.decode_sessions(&mut pending)?;
        }
    }

    let eos_id = engine.tokenizer().eos_token_id();
    sessions
        .into_iter()
        .zip(metadata)
        .map(|(session, (prompt, prompt_token_ids, rollout_index))| {
            let output = session.into_output()?;
            let mut completion_token_ids = output.token_ids;
            let mut loss_mask = output
                .steps
                .iter()
                .map(|step| !step.forced)
                .collect::<Vec<_>>();
            if output.finish_reason == FinishReason::EosToken {
                completion_token_ids.push(eos_id);
                loss_mask.push(true);
            }
            if completion_token_ids.len() != loss_mask.len() {
                return Err(AarambhError::Config(
                    "rollout token and forced-token metadata lengths differ".into(),
                ));
            }
            Ok(StudentRollout {
                prompt_id: prompt.id,
                prompt: prompt.prompt,
                prompt_token_ids,
                completion_token_ids,
                completion_text: output.raw_text,
                loss_mask,
                rollout_index,
                finish_reason: output.finish_reason.into(),
            })
        })
        .collect()
}

fn generation_config(config: &DistillConfig, seed: u64) -> Result<GenerationConfig> {
    Ok(GenerationConfig {
        max_new_tokens: config.max_new_tokens,
        sampler: Sampler::top_k_top_p(config.temperature, config.top_k, config.top_p, Some(seed))?,
        thinking_mode: config.thinking.into(),
        top_candidates: 0,
        tool_calling: None,
        stop_sequences: Vec::new(),
        capture_steps: true,
    })
}

fn rollout_seed(base: u64, step: usize, prompt_id: &str, rollout_index: usize) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in prompt_id.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    base ^ hash.rotate_left(17) ^ (step as u64).rotate_left(31) ^ rollout_index as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use aarambh_studio_core::ModelConfig;
    use aarambh_studio_model::AarambhModel;
    use aarambh_studio_tokenizer::{
        ASSISTANT, ASSISTANT_ID, BOS, BOS_ID, BpeTokenizer, ENDOFTEXT, ENDOFTEXT_ID, PAD, PAD_ID,
        THINK_END, THINK_END_ID, THINK_START, THINK_START_ID, USER, USER_ID, Vocab,
    };
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;
    use std::collections::HashMap;

    fn engine() -> InferenceEngine {
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
        let tokenizer = BpeTokenizer {
            vocab: Vocab {
                token_to_id,
                id_to_token,
            },
            merges: Vec::new(),
            merge_rank: HashMap::new(),
            chat_template_version: None,
        };
        let config = ModelConfig {
            vocab_size: 12,
            hidden_dim: 64,
            ffn_dim: 128,
            n_layers: 1,
            n_heads: 1,
            n_kv_heads: 1,
            max_seq_len: 16,
            rope_theta: 10_000.0,
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
        let device = Device::Cpu;
        let model = AarambhModel::new(&config, VarBuilder::zeros(DType::F32, &device)).unwrap();
        InferenceEngine::new(model, tokenizer, device).unwrap()
    }

    #[test]
    fn rollouts_reuse_inference_decode_and_are_seed_deterministic() {
        let engine = engine();
        let prompts = vec![PromptExample {
            id: "hello".into(),
            prompt: "Hello".into(),
        }];
        let config = DistillConfig {
            rollouts_per_prompt: 2,
            max_new_tokens: 3,
            temperature: 0.8,
            top_p: Some(0.95),
            top_k: Some(8),
            ..DistillConfig::default()
        };
        let first = generate_student_rollouts(&engine, &prompts, &config, 7, 42).unwrap();
        let second = generate_student_rollouts(&engine, &prompts, &config, 7, 42).unwrap();

        assert_eq!(first.len(), 2);
        assert_eq!(
            first
                .iter()
                .map(|rollout| &rollout.completion_token_ids)
                .collect::<Vec<_>>(),
            second
                .iter()
                .map(|rollout| &rollout.completion_token_ids)
                .collect::<Vec<_>>()
        );
        let encoded = engine.encode_prompt("Hello").unwrap();
        assert!(
            first
                .iter()
                .all(|rollout| rollout.prompt_token_ids == encoded)
        );
        assert!(
            first
                .iter()
                .all(|rollout| { rollout.completion_token_ids.len() == rollout.loss_mask.len() })
        );
    }
}
