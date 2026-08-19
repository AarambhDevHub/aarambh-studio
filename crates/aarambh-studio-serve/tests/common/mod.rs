//! Shared test helpers for the Phase 51 acceptance tests.
//!
//! Not every helper is used by every test binary; allow dead code to keep
//! the shared module ergonomic across `auth`, `prefix_cache`, and
//! `tenant_isolation`.
#![allow(dead_code)]

use std::collections::HashMap;
use std::time::Duration;

use aarambh_studio_core::ModelConfig;
use aarambh_studio_inference::{
    GenerationConfig, GenerationOutput, GenerationSession, InferenceEngine,
};
use aarambh_studio_model::AarambhModel;
use aarambh_studio_tokenizer::{
    ASSISTANT, ASSISTANT_ID, BOS, BOS_ID, BpeTokenizer, ENDOFTEXT, ENDOFTEXT_ID, PAD, PAD_ID,
    THINK_END, THINK_END_ID, THINK_START, THINK_START_ID, USER, USER_ID, Vocab,
};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

/// Build a tiny CPU inference engine suitable for tests, with `n_layers`
/// transformer layers.
pub fn test_engine(n_layers: usize) -> InferenceEngine {
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
        chat_template_version: None,
    };
    let config = ModelConfig {
        vocab_size: 12,
        hidden_dim: 64,
        ffn_dim: 128,
        n_layers,
        n_heads: 1,
        n_kv_heads: 1,
        max_seq_len: 256,
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
    let device = Device::Cpu;
    let model = AarambhModel::new(&config, VarBuilder::zeros(DType::F32, &device)).unwrap();
    InferenceEngine::new(model, tokenizer, device).unwrap()
}

/// Return a small greedy generation config that produces `max_new_tokens`
/// answer tokens.
pub fn greedy_config(max_new_tokens: usize) -> GenerationConfig {
    GenerationConfig::greedy(max_new_tokens)
}

/// Drive `session` to completion on `engine`, returning the final output.
/// Takes ownership of the session so it can call `into_output`.
pub fn drive_to_completion(
    engine: &InferenceEngine,
    mut session: GenerationSession,
) -> GenerationOutput {
    while !session.is_finished() {
        if session.advance(engine.tokenizer()).unwrap().is_some() {
            // step consumed
        }
        if !session.is_finished() {
            engine.decode_sessions(&mut [&mut session]).unwrap();
        }
    }
    session.into_output().unwrap()
}

/// A prompt with a long shared prefix suitable for prefix-cache tests: the
/// first ~20 chars are identical across calls (system prompt + role start),
/// and only the trailing user content differs.
pub fn repeated_prompt(suffix: &str) -> String {
    let prefix = "System: You are a precise assistant.\n\
                  Assistant: I will answer precisely.\n\
                  User: ";
    format!("{prefix}{suffix}\n")
}

/// Convenience: a server batcher config with tiny capacities for fast tests.
pub fn test_batcher_config() -> aarambh_studio_serve::BatcherConfig {
    aarambh_studio_serve::BatcherConfig {
        max_batch_size: 4,
        queue_capacity: 16,
        batch_wait: Duration::from_millis(1),
        prefill_chunk_size: 32,
    }
}
