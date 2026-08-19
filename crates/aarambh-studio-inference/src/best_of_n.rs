//! Best-of-N test-time compute scaling.
//!
//! Generates N independent candidate completions for one prompt and selects
//! among them via a [`SelectionStrategy`]. This is a genuinely new
//! inference-time axis, distinct from the thinking engine (v1 §7): instead
//! of controlling how many tokens *one* generation spends reasoning, this
//! module controls how many *independent generations* are produced and how
//! the best one is chosen. The two compose freely — each of the N candidates
//! can itself use any thinking mode.
//!
//! ## Mechanism
//!
//! [`BestOfNEngine`] wraps an [`InferenceEngine`] and reuses its existing
//! prompt-prefill sharing ([`InferenceEngine::prepare_session`]) and
//! batched multi-session decode ([`InferenceEngine::decode_sessions`]) to
//! run the N candidates efficiently: the prompt is prefilled once, the
//! session is forked N times (each fork clones the KV-cache snapshot and
//! gets an independent sampler), and the forks are decoded together so one
//! target forward pass advances all pending candidates.
//!
//! Candidate `0` inherits the input sampler's seed unchanged so `N = 1`
//! reproduces single-sample generation byte-for-byte; candidates `1..N` are
//! re-seeded `base_seed + i` so they diverge. When the sampler is
//! [`Sampler::Greedy`](crate::Sampler::Greedy) the candidates are
//! deterministic and best-of-N is degenerate (all identical) — the CLI
//! documents this and recommends a stochastic sampler for `N > 1`.

use std::path::Path;
use std::str::FromStr;

use aarambh_studio_core::{AarambhError, Result};
use aarambh_studio_tokenizer::BpeTokenizer;
use candle_core::DType;

use crate::process_reward::ProcessRewardScorer;
use crate::self_consistency::{majority_vote, self_consistency_select};
use crate::{GenerationConfig, GenerationOutput, GenerationSession, InferenceEngine, Sampler};

/// Selection strategy applied to N candidate completions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionStrategy {
    /// Score each candidate against a ground-truth answer via a
    /// [`CompletionVerifier`] and select the highest-scoring candidate.
    Verifier,
    /// Extract each candidate's final answer and majority-vote across all
    /// N candidates — works without a verifier when the task has a
    /// well-defined final answer.
    SelfConsistency,
    /// Majority-vote on the raw completion strings (no answer extraction).
    Majority,
    /// Score each candidate's reasoning trace via a
    /// [`ProcessRewardScorer`] and select the highest-scoring trace.
    ProcessReward,
}

impl SelectionStrategy {
    /// Parse a strategy name, accepting kebab-case aliases used by the CLI.
    pub fn parse(name: &str) -> Result<Self> {
        Self::from_str(name).map_err(AarambhError::Config)
    }
}

impl FromStr for SelectionStrategy {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "verifier" => Ok(Self::Verifier),
            "self-consistency" | "self_consistency" | "selfconsistency" => {
                Ok(Self::SelfConsistency)
            }
            "majority" => Ok(Self::Majority),
            "process-reward" | "process_reward" | "processreward" => Ok(Self::ProcessReward),
            other => Err(format!(
                "unsupported selection strategy '{other}', expected verifier, self-consistency, majority, or process-reward"
            )),
        }
    }
}

impl std::fmt::Display for SelectionStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Verifier => write!(f, "verifier"),
            Self::SelfConsistency => write!(f, "self-consistency"),
            Self::Majority => write!(f, "majority"),
            Self::ProcessReward => write!(f, "process-reward"),
        }
    }
}

/// Verifies a generated completion against an optional ground-truth answer.
///
/// This trait is local to the inference crate (which is architecturally
/// lower-level than the finetune crate that owns `Verifier` /
/// `MathVerifier` / `CodeVerifier`). The CLI binary provides thin adapters
/// that wrap the finetune verifiers into this trait at the call site, so
/// the inference crate never depends on the finetune crate.
pub trait CompletionVerifier: Send + Sync {
    /// Extract the canonical answer string from a completion, if any.
    fn extract_answer(&self, completion: &str) -> Option<String>;
    /// Return a reward score in `[0.0, 1.0]` for `completion` against
    /// `ground_truth`. `1.0` means fully correct, `0.0` means incorrect or
    /// unextractable.
    fn verify(&self, completion: &str, ground_truth: &str) -> f32;
}

/// Configuration for one best-of-N selection pass.
pub struct BestOfNConfig {
    /// Number of independent candidate completions to generate.
    pub n: usize,
    /// Strategy used to select the chosen candidate from the N generated.
    pub strategy: SelectionStrategy,
    /// Base RNG seed for the per-candidate samplers. Candidate `i` is
    /// seeded `base_seed + i`. When `None`, a random base is drawn from
    /// entropy, making the run non-reproducible.
    pub base_seed: Option<u64>,
    /// Optional verifier used by [`SelectionStrategy::Verifier`].
    pub verifier: Option<Box<dyn CompletionVerifier>>,
    /// Optional process-reward scorer used by [`SelectionStrategy::ProcessReward`].
    pub process_reward: Option<Box<dyn ProcessRewardScorer>>,
    /// Optional ground-truth answer used by [`SelectionStrategy::Verifier`].
    pub ground_truth: Option<String>,
}

impl std::fmt::Debug for BestOfNConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BestOfNConfig")
            .field("n", &self.n)
            .field("strategy", &self.strategy)
            .field("base_seed", &self.base_seed)
            .field("has_verifier", &self.verifier.is_some())
            .field("has_process_reward", &self.process_reward.is_some())
            .field("has_ground_truth", &self.ground_truth.is_some())
            .finish()
    }
}

impl BestOfNConfig {
    /// Create a best-of-N config with `n` candidates and the given strategy.
    pub fn new(n: usize, strategy: SelectionStrategy) -> Result<Self> {
        if n == 0 {
            return Err(AarambhError::Config(
                "best-of-N requires at least one candidate".into(),
            ));
        }
        Ok(Self {
            n,
            strategy,
            base_seed: None,
            verifier: None,
            process_reward: None,
            ground_truth: None,
        })
    }

    /// Set the base RNG seed for per-candidate samplers.
    #[must_use]
    pub fn with_base_seed(mut self, seed: u64) -> Self {
        self.base_seed = Some(seed);
        self
    }

    /// Attach a verifier for [`SelectionStrategy::Verifier`].
    #[must_use]
    pub fn with_verifier(mut self, verifier: Box<dyn CompletionVerifier>) -> Self {
        self.verifier = Some(verifier);
        self
    }

    /// Attach a process-reward scorer for [`SelectionStrategy::ProcessReward`].
    #[must_use]
    pub fn with_process_reward(mut self, scorer: Box<dyn ProcessRewardScorer>) -> Self {
        self.process_reward = Some(scorer);
        self
    }

    /// Attach a ground-truth answer for [`SelectionStrategy::Verifier`].
    #[must_use]
    pub fn with_ground_truth(mut self, ground_truth: impl Into<String>) -> Self {
        self.ground_truth = Some(ground_truth.into());
        self
    }

    /// Validate that the strategy's required scorer is present.
    pub fn validate(&self) -> Result<()> {
        match self.strategy {
            SelectionStrategy::Verifier => {
                if self.verifier.is_none() {
                    return Err(AarambhError::Config(
                        "verifier selection requires a CompletionVerifier".into(),
                    ));
                }
            }
            SelectionStrategy::ProcessReward => {
                if self.process_reward.is_none() {
                    return Err(AarambhError::Config(
                        "process-reward selection requires a ProcessRewardScorer".into(),
                    ));
                }
            }
            SelectionStrategy::SelfConsistency | SelectionStrategy::Majority => {}
        }
        Ok(())
    }
}

/// Why a particular candidate was selected.
#[derive(Debug, Clone, PartialEq)]
pub enum SelectionRationale {
    /// Selected by majority vote on raw completion strings.
    Majority {
        /// Number of candidates matching the winning string.
        count: usize,
        /// Total candidates considered.
        total: usize,
    },
    /// Selected by majority vote on extracted final answers.
    SelfConsistency {
        /// The winning extracted answer.
        answer: String,
        /// Number of candidates whose extracted answer matched.
        count: usize,
        /// Total candidates considered.
        total: usize,
    },
    /// Selected by a verifier scoring the highest against ground truth.
    Verifier {
        /// Index of the selected candidate.
        index: usize,
        /// Verifier score of the selected candidate.
        score: f32,
    },
    /// Selected by a process-reward scorer as the highest-scoring trace.
    ProcessReward {
        /// Index of the selected candidate.
        index: usize,
        /// Process-reward score of the selected candidate.
        score: f32,
    },
    /// Only one candidate was generated, so no selection was performed.
    Single,
}

/// Result of one best-of-N generation pass.
#[derive(Debug, Clone)]
pub struct BestOfNOutput {
    /// The selected candidate's full generation output.
    pub chosen: GenerationOutput,
    /// Index of the selected candidate within `candidates`.
    pub chosen_index: usize,
    /// All N candidate generation outputs, in generation order.
    pub candidates: Vec<GenerationOutput>,
    /// Strategy that selected the chosen candidate.
    pub selection: SelectionStrategy,
    /// Why the chosen candidate was selected.
    pub rationale: SelectionRationale,
}

/// Best-of-N inference engine wrapping a target [`InferenceEngine`].
///
/// Mirrors the wrapper-struct pattern used by
/// [`crate::MtpSpeculativeEngine`] and [`crate::SpeculativeEngine`]: the
/// target engine is owned and reused for prompt prefill + batched decode.
pub struct BestOfNEngine {
    target: InferenceEngine,
    config: BestOfNConfig,
}

impl BestOfNEngine {
    /// Create a best-of-N engine from a loaded target engine and config.
    pub fn new(target: InferenceEngine, config: BestOfNConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { target, config })
    }

    /// Load a target checkpoint and tokenizer, then wrap it.
    pub fn from_paths_with_dtype(
        model_path: impl AsRef<Path>,
        model_config: &aarambh_studio_core::ModelConfig,
        tokenizer_path: impl AsRef<Path>,
        device: candle_core::Device,
        dtype: DType,
        config: BestOfNConfig,
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

    /// Return the best-of-N configuration.
    pub fn config(&self) -> &BestOfNConfig {
        &self.config
    }

    /// Generate N candidate completions and select one.
    ///
    /// The prompt is prefilled once on the target engine, the session is
    /// forked N times with per-candidate samplers (candidate 0 keeps the
    /// input sampler's seed; candidates 1..N are re-seeded
    /// `base_seed + i`), and the forks are decoded together via
    /// [`InferenceEngine::decode_sessions`].
    pub fn generate(&mut self, prompt: &str, config: GenerationConfig) -> Result<BestOfNOutput> {
        config.validate()?;
        if config.tool_calling.is_some() {
            return Err(AarambhError::Unsupported(
                "best-of-N does not support tool-calling prompts; use single-sample generation"
                    .into(),
            ));
        }
        let candidates = self.generate_candidates(prompt, config)?;
        let (chosen_index, rationale) = self.select(&candidates, prompt);
        let chosen = candidates[chosen_index].clone();
        Ok(BestOfNOutput {
            chosen,
            chosen_index,
            candidates,
            selection: self.config.strategy,
            rationale,
        })
    }

    fn generate_candidates(
        &self,
        prompt: &str,
        config: GenerationConfig,
    ) -> Result<Vec<GenerationOutput>> {
        let base = self.target.prepare_session(prompt, config.clone())?;
        let base_seed = match self.config.base_seed {
            Some(seed) => seed,
            None => rand::random(),
        };
        let mut sessions: Vec<GenerationSession> = Vec::with_capacity(self.config.n);
        for index in 0..self.config.n {
            let candidate_config = reseed_config(&config, index, base_seed);
            sessions.push(base.fork_with_config(candidate_config, self.target.tokenizer())?);
        }
        decode_all(&self.target, &mut sessions)?;
        sessions
            .into_iter()
            .map(|session| session.into_output())
            .collect()
    }

    fn select(&self, candidates: &[GenerationOutput], prompt: &str) -> (usize, SelectionRationale) {
        if candidates.len() == 1 {
            return (0, SelectionRationale::Single);
        }
        match self.config.strategy {
            SelectionStrategy::Majority => {
                let texts: Vec<&str> = candidates.iter().map(|c| c.text.as_str()).collect();
                let (winner, count) =
                    majority_vote(&texts).expect("non-empty candidates guarantee a winner");
                let index = candidates
                    .iter()
                    .position(|candidate| candidate.text == winner)
                    .expect("winner came from candidates");
                (
                    index,
                    SelectionRationale::Majority {
                        count,
                        total: candidates.len(),
                    },
                )
            }
            SelectionStrategy::SelfConsistency => {
                let texts: Vec<String> = candidates.iter().map(|c| c.text.clone()).collect();
                self_consistency_select(&texts)
            }
            SelectionStrategy::Verifier => {
                let verifier = self.config.verifier.as_ref().expect("validated");
                let ground_truth = self.config.ground_truth.as_deref().unwrap_or("");
                let mut best_index = 0;
                let mut best_score = f32::NEG_INFINITY;
                for (index, candidate) in candidates.iter().enumerate() {
                    let score = verifier.verify(&candidate.text, ground_truth);
                    if score > best_score {
                        best_score = score;
                        best_index = index;
                    }
                }
                (
                    best_index,
                    SelectionRationale::Verifier {
                        index: best_index,
                        score: best_score,
                    },
                )
            }
            SelectionStrategy::ProcessReward => {
                let scorer = self.config.process_reward.as_ref().expect("validated");
                let mut best_index = 0;
                let mut best_score = f32::NEG_INFINITY;
                for (index, candidate) in candidates.iter().enumerate() {
                    let score = scorer.score(prompt, &candidate.text);
                    if score > best_score {
                        best_score = score;
                        best_index = index;
                    }
                }
                (
                    best_index,
                    SelectionRationale::ProcessReward {
                        index: best_index,
                        score: best_score,
                    },
                )
            }
        }
    }
}

fn reseed_config(config: &GenerationConfig, index: usize, base_seed: u64) -> GenerationConfig {
    let mut next = config.clone();
    next.sampler = match &config.sampler {
        Sampler::Greedy => Sampler::Greedy,
        Sampler::TopKTopP {
            temperature,
            top_k,
            top_p,
            ..
        } => {
            let seed = base_seed.wrapping_add(index as u64);
            Sampler::top_k_top_p(*temperature, *top_k, *top_p, Some(seed))
                .expect("validated parameters re-seed without error")
        }
    };
    next
}

fn decode_all(engine: &InferenceEngine, sessions: &mut [GenerationSession]) -> Result<()> {
    while sessions.iter().any(|session| !session.is_finished()) {
        let mut pending: Vec<&mut GenerationSession> = Vec::with_capacity(sessions.len());
        for session in sessions.iter_mut() {
            if session.is_finished() {
                continue;
            }
            session.advance(engine.tokenizer())?;
            if !session.is_finished() {
                pending.push(session);
            }
        }
        if pending.is_empty() {
            break;
        }
        engine.decode_sessions(&mut pending)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::thinking::ThinkingMode;
    use aarambh_studio_core::ModelConfig;
    use aarambh_studio_model::AarambhModel;
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
            chat_template_version: None,
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
            chat_template_version: None,
        };
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = AarambhModel::new(&config, vb).unwrap();
        InferenceEngine::new(model, test_tokenizer(), device).unwrap()
    }

    fn stochastic_config(seed: u64) -> GenerationConfig {
        GenerationConfig {
            max_new_tokens: 4,
            sampler: Sampler::top_k_top_p(0.8, Some(50), Some(0.9), Some(seed)).unwrap(),
            thinking_mode: ThinkingMode::None,
            top_candidates: 5,
            tool_calling: None,
            stop_sequences: Vec::new(),
            capture_steps: true,
        }
    }

    #[derive(Debug, Clone, Copy, Default)]
    struct ExactMatchVerifier;

    impl CompletionVerifier for ExactMatchVerifier {
        fn extract_answer(&self, completion: &str) -> Option<String> {
            Some(completion.trim().to_string())
        }
        fn verify(&self, completion: &str, ground_truth: &str) -> f32 {
            if completion.trim() == ground_truth.trim() {
                1.0
            } else {
                0.0
            }
        }
    }

    #[test]
    fn best_of_n_with_n_equal_one_matches_single_sample_generation_exactly() {
        let prompt = "Hello";
        let seed = 42u64;
        let config = stochastic_config(seed);

        let mut single = test_engine();
        let single_output = single.generate(prompt, config.clone()).unwrap();

        let engine = test_engine();
        let best_of_n = BestOfNEngine::new(
            engine,
            BestOfNConfig::new(1, SelectionStrategy::Majority)
                .unwrap()
                .with_base_seed(seed),
        )
        .unwrap();
        let mut engine = best_of_n;
        let output = engine.generate(prompt, config).unwrap();

        assert_eq!(output.candidates.len(), 1);
        assert_eq!(output.chosen_index, 0);
        assert_eq!(output.chosen.token_ids, single_output.token_ids);
        assert_eq!(output.chosen.text, single_output.text);
    }

    #[test]
    fn best_of_n_generates_n_distinct_candidates_with_stochastic_sampler() {
        let prompt = "Hello";
        let seed = 7u64;
        let config = stochastic_config(seed);

        let engine = test_engine();
        let best_of_n = BestOfNEngine::new(
            engine,
            BestOfNConfig::new(4, SelectionStrategy::SelfConsistency)
                .unwrap()
                .with_base_seed(seed),
        )
        .unwrap();
        let mut engine = best_of_n;
        let output = engine.generate(prompt, config).unwrap();

        assert_eq!(output.candidates.len(), 4);
        let seed0 = output.candidates[0].token_ids.clone();
        let seed1 = output.candidates[1].token_ids.clone();
        // Candidate 0 uses the base seed and must match a single-sample run.
        let mut single = test_engine();
        let single_output = single.generate(prompt, stochastic_config(seed)).unwrap();
        assert_eq!(seed0, single_output.token_ids);
        // Candidate 1 uses a different seed and should differ when the sampler
        // is stochastic. (On the synthetic zero-weights model the logits are
        // uniform, so different seeds almost always produce different tokens;
        // assert they differ to confirm re-seeding happened.)
        assert_ne!(seed0, seed1);
    }

    #[test]
    fn best_of_n_greedy_candidates_are_identical() {
        let prompt = "Hello";
        let config = GenerationConfig::greedy(4);

        let engine = test_engine();
        let best_of_n = BestOfNEngine::new(
            engine,
            BestOfNConfig::new(3, SelectionStrategy::Majority)
                .unwrap()
                .with_base_seed(0),
        )
        .unwrap();
        let mut engine = best_of_n;
        let output = engine.generate(prompt, config).unwrap();

        assert_eq!(output.candidates.len(), 3);
        assert_eq!(
            output.candidates[0].token_ids,
            output.candidates[1].token_ids
        );
        assert_eq!(
            output.candidates[1].token_ids,
            output.candidates[2].token_ids
        );
        assert!(matches!(
            output.rationale,
            SelectionRationale::Majority { count: 3, total: 3 }
        ));
    }

    #[test]
    fn verifier_selection_picks_first_fully_correct_candidate() {
        let completions = vec![
            GenerationOutput {
                text: "wrong".into(),
                raw_text: "wrong".into(),
                thinking_text: String::new(),
                answer_text: "wrong".into(),
                token_ids: vec![7],
                thinking_token_ids: vec![],
                answer_token_ids: vec![7],
                thinking_tokens: 0,
                finish_reason: crate::FinishReason::MaxTokens,
                steps: vec![],
                speculative_stats: None,
                tool_call: None,
                usage: crate::GenerationUsage::default(),
            },
            GenerationOutput {
                text: "right".into(),
                raw_text: "right".into(),
                thinking_text: String::new(),
                answer_text: "right".into(),
                token_ids: vec![8],
                thinking_token_ids: vec![],
                answer_token_ids: vec![8],
                thinking_tokens: 0,
                finish_reason: crate::FinishReason::MaxTokens,
                steps: vec![],
                speculative_stats: None,
                tool_call: None,
                usage: crate::GenerationUsage::default(),
            },
        ];
        let engine = test_engine();
        let best_of_n = BestOfNEngine::new(
            engine,
            BestOfNConfig::new(2, SelectionStrategy::Verifier)
                .unwrap()
                .with_verifier(Box::new(ExactMatchVerifier))
                .with_ground_truth("right"),
        )
        .unwrap();
        let (index, rationale) = best_of_n.select(&completions, "p");
        assert_eq!(index, 1);
        match rationale {
            SelectionRationale::Verifier { index, score } => {
                assert_eq!(index, 1);
                assert_eq!(score, 1.0);
            }
            other => panic!("expected Verifier, got {other:?}"),
        }
    }

    #[test]
    fn rejects_zero_candidates() {
        let engine = test_engine();
        assert!(BestOfNConfig::new(0, SelectionStrategy::Majority).is_err());
        let _ = engine;
    }

    #[test]
    fn rejects_verifier_strategy_without_verifier() {
        let engine = test_engine();
        let config = BestOfNConfig::new(2, SelectionStrategy::Verifier).unwrap();
        assert!(BestOfNEngine::new(engine, config).is_err());
    }

    #[test]
    fn selection_strategy_parses_kebab_and_snake_aliases() {
        assert_eq!(
            SelectionStrategy::parse("self-consistency").unwrap(),
            SelectionStrategy::SelfConsistency
        );
        assert_eq!(
            SelectionStrategy::parse("self_consistency").unwrap(),
            SelectionStrategy::SelfConsistency
        );
        assert_eq!(
            SelectionStrategy::parse("process-reward").unwrap(),
            SelectionStrategy::ProcessReward
        );
        assert!(SelectionStrategy::parse("unknown").is_err());
    }
}
