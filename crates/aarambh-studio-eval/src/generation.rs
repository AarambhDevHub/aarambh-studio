use aarambh_studio_core::{AarambhError, Result, TokenizerLike};
use aarambh_studio_inference::{ThinkingController, ThinkingMode};
use aarambh_studio_tokenizer::{THINK_END_ID, THINK_START_ID};
use candle_core::Tensor;

use crate::harness::EvalContext;

/// Generate text greedily from a prompt.
pub fn greedy_generate(
    context: &EvalContext,
    prompt: &str,
    max_new_tokens: usize,
) -> Result<String> {
    let mut prompt_ids = context.tokenizer().encode(prompt)?;
    if prompt_ids.is_empty() {
        if let Some(bos) = context.tokenizer().bos_token_id() {
            prompt_ids.push(bos);
        } else {
            return Err(AarambhError::Tokenizer(
                "prompt produced no tokens and tokenizer has no BOS token".into(),
            ));
        }
    }
    if prompt_ids.len() >= context.max_seq_len() {
        return Err(AarambhError::Shape(format!(
            "prompt length {} leaves no room in max_seq_len {}",
            prompt_ids.len(),
            context.max_seq_len()
        )));
    }

    let budget = max_new_tokens.min(context.max_seq_len() - prompt_ids.len());
    let mut caches = context.model().empty_kv_cache();
    let input = Tensor::from_vec(prompt_ids.clone(), (1, prompt_ids.len()), context.device())?;
    let logits = context.model().forward_with_cache(&input, 0, &mut caches)?;
    let mut next_logits = last_logits(&logits)?;
    let mut generated = Vec::with_capacity(budget);

    for step in 0..budget {
        let logits_vec = next_logits.to_vec1::<f32>()?;
        let token_id = argmax(&logits_vec) as u32;
        if token_id == context.tokenizer().eos_token_id() {
            break;
        }
        generated.push(token_id);
        context.record_context_len(prompt_ids.len() + generated.len());
        if step + 1 == budget {
            break;
        }
        let offset = prompt_ids.len() + generated.len() - 1;
        let input = Tensor::from_vec(vec![token_id], (1, 1), context.device())?;
        let logits = context
            .model()
            .forward_with_cache(&input, offset, &mut caches)?;
        next_logits = last_logits(&logits)?;
    }

    context.tokenizer().decode(&generated)
}

/// Token accounting for a thinking-aware greedy generation.
///
/// `thinking_tokens` counts content tokens emitted inside the `<think>` block
/// (excluding the markers themselves), `completion_tokens` counts answer
/// tokens emitted after the block closes, and `total_tokens` is their sum.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ThinkingGenerationResult {
    /// Decoded completion text (markers stripped from the visible answer).
    pub text: String,
    /// Content tokens spent inside the thinking block.
    pub thinking_tokens: usize,
    /// Answer tokens emitted after the thinking block closed.
    pub completion_tokens: usize,
}

impl ThinkingGenerationResult {
    /// Total content tokens generated (thinking + completion, excluding markers).
    pub fn total_tokens(&self) -> usize {
        self.thinking_tokens + self.completion_tokens
    }
}

/// Generate text greedily while reusing the inference crate's
/// [`ThinkingController`] for forced `<think>`/`</think>` markers and budget
/// enforcement. Generation is deterministic (greedy argmax) regardless of the
/// thinking mode, matching the eval harness's deterministic-defaults policy.
///
/// The effective thinking budget is clamped by the controller to
/// `min(mode.budget(), max_new_tokens - reserve)` exactly as it is during
/// normal inference, so Max mode never exceeds the configured generation
/// budget.
pub fn greedy_generate_with_thinking(
    context: &EvalContext,
    prompt: &str,
    max_new_tokens: usize,
    thinking_mode: ThinkingMode,
) -> Result<ThinkingGenerationResult> {
    let mut prompt_ids = context.tokenizer().encode(prompt)?;
    if prompt_ids.is_empty() {
        if let Some(bos) = context.tokenizer().bos_token_id() {
            prompt_ids.push(bos);
        } else {
            return Err(AarambhError::Tokenizer(
                "prompt produced no tokens and tokenizer has no BOS token".into(),
            ));
        }
    }
    if prompt_ids.len() >= context.max_seq_len() {
        return Err(AarambhError::Shape(format!(
            "prompt length {} leaves no room in max_seq_len {}",
            prompt_ids.len(),
            context.max_seq_len()
        )));
    }

    let budget = max_new_tokens.min(context.max_seq_len() - prompt_ids.len());
    let mut thinking = ThinkingController::for_generation(thinking_mode, max_new_tokens);
    let mut caches = context.model().empty_kv_cache();
    let input = Tensor::from_vec(prompt_ids.clone(), (1, prompt_ids.len()), context.device())?;
    let logits = context.model().forward_with_cache(&input, 0, &mut caches)?;
    let mut next_logits = last_logits(&logits)?;
    let mut generated = Vec::with_capacity(budget);
    let mut completion_tokens = 0usize;
    let eos = context.tokenizer().eos_token_id();

    for step in 0..budget {
        let (mut token_id, forced) = match thinking.take_forced_token() {
            Some(force) => (force.token_id(), true),
            None => {
                let logits_vec = next_logits.to_vec1::<f32>()?;
                (argmax(&logits_vec) as u32, false)
            }
        };

        // An EOS sampled inside an open thinking block forces the closing
        // marker instead of ending generation, exactly as the inference engine
        // does. A forced or naturally-sampled EOS outside thinking ends the
        // turn.
        if token_id == eos && !forced {
            if thinking.in_thinking_block() {
                token_id = THINK_END_ID;
            } else {
                break;
            }
        }

        let is_marker = token_id == THINK_START_ID || token_id == THINK_END_ID;
        let in_block_before = thinking.in_thinking_block();
        // on_token advances controller state (opens/closes the block, counts
        // thinking content, and may queue the next forced close marker).
        let _ = thinking.on_token(token_id);
        if !is_marker && !in_block_before {
            completion_tokens += 1;
        }

        generated.push(token_id);
        context.record_context_len(prompt_ids.len() + generated.len());
        if step + 1 == budget {
            break;
        }
        let offset = prompt_ids.len() + generated.len() - 1;
        let input = Tensor::from_vec(vec![token_id], (1, 1), context.device())?;
        let logits = context
            .model()
            .forward_with_cache(&input, offset, &mut caches)?;
        next_logits = last_logits(&logits)?;
    }

    let thinking_tokens = thinking.tokens_used();
    let text = context.tokenizer().decode(&generated)?;
    Ok(ThinkingGenerationResult {
        text,
        thinking_tokens,
        completion_tokens,
    })
}

fn last_logits(logits: &Tensor) -> Result<Tensor> {
    let dims = logits.dims();
    if dims.len() != 3 || dims[1] == 0 {
        return Err(AarambhError::Shape(format!(
            "expected logits [batch, seq, vocab], got {dims:?}"
        )));
    }
    Ok(logits.narrow(1, dims[1] - 1, 1)?.squeeze(1)?.squeeze(0)?)
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

/// Result of one best-of-N generation pass in the eval harness.
///
/// `candidates` holds the N decoded completion strings (in generation
/// order); `chosen_index` is the candidate selected by the configured
/// [`aarambh_studio_inference::SelectionStrategy`]; `rationale` describes
/// why it was chosen.
#[derive(Debug, Clone)]
pub struct BestOfNResult {
    /// All N candidate completion strings, in generation order.
    pub candidates: Vec<String>,
    /// Index of the selected candidate within `candidates`.
    pub chosen_index: usize,
    /// Why the chosen candidate was selected.
    pub rationale: aarambh_studio_inference::SelectionRationale,
}

/// Function scoring a candidate completion against a ground-truth answer.
pub type VerifierFn<'a> = &'a dyn Fn(&str, &str) -> f32;

/// Options for one [`best_of_n_generate`] call.
///
/// Groups the best-of-N generation parameters so the function signature
/// stays below clippy's argument-count threshold and the verifier closure
/// type is named once.
#[derive(Clone)]
pub struct BestOfNOptions<'a> {
    /// Number of independent candidate completions to generate.
    pub n: usize,
    /// Selection strategy applied to the N candidates.
    pub strategy: aarambh_studio_inference::SelectionStrategy,
    /// Base RNG seed; candidate `i` is seeded `base_seed + i`.
    pub base_seed: u64,
    /// Sampling temperature for each candidate.
    pub temperature: f32,
    /// Optional top-k filter.
    pub top_k: Option<usize>,
    /// Optional nucleus (top-p) filter.
    pub top_p: Option<f32>,
    /// Optional verifier scoring each candidate against a ground-truth
    /// answer; required when `strategy` is `Verifier`.
    pub verifier: Option<VerifierFn<'a>>,
    /// Optional ground-truth answer passed to the verifier.
    pub ground_truth: Option<&'a str>,
}

impl<'a> std::fmt::Debug for BestOfNOptions<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BestOfNOptions")
            .field("n", &self.n)
            .field("strategy", &self.strategy)
            .field("base_seed", &self.base_seed)
            .field("temperature", &self.temperature)
            .field("top_k", &self.top_k)
            .field("top_p", &self.top_p)
            .field("has_verifier", &self.verifier.is_some())
            .field("has_ground_truth", &self.ground_truth.is_some())
            .finish()
    }
}

/// Generate N independent stochastic completions for one prompt and select
/// among them via `options.strategy`.
///
/// Each candidate is decoded with a [`Sampler::TopKTopP`](aarambh_studio_inference::Sampler)
/// re-seeded `base_seed + i`, so candidate 0 reproduces a single-sample
/// stochastic decode with `base_seed`. When `strategy` is
/// [`SelectionStrategy::Verifier`](aarambh_studio_inference::SelectionStrategy), the
/// `verifier` callback scores each candidate against `ground_truth` and the
/// highest-scoring candidate is selected (ties broken by first
/// occurrence). When `strategy` is `ProcessReward`, the
/// [`HeuristicProcessRewardScorer`](aarambh_studio_inference::HeuristicProcessRewardScorer)
/// scores each reasoning trace.
pub fn best_of_n_generate(
    context: &EvalContext,
    prompt: &str,
    max_new_tokens: usize,
    options: &BestOfNOptions<'_>,
) -> Result<BestOfNResult> {
    use aarambh_studio_inference::{HeuristicProcessRewardScorer, ProcessRewardScorer};

    let BestOfNOptions {
        n,
        strategy,
        base_seed,
        temperature,
        top_k,
        top_p,
        verifier,
        ground_truth,
    } = *options;

    let mut candidates = Vec::with_capacity(n);
    for index in 0..n {
        let seed = base_seed.wrapping_add(index as u64);
        let candidate = sample_generate(
            context,
            prompt,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            seed,
        )?;
        candidates.push(candidate);
    }
    let scorer = HeuristicProcessRewardScorer::new();
    let (chosen_index, rationale) = match strategy {
        aarambh_studio_inference::SelectionStrategy::Majority => {
            let refs: Vec<&str> = candidates.iter().map(String::as_str).collect();
            let (winner, count) = aarambh_studio_inference::majority_vote(&refs)
                .expect("non-empty candidates guarantee a winner");
            let idx = candidates
                .iter()
                .position(|candidate| candidate.as_str() == winner)
                .expect("winner came from candidates");
            (
                idx,
                aarambh_studio_inference::SelectionRationale::Majority { count, total: n },
            )
        }
        aarambh_studio_inference::SelectionStrategy::SelfConsistency => {
            aarambh_studio_inference::self_consistency_select(&candidates)
        }
        aarambh_studio_inference::SelectionStrategy::Verifier => {
            let verifier_fn = verifier.ok_or_else(|| {
                AarambhError::Config("verifier selection requires a verifier callback".into())
            })?;
            let truth = ground_truth.unwrap_or("");
            let mut best_index = 0;
            let mut best_score = f32::NEG_INFINITY;
            for (index, candidate) in candidates.iter().enumerate() {
                let score = verifier_fn(candidate, truth);
                if score > best_score {
                    best_score = score;
                    best_index = index;
                }
            }
            (
                best_index,
                aarambh_studio_inference::SelectionRationale::Verifier {
                    index: best_index,
                    score: best_score,
                },
            )
        }
        aarambh_studio_inference::SelectionStrategy::ProcessReward => {
            let mut best_index = 0;
            let mut best_score = f32::NEG_INFINITY;
            for (index, candidate) in candidates.iter().enumerate() {
                let score = scorer.score(prompt, candidate);
                if score > best_score {
                    best_score = score;
                    best_index = index;
                }
            }
            (
                best_index,
                aarambh_studio_inference::SelectionRationale::ProcessReward {
                    index: best_index,
                    score: best_score,
                },
            )
        }
    };
    Ok(BestOfNResult {
        candidates,
        chosen_index,
        rationale,
    })
}

/// Generate one stochastic completion from a prompt with a seeded sampler.
///
/// Mirrors [`greedy_generate`] but replaces the argmax step with
/// [`Sampler::sample`](aarambh_studio_inference::Sampler) so each call with a
/// distinct seed produces an independent candidate.
pub fn sample_generate(
    context: &EvalContext,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    seed: u64,
) -> Result<String> {
    use aarambh_studio_inference::Sampler;

    let mut prompt_ids = context.tokenizer().encode(prompt)?;
    if prompt_ids.is_empty() {
        if let Some(bos) = context.tokenizer().bos_token_id() {
            prompt_ids.push(bos);
        } else {
            return Err(AarambhError::Tokenizer(
                "prompt produced no tokens and tokenizer has no BOS token".into(),
            ));
        }
    }
    if prompt_ids.len() >= context.max_seq_len() {
        return Err(AarambhError::Shape(format!(
            "prompt length {} leaves no room in max_seq_len {}",
            prompt_ids.len(),
            context.max_seq_len()
        )));
    }

    let budget = max_new_tokens.min(context.max_seq_len() - prompt_ids.len());
    let mut sampler = Sampler::top_k_top_p(temperature, top_k, top_p, Some(seed))?;
    let mut caches = context.model().empty_kv_cache();
    let input = Tensor::from_vec(prompt_ids.clone(), (1, prompt_ids.len()), context.device())?;
    let logits = context.model().forward_with_cache(&input, 0, &mut caches)?;
    let mut next_logits = last_logits(&logits)?;
    let mut generated = Vec::with_capacity(budget);
    let eos = context.tokenizer().eos_token_id();

    for step in 0..budget {
        let logits_vec = next_logits.to_vec1::<f32>()?;
        let token_id = sampler.sample(&logits_vec)?;
        if token_id == eos {
            break;
        }
        generated.push(token_id);
        context.record_context_len(prompt_ids.len() + generated.len());
        if step + 1 == budget {
            break;
        }
        let offset = prompt_ids.len() + generated.len() - 1;
        let input = Tensor::from_vec(vec![token_id], (1, 1), context.device())?;
        let logits = context
            .model()
            .forward_with_cache(&input, offset, &mut caches)?;
        next_logits = last_logits(&logits)?;
    }

    context.tokenizer().decode(&generated)
}

#[cfg(test)]
mod tests {
    use aarambh_studio_inference::ForceToken;

    use super::*;

    #[test]
    fn thinking_generation_result_total_sums_components() {
        let result = ThinkingGenerationResult {
            text: "answer".into(),
            thinking_tokens: 12,
            completion_tokens: 7,
        };
        assert_eq!(result.total_tokens(), 19);
    }

    #[test]
    fn force_token_marker_ids_match_tokenizer_constants() {
        assert_eq!(ForceToken::ThinkStart.token_id(), THINK_START_ID);
        assert_eq!(ForceToken::ThinkEnd.token_id(), THINK_END_ID);
    }

    #[test]
    fn selection_strategy_round_trips_through_display() {
        use std::str::FromStr;
        for name in ["verifier", "self-consistency", "majority", "process-reward"] {
            let strategy = aarambh_studio_inference::SelectionStrategy::from_str(name).unwrap();
            assert_eq!(strategy.to_string(), name);
        }
    }
}
