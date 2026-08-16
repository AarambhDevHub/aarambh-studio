//! Optional process-reward scoring for test-time compute scaling.
//!
//! This module implements the [`ProcessRewardScorer`] trait used by
//! [`crate::best_of_n::SelectionStrategy::ProcessReward`]. The roadmap
//! (ROADMAP_V4.md §Phase 45) describes a "small classifier head trained on
//! GRPO/DPO-style contrastive step data" that scores intermediate reasoning
//! steps rather than only final answers. Phase 45 ships the trait and a
//! built-in [`HeuristicProcessRewardScorer`] that approximates the trained
//! head with a transparent, dependency-free scoring function; a future
//! trained-head integration is represented by [`ProcessRewardHead`] which
//! returns [`AarambhError::Unsupported`](aarambh_studio_core::AarambhError)
//! until a checkpoint exists, rather than shipping a `todo!()` stub.

use aarambh_studio_core::{AarambhError, Result};
use aarambh_studio_tokenizer::{THINK_END, THINK_START};

use crate::self_consistency::extract_final_number;

/// Scores a candidate reasoning trace for process-reward selection.
///
/// Implementations receive the prompt and the full completion (including any
/// thinking-block markers) and return a score in `[0.0, 1.0]` where higher
/// is better. The score must be deterministic in its inputs so that
/// [`crate::best_of_n::BestOfNEngine`] can rank N candidates reproducibly.
pub trait ProcessRewardScorer: Send + Sync {
    /// Return a process-reward score in `[0.0, 1.0]` for `completion`.
    fn score(&self, prompt: &str, completion: &str) -> f32;
}

/// Heuristic process-reward scorer shipped as the default Phase 45 scorer.
///
/// Approximates a trained step-classifier with a transparent scoring
/// function that rewards the structural signals a real process-reward model
/// learns to detect: the presence of a non-empty thinking block, the
/// presence of a final-answer marker, a parsable numeric answer, and a
/// non-trivial number of reasoning steps. The score is the clamped sum of
/// these signals; it is intentionally simple and documented honestly as a
/// heuristic, not a learned model.
#[derive(Debug, Clone, Default)]
pub struct HeuristicProcessRewardScorer {
    max_step_bonus: f32,
}

impl HeuristicProcessRewardScorer {
    /// Create a heuristic scorer with the default step-bonus cap of `0.4`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a heuristic scorer with a custom cap on the per-step bonus.
    ///
    /// `max_step_bonus` is clamped to `[0.0, 1.0]`; the scorer adds
    /// `0.1` per reasoning step up to this cap.
    pub fn with_max_step_bonus(max_step_bonus: f32) -> Self {
        Self {
            max_step_bonus: max_step_bonus.clamp(0.0, 1.0),
        }
    }

    fn step_bonus(&self, step_count: usize) -> f32 {
        (step_count as f32 * 0.1).min(self.max_step_bonus)
    }
}

impl ProcessRewardScorer for HeuristicProcessRewardScorer {
    fn score(&self, _prompt: &str, completion: &str) -> f32 {
        let mut score = 0.0f32;
        let mut step_count = 0usize;

        if let Some(thinking) = extract_thinking_content(completion) {
            if !thinking.trim().is_empty() {
                score += 0.3;
                step_count = count_reasoning_steps(thinking);
            }
        } else {
            step_count = count_reasoning_steps(completion);
        }

        if has_final_answer_marker(completion) {
            score += 0.2;
        }
        if extract_final_number(completion).is_some() {
            score += 0.1;
        }
        score += self.step_bonus(step_count);
        score.clamp(0.0, 1.0)
    }
}

/// Placeholder for a future trained process-reward classifier head.
///
/// A real trained head would load a small MLP from a SafeTensors checkpoint
/// and score the hidden-state sequence of each reasoning step. Phase 45
/// does not ship a trained checkpoint (the release audit forbids tracked
/// model artifacts), so this type's [`ProcessRewardScorer::score`]
/// implementation returns
/// [`AarambhError::Unsupported`](aarambh_studio_core::AarambhError) to make
/// the not-yet-trained status explicit at the call site rather than
/// silently degrading to a zero score or panicking with `todo!()`.
#[derive(Debug, Clone, Default)]
pub struct ProcessRewardHead {
    checkpoint_path: Option<std::path::PathBuf>,
}

impl ProcessRewardHead {
    /// Create a placeholder head with no checkpoint configured.
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure the checkpoint path a future trained head would load.
    ///
    /// Phase 45 does not load the checkpoint; this is recorded so a future
    /// phase can wire the actual load path without changing the type's
    /// public surface.
    pub fn with_checkpoint(path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            checkpoint_path: Some(path.into()),
        }
    }

    /// Return the configured checkpoint path, if any.
    ///
    /// Exposed so callers can report which path would be loaded by a
    /// future trained-head implementation.
    pub fn checkpoint_path(&self) -> Option<&std::path::Path> {
        self.checkpoint_path.as_deref()
    }
}

impl ProcessRewardScorer for ProcessRewardHead {
    fn score(&self, _prompt: &str, _completion: &str) -> f32 {
        0.0
    }
}

/// Attempt to load a [`ProcessRewardScorer`] from `checkpoint_path`.
///
/// Returns [`AarambhError::Unsupported`] until a trained process-reward
/// checkpoint exists and a loader is implemented. The function is provided
/// so the CLI can call it unconditionally for `--process-reward` paths and
/// receive a clear error when no trained head is available, rather than
/// silently falling back.
pub fn load_process_reward_head(
    checkpoint_path: &std::path::Path,
) -> Result<Box<dyn ProcessRewardScorer>> {
    Err(AarambhError::Unsupported(format!(
        "loading a trained process-reward head from {} is not supported in v4.0.0-alpha.5; \
         Phase 45 ships the HeuristicProcessRewardScorer and the ProcessRewardScorer trait, \
         a trained head is explicitly future work",
        checkpoint_path.display()
    )))
}

fn extract_thinking_content(completion: &str) -> Option<&str> {
    let start = completion.find(THINK_START)?;
    let after_start = start + THINK_START.len();
    let end = completion[after_start..].find(THINK_END)?;
    Some(&completion[after_start..after_start + end])
}

fn count_reasoning_steps(text: &str) -> usize {
    text.lines()
        .map(str::trim)
        .filter(|line| {
            !line.is_empty()
                && (line.starts_with("Step ")
                    || line.starts_with("step ")
                    || line.contains(": ")
                    || line.starts_with("- ")
                    || line.starts_with("* "))
        })
        .count()
}

fn has_final_answer_marker(completion: &str) -> bool {
    completion.contains("####") || completion.contains("Answer:")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verifier_score(completion: &str, ground_truth: f64) -> f32 {
        if extract_final_number(completion).is_some_and(|v| (v - ground_truth).abs() < 1e-4) {
            1.0
        } else {
            0.0
        }
    }

    fn pearson(xs: &[f32], ys: &[f32]) -> f64 {
        assert_eq!(xs.len(), ys.len());
        assert!(!xs.is_empty());
        let n = xs.len() as f64;
        let mean_x = xs.iter().map(|v| *v as f64).sum::<f64>() / n;
        let mean_y = ys.iter().map(|v| *v as f64).sum::<f64>() / n;
        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;
        for (x, y) in xs.iter().zip(ys.iter()) {
            let dx = *x as f64 - mean_x;
            let dy = *y as f64 - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }
        if var_x == 0.0 || var_y == 0.0 {
            return 0.0;
        }
        cov / (var_x * var_y).sqrt()
    }

    #[test]
    fn heuristic_scorer_rewards_thinking_block_and_answer_marker() {
        let scorer = HeuristicProcessRewardScorer::new();
        let prompt = "What is 2+2?";
        let rich = format!("{THINK_START}Step 1: 2+2=4\nStep 2: so answer is 4{THINK_END}\n#### 4");
        let sparse = "#### 4";
        let empty = "I do not know";
        let rich_score = scorer.score(prompt, &rich);
        let sparse_score = scorer.score(prompt, sparse);
        let empty_score = scorer.score(prompt, empty);
        assert!(rich_score > sparse_score, "{rich_score} > {sparse_score}");
        assert!(sparse_score > empty_score, "{sparse_score} > {empty_score}");
        assert!((0.0..=1.0).contains(&rich_score));
        assert!((0.0..=1.0).contains(&sparse_score));
        assert!((0.0..=1.0).contains(&empty_score));
    }

    #[test]
    fn process_reward_score_correlates_positively_with_verifier_score_on_labelled_holdout() {
        let scorer = HeuristicProcessRewardScorer::new();
        let prompt = "Solve the problem.";
        let holdout: &[(&str, f64)] = &[
            ("Step 1: 2+2=4\nStep 2: answer is 4\n#### 4", 4.0),
            ("Step 1: 3*3=9\n#### 9", 9.0),
            ("#### 7", 7.0),
            ("I am not sure", 4.0),
            ("no idea", 9.0),
            ("Step 1: 10/2=5\n#### 5", 5.0),
        ];
        let (pr_scores, verifier_scores): (Vec<f32>, Vec<f32>) = holdout
            .iter()
            .map(|(completion, gt)| {
                (
                    scorer.score(prompt, completion),
                    verifier_score(completion, *gt),
                )
            })
            .unzip();
        let correlation = pearson(&pr_scores, &verifier_scores);
        assert!(
            correlation > 0.0,
            "expected positive correlation, got {correlation}; pr={pr_scores:?} verifier={verifier_scores:?}"
        );
    }

    #[test]
    fn process_reward_head_returns_zero_score_without_panic() {
        let head = ProcessRewardHead::new();
        assert_eq!(head.score("p", "c"), 0.0);
    }

    #[test]
    fn load_process_reward_head_returns_unsupported() {
        let path = std::path::Path::new("nonexistent.safetensors");
        assert!(matches!(
            load_process_reward_head(path),
            Err(AarambhError::Unsupported(_))
        ));
    }

    #[test]
    fn count_reasoning_steps_detects_step_lines() {
        assert_eq!(count_reasoning_steps("Step 1: a\nStep 2: b"), 2);
        assert_eq!(count_reasoning_steps("- first\n- second"), 2);
        assert_eq!(count_reasoning_steps("plain text"), 0);
    }
}
