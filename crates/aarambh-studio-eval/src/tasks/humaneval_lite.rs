use aarambh_studio_core::Result;
use aarambh_studio_finetune::{CodeVerifier, Verifier};
use serde::Deserialize;

use crate::generation::{BestOfNOptions, best_of_n_generate, greedy_generate};
use crate::harness::{EvalConfig, EvalContext, EvalTask};
use crate::report::TaskScore;
use crate::tasks::read_jsonl;

#[derive(Debug, Clone, Deserialize)]
struct HumanEvalExample {
    prompt: String,
    #[serde(alias = "ground_truth")]
    test: String,
}

/// HumanEval-lite pass@1 task.
pub struct HumanEvalLiteTask;

impl EvalTask for HumanEvalLiteTask {
    fn name(&self) -> &'static str {
        "humaneval"
    }

    fn run(&self, context: &EvalContext, config: &EvalConfig) -> Result<TaskScore> {
        let path = config.data_dir.join("humaneval_lite").join("data.jsonl");
        let examples = read_jsonl::<HumanEvalExample>(&path, config.max_examples)?;
        let verifier = CodeVerifier::default();
        let mut passed = 0usize;

        let mut best_of_n_passed = 0usize;
        let mut best_of_n_enabled = false;
        for example in &examples {
            let completion = greedy_generate(context, &example.prompt, config.max_new_tokens)?;
            let candidate = format!("{}{}", example.prompt, completion);
            if verifier.score(&candidate, &example.test) >= 1.0 {
                passed += 1;
            }
            if let Some(n) = config.best_of_n {
                best_of_n_enabled = true;
                let prompt_text = example.prompt.clone();
                let verifier_fn = move |candidate_completion: &str, test: &str| {
                    let full = format!("{prompt_text}{candidate_completion}");
                    CodeVerifier::default().score(&full, test)
                };
                let options = BestOfNOptions {
                    n,
                    strategy: config.best_of_n_selection,
                    base_seed: config.best_of_n_seed,
                    temperature: 0.8,
                    top_k: Some(50),
                    top_p: Some(0.9),
                    verifier: Some(&verifier_fn),
                    ground_truth: Some(&example.test),
                };
                let result =
                    best_of_n_generate(context, &example.prompt, config.max_new_tokens, &options)?;
                let chosen_completion = &result.candidates[result.chosen_index];
                let chosen_candidate = format!("{}{}", example.prompt, chosen_completion);
                if verifier.score(&chosen_candidate, &example.test) >= 1.0 {
                    best_of_n_passed += 1;
                }
            }
        }

        let mut score = TaskScore::pass_at_1("humaneval", passed, examples.len());
        if best_of_n_enabled {
            let single_sample = if examples.is_empty() {
                0.0
            } else {
                passed as f64 / examples.len() as f64
            };
            let best_of_n = if examples.is_empty() {
                0.0
            } else {
                best_of_n_passed as f64 / examples.len() as f64
            };
            score = score
                .with_detail("single_sample_accuracy", single_sample)
                .with_detail("best_of_n_accuracy", best_of_n)
                .with_detail("best_of_n_delta", best_of_n - single_sample);
        }
        Ok(score)
    }
}
