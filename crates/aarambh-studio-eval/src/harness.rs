use std::cell::Cell;
use std::path::PathBuf;

use aarambh_studio_core::{AarambhError, Configurable, Result};
use aarambh_studio_inference::{SelectionStrategy, ThinkingMode};
use aarambh_studio_model::AarambhModel;
use aarambh_studio_tokenizer::BpeTokenizer;
use candle_core::{DType, Device};

use crate::report::{Scorecard, TaskScore};
use crate::tasks::{
    AssociativeRecallTask, AudioQaTask, DocumentQaTask, Gsm8kSubsetTask, HardProblemsTask,
    HellaSwagTask, HumanEvalLiteTask, ImageCaptionTask, MmluLiteTask, PplTask, PreferenceTask,
    RagTask, ToolCallingTask, ToolChainTask, VideoQaTask, VqaTask,
};

/// Evaluation run configuration.
#[derive(Debug, Clone)]
pub struct EvalConfig {
    /// Task selectors such as `ppl`, `mmlu`, `hellaswag`, `gsm8k`, `hard-problems`, `humaneval`, `image-caption`, or `all`.
    pub tasks: Vec<String>,
    /// Root directory containing normalized eval data.
    pub data_dir: PathBuf,
    /// Optional maximum number of examples per task.
    pub max_examples: Option<usize>,
    /// Maximum generated tokens for generative tasks.
    pub max_new_tokens: usize,
    /// Maximum caller-result steps for tool-chain evaluation.
    pub agent_max_steps: usize,
    /// Whether HumanEval-lite may execute generated Python code.
    pub allow_code_exec: bool,
    /// Thinking mode applied to thinking-aware generative tasks (Phase 39).
    pub thinking_mode: ThinkingMode,
    /// Optional best-of-N candidate count for generative tasks (Phase 45).
    /// When set, supported tasks compute both single-sample and best-of-N
    /// accuracy and record the delta in their `TaskScore::details` map.
    pub best_of_n: Option<usize>,
    /// Selection strategy for best-of-N evaluation (Phase 45).
    pub best_of_n_selection: SelectionStrategy,
    /// Base RNG seed for best-of-N candidate sampling (Phase 45).
    pub best_of_n_seed: u64,
    /// Optional model path stored in scorecards.
    pub model_path: Option<String>,
    /// Optional tokenizer path stored in scorecards.
    pub tokenizer_path: Option<String>,
    /// Optional config path stored in scorecards.
    pub config_path: Option<String>,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            tasks: vec!["ppl".into()],
            data_dir: PathBuf::from("data/eval"),
            max_examples: None,
            max_new_tokens: 128,
            agent_max_steps: 8,
            allow_code_exec: false,
            thinking_mode: ThinkingMode::None,
            best_of_n: None,
            best_of_n_selection: SelectionStrategy::SelfConsistency,
            best_of_n_seed: 0,
            model_path: None,
            tokenizer_path: None,
            config_path: None,
        }
    }
}

/// Loaded model/tokenizer/device context shared by eval tasks.
pub struct EvalContext {
    model: AarambhModel,
    tokenizer: BpeTokenizer,
    device: Device,
    dtype: DType,
    max_seq_len: usize,
    context_len_used: Cell<usize>,
}

impl EvalContext {
    /// Create an evaluation context from loaded components.
    pub fn new(model: AarambhModel, tokenizer: BpeTokenizer, device: Device, dtype: DType) -> Self {
        let max_seq_len = model.config().max_seq_len;
        Self {
            model,
            tokenizer,
            device,
            dtype,
            max_seq_len,
            context_len_used: Cell::new(0),
        }
    }

    /// Return the loaded model.
    pub fn model(&self) -> &AarambhModel {
        &self.model
    }

    /// Return the tokenizer.
    pub fn tokenizer(&self) -> &BpeTokenizer {
        &self.tokenizer
    }

    /// Return the Candle device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Return the evaluation dtype.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Return the configured model context length.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Record a context length used by a task.
    pub fn record_context_len(&self, len: usize) {
        self.context_len_used
            .set(self.context_len_used.get().max(len));
    }

    /// Return the largest context length used by this run.
    pub fn context_len_used(&self) -> usize {
        self.context_len_used.get()
    }
}

/// A runnable evaluation task.
pub trait EvalTask {
    /// Stable task name.
    fn name(&self) -> &'static str;

    /// Execute the task and return a score.
    fn run(&self, context: &EvalContext, config: &EvalConfig) -> Result<TaskScore>;
}

/// Run all selected tasks and produce a scorecard.
pub fn run_all(context: &EvalContext, config: &EvalConfig) -> Result<Scorecard> {
    let tasks = selected_tasks(&config.tasks, config.allow_code_exec)?;
    let mut scores = Vec::with_capacity(tasks.len());
    for task in tasks {
        scores.push(task.run(context, config)?);
    }
    Ok(Scorecard::new(
        scores,
        context.context_len_used(),
        config.max_new_tokens,
        config.model_path.clone(),
        config.tokenizer_path.clone(),
        config.config_path.clone(),
    ))
}

fn selected_tasks(selectors: &[String], allow_code_exec: bool) -> Result<Vec<Box<dyn EvalTask>>> {
    let expanded = if selectors
        .iter()
        .any(|task| task.eq_ignore_ascii_case("all"))
    {
        vec![
            "ppl",
            "mmlu",
            "hellaswag",
            "gsm8k",
            "hard-problems",
            "humaneval",
            "rag",
        ]
    } else {
        selectors.iter().map(String::as_str).collect::<Vec<_>>()
    };
    let mut tasks: Vec<Box<dyn EvalTask>> = Vec::new();
    for selector in expanded {
        match selector.trim().to_ascii_lowercase().as_str() {
            "ppl" | "perplexity" => tasks.push(Box::new(PplTask)),
            "mmlu" | "mmlu-lite" | "mmlu_lite" => tasks.push(Box::new(MmluLiteTask)),
            "hellaswag" => tasks.push(Box::new(HellaSwagTask)),
            "gsm8k" | "gsm8k-subset" | "gsm8k_subset" => tasks.push(Box::new(Gsm8kSubsetTask)),
            "hard-problems" | "hard_problems" | "hard" => {
                tasks.push(Box::new(HardProblemsTask));
            }
            "humaneval" | "humaneval-lite" | "humaneval_lite" => {
                if !allow_code_exec {
                    return Err(AarambhError::Config(
                        "HumanEval-lite requires --allow-code-exec".into(),
                    ));
                }
                tasks.push(Box::new(HumanEvalLiteTask));
            }
            "image-caption" | "image_caption" | "caption" | "vlm-smoke" => {
                tasks.push(Box::new(ImageCaptionTask));
            }
            "vqa" | "vision-qa" | "vision_qa" | "vqa-smoke" | "vqa_smoke" => {
                tasks.push(Box::new(VqaTask));
            }
            "video-qa" | "video_qa" | "nextqa" | "video-qa-smoke" | "video_qa_smoke" => {
                tasks.push(Box::new(VideoQaTask));
            }
            "document-qa" | "document_qa" | "docvqa" | "document-qa-smoke"
            | "document_qa_smoke" => tasks.push(Box::new(DocumentQaTask)),
            "audio-qa" | "audio_qa" | "audio-qa-smoke" | "audio_qa_smoke" => {
                tasks.push(Box::new(AudioQaTask));
            }
            "preference" | "dpo" | "preference-win-rate" | "preference_win_rate" => {
                tasks.push(Box::new(PreferenceTask));
            }
            "rag" | "rag-qa" | "rag_qa" | "retrieval" | "retrieval-augmented" => {
                tasks.push(Box::new(RagTask));
            }
            "tool-calling" | "tool_calling" | "function-calling" | "function_calling" => {
                tasks.push(Box::new(ToolCallingTask));
            }
            "tool-chain" | "tool_chain" | "agent-chain" | "agent_chain" | "bfcl-multistep"
            | "bfcl_multistep" => tasks.push(Box::new(ToolChainTask)),
            "associative-recall" | "associative_recall" | "assoc-recall" | "assoc_recall" => {
                tasks.push(Box::new(AssociativeRecallTask));
            }
            other => {
                return Err(AarambhError::Config(format!(
                    "unknown eval task '{other}', expected ppl,mmlu,hellaswag,gsm8k,hard-problems,humaneval,preference,image-caption,vqa,video-qa,document-qa,audio-qa,tool-calling,tool-chain,associative-recall,rag,all"
                )));
            }
        }
    }
    Ok(tasks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_tasks_accepts_hard_problems_under_all_aliases() {
        for selector in ["hard-problems", "hard_problems", "hard"] {
            let tasks = selected_tasks(&[selector.into()], false).unwrap();
            assert_eq!(tasks.len(), 1);
            assert_eq!(tasks[0].name(), "hard-problems");
        }
    }

    #[test]
    fn selected_tasks_rejects_unknown_selector() {
        assert!(selected_tasks(&["nope".into()], false).is_err());
    }

    #[test]
    fn eval_config_default_thinking_mode_is_none() {
        assert_eq!(EvalConfig::default().thinking_mode, ThinkingMode::None);
    }
}
