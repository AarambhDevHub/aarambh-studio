use std::fs;
use std::path::Path;

use aarambh_studio_core::{AarambhError, Result};
use serde::de::DeserializeOwned;

/// Associative key-value recall task.
pub mod associative_recall;
/// Native audio question-answering task (Phase 42).
pub mod audio_qa;
/// Native document question-answering task.
pub mod document_qa;
/// GSM8K subset task.
pub mod gsm8k_subset;
/// Hard-problems task for Max-mode thinking validation (Phase 39).
pub mod hard_problems;
/// HellaSwag task.
pub mod hellaswag;
/// HumanEval-lite task.
pub mod humaneval_lite;
/// Image-captioning smoke task.
pub mod image_caption;
/// MMLU-lite task.
pub mod mmlu_lite;
/// Perplexity task wrapper.
pub mod ppl_task;
/// Pairwise preference-ranking task.
pub mod preference;
/// Retrieval-augmented generation task (Phase 49).
pub mod rag;
/// Grammar-constrained function-selection task.
pub mod tool_calling;
/// Scripted caller-response long-horizon tool-chain task.
pub mod tool_chain;
/// Native video question-answering task.
pub mod video_qa;
/// Vision-question-answering task.
pub mod vqa;

pub use associative_recall::AssociativeRecallTask;
pub use audio_qa::AudioQaTask;
pub use document_qa::DocumentQaTask;
pub use gsm8k_subset::Gsm8kSubsetTask;
pub use hard_problems::HardProblemsTask;
pub use hellaswag::HellaSwagTask;
pub use humaneval_lite::HumanEvalLiteTask;
pub use image_caption::ImageCaptionTask;
pub use mmlu_lite::MmluLiteTask;
pub use ppl_task::PplTask;
pub use preference::PreferenceTask;
pub use rag::RagTask;
pub use tool_calling::ToolCallingTask;
pub use tool_chain::ToolChainTask;
pub use video_qa::VideoQaTask;
pub use vqa::VqaTask;

fn read_jsonl<T: DeserializeOwned>(path: &Path, max_examples: Option<usize>) -> Result<Vec<T>> {
    let content = fs::read_to_string(path).map_err(|err| {
        AarambhError::Io(std::io::Error::new(
            err.kind(),
            format!("failed to read {}: {err}", path.display()),
        ))
    })?;
    let mut out = Vec::new();
    for (line_idx, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value = serde_json::from_str(line).map_err(|err| {
            AarambhError::Config(format!(
                "failed to parse {} line {}: {err}",
                path.display(),
                line_idx + 1
            ))
        })?;
        out.push(value);
        if max_examples.is_some_and(|max| out.len() >= max) {
            break;
        }
    }
    if out.is_empty() {
        return Err(AarambhError::Config(format!(
            "{} contains no examples",
            path.display()
        )));
    }
    Ok(out)
}

fn first_existing(paths: &[std::path::PathBuf]) -> Result<std::path::PathBuf> {
    paths
        .iter()
        .find(|path| path.exists())
        .cloned()
        .ok_or_else(|| {
            AarambhError::Config(format!(
                "none of the expected eval data files exist: {}",
                paths
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })
}
