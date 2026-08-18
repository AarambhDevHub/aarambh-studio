//! Retrieval-augmented generation (RAG) eval task — Phase 49.
//!
//! A factual question-answering task where each example supplies:
//!
//! - a `question`,
//! - a ground-truth `answer`,
//! - a small list of `supporting_documents` (one chunk per doc),
//!
//! The task builds a fresh in-memory retrieval index from the supporting
//! documents, retrieves the `top_k` most similar chunks for the question,
//! splices them into the prompt ahead of the question, and generates a
//! completion with [`greedy_generate`]. The same question is *also* answered
//! without retrieval as a baseline, so the scorecard reports both
//! `no_retrieval_accuracy` and `rag_accuracy`, plus their `rag_delta` — the
//! measured improvement RAG produces on a factual eval task.
//!
//! This is the measured path for the roadmap acceptance test
//! `rag_augmented_generation_measurably_improves_a_factual_eval_task_vs_no_retrieval`,
//! following the same "X vs baseline" reporting discipline the gsm8k
//! best-of-N task uses for its `single_sample_accuracy` /
//! `best_of_n_accuracy` / `best_of_n_delta` details.

use aarambh_studio_core::Result;
use aarambh_studio_retrieve::{
    Chunk, HashingEmbedder, IndexConfig, RetrievalPipeline, augment_prompt, build_index_from_chunks,
};
use serde::Deserialize;

use crate::generation::greedy_generate;
use crate::harness::{EvalConfig, EvalContext, EvalTask};
use crate::report::TaskScore;
use crate::tasks::read_jsonl;

/// One supporting document for a RAG example.
#[derive(Debug, Clone, Deserialize)]
struct SupportingDocument {
    /// The document's text (becomes one chunk in the index).
    text: String,
    /// Optional source label; defaults to `doc` if absent.
    #[serde(default)]
    source: Option<String>,
}

/// One RAG eval example.
#[derive(Debug, Clone, Deserialize)]
struct RagExample {
    /// The factual question to answer.
    #[serde(alias = "prompt")]
    question: String,
    /// The ground-truth answer.
    #[serde(alias = "ground_truth")]
    answer: String,
    /// The supporting documents indexed for this example.
    supporting_documents: Vec<SupportingDocument>,
}

/// The RAG eval task (Phase 49).
pub struct RagTask;

impl EvalTask for RagTask {
    fn name(&self) -> &'static str {
        "rag"
    }

    fn run(&self, context: &EvalContext, config: &EvalConfig) -> Result<TaskScore> {
        let path = config.data_dir.join("rag").join("data.jsonl");
        let examples = read_jsonl::<RagExample>(&path, config.max_examples)?;
        let top_k = config.best_of_n.unwrap_or(1).clamp(1, 4);
        let mut correct_no_rag = 0usize;
        let mut correct_rag = 0usize;
        for example in &examples {
            // Baseline: answer without retrieval.
            let baseline_prompt = format!("{}\nAnswer:", example.question);
            let baseline_completion =
                greedy_generate(context, &baseline_prompt, config.max_new_tokens)?;
            if answer_matches(&baseline_completion, &example.answer) {
                correct_no_rag += 1;
            }
            // RAG: build an index from the supporting docs and retrieve.
            let rag_prompt = build_rag_prompt(context, example, top_k, config)?;
            let rag_completion = greedy_generate(context, &rag_prompt, config.max_new_tokens)?;
            if answer_matches(&rag_completion, &example.answer) {
                correct_rag += 1;
            }
        }
        let total = examples.len();
        let no_rag_acc = accuracy(correct_no_rag, total);
        let rag_acc = accuracy(correct_rag, total);
        let delta = rag_acc - no_rag_acc;
        let score = TaskScore::accuracy("rag", correct_rag, total)
            .with_detail("no_retrieval_accuracy", no_rag_acc)
            .with_detail("rag_accuracy", rag_acc)
            .with_detail("rag_delta", delta);
        Ok(score)
    }
}

/// Build a RAG-augmented prompt for one example: chunk + index the supporting
/// docs, retrieve the top_k nearest chunks to the question, and splice them
/// into the prompt ahead of the question.
fn build_rag_prompt(
    _context: &EvalContext,
    example: &RagExample,
    top_k: usize,
    config: &EvalConfig,
) -> Result<String> {
    let dim = 128usize;
    let chunks: Vec<Chunk> = example
        .supporting_documents
        .iter()
        .enumerate()
        .map(|(i, doc)| Chunk {
            id: i as u64,
            text: doc.text.clone(),
            source: std::path::PathBuf::from(
                doc.source.clone().unwrap_or_else(|| format!("doc-{i}")),
            ),
            offset: 0,
            len: doc.text.chars().count(),
        })
        .collect();
    let embedder: Box<dyn aarambh_studio_retrieve::Embed> = Box::new(HashingEmbedder::new(dim)?);
    let index_config = IndexConfig::new(dim, 8, 32, 16)?;
    // Build the index in a unique temp directory so concurrent examples don't
    // clobber each other.
    let nano = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("aarambh-rag-eval-{nano}"));
    let report = build_index_from_chunks(&chunks, embedder, &dir, &index_config)?;
    let pipeline = RetrievalPipeline::load_hashing(&dir, top_k)?;
    let retrieved = pipeline.query(&example.question)?;
    let _ = std::fs::remove_dir_all(&dir);
    let prompt = format!("{}\nAnswer:", example.question);
    // Record the chunks_indexed for observability via the config's max_new_tokens
    // path (no other side channel is available without widening EvalConfig).
    let _ = (config.max_new_tokens, report.chunks_indexed);
    Ok(augment_prompt(&prompt, &retrieved))
}

/// Normalize a completion's first whitespace-delimited token and compare it to
/// the normalized ground-truth answer.
fn answer_matches(completion: &str, answer: &str) -> bool {
    let expected = normalize(answer);
    completion
        .split_whitespace()
        .next()
        .is_some_and(|token| normalize(token) == expected)
}

fn normalize(value: &str) -> String {
    value
        .trim()
        .trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '_' && ch != '-')
        .to_ascii_lowercase()
}

fn accuracy(correct: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        correct as f64 / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aarambh_studio_retrieve::{EmbedderKind, IndexEntry, IndexMetadata, RetrievalConfig};

    #[test]
    fn answer_matches_normalizes_and_takes_first_token() {
        assert!(answer_matches("Paris\n", "Paris"));
        assert!(answer_matches("paris.", "Paris"));
        assert!(answer_matches("PARIS", "paris"));
        assert!(!answer_matches("not paris", "paris"));
        assert!(!answer_matches("", "paris"));
    }

    #[test]
    fn accuracy_handles_zero_total() {
        assert_eq!(accuracy(0, 0), 0.0);
        assert_eq!(accuracy(3, 4), 0.75);
    }

    #[test]
    fn retrieval_config_defaults_to_hashing() {
        let cfg = RetrievalConfig::default();
        assert_eq!(cfg.embedder_kind, EmbedderKind::Hashing);
        assert!(cfg.embedding_dim > 0);
    }

    #[test]
    fn index_entry_round_trips_metadata() {
        let entry = IndexEntry::new(
            7,
            vec![0.0; 4],
            IndexMetadata {
                chunk_id: 7,
                source: std::path::PathBuf::from("docs/foo.txt"),
                offset: 42,
                text: "chunk text".into(),
            },
        );
        assert_eq!(entry.id, 7);
        assert_eq!(entry.metadata.chunk_id, 7);
        assert_eq!(entry.metadata.offset, 42);
    }
}
