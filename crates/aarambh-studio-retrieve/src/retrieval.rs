//! The retrieval pipeline: chunk → embed → index → query → augment prompt.
//!
//! [`RetrievalPipeline`] owns an embedder and a [`VectorIndex`] and exposes a
//! single [`RetrievalPipeline::query`] entry point. [`augment_prompt`] splices
//! retrieved chunks into the existing prompt-construction path **ahead of the
//! user's question** — the same mechanism that already assembles the system
//! prompt, chat history, and user turn. RAG augments the prompt; it does not
//! change how the decoder processes it.

use std::path::{Path, PathBuf};

use aarambh_studio_core::{AarambhError, Result, TokenizerLike};
use serde::{Deserialize, Serialize};

use crate::chunking::{Chunk, Chunker, ChunkingConfig};
use crate::embedding::{Embed, EmbedderKind, HashingEmbedder, build_embedder};
use crate::index::{IndexConfig, IndexEntry, IndexMetadata, SearchResult, VectorIndex};

/// One retrieved chunk, with its similarity score.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RetrievedChunk {
    /// The chunk's stable id.
    pub chunk_id: u64,
    /// The chunk's decoded text.
    pub text: String,
    /// The source file the chunk came from.
    pub source: PathBuf,
    /// Byte offset of the chunk in its source.
    pub offset: usize,
    /// Cosine similarity to the query (higher is better).
    pub score: f32,
}

/// Configuration for the retrieval pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    /// How many chunks to retrieve per query.
    pub top_k: usize,
    /// The chunking policy used when building an index.
    pub chunking: ChunkingConfig,
    /// The ANN index parameters.
    pub index: IndexConfig,
    /// The embedder kind.
    pub embedder_kind: EmbedderKind,
    /// The embedding dimension (vector width).
    pub embedding_dim: usize,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            top_k: 4,
            chunking: ChunkingConfig::default(),
            index: IndexConfig::default(),
            embedder_kind: EmbedderKind::Hashing,
            embedding_dim: 64,
        }
    }
}

/// A retrieval pipeline: an embedder plus a loaded [`VectorIndex`].
pub struct RetrievalPipeline {
    embedder: Box<dyn Embed>,
    index: VectorIndex,
    top_k: usize,
}

impl RetrievalPipeline {
    /// Construct a pipeline from an embedder and a loaded index.
    pub fn new(embedder: Box<dyn Embed>, index: VectorIndex, top_k: usize) -> Result<Self> {
        if embedder.dim() != index.config().dim {
            return Err(AarambhError::Shape(format!(
                "embedder dimension {} does not match index dimension {}",
                embedder.dim(),
                index.config().dim
            )));
        }
        if top_k == 0 {
            return Err(AarambhError::Config("top_k must be > 0".into()));
        }
        Ok(Self {
            embedder,
            index,
            top_k,
        })
    }

    /// Load a pipeline from an on-disk index directory, using a hashing
    /// embedder (the weight-free default) sized to the index's dimension.
    pub fn load_hashing(index_dir: &Path, top_k: usize) -> Result<Self> {
        let index = VectorIndex::load(index_dir)?;
        let dim = index.config().dim;
        let embedder: Box<dyn Embed> = Box::new(HashingEmbedder::new(dim)?);
        Self::new(embedder, index, top_k)
    }

    /// Query the index for the `top_k` chunks nearest to `query`.
    pub fn query(&self, query: &str) -> Result<Vec<RetrievedChunk>> {
        let query_vec = self.embedder.embed(query)?;
        let results = self.index.search(&query_vec, self.top_k)?;
        Ok(results
            .into_iter()
            .map(
                |SearchResult {
                     id,
                     score,
                     metadata,
                 }| RetrievedChunk {
                    chunk_id: id,
                    text: metadata.text,
                    source: metadata.source,
                    offset: metadata.offset,
                    score,
                },
            )
            .collect())
    }

    /// Return a reference to the underlying index (for inspection / metrics).
    pub fn index(&self) -> &VectorIndex {
        &self.index
    }

    /// Return the configured top-k.
    pub fn top_k(&self) -> usize {
        self.top_k
    }
}

/// Splice retrieved chunks into a prompt ahead of the user's question.
///
/// The retrieved context is rendered as a fenced block between a
/// `Retrieved context:` header and the original prompt, so the model sees:
///
/// ```text
/// Retrieved context:
/// [1] (source: docs/foo.txt) <chunk text>
/// [2] (source: docs/bar.txt) <chunk text>
///
/// <original prompt>
/// ```
///
/// This is the **same** prompt-construction path the inference engine already
/// uses (system prompt + chat history + user turn) — RAG only prepends
/// retrieved context. It does not change how the decoder processes the prompt.
pub fn augment_prompt(prompt: &str, retrieved: &[RetrievedChunk]) -> String {
    if retrieved.is_empty() {
        return prompt.to_string();
    }
    let mut out = String::with_capacity(prompt.len() + 256 * retrieved.len());
    out.push_str("Retrieved context:\n");
    for (i, chunk) in retrieved.iter().enumerate() {
        out.push_str(&format!(
            "[{}] (source: {}) {}\n",
            i + 1,
            chunk.source.display(),
            chunk.text.trim_end()
        ));
    }
    out.push('\n');
    out.push_str(prompt);
    out
}

/// Build an index from a corpus directory, writing it to `output_dir`.
///
/// This is the entry point for the `aarambh-studio retrieve build-index` CLI:
/// it chunks every text file under `corpus_dir`, embeds each chunk with the
/// configured embedder, inserts it into a fresh [`VectorIndex`], and persists
/// the index to `output_dir/index.json`.
pub fn build_index<T: TokenizerLike>(
    tokenizer: &T,
    corpus_dir: &Path,
    output_dir: &Path,
    config: &RetrievalConfig,
) -> Result<BuildIndexReport> {
    let chunker = Chunker::new(config.chunking.clone());
    let chunks = chunker.chunk_corpus(tokenizer, corpus_dir)?;
    if chunks.is_empty() {
        return Err(AarambhError::Config(format!(
            "no chunks produced from corpus {}",
            corpus_dir.display()
        )));
    }
    let embedder: Box<dyn Embed> = match config.embedder_kind {
        EmbedderKind::Hashing => Box::new(HashingEmbedder::new(config.embedding_dim)?),
        EmbedderKind::Text => {
            return Err(AarambhError::Config(
                "build_index with TextEmbedder requires weights; use HashingEmbedder via build_index_hashing or pass a VarBuilder through a higher layer".into(),
            ));
        }
    };
    build_index_from_chunks(&chunks, embedder, output_dir, &config.index)
}

/// Build an index from already-chunked text using a supplied embedder.
///
/// Separated from [`build_index`] so the eval harness and tests can supply a
/// pre-built chunk list and a specific embedder (including the candle
/// [`crate::TextEmbedder`] when weights are available) without re-reading the
/// corpus.
pub fn build_index_from_chunks(
    chunks: &[Chunk],
    embedder: Box<dyn Embed>,
    output_dir: &Path,
    index_config: &IndexConfig,
) -> Result<BuildIndexReport> {
    if embedder.dim() != index_config.dim {
        return Err(AarambhError::Shape(format!(
            "embedder dimension {} does not match index dimension {}",
            embedder.dim(),
            index_config.dim
        )));
    }
    let mut index = VectorIndex::new(index_config.clone())?;
    for chunk in chunks {
        let vector = embedder.embed(&chunk.text)?;
        let entry = IndexEntry::new(
            chunk.id,
            vector,
            IndexMetadata {
                chunk_id: chunk.id,
                source: chunk.source.clone(),
                offset: chunk.offset,
                text: chunk.text.clone(),
            },
        );
        index.insert(entry)?;
    }
    index.save(output_dir)?;
    Ok(BuildIndexReport {
        chunks_indexed: chunks.len(),
        embedding_dim: embedder.dim(),
        index_path: output_dir.join("index.json"),
    })
}

/// A summary of a `build_index` run, returned for CLI / metrics reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildIndexReport {
    /// Number of chunks indexed.
    pub chunks_indexed: usize,
    /// The embedding dimension.
    pub embedding_dim: usize,
    /// Path to the written `index.json`.
    pub index_path: PathBuf,
}

/// Build a default hashing-embedder pipeline config for a smoke run.
///
/// Exposed so the CLI and tests share one honest default: the hashing embedder,
/// small dimensions, modest top-k — the weight-free path the whole pipeline is
/// testable on.
pub fn default_hashing_config(embedding_dim: usize, top_k: usize) -> RetrievalConfig {
    RetrievalConfig {
        top_k,
        chunking: ChunkingConfig::default(),
        index: IndexConfig {
            dim: embedding_dim,
            ..IndexConfig::default()
        },
        embedder_kind: EmbedderKind::Hashing,
        embedding_dim,
    }
}

/// Construct an embedder for the pipeline from the configured kind.
///
/// For [`EmbedderKind::Hashing`] this returns a weight-free embedder. For
/// [`EmbedderKind::Text`] it requires a [`candle_nn::VarBuilder`] (carried by
/// the caller) and the tokenizer's vocabulary size.
pub fn pipeline_embedder(
    kind: EmbedderKind,
    dim: usize,
    vocab_size: usize,
    hidden_dim: usize,
) -> Result<Box<dyn Embed>> {
    build_embedder(
        kind,
        dim,
        vocab_size,
        hidden_dim,
        &candle_core::Device::Cpu,
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use aarambh_studio_tokenizer::BpeTokenizer;

    /// A tiny character-level stub tokenizer for self-contained tests.
    struct StubTokenizer;
    impl TokenizerLike for StubTokenizer {
        fn encode(&self, text: &str) -> Result<Vec<u32>> {
            Ok(text.bytes().map(|b| b as u32).collect())
        }
        fn decode(&self, ids: &[u32]) -> Result<String> {
            Ok(ids.iter().map(|&i| i as u8 as char).collect())
        }
        fn vocab_size(&self) -> usize {
            256
        }
        fn eos_token_id(&self) -> u32 {
            0
        }
        fn bos_token_id(&self) -> Option<u32> {
            None
        }
    }

    fn make_corpus() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "aarambh-rag-corpus-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("alpha.txt"),
            "alpha document about cats and kittens playing in the garden",
        )
        .unwrap();
        std::fs::write(
            dir.join("bravo.txt"),
            "bravo document about dogs and puppies running in the park",
        )
        .unwrap();
        std::fs::write(
            dir.join("charlie.txt"),
            "charlie document about birds flying over the ocean at sunset",
        )
        .unwrap();
        dir
    }

    #[test]
    fn build_index_then_query_returns_relevant_chunk() {
        let corpus = make_corpus();
        let out = std::env::temp_dir().join(format!(
            "aarambh-rag-out-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let config = default_hashing_config(128, 2);
        let report = build_index(&StubTokenizer, &corpus, &out, &config).unwrap();
        assert!(report.chunks_indexed > 0);
        let pipeline = RetrievalPipeline::load_hashing(&out, 2).unwrap();
        let results = pipeline.query("cats kittens garden").unwrap();
        assert!(!results.is_empty());
        // The top hit must come from alpha.txt (the cats document).
        assert_eq!(
            results[0].source.file_name().unwrap().to_string_lossy(),
            "alpha.txt",
            "top result was {:?}",
            results[0].source
        );
        let _ = std::fs::remove_dir_all(&corpus);
        let _ = std::fs::remove_dir_all(&out);
    }

    #[test]
    fn rag_augmented_prompt_contains_retrieved_context_and_preserves_user_query() {
        let retrieved = vec![RetrievedChunk {
            chunk_id: 0,
            text: "The capital of France is Paris.".into(),
            source: PathBuf::from("docs/geography.txt"),
            offset: 0,
            score: 0.9,
        }];
        let augmented = augment_prompt("What is the capital of France?", &retrieved);
        assert!(
            augmented.contains("Retrieved context:"),
            "augmented prompt has context header"
        );
        assert!(
            augmented.contains("The capital of France is Paris."),
            "retrieved text is present"
        );
        assert!(
            augmented.contains("What is the capital of France?"),
            "user query is preserved"
        );
        assert!(
            augmented.find("Retrieved context:").unwrap()
                < augmented.find("What is the capital").unwrap(),
            "context precedes the user query"
        );
    }

    #[test]
    fn augment_prompt_with_no_results_is_a_noop() {
        let augmented = augment_prompt("just a question", &[]);
        assert_eq!(augmented, "just a question");
    }

    #[test]
    fn pipeline_rejects_dimension_mismatch() {
        let mut idx = VectorIndex::new(IndexConfig::new(4, 8, 16, 8).unwrap()).unwrap();
        idx.insert(IndexEntry::new(
            0,
            vec![1.0, 0.0, 0.0, 0.0],
            IndexMetadata {
                chunk_id: 0,
                source: PathBuf::from("x"),
                offset: 0,
                text: "x".into(),
            },
        ))
        .unwrap();
        let wrong_dim_embedder: Box<dyn Embed> = Box::new(HashingEmbedder::new(8).unwrap());
        match RetrievalPipeline::new(wrong_dim_embedder, idx, 2) {
            Err(AarambhError::Shape(_)) => {}
            Err(other) => panic!("expected Shape error, got {other:?}"),
            Ok(_) => panic!("expected Shape error, but pipeline built with mismatched dims"),
        }
    }

    #[test]
    fn build_index_rejects_text_embedder_without_weights() {
        let corpus = make_corpus();
        let out = std::env::temp_dir().join("aarambh-rag-text-fail");
        let mut config = default_hashing_config(64, 2);
        config.embedder_kind = EmbedderKind::Text;
        match build_index(&StubTokenizer, &corpus, &out, &config) {
            Err(AarambhError::Config(_)) => {}
            Err(other) => panic!("expected Config error, got {other:?}"),
            Ok(_) => panic!("expected Config error, but build_index accepted TextEmbedder"),
        }
        let _ = std::fs::remove_dir_all(&corpus);
        let _ = std::fs::remove_dir_all(&out);
    }

    #[test]
    fn default_hashing_config_is_weight_free() {
        let config = default_hashing_config(64, 4);
        assert_eq!(config.embedder_kind, EmbedderKind::Hashing);
        assert_eq!(config.embedding_dim, 64);
        assert_eq!(config.top_k, 4);
    }

    #[test]
    fn full_rag_round_trip_with_real_tokenizer_when_available() {
        // This test exercises the pipeline against a real BPE tokenizer if a
        // smoke fixture is present, and otherwise is skipped. It keeps the
        // crate honest about working with the project tokenizer, not just a
        // stub.
        let tokenizer_path = std::env::var("AARAMBH_RAG_SMOKE_TOKENIZER")
            .map(PathBuf::from)
            .ok();
        let Some(path) = tokenizer_path else {
            eprintln!(
                "[rag] skipping real-tokenizer round trip (set AARAMBH_RAG_SMOKE_TOKENIZER to enable)"
            );
            return;
        };
        let tokenizer = BpeTokenizer::from_pretrained(&path).unwrap();
        let corpus = make_corpus();
        let out = std::env::temp_dir().join("aarambh-rag-real-tok-out");
        let config = default_hashing_config(128, 2);
        build_index(&tokenizer, &corpus, &out, &config).unwrap();
        let pipeline = RetrievalPipeline::load_hashing(&out, 2).unwrap();
        let results = pipeline.query("dogs puppies park").unwrap();
        assert!(!results.is_empty());
        let _ = std::fs::remove_dir_all(&corpus);
        let _ = std::fs::remove_dir_all(&out);
    }
}
