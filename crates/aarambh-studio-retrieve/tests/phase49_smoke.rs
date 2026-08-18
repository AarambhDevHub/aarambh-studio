//! Phase 49 end-to-end smoke integration test.
//!
//! Builds a retrieval index from the checked-in `data/rag_smoke_corpus/`
//! fixture using a stub byte tokenizer (so it runs without a trained
//! tokenizer checkpoint), queries it for a known fact, and asserts the
//! top-1 retrieved chunk comes from the expected source file. This is the
//! `scripts/phase49_smoke.sh` end-to-end proof, runnable as a plain
//! `cargo test` without any CLI or model dependency.

use std::path::{Path, PathBuf};

use aarambh_studio_core::{Result, TokenizerLike};
use aarambh_studio_retrieve::{
    Embed, EmbedderKind, HashingEmbedder, IndexConfig, RetrievalConfig, RetrievalPipeline,
    build_index, default_hashing_config,
};

/// A minimal byte-level tokenizer for the smoke test — encodes each UTF-8
/// byte as its own token id. This is sufficient for chunking plain-ASCII
/// corpus fixtures and keeps the smoke self-contained (no trained tokenizer
/// checkpoint required).
struct ByteTokenizer;

impl TokenizerLike for ByteTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(text.bytes().map(|b| b as u32).collect())
    }
    fn decode(&self, ids: &[u32]) -> Result<String> {
        let bytes: Vec<u8> = ids.iter().map(|&i| i as u8).collect();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
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

fn workspace_root() -> PathBuf {
    // tests/ is two levels under the crate root; the workspace root is three
    // levels up from the crate root.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap()
        .to_path_buf()
}

fn corpus_dir() -> PathBuf {
    workspace_root().join("data/rag_smoke_corpus")
}

fn output_dir() -> PathBuf {
    let nano = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("aarambh-phase49-smoke-it-{nano}"))
}

#[test]
fn phase49_end_to_end_build_and_query_smoke_corpus() {
    let corpus = corpus_dir();
    assert!(corpus.is_dir(), "smoke corpus dir exists: {}", corpus.display());
    let output = output_dir();
    let config = RetrievalConfig {
        embedding_dim: 256,
        chunking: aarambh_studio_retrieve::ChunkingConfig::new(128, 24).unwrap(),
        index: IndexConfig::new(256, 16, 64, 32).unwrap(),
        embedder_kind: EmbedderKind::Hashing,
        top_k: 4,
    };
    let report = build_index(&ByteTokenizer, &corpus, &output, &config).unwrap();
    assert!(report.chunks_indexed > 0, "indexed at least one chunk");
    assert_eq!(report.embedding_dim, 256);

    let pipeline = RetrievalPipeline::load_hashing(&output, 4).unwrap();
    // Query for a fact that appears in geography.txt.
    let results = pipeline.query("What is the capital of France?").unwrap();
    assert!(!results.is_empty(), "query returned at least one chunk");
    let top = &results[0];
    let top_source = top.source.file_name().unwrap().to_string_lossy().to_string();
    assert!(
        top_source == "geography.txt",
        "top-1 retrieved chunk for 'capital of France' came from {top_source}, expected geography.txt"
    );
    // The retrieved chunk's text must mention Paris.
    assert!(
        top.text.to_lowercase().contains("paris"),
        "top-1 chunk text mentions Paris: {}",
        top.text.chars().take(120).collect::<String>()
    );
    let _ = std::fs::remove_dir_all(&output);
}

#[test]
fn phase49_augment_prompt_smoke() {
    // Smoke that the prompt augmentation path used by `infer --rag` produces
    // a prompt whose retrieved-context block precedes the user's question and
    // preserves the question verbatim.
    let retrieved = vec![aarambh_studio_retrieve::RetrievedChunk {
        chunk_id: 0,
        text: "The capital of France is Paris.".into(),
        source: PathBuf::from("data/rag_smoke_corpus/geography.txt"),
        offset: 0,
        score: 0.95,
    }];
    let augmented =
        aarambh_studio_retrieve::augment_prompt("What is the capital of France?", &retrieved);
    assert!(augmented.starts_with("Retrieved context:"));
    assert!(augmented.contains("The capital of France is Paris."));
    assert!(augmented.contains("What is the capital of France?"));
    let ctx_pos = augmented.find("Retrieved context:").unwrap();
    let q_pos = augmented.find("What is the capital").unwrap();
    assert!(ctx_pos < q_pos, "context precedes the user query");
}

#[test]
fn phase49_default_hashing_config_is_weight_free() {
    let config = default_hashing_config(128, 4);
    assert_eq!(config.embedder_kind, EmbedderKind::Hashing);
    assert_eq!(config.embedding_dim, 128);
    // Sanity: the hashing embedder itself constructs and embeds.
    let emb = HashingEmbedder::new(128).unwrap();
    let v = emb.embed("hello world").unwrap();
    assert_eq!(v.len(), 128);
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5, "L2-normalized (norm={norm})");
}
