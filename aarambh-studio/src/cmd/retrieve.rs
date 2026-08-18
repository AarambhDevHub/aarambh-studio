//! The `retrieve` subcommand — Phase 49 RAG index building.
//!
//! `aarambh-studio retrieve build-index --corpus docs/ --output my_index/`
//! chunks every text file under `--corpus`, embeds each chunk, and writes a
//! navigable small-world graph index to `--output/index.json`. The index is
//! later consumed by `aarambh-studio infer --rag --index my_index/`.

use std::path::PathBuf;

use aarambh_studio_retrieve::{
    ChunkingConfig, EmbedderKind, IndexConfig, RetrievalConfig, build_index,
};
use aarambh_studio_tokenizer::BpeTokenizer;
use clap::{Args, Subcommand};

/// Build or inspect a retrieval index (Phase 49 RAG).
#[derive(Debug, Args)]
pub struct RetrieveArgs {
    #[command(subcommand)]
    pub command: RetrieveCommand,
}

/// Subcommands of `aarambh-studio retrieve`.
#[derive(Debug, Subcommand)]
pub enum RetrieveCommand {
    /// Build a retrieval index from a corpus directory.
    BuildIndex(BuildIndexArgs),
}

/// Arguments for `aarambh-studio retrieve build-index`.
#[derive(Debug, Args)]
pub struct BuildIndexArgs {
    /// Directory containing `.txt`/`.md`/`.jsonl` corpus files (searched
    /// recursively).
    #[arg(long)]
    pub corpus: PathBuf,
    /// Output directory for the written `index.json`.
    #[arg(long)]
    pub output: PathBuf,
    /// Tokenizer JSON path; falls back to the project default tokenizer.
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,
    /// Embedder kind: `hashing` (default, weight-free) or `text` (candle head,
    /// requires `--embedding-model`).
    #[arg(long, default_value = "hashing")]
    pub embedder: String,
    /// Embedding dimension (vector width). Default 64.
    #[arg(long, default_value_t = 64)]
    pub embedding_dim: usize,
    /// Trained text-embedding checkpoint path (only used with `--embedder text`).
    #[arg(long)]
    pub embedding_model: Option<PathBuf>,
    /// Target tokens per chunk. Default 256.
    #[arg(long, default_value_t = 256)]
    pub chunk_size: usize,
    /// Token overlap between consecutive chunks. Default 32.
    #[arg(long, default_value_t = 32)]
    pub overlap: usize,
    /// Number of chunks to retrieve per query (recorded in the index config).
    #[arg(long, default_value_t = 4)]
    pub top_k: usize,
    /// Maximum graph edges per node (NSW "M"). Default 16.
    #[arg(long, default_value_t = 16)]
    pub max_neighbors: usize,
    /// Beam-search width during insertion (NSW "ef_construction"). Default 64.
    #[arg(long, default_value_t = 64)]
    pub ef_construction: usize,
    /// Beam-search width during queries (NSW "ef_search"). Default 32.
    #[arg(long, default_value_t = 32)]
    pub ef_search: usize,
}

impl RetrieveArgs {
    /// Dispatch to the selected subcommand.
    pub fn run(self) -> anyhow::Result<()> {
        match self.command {
            RetrieveCommand::BuildIndex(args) => run_build_index(args),
        }
    }
}

/// Convenience entry point matching the `run(args)` convention of the other
/// command modules.
pub fn run(args: RetrieveArgs) -> anyhow::Result<()> {
    args.run()
}

fn run_build_index(args: BuildIndexArgs) -> anyhow::Result<()> {
    let embedder_kind = parse_embedder_kind(&args.embedder)?;
    if embedder_kind == EmbedderKind::Text && args.embedding_model.is_none() {
        return Err(anyhow::anyhow!(
            "--embedder text requires --embedding-model <PATH> (no trained embedding checkpoint provided)"
        ));
    }
    let chunking = ChunkingConfig::new(args.chunk_size, args.overlap)?;
    let index_config = IndexConfig::new(
        args.embedding_dim,
        args.max_neighbors,
        args.ef_construction,
        args.ef_search,
    )?;
    let config = RetrievalConfig {
        top_k: args.top_k,
        chunking,
        index: index_config,
        embedder_kind,
        embedding_dim: args.embedding_dim,
    };
    let tokenizer_path = match &args.tokenizer {
        Some(path) => path.clone(),
        None => default_tokenizer_path()?,
    };
    let tokenizer = BpeTokenizer::from_pretrained(&tokenizer_path)?;
    eprintln!(
        "[retrieve] building index: corpus={} output={} embedder={} dim={} chunk_size={} overlap={} top_k={}",
        args.corpus.display(),
        args.output.display(),
        args.embedder,
        args.embedding_dim,
        args.chunk_size,
        args.overlap,
        args.top_k
    );
    let report = build_index(&tokenizer, &args.corpus, &args.output, &config)?;
    eprintln!(
        "[retrieve] indexed {} chunks (dim={}) → {}",
        report.chunks_indexed,
        report.embedding_dim,
        report.index_path.display()
    );
    Ok(())
}

fn parse_embedder_kind(value: &str) -> anyhow::Result<EmbedderKind> {
    match value.to_ascii_lowercase().as_str() {
        "hashing" | "hash" => Ok(EmbedderKind::Hashing),
        "text" | "candle" => Ok(EmbedderKind::Text),
        other => Err(anyhow::anyhow!(
            "unknown embedder kind '{other}', expected 'hashing' or 'text'"
        )),
    }
}

/// Resolve a default tokenizer path. Mirrors the inference CLI's fallback: the
/// `configs/tiny_shakespeare.toml` training config's tokenizer, if present.
fn default_tokenizer_path() -> anyhow::Result<PathBuf> {
    // The smoke corpus is exercised with the project's default tokenizer; if
    // none is configured, surface a clear error rather than a panic.
    let candidate = PathBuf::from("checkpoints/tiny_shakespeare/tokenizer.json");
    if candidate.exists() {
        return Ok(candidate);
    }
    Err(anyhow::anyhow!(
        "no --tokenizer given and default {} not found; pass --tokenizer <PATH>",
        candidate.display()
    ))
}
