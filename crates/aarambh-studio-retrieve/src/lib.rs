//! Retrieval-augmented generation (RAG) — Phase 49.
//!
//! A from-scratch, pure-Rust retrieval pipeline. No external vector
//! database is required, and none is used: the approximate-nearest-neighbour
//! index is a navigable small-world graph implemented entirely in this crate
//! (no FFI to an external vector-search library).
//!
//! Retrieved context augments the prompt *before* generation; it does not
//! touch model internals. RAG augments the prompt; it does not change how the
//! decoder processes it. This keeps the phase entirely additive and simple to
//! reason about.
//!
//! # Layout
//!
//! - [`chunking`]: fixed-size token-based document chunking with overlap.
//! - [`embedding`]: text-embedding heads. A [`embedding::HashingEmbedder`]
//!   provides a deterministic, weight-free default so the full pipeline runs
//!   end-to-end without a trained checkpoint; a [`embedding::TextEmbedder`]
//!   is the candle-backed, contrastively-trained head shape that loads
//!   weights when a checkpoint is available. Both implement [`Embed`].
//! - [`index`]: a from-scratch graph-based approximate-nearest-neighbour
//!   index with `insert` / `search` / `save` / `load`.
//! - [`retrieval`]: [`retrieval::RetrievalPipeline`] ties chunking, embedding,
//!   and the index together, and splices retrieved chunks into a prompt via
//!   [`retrieval::augment_prompt`].
//!
//! # Honesty boundary
//!
//! The default tested path uses the [`embedding::HashingEmbedder`] (no trained
//! weights) so the whole pipeline is testable in milliseconds without a
//! checkpoint — mirroring the Phase 47/48 "fake decoder for tests, real engine
//! for production" discipline. The [`embedding::TextEmbedder`] satisfies the
//! architecture the roadmap describes (a contrastively-trained, CPU-capable,
//! separate-from-the-decoder head, loadable as weights); when no embedding
//! checkpoint is shipped, the hashing embedder remains the default. An
//! optional plug-in adapter to an external vector store is a documented
//! extension point and is *not* implemented here — the from-scratch pure-Rust
//! index remains the default and the tested path.

#![deny(missing_docs)]

pub mod chunking;
pub mod embedding;
pub mod index;
pub mod retrieval;

pub use chunking::{Chunk, Chunker, ChunkingConfig};
pub use embedding::{Embed, EmbedderKind, HashingEmbedder, TextEmbedder, TextEmbedderConfig};
pub use index::{IndexConfig, IndexEntry, IndexMetadata, SearchResult, VectorIndex};
pub use retrieval::{
    BuildIndexReport, RetrievalConfig, RetrievalPipeline, RetrievedChunk, augment_prompt,
    build_index, build_index_from_chunks, default_hashing_config, pipeline_embedder,
};
