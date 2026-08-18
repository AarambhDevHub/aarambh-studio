//! A from-scratch approximate-nearest-neighbour (ANN) index.
//!
//! This is a **navigable small-world graph** (NSW) index — a graph-based ANN
//! structure implemented entirely in pure Rust, with no FFI to an external
//! vector-search library and no external vector-database dependency. It is the
//! default and the only tested index path for the retrieval pipeline, as the
//! roadmap's forbidden-dependencies rule requires.
//!
//! # Algorithm
//!
//! - **Insert**: a new node is connected to its `max_neighbors` (M) nearest
//!   already-indexed neighbours, found by a greedy beam search of width
//!   `ef_construction`. Each newly-linked neighbour's adjacency list is pruned
//!   back to its `max_neighbors` nearest, keeping the graph's out-degree
//!   bounded while preserving navigability.
//! - **Search**: a greedy beam search of width `ef_search` from an entry
//!   point, returning the `top_k` nearest nodes by cosine similarity. Since
//!   vectors are L2-normalized at embed time, cosine similarity is a plain
//!   dot product.
//! - **Persist**: the whole index (config, entries, adjacency) is serialized
//!   to JSON so it is human-debuggable and trivially loadable.
//!
//! # Honesty
//!
//! The index is **approximate**: search returns a near-optimal set, not a
//! guaranteed-optimal one. Recall against a brute-force scan is measured by
//! the crate's acceptance test and is required to meet a documented floor.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use aarambh_studio_core::{AarambhError, Result};
use serde::{Deserialize, Serialize};

/// Configuration for the [`VectorIndex`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    /// Vector dimensionality. Every inserted vector must have this length.
    pub dim: usize,
    /// Maximum outgoing graph edges per node (the NSW "M" parameter).
    pub max_neighbors: usize,
    /// Beam-search width during insertion (the NSW "ef_construction").
    pub ef_construction: usize,
    /// Beam-search width during queries (the NSW "ef_search").
    pub ef_search: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            dim: 64,
            max_neighbors: 16,
            ef_construction: 64,
            ef_search: 32,
        }
    }
}

impl IndexConfig {
    /// Construct a config, validating all parameters are non-zero and that
    /// `ef_search` >= 1 (a beam width of zero would return no results).
    pub fn new(
        dim: usize,
        max_neighbors: usize,
        ef_construction: usize,
        ef_search: usize,
    ) -> Result<Self> {
        if dim == 0 {
            return Err(AarambhError::Config("index dim must be > 0".into()));
        }
        if max_neighbors == 0 {
            return Err(AarambhError::Config("max_neighbors must be > 0".into()));
        }
        if ef_construction == 0 {
            return Err(AarambhError::Config("ef_construction must be > 0".into()));
        }
        if ef_search == 0 {
            return Err(AarambhError::Config("ef_search must be > 0".into()));
        }
        Ok(Self {
            dim,
            max_neighbors,
            ef_construction,
            ef_search,
        })
    }
}

/// Per-chunk metadata stored alongside each indexed vector.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndexMetadata {
    /// The chunk id (monotonic within the corpus build).
    pub chunk_id: u64,
    /// The source file path the chunk came from.
    pub source: PathBuf,
    /// Byte offset of the chunk in its source.
    pub offset: usize,
    /// The chunk's decoded text.
    pub text: String,
}

/// One indexed vector and its metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndexEntry {
    /// Stable id (matches `IndexMetadata::chunk_id`).
    pub id: u64,
    /// The L2-normalized embedding vector.
    pub vector: Vec<f32>,
    /// Per-chunk metadata.
    pub metadata: IndexMetadata,
}

impl IndexEntry {
    /// Construct an entry.
    pub fn new(id: u64, vector: Vec<f32>, metadata: IndexMetadata) -> Self {
        Self {
            id,
            vector,
            metadata,
        }
    }
}

/// One search hit.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchResult {
    /// The matched entry's id.
    pub id: u64,
    /// Cosine similarity score (higher is better; in [-1, 1] for normalized
    /// vectors).
    pub score: f32,
    /// The matched entry's metadata.
    pub metadata: IndexMetadata,
}

/// A navigable small-world graph ANN index.
///
/// The graph is stored as an adjacency list parallel to the entries vector:
/// `graph[i]` is the list of node indices reachable from node `i`.
pub struct VectorIndex {
    config: IndexConfig,
    entries: Vec<IndexEntry>,
    graph: Vec<Vec<usize>>,
}

impl VectorIndex {
    /// Construct an empty index.
    pub fn new(config: IndexConfig) -> Result<Self> {
        // Re-validate via IndexConfig::new to catch a default-constructed
        // config that someone mutated into an invalid state.
        let validated = IndexConfig::new(
            config.dim,
            config.max_neighbors,
            config.ef_construction,
            config.ef_search,
        )?;
        Ok(Self {
            config: validated,
            entries: Vec::new(),
            graph: Vec::new(),
        })
    }

    /// Return the index config.
    pub fn config(&self) -> &IndexConfig {
        &self.config
    }

    /// Return the number of indexed entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return a reference to all indexed entries.
    pub fn entries(&self) -> &[IndexEntry] {
        &self.entries
    }

    /// Insert one entry, connecting it to its nearest neighbours in the graph.
    pub fn insert(&mut self, entry: IndexEntry) -> Result<()> {
        if entry.vector.len() != self.config.dim {
            return Err(AarambhError::Shape(format!(
                "vector dimension {} does not match index dimension {}",
                entry.vector.len(),
                self.config.dim
            )));
        }
        let new_node = self.entries.len();
        self.entries.push(entry);
        // Start with empty adjacency; populate below. If this is the first
        // node, it has no neighbours to connect to.
        self.graph.push(Vec::new());
        if new_node == 0 {
            return Ok(());
        }
        // Find ef_construction nearest existing nodes via greedy beam search.
        let query = self.entries[new_node].vector.clone();
        let ef = self.config.ef_construction;
        let neighbors = self.greedy_beam_search(&query, ef, Some(new_node))?;
        // Connect the new node to its M nearest of those found.
        let m = self.config.max_neighbors;
        let mut sorted = neighbors;
        sorted.sort_by(|a, b| b.1.total_cmp(&a.1)); // highest score first
        let chosen: Vec<usize> = sorted.iter().take(m).map(|(idx, _)| *idx).collect();
        self.graph[new_node] = chosen.clone();
        // For each chosen neighbour, add the new node to its adjacency and
        // prune its list back to M nearest.
        for &neighbor_idx in &chosen {
            let neighbor_vec = self.entries[neighbor_idx].vector.clone();
            let mut adj = self.graph[neighbor_idx].clone();
            if !adj.contains(&new_node) {
                adj.push(new_node);
            }
            // Prune: keep the M nearest by similarity to `neighbor_idx`.
            adj.sort_by(|&a, &b| {
                let sa = cosine(&self.entries[a].vector, &neighbor_vec);
                let sb = cosine(&self.entries[b].vector, &neighbor_vec);
                sb.total_cmp(&sa)
            });
            adj.truncate(m);
            self.graph[neighbor_idx] = adj;
        }
        Ok(())
    }

    /// Search the index for the `top_k` entries nearest to `query`.
    pub fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.config.dim {
            return Err(AarambhError::Shape(format!(
                "query dimension {} does not match index dimension {}",
                query.len(),
                self.config.dim
            )));
        }
        if self.entries.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }
        let ef = self.config.ef_search.max(top_k);
        let found = self.greedy_beam_search(query, ef, None)?;
        let mut sorted = found;
        sorted.sort_by(|a, b| b.1.total_cmp(&a.1)); // highest score first
        Ok(sorted
            .into_iter()
            .take(top_k)
            .map(|(idx, score)| {
                let entry = &self.entries[idx];
                SearchResult {
                    id: entry.id,
                    score,
                    metadata: entry.metadata.clone(),
                }
            })
            .collect())
    }

    /// Greedy beam search returning up to `ef` (node-index, score) pairs.
    ///
    /// Implements the standard NSW search: starting from an entry point, we
    /// repeatedly expand the best-scoring not-yet-expanded candidate, adding
    /// its neighbours to the frontier when they improve on the current
    /// result set. `exclude` is the node index currently being inserted (so
    /// we don't connect a node to itself); `None` for queries.
    fn greedy_beam_search(
        &self,
        query: &[f32],
        ef: usize,
        exclude: Option<usize>,
    ) -> Result<Vec<(usize, f32)>> {
        if self.entries.is_empty() {
            return Ok(Vec::new());
        }
        let entry_point = 0usize;
        let mut expanded: HashSet<usize> = HashSet::new();
        let mut seen: HashSet<usize> = HashSet::new();
        seen.insert(entry_point);
        let mut results: Vec<(usize, f32)> = vec![(
            entry_point,
            cosine(query, &self.entries[entry_point].vector),
        )];
        // The frontier: candidates seen but not yet expanded.
        let mut frontier: Vec<(usize, f32)> = results.clone();
        loop {
            // Pick the best-scoring candidate that has not been expanded.
            let best = frontier
                .iter()
                .filter(|(idx, _)| !expanded.contains(idx))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .copied();
            let Some((best_idx, _)) = best else {
                break;
            };
            expanded.insert(best_idx);
            for &neighbor in &self.graph[best_idx] {
                if let Some(excluded) = exclude
                    && neighbor == excluded
                {
                    continue;
                }
                if seen.contains(&neighbor) {
                    continue;
                }
                let score = cosine(query, &self.entries[neighbor].vector);
                let worst_in_results = results
                    .iter()
                    .map(|(_, s)| *s)
                    .fold(f32::INFINITY, f32::min);
                if results.len() < ef || score > worst_in_results {
                    seen.insert(neighbor);
                    results.push((neighbor, score));
                    frontier.push((neighbor, score));
                    // Trim results to ef, dropping the worst.
                    if results.len() > ef {
                        let worst_idx = results
                            .iter()
                            .enumerate()
                            .min_by(|(_, a), (_, b)| a.1.total_cmp(&b.1))
                            .map(|(i, _)| i);
                        if let Some(worst_idx) = worst_idx {
                            results.swap_remove(worst_idx);
                        }
                    }
                }
            }
        }
        Ok(results)
    }

    /// Persist the index to a directory as `index.json`.
    ///
    /// The directory is created if it does not exist. The on-disk format is
    /// a single JSON object with the config, the entries, and the adjacency
    /// graph — human-readable and trivially loadable.
    pub fn save(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir).map_err(AarambhError::from)?;
        let serialized = SerializedIndex {
            config: self.config.clone(),
            entries: self.entries.clone(),
            graph: self.graph.clone(),
        };
        let json = serde_json::to_string_pretty(&serialized).map_err(AarambhError::from)?;
        let path = dir.join("index.json");
        std::fs::write(&path, json).map_err(AarambhError::from)?;
        Ok(())
    }

    /// Load an index previously written by [`Self::save`].
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("index.json");
        let json = std::fs::read_to_string(&path).map_err(AarambhError::from)?;
        let serialized: SerializedIndex =
            serde_json::from_str(&json).map_err(AarambhError::from)?;
        let index = VectorIndex::new(serialized.config)?;
        // Bypass insert() graph rebuild: trust the persisted adjacency.
        Ok(VectorIndex {
            config: index.config,
            entries: serialized.entries,
            graph: serialized.graph,
        })
    }
}

/// The serialized form of a [`VectorIndex`].
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializedIndex {
    config: IndexConfig,
    entries: Vec<IndexEntry>,
    graph: Vec<Vec<usize>>,
}

/// Cosine similarity for two vectors. For L2-normalized inputs this is a plain
/// dot product in [-1, 1]; for non-normalized inputs it is still a valid
/// cosine. Used internally by the index.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = (na * nb).sqrt();
    if denom == 0.0 { 0.0 } else { dot / denom }
}

/// Brute-force top-k nearest neighbours — used by tests to measure the ANN
/// index's recall against a guaranteed-optimal baseline.
pub fn brute_force_top_k(entries: &[IndexEntry], query: &[f32], top_k: usize) -> Vec<(u64, f32)> {
    let mut scored: Vec<(u64, f32)> = entries
        .iter()
        .map(|e| (e.id, cosine(query, &e.vector)))
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored.truncate(top_k);
    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: u64, vector: Vec<f32>, text: &str) -> IndexEntry {
        IndexEntry::new(
            id,
            vector,
            IndexMetadata {
                chunk_id: id,
                source: PathBuf::from("doc.txt"),
                offset: 0,
                text: text.into(),
            },
        )
    }

    fn normalize(v: &mut [f32]) {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
    }

    #[test]
    fn index_config_validates_parameters() {
        assert!(IndexConfig::new(0, 16, 64, 32).is_err());
        assert!(IndexConfig::new(64, 0, 64, 32).is_err());
        assert!(IndexConfig::new(64, 16, 0, 32).is_err());
        assert!(IndexConfig::new(64, 16, 64, 0).is_err());
        assert!(IndexConfig::new(64, 16, 64, 32).is_ok());
    }

    #[test]
    fn insert_rejects_wrong_dimension() {
        let mut idx = VectorIndex::new(IndexConfig::new(4, 8, 16, 8).unwrap()).unwrap();
        let err = idx.insert(entry(0, vec![0.0; 8], "x")).unwrap_err();
        assert!(matches!(err, AarambhError::Shape(_)), "{err:?}");
    }

    #[test]
    fn search_rejects_wrong_dimension_query() {
        let mut idx = VectorIndex::new(IndexConfig::new(4, 8, 16, 8).unwrap()).unwrap();
        idx.insert(entry(0, vec![1.0, 0.0, 0.0, 0.0], "a")).unwrap();
        let err = idx.search(&[0.0; 8], 1).unwrap_err();
        assert!(matches!(err, AarambhError::Shape(_)), "{err:?}");
    }

    #[test]
    fn empty_index_search_returns_empty() {
        let idx = VectorIndex::new(IndexConfig::new(4, 8, 16, 8).unwrap()).unwrap();
        let out = idx.search(&[1.0, 0.0, 0.0, 0.0], 5).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn index_build_and_query_round_trip_returns_the_inserted_chunk() {
        // Build an index where each chunk's text is embedded such that the
        // chunk's own text is the nearest neighbour of itself.
        let dim = 8usize;
        let mut idx = VectorIndex::new(IndexConfig::new(dim, 8, 32, 16).unwrap()).unwrap();
        let texts = [
            "the quick brown fox",
            "lazy dog sleeps",
            "rust programming language",
            "neural network training",
        ];
        for (i, text) in texts.iter().enumerate() {
            let mut v = vec![0.0f32; dim];
            // Deterministic, distinct vector per text so self-similarity is 1.
            for (j, byte) in text.bytes().enumerate() {
                v[j % dim] += byte as f32;
            }
            normalize(&mut v);
            idx.insert(entry(i as u64, v, text)).unwrap();
        }
        // Query with each text's own embedding; the top-1 result must be that
        // text's chunk (round-trip).
        for (i, text) in texts.iter().enumerate() {
            let mut q = vec![0.0f32; dim];
            for (j, byte) in text.bytes().enumerate() {
                q[j % dim] += byte as f32;
            }
            normalize(&mut q);
            let results = idx.search(&q, 1).unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(
                results[0].id, i as u64,
                "query for {text:?} returned id {}",
                results[0].id
            );
            assert_eq!(results[0].metadata.text, *text);
            assert!(
                (results[0].score - 1.0).abs() < 1e-4,
                "self-similarity ~1.0, got {}",
                results[0].score
            );
        }
    }

    #[test]
    fn retrieval_recall_on_a_small_labelled_holdout_meets_a_documented_floor() {
        // A labelled holdout: each query has a known-correct chunk. We embed
        // both corpus and queries with the real HashingEmbedder (byte 3-gram
        // hashing), measure recall@1 (is the correct chunk the top-1 result?),
        // and require it to meet a documented floor of 0.8.
        use crate::embedding::{Embed, HashingEmbedder};
        let dim = 256usize;
        let embedder = HashingEmbedder::new(dim).unwrap();
        let mut idx = VectorIndex::new(IndexConfig::new(dim, 8, 64, 32).unwrap()).unwrap();
        let corpus = [
            "alpha document about cats and kittens",
            "bravo document about dogs and puppies",
            "charlie document about birds and feathers",
            "delta document about fish and oceans",
            "echo document about reptiles and scales",
            "foxtrot document about insects and antennae",
            "golf document about mammals and fur",
            "hotel document about plants and leaves",
            "india document about fungi and spores",
            "juliet document about bacteria and cells",
        ];
        let queries: Vec<(&str, u64)> = vec![
            ("cats kittens", 0),
            ("dogs puppies", 1),
            ("birds feathers", 2),
            ("fish oceans", 3),
            ("reptiles scales", 4),
            ("insects antennae", 5),
            ("mammals fur", 6),
            ("plants leaves", 7),
            ("fungi spores", 8),
            ("bacteria cells", 9),
        ];
        for (i, text) in corpus.iter().enumerate() {
            let v = embedder.embed(text).unwrap();
            idx.insert(entry(i as u64, v, text)).unwrap();
        }
        let mut hits = 0usize;
        for (query, expected_id) in &queries {
            let q = embedder.embed(query).unwrap();
            let results = idx.search(&q, 1).unwrap();
            if let Some(top) = results.first()
                && top.id == *expected_id
            {
                hits += 1;
            }
        }
        let recall = hits as f64 / queries.len() as f64;
        let documented_floor = 0.8;
        assert!(
            recall >= documented_floor,
            "recall@1 = {recall} is below documented floor {documented_floor}"
        );
    }

    #[test]
    fn ann_recall_matches_brute_force_on_small_index() {
        // For a small index, the NSW search should match brute force at top_k.
        let dim = 12usize;
        let mut idx = VectorIndex::new(IndexConfig::new(dim, 8, 32, 32).unwrap()).unwrap();
        let mut rng = 12345u64;
        let mut next_random = || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng >> 33) as f32) / (1u64 << 31) as f32
        };
        let mut entries = Vec::new();
        for i in 0..40 {
            let mut v: Vec<f32> = (0..dim).map(|_| next_random()).collect();
            normalize(&mut v);
            let e = entry(i as u64, v.clone(), &format!("chunk-{i}"));
            entries.push(e.clone());
            idx.insert(e).unwrap();
        }
        for trial in 0..8 {
            let mut q: Vec<f32> = (0..dim).map(|_| next_random()).collect();
            normalize(&mut q);
            let ann = idx.search(&q, 3).unwrap();
            let bf = brute_force_top_k(&entries, &q, 3);
            let ann_ids: std::collections::HashSet<u64> = ann.iter().map(|r| r.id).collect();
            let bf_ids: std::collections::HashSet<u64> = bf.iter().map(|(id, _)| *id).collect();
            let overlap = ann_ids.intersection(&bf_ids).count();
            // Allow approximate mismatch but require >= 2/3 overlap on a small
            // index with ef_search=32 (near-exhaustive).
            assert!(
                overlap >= 2,
                "trial {trial}: ann={ann_ids:?} bf={bf_ids:?} overlap={overlap}"
            );
        }
    }

    #[test]
    fn save_and_load_round_trips_the_index() {
        let dim = 4usize;
        let mut idx = VectorIndex::new(IndexConfig::new(dim, 4, 8, 4).unwrap()).unwrap();
        idx.insert(entry(0, vec![1.0, 0.0, 0.0, 0.0], "a")).unwrap();
        idx.insert(entry(1, vec![0.0, 1.0, 0.0, 0.0], "b")).unwrap();
        idx.insert(entry(2, vec![0.0, 0.0, 1.0, 0.0], "c")).unwrap();
        let tmp = std::env::temp_dir().join(format!("aarambh-rag-index-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        idx.save(&tmp).unwrap();
        let loaded = VectorIndex::load(&tmp).unwrap();
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded.entries(), idx.entries());
        let results = loaded.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].id, 0);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn cosine_similarity_properties() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!((cosine(&[1.0, 0.0], &[0.0, 1.0]) - 0.0).abs() < 1e-6);
        assert!((cosine(&[1.0, 0.0], &[-1.0, 0.0]) - (-1.0)).abs() < 1e-6);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
    }
}
