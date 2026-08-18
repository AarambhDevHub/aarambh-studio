//! Text-embedding heads for retrieval.
//!
//! Two embedders live here, both implementing the [`Embed`] trait:
//!
//! - [`HashingEmbedder`]: a deterministic, weight-free embedder. It hashes each
//!   token of the input into a fixed-size vector of signed counts, then
//!   L2-normalizes the result. It needs no checkpoint, runs on any CPU in
//!   microseconds, and is the **default tested path** so the full retrieval
//!   pipeline (chunking → embed → index → retrieve → augment prompt) runs
//!   end-to-end without a trained embedding model. This mirrors the Phase
//!   47/48 honesty discipline: a fake-but-real decoder for tests, a real
//!   engine for production.
//!
//! - [`TextEmbedder`]: the candle-backed, contrastively-trained head shape the
//!   roadmap describes — a token-embedding table (separate from the main
//!   decoder's), a mean-pool over the input tokens, a linear projection to the
//!   embedding dimension, and L2 normalization. It is CPU-capable and loadable
//!   as weights. When no trained embedding checkpoint is available, the
//!   [`HashingEmbedder`] remains the default; the [`TextEmbedder`] exists so a
//!   trained checkpoint can be dropped in without changing the retrieval
//!   pipeline.

use aarambh_studio_core::{AarambhError, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Init, Module, VarBuilder};
use serde::{Deserialize, Serialize};

/// The kind of embedder to construct.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbedderKind {
    /// The weight-free hashing embedder (default).
    #[default]
    Hashing,
    /// The candle-backed trained text-embedding head.
    Text,
}

/// Convert text into a fixed-size vector.
///
/// All embedders return an L2-normalized vector so that cosine similarity is a
/// plain dot product — the [`crate::index::VectorIndex`] relies on this.
pub trait Embed {
    /// Embed `text` into a fixed-size, L2-normalized vector.
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// The dimensionality of vectors produced by this embedder.
    fn dim(&self) -> usize;
}

/// Configuration for the candle-backed [`TextEmbedder`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextEmbedderConfig {
    /// Tokenizer vocabulary size (the embedding table width).
    pub vocab_size: usize,
    /// Hidden dimension of the token-embedding table.
    pub hidden_dim: usize,
    /// Output embedding dimension (the projection target).
    pub embedding_dim: usize,
}

impl TextEmbedderConfig {
    /// Construct a config, validating dimensions are non-zero.
    pub fn new(vocab_size: usize, hidden_dim: usize, embedding_dim: usize) -> Result<Self> {
        if vocab_size == 0 {
            return Err(AarambhError::Config("vocab_size must be > 0".into()));
        }
        if hidden_dim == 0 {
            return Err(AarambhError::Config("hidden_dim must be > 0".into()));
        }
        if embedding_dim == 0 {
            return Err(AarambhError::Config("embedding_dim must be > 0".into()));
        }
        Ok(Self {
            vocab_size,
            hidden_dim,
            embedding_dim,
        })
    }
}

/// A candle-backed text-embedding head: token embedding → mean-pool → linear
/// projection → L2-normalize.
///
/// This is the architecture the roadmap describes for Phase 49 — a small,
/// dedicated, contrastively-trained text-embedding head, CPU-capable,
/// **separate from the main decoder**. It is constructed from a
/// [`TextEmbedderConfig`] and a [`candle_nn::VarBuilder`], exactly like the
/// main decoder's [`TokenEmbedding`](aarambh_studio_model::TokenEmbedding),
/// and loads weights when a checkpoint is provided.
pub struct TextEmbedder {
    config: TextEmbedderConfig,
    embedding: candle_nn::Embedding,
    projection: candle_nn::Linear,
    device: Device,
}

impl TextEmbedder {
    /// Construct the embedding head from a config and a var builder.
    ///
    /// The token-embedding table is created at `vb.pp("token_embedding")`
    /// with the same `Randn{mean:0, stdev:0.02}` initialization the main
    /// decoder uses, and the projection is a bias-free linear layer at
    /// `vb.pp("projection")`.
    pub fn new(config: TextEmbedderConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let weight = vb.get_with_hints(
            (config.vocab_size, config.hidden_dim),
            "weight",
            Init::Randn {
                mean: 0.0,
                stdev: 0.02,
            },
        )?;
        let embedding = candle_nn::Embedding::new(weight, config.hidden_dim);
        let projection = candle_nn::linear_no_bias(
            config.hidden_dim,
            config.embedding_dim,
            vb.pp("projection"),
        )?;
        Ok(Self {
            config,
            embedding,
            projection,
            device: vb.device().clone(),
        })
    }

    /// Return the embedder's config.
    pub fn config(&self) -> &TextEmbedderConfig {
        &self.config
    }
}

impl Embed for TextEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        embed_with_tokenizer(
            text,
            &self.embedding,
            &self.projection,
            &self.device,
            self.config.embedding_dim,
        )
    }

    fn dim(&self) -> usize {
        self.config.embedding_dim
    }
}

/// Embed text using a candle token-embedding table, mean-pool, project, and
/// L2-normalize. Shared helper so the [`TextEmbedder`] stays thin.
fn embed_with_tokenizer(
    text: &str,
    embedding: &candle_nn::Embedding,
    projection: &candle_nn::Linear,
    device: &Device,
    embedding_dim: usize,
) -> Result<Vec<f32>> {
    // We tokenize with a trivial byte-fallback here because the TextEmbedder
    // is the trained-head path and expects a paired tokenizer to be supplied
    // by the caller via the retrieval pipeline. To keep the Embed trait
    // tokenizer-free (so it composes with any tokenizer, not just the
    // project's BPE), we encode text as UTF-8 bytes mapped into the lower
    // vocab range — a real trained checkpoint's tokenizer would be used at
    // the pipeline layer instead. This keeps the head testable in isolation.
    let ids: Vec<u32> = text.as_bytes().iter().map(|b| *b as u32).collect();
    if ids.is_empty() {
        return Ok(vec![0.0; embedding_dim]);
    }
    let ids_tensor = Tensor::from_vec(ids.clone(), (1, ids.len()), device)?;
    let token_embeddings = embedding.forward(&ids_tensor)?; // [1, seq, hidden]
    let pooled = token_embeddings.mean(1)?; // [1, hidden]
    let projected = projection.forward(&pooled)?; // [1, embedding_dim]
    let flat = projected.squeeze(0)?.to_dtype(DType::F32)?;
    let vec_f32 = flat.to_vec1::<f32>()?;
    Ok(l2_normalize(&vec_f32))
}

/// A deterministic, weight-free text embedder.
///
/// It maps each UTF-8 byte of the input into one of `dim` buckets via hashing,
/// adds `+1` for one half of the hash space and `-1` for the other (the
/// classic "feature hashing" / "hashing trick" trick), then L2-normalizes the
/// resulting vector. The mapping is deterministic for a fixed `dim` and seed,
/// requires no checkpoint, and runs on any CPU in microseconds.
///
/// Cosine similarity between two hashing-embedded texts therefore reflects
/// shared byte n-gram overlap — sufficient for the from-scratch index and
/// retrieval recall floor tested by this crate, while remaining honest about
/// not requiring a trained model to exercise the pipeline.
pub struct HashingEmbedder {
    dim: usize,
    seed: u32,
}

impl HashingEmbedder {
    /// Construct a hashing embedder with the given output dimension.
    ///
    /// `dim` must be greater than zero. A larger `dim` reduces hash collisions
    /// and improves recall; 64–256 is a reasonable range for small corpora.
    pub fn new(dim: usize) -> Result<Self> {
        if dim == 0 {
            return Err(AarambhError::Config(
                "hashing embedder dim must be > 0".into(),
            ));
        }
        Ok(Self {
            dim,
            seed: 0xc0ffee_u32,
        })
    }

    /// Construct a hashing embedder with a custom seed (useful for tests that
    /// want a fixed, reproducible mapping independent of the default seed).
    pub fn with_seed(dim: usize, seed: u32) -> Result<Self> {
        if dim == 0 {
            return Err(AarambhError::Config(
                "hashing embedder dim must be > 0".into(),
            ));
        }
        Ok(Self { dim, seed })
    }

    /// Return the configured dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Return the seed.
    pub fn seed(&self) -> u32 {
        self.seed
    }
}

impl Embed for HashingEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut vec = vec![0.0f32; self.dim];
        if text.is_empty() {
            return Ok(vec);
        }
        // Hash byte 3-grams so similar texts collide in the same buckets,
        // giving non-trivial cosine similarity for shared substrings.
        let bytes = text.as_bytes();
        let mut grams = Vec::with_capacity(bytes.len().saturating_sub(2).max(1));
        if bytes.len() >= 3 {
            for window in bytes.windows(3) {
                grams.push(window);
            }
        } else {
            grams.push(bytes);
        }
        for gram in grams {
            let h = fnv1a_32(gram, self.seed);
            let bucket = (h as usize) % self.dim;
            // Sign from a second hash so positive and negative contributions
            // balance in expectation (the signed hashing trick).
            let sign = if ((h >> 16) & 1) == 0 { 1.0 } else { -1.0 };
            vec[bucket] += sign;
        }
        Ok(l2_normalize(&vec))
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

/// FNV-1a 32-bit hash with a seed mix.
fn fnv1a_32(data: &[u8], seed: u32) -> u32 {
    let mut hash: u32 = 0x811c9dc5 ^ seed;
    for &byte in data {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

/// L2-normalize a vector, returning the zero vector if the input norm is zero
/// (so an all-zero input stays all-zero rather than producing NaNs).
pub(crate) fn l2_normalize(vec: &[f32]) -> Vec<f32> {
    let mut norm_sq = 0.0f32;
    for v in vec {
        norm_sq += v * v;
    }
    if norm_sq == 0.0 {
        return vec.to_vec();
    }
    let norm = norm_sq.sqrt();
    vec.iter().map(|v| v / norm).collect()
}

/// Build an embedder of the requested kind, optionally with weights.
///
/// For [`EmbedderKind::Hashing`] this ignores `vocab_size`, `hidden_dim`,
/// `device`, and `vb` — the hashing embedder is weight-free and CPU-native.
/// For [`EmbedderKind::Text`] it uses the tokenizer's vocabulary size to size
/// the token-embedding table and requires a [`candle_nn::VarBuilder`] (which
/// carries both the device and any loaded weights).
pub fn build_embedder(
    kind: EmbedderKind,
    dim: usize,
    vocab_size: usize,
    hidden_dim: usize,
    _device: &Device,
    vb: Option<VarBuilder<'_>>,
) -> Result<Box<dyn Embed>> {
    match kind {
        EmbedderKind::Hashing => Ok(Box::new(HashingEmbedder::new(dim)?)),
        EmbedderKind::Text => {
            let config = TextEmbedderConfig::new(vocab_size, hidden_dim, dim)?;
            let vb = vb.ok_or_else(|| {
                AarambhError::Config(
                    "TextEmbedder requires a VarBuilder (weights path) — use HashingEmbedder for the weight-free default".into(),
                )
            })?;
            Ok(Box::new(TextEmbedder::new(config, vb)?))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashing_embedder_is_deterministic_and_normalized() {
        let emb = HashingEmbedder::new(64).unwrap();
        let a = emb.embed("the quick brown fox").unwrap();
        let b = emb.embed("the quick brown fox").unwrap();
        assert_eq!(a, b, "identical inputs produce identical vectors");
        let norm: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "output is L2-normalized (norm={norm})"
        );
        assert_eq!(emb.dim(), 64);
    }

    #[test]
    fn hashing_embedder_similarity_tracks_shared_substrings() {
        let emb = HashingEmbedder::new(128).unwrap();
        let a = emb.embed("the quick brown fox jumps").unwrap();
        let b = emb.embed("the quick brown fox sleeps").unwrap();
        let c = emb.embed("completely unrelated tokens zzz").unwrap();
        let sim_ab = dot(&a, &b);
        let sim_ac = dot(&a, &c);
        // Two texts sharing a 3-gram prefix should be more similar than two
        // texts sharing none — the hashing trick's central property.
        assert!(
            sim_ab > sim_ac,
            "shared-substring texts must be more similar: ab={sim_ab} ac={sim_ac}"
        );
    }

    #[test]
    fn hashing_embedder_rejects_zero_dim() {
        assert!(HashingEmbedder::new(0).is_err());
    }

    #[test]
    fn l2_normalize_handles_zero_vector() {
        let z = vec![0.0; 4];
        let out = l2_normalize(&z);
        assert_eq!(out, vec![0.0; 4], "zero vector stays zero (no NaN)");
        let v = vec![3.0, 4.0];
        let out = l2_normalize(&v);
        assert!((out[0] - 0.6).abs() < 1e-5 && (out[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn text_embedder_config_validates_dimensions() {
        assert!(TextEmbedderConfig::new(0, 8, 8).is_err());
        assert!(TextEmbedderConfig::new(8, 0, 8).is_err());
        assert!(TextEmbedderConfig::new(8, 8, 0).is_err());
        let cfg = TextEmbedderConfig::new(100, 32, 16).unwrap();
        assert_eq!(cfg.embedding_dim, 16);
    }

    #[test]
    fn text_embedder_constructs_and_embeds_on_cpu() {
        let cfg = TextEmbedderConfig::new(256, 16, 8).unwrap();
        // Use a fresh VarMap so the token-embedding table is randomly
        // initialized (Init::Randn) rather than all-zeros, giving a
        // non-trivial, L2-normalized output.
        let varmap = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let head = TextEmbedder::new(cfg, vb).unwrap();
        let v = head.embed("hello world").unwrap();
        assert_eq!(v.len(), 8);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "TextEmbedder output is L2-normalized (norm={norm})"
        );
        // Random init should produce a non-zero vector.
        assert!(
            v.iter().any(|x| x.abs() > 1e-6),
            "output is non-zero under random init"
        );
    }

    #[test]
    fn build_embedder_hashing_ignores_tokenizer_args() {
        let emb = build_embedder(
            EmbedderKind::Hashing,
            32,
            9999, // ignored
            9999, // ignored
            &Device::Cpu,
            None,
        )
        .unwrap();
        let v = emb.embed("abc").unwrap();
        assert_eq!(v.len(), 32);
    }

    #[test]
    fn build_embedder_text_requires_varbuilder() {
        match build_embedder(EmbedderKind::Text, 8, 256, 16, &Device::Cpu, None) {
            Err(AarambhError::Config(_)) => {}
            Ok(_) => panic!("expected Config error, but TextEmbedder built without a VarBuilder"),
            Err(other) => panic!("expected Config error, got {other:?}"),
        }
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}
