//! Neural-network layers used by the Aarambh transformer stack.
#![deny(missing_docs)]

/// Grouped-query attention layer.
pub mod attention;
/// Transformer block composition.
pub mod block;
/// Expert dispatch helpers.
pub mod dispatch;
/// SwiGLU feed-forward layer.
pub mod ffn;
/// Gated DeltaNet linear-attention layer and recurrent state.
pub mod gated_deltanet;
/// Training/inference KV cache helper.
pub mod kvcache;
/// Multi-Head Latent Attention layer and compressed-latent cache (v4 Phase 41).
pub mod mla;
/// Mixture-of-Experts feed-forward layer.
pub mod moe;
/// Multi-token prediction auxiliary head.
pub mod mtp;
/// RMSNorm layer.
pub mod norm;
/// Rotary-position embedding cache.
pub mod rope;
/// RoPE long-context scaling helpers.
pub mod rope_scaling;
/// Learned block-sparse attention and DSA indexer training helpers.
pub mod sparse_attention;

pub use attention::GroupedQueryAttention;
pub use block::{FeedForwardLayer, TokenMixer, TransformerBlock};
pub use dispatch::{dense_weighted_dispatch, effective_dispatch_kind, sparse_grouped_dispatch};
pub use ffn::SwiGluFfn;
pub use gated_deltanet::{DeltaNetForm, DeltaNetState, GatedDeltaNetLayer};
pub use kvcache::{DsaKvCache, HybridKvCache, KVCache};
pub use mla::{MlaAttention, MlaCache};
pub use moe::{
    GatingOutput, MoeFfn, MoeForwardStats, SharedExpertPath, load_balancing_loss_from_stats,
    top_k_gating,
};
pub use mtp::MtpHead;
pub use norm::RMSNorm;
pub use rope::RopeCache;
pub use sparse_attention::{DsaAttention, DsaForwardStats, DsaTeacherOutput};
