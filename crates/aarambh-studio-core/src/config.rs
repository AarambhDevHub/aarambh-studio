use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::error::{AarambhError, Result};

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
/// RoPE scaling strategy used to extend context length.
pub enum RopeScalingMethod {
    /// YaRN interpolation with beta correction ramp.
    #[default]
    Yarn,
    /// NTK-aware theta rescaling.
    Ntk,
    /// Linear inverse-frequency scaling.
    Linear,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
/// Configuration for long-context RoPE scaling.
pub struct RopeScalingConfig {
    /// Scaling method to apply.
    pub method: RopeScalingMethod,
    /// Context extension factor relative to the original context length.
    pub factor: f64,
    /// Context length used during the base model's original training.
    pub original_max_seq_len: usize,
    /// YaRN high-frequency correction boundary.
    pub beta_fast: f64,
    /// YaRN low-frequency correction boundary.
    pub beta_slow: f64,
    /// Multiplicative YaRN attention scale applied to cos/sin tables.
    pub attn_factor: f64,
}

impl Default for RopeScalingConfig {
    fn default() -> Self {
        Self {
            method: RopeScalingMethod::Yarn,
            factor: 1.0,
            original_max_seq_len: 0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attn_factor: 1.0,
        }
    }
}

impl RopeScalingConfig {
    /// Validate this scaling config against a model context length and RoPE head dimension.
    pub fn validate(&self, max_seq_len: usize, head_dim: usize) -> Result<()> {
        if self.factor <= 1.0 || !self.factor.is_finite() {
            return Err(AarambhError::Config(
                "rope_scaling.factor must be finite and greater than 1.0".into(),
            ));
        }
        if self.original_max_seq_len == 0 {
            return Err(AarambhError::Config(
                "rope_scaling.original_max_seq_len must be non-zero".into(),
            ));
        }
        if max_seq_len < self.original_max_seq_len {
            return Err(AarambhError::Config(format!(
                "max_seq_len {max_seq_len} must be >= rope_scaling.original_max_seq_len {}",
                self.original_max_seq_len
            )));
        }
        if head_dim <= 2 {
            return Err(AarambhError::Config(
                "head_dim must be greater than 2 for RoPE scaling".into(),
            ));
        }
        if self.attn_factor <= 0.0 || !self.attn_factor.is_finite() {
            return Err(AarambhError::Config(
                "rope_scaling.attn_factor must be finite and positive".into(),
            ));
        }
        if matches!(self.method, RopeScalingMethod::Yarn) {
            if self.beta_fast <= 0.0
                || self.beta_slow <= 0.0
                || !self.beta_fast.is_finite()
                || !self.beta_slow.is_finite()
            {
                return Err(AarambhError::Config(
                    "rope_scaling beta values must be finite and positive".into(),
                ));
            }
            if self.beta_fast <= self.beta_slow {
                return Err(AarambhError::Config(
                    "rope_scaling.beta_fast must be greater than beta_slow".into(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
/// Mixture-of-Experts feed-forward configuration.
pub struct MoeConfig {
    /// Number of coarse expert groups before fine-grained subdivision.
    pub num_experts: usize,
    /// Number of routed fine-grained experts selected per token.
    pub top_k: usize,
    /// Intermediate width of one coarse expert before subdivision.
    pub expert_ffn_dim: usize,
    /// Weight applied to the load-balancing auxiliary loss.
    pub aux_loss_weight: f64,
    /// Use MoE every Nth layer, selecting zero-based layers `N - 1, 2N - 1, ...`.
    pub every_n_layers: usize,
    /// Number of fine experts created from each coarse expert group.
    pub fine_grained_factor: usize,
    /// Number of always-active fine-width shared experts.
    pub num_shared_experts: usize,
    /// v4 Phase 43 dispatch strategy selecting how routed experts are
    /// evaluated. Defaults to [`DispatchKind::DenseMasked`] for exact
    /// backward compatibility with every existing MoE checkpoint.
    pub dispatch: DispatchKind,
}

impl Default for MoeConfig {
    fn default() -> Self {
        Self {
            num_experts: 8,
            top_k: 2,
            expert_ffn_dim: 0,
            aux_loss_weight: 0.01,
            every_n_layers: 2,
            fine_grained_factor: 1,
            num_shared_experts: 0,
            dispatch: DispatchKind::DenseMasked,
        }
    }
}

impl MoeConfig {
    /// Return the number of independently routed fine-grained experts.
    pub fn routed_expert_count(&self) -> Result<usize> {
        self.num_experts
            .checked_mul(self.fine_grained_factor)
            .ok_or_else(|| {
                AarambhError::Config("moe.num_experts * fine_grained_factor overflows usize".into())
            })
    }

    /// Return the intermediate width of one routed or shared fine expert.
    pub fn fine_grained_expert_dim(&self) -> Result<usize> {
        if self.fine_grained_factor == 0 {
            return Err(AarambhError::Config(
                "moe.fine_grained_factor must be non-zero".into(),
            ));
        }
        if self.expert_ffn_dim == 0 || !self.expert_ffn_dim.is_multiple_of(self.fine_grained_factor)
        {
            return Err(AarambhError::Config(format!(
                "moe.expert_ffn_dim {} must be non-zero and divisible by fine_grained_factor {}",
                self.expert_ffn_dim, self.fine_grained_factor
            )));
        }
        Ok(self.expert_ffn_dim / self.fine_grained_factor)
    }

    /// Return the summed routed intermediate width across the full expert pool.
    pub fn routed_capacity_width(&self) -> Result<usize> {
        self.routed_expert_count()?
            .checked_mul(self.fine_grained_expert_dim()?)
            .ok_or_else(|| AarambhError::Config("MoE routed capacity overflows usize".into()))
    }

    /// Return the routed intermediate width activated conceptually per token.
    pub fn active_routed_width(&self) -> Result<usize> {
        self.top_k
            .checked_mul(self.fine_grained_expert_dim()?)
            .ok_or_else(|| AarambhError::Config("MoE active width overflows usize".into()))
    }

    /// Return true when the zero-based layer index should use an MoE FFN.
    pub fn applies_to_layer(&self, layer_idx: usize) -> bool {
        self.every_n_layers > 0 && (layer_idx + 1).is_multiple_of(self.every_n_layers)
    }

    /// Validate MoE routing and expert dimensions for a model with `n_layers`.
    pub fn validate(&self, n_layers: usize) -> Result<()> {
        if self.num_experts < 2 {
            return Err(AarambhError::Config(
                "moe.num_experts must be at least 2".into(),
            ));
        }
        let routed_experts = self.routed_expert_count()?;
        let _fine_dim = self.fine_grained_expert_dim()?;
        if self.top_k == 0 || self.top_k > routed_experts {
            return Err(AarambhError::Config(format!(
                "moe.top_k must be in 1..={routed_experts} for num_experts={} and fine_grained_factor={}",
                self.num_experts, self.fine_grained_factor
            )));
        }
        if self.aux_loss_weight < 0.0 || !self.aux_loss_weight.is_finite() {
            return Err(AarambhError::Config(
                "moe.aux_loss_weight must be finite and non-negative".into(),
            ));
        }
        if self.every_n_layers == 0 {
            return Err(AarambhError::Config(
                "moe.every_n_layers must be non-zero".into(),
            ));
        }
        if n_layers == 0 || !(0..n_layers).any(|idx| self.applies_to_layer(idx)) {
            return Err(AarambhError::Config(
                "moe.every_n_layers does not select any model layer".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
/// Token-mixing implementation selected for one decoder layer.
pub enum AttentionKind {
    /// Existing grouped-query causal attention with RoPE.
    #[default]
    Full,
    /// Block-sparse grouped-query attention selected by a learned DSA indexer.
    Sparse,
    /// Fixed-state Gated DeltaNet linear attention.
    GatedDeltaNet,
    /// Multi-Head Latent Attention with compressed-latent KV cache (v4 Phase 41).
    LatentMLA,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
/// Routed-expert dispatch strategy used by a Mixture-of-Experts layer.
///
/// v2 Phase 22 and v3 Phase 31 shipped only [`DispatchKind::DenseMasked`]:
/// every routed expert computes on every token, then the result is masked
/// and weighted by the router. v4 Phase 43 introduces
/// [`DispatchKind::Sparse`], where each token's forward pass only computes
/// its assigned top-k experts — resolving the "documented future
/// optimisation" carried forward unresolved since v2 §35 and v3's
/// out-of-scope list.
pub enum DispatchKind {
    /// Dense masked dispatch: every routed expert runs on every token and
    /// the router weights mask the non-selected contributions. This is the
    /// v2/v3 behaviour, kept as the CPU fallback and correctness reference.
    #[default]
    DenseMasked,
    /// Sparse grouped dispatch: tokens are grouped by router assignment
    /// into per-expert contiguous batches and each expert's feed-forward
    /// matmul executes only on its assigned token group. The real
    /// throughput win lives on CUDA (grouped GEMM via cuBLAS); the CPU
    /// path keeps [`DispatchKind::DenseMasked`] regardless of
    /// configuration, documented plainly as "GPU only pays off."
    Sparse,
}

impl DispatchKind {
    /// Return the snake_case identifier used in TOML/JSON configs.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DenseMasked => "dense_masked",
            Self::Sparse => "sparse",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
/// DeepSeek-style learned block-sparse attention settings.
pub struct DsaConfig {
    /// Number of contiguous tokens represented by one sparse-attention block.
    pub block_size: usize,
    /// Maximum number of causal blocks read for each query block.
    pub top_k_blocks: usize,
    /// Sequence length below which exact dense attention is used.
    pub min_seq_len_for_sparsity: usize,
}

impl Default for DsaConfig {
    fn default() -> Self {
        Self {
            block_size: 64,
            top_k_blocks: 16,
            min_seq_len_for_sparsity: 2048,
        }
    }
}

impl DsaConfig {
    /// Validate block geometry and the dense-fallback threshold.
    pub fn validate(&self) -> Result<()> {
        if self.block_size < 16 || self.block_size > 256 || !self.block_size.is_power_of_two() {
            return Err(AarambhError::Config(
                "dsa.block_size must be a power of two in 16..=256".into(),
            ));
        }
        if self.top_k_blocks == 0 {
            return Err(AarambhError::Config(
                "dsa.top_k_blocks must be non-zero".into(),
            ));
        }
        if self.min_seq_len_for_sparsity < self.block_size {
            return Err(AarambhError::Config(
                "dsa.min_seq_len_for_sparsity must be at least dsa.block_size".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
/// Multi-token prediction auxiliary-head settings.
pub struct MtpConfig {
    /// Total prediction horizon including the main next-token head.
    ///
    /// A value of two means the main head predicts `t+1` and one auxiliary
    /// head predicts `t+2`. A value of three adds a second auxiliary head for
    /// `t+3`.
    pub num_future_tokens: usize,
    /// Weight applied to the mean auxiliary-head loss.
    pub aux_loss_weight: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
/// Integer precision simulated during quantization-aware training.
pub enum QuantBits {
    /// Four-bit weight quantization.
    Int4,
    /// Eight-bit weight quantization.
    Int8,
}

impl QuantBits {
    /// Return the number of bits represented by this variant.
    pub const fn bits(self) -> u8 {
        match self {
            Self::Int4 => 4,
            Self::Int8 => 8,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
/// Scale granularity used by fake weight quantization.
pub enum QuantGranularity {
    /// Match the existing GGUF exporter: Q4_K_M blocks or global Q8 absmax.
    #[default]
    ExportAligned,
    /// Use one scale for the complete weight tensor.
    PerTensor,
    /// Use one independent scale for each output row of a linear weight.
    PerOutputChannel,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
/// Class of linear projection eligible for quantization-aware training.
pub enum QatTarget {
    /// Query, key, value, and output attention projections.
    Attention,
    /// Dense, routed-expert, and shared-expert feed-forward projections.
    Ffn,
    /// Mixture-of-Experts router projections.
    MoeRouter,
    /// Gated DeltaNet projection matrices.
    DeltaNet,
    /// DeepSeek Sparse Attention indexer projections.
    DsaIndexer,
    /// Multi-token prediction refinement projections.
    Mtp,
    /// Multi-Head Latent Attention down/up and rope projections (v4 Phase 41).
    Mla,
    /// The language-model output projection.
    LmHead,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
/// Weight-only quantization-aware training settings.
pub struct QatConfig {
    /// Target integer precision.
    pub bits: QuantBits,
    /// Scale granularity used by fake quantization.
    pub granularity: QuantGranularity,
    /// Projection classes wrapped by fake quantization.
    pub targets: std::collections::BTreeSet<QatTarget>,
}

impl Default for QatConfig {
    fn default() -> Self {
        Self {
            bits: QuantBits::Int4,
            granularity: QuantGranularity::ExportAligned,
            targets: [
                QatTarget::Attention,
                QatTarget::Ffn,
                QatTarget::MoeRouter,
                QatTarget::DeltaNet,
                QatTarget::DsaIndexer,
                QatTarget::Mtp,
                QatTarget::Mla,
            ]
            .into_iter()
            .collect(),
        }
    }
}

impl QatConfig {
    /// Validate that at least one trainable projection class is selected.
    pub fn validate(&self) -> Result<()> {
        if self.targets.is_empty() {
            return Err(AarambhError::Config(
                "model.qat.targets must contain at least one projection class".into(),
            ));
        }
        Ok(())
    }

    /// Return whether a projection class should use fake quantization.
    pub fn applies_to(&self, target: QatTarget) -> bool {
        self.targets.contains(&target)
    }

    /// Return the effective precision for a target under the GGUF-aligned policy.
    pub fn effective_bits(&self, target: QatTarget) -> QuantBits {
        if self.granularity == QuantGranularity::ExportAligned && target == QatTarget::DsaIndexer {
            QuantBits::Int8
        } else {
            self.bits
        }
    }
}

impl Default for MtpConfig {
    fn default() -> Self {
        Self {
            num_future_tokens: 2,
            aux_loss_weight: 0.3,
        }
    }
}

impl MtpConfig {
    /// Return the number of auxiliary heads implied by the total horizon.
    pub fn auxiliary_head_count(&self) -> usize {
        self.num_future_tokens.saturating_sub(1)
    }

    /// Validate the prediction horizon and auxiliary-loss scale.
    pub fn validate(&self, max_seq_len: usize) -> Result<()> {
        if self.num_future_tokens < 2 {
            return Err(AarambhError::Config(
                "mtp.num_future_tokens must be at least 2 (main t+1 plus one auxiliary head)"
                    .into(),
            ));
        }
        if self.num_future_tokens > max_seq_len {
            return Err(AarambhError::Config(format!(
                "mtp.num_future_tokens {} exceeds max_seq_len {max_seq_len}",
                self.num_future_tokens
            )));
        }
        if !(0.0..=1.0).contains(&self.aux_loss_weight)
            || self.aux_loss_weight == 0.0
            || !self.aux_loss_weight.is_finite()
        {
            return Err(AarambhError::Config(
                "mtp.aux_loss_weight must be finite and in (0, 1]".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
/// Shape and execution settings for Gated DeltaNet layers.
pub struct GatedDeltaNetConfig {
    /// Number of recurrent state heads, or zero to derive half the GQA head count.
    pub n_heads: usize,
    /// Key width per recurrent head, or zero to derive a parameter-balanced width.
    pub key_head_dim: usize,
    /// Value width per recurrent head, or zero to use twice the resolved key width.
    pub value_head_dim: usize,
    /// Width of the causal depthwise short convolution.
    pub conv_kernel_size: usize,
    /// Sequence chunk size used by the training and prefill implementation.
    pub chunk_size: usize,
}

impl Default for GatedDeltaNetConfig {
    fn default() -> Self {
        Self {
            n_heads: 0,
            key_head_dim: 0,
            value_head_dim: 0,
            conv_kernel_size: 4,
            chunk_size: 64,
        }
    }
}

impl GatedDeltaNetConfig {
    /// Resolve automatic dimensions against a transformer model.
    pub fn resolve(&self, hidden_dim: usize, transformer_heads: usize) -> Result<Self> {
        let n_heads = if self.n_heads == 0 {
            (transformer_heads / 2).max(1)
        } else {
            self.n_heads
        };
        let key_head_dim = if self.key_head_dim == 0 {
            hidden_dim
                .checked_mul(3)
                .and_then(|value| value.checked_div(4 * n_heads))
                .unwrap_or(0)
        } else {
            self.key_head_dim
        };
        let value_head_dim = if self.value_head_dim == 0 {
            key_head_dim.saturating_mul(2)
        } else {
            self.value_head_dim
        };
        let resolved = Self {
            n_heads,
            key_head_dim,
            value_head_dim,
            conv_kernel_size: self.conv_kernel_size,
            chunk_size: self.chunk_size,
        };
        resolved.validate()?;
        Ok(resolved)
    }

    /// Validate resolved Gated DeltaNet dimensions and execution settings.
    pub fn validate(&self) -> Result<()> {
        if self.n_heads == 0 || self.key_head_dim == 0 || self.value_head_dim == 0 {
            return Err(AarambhError::Config(
                "gated_deltanet head counts and dimensions must resolve to non-zero values".into(),
            ));
        }
        if self.conv_kernel_size < 2 {
            return Err(AarambhError::Config(
                "gated_deltanet.conv_kernel_size must be at least 2".into(),
            ));
        }
        if self.chunk_size < 16 || self.chunk_size > 256 || !self.chunk_size.is_power_of_two() {
            return Err(AarambhError::Config(
                "gated_deltanet.chunk_size must be a power of two in 16..=256".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
/// Multi-Head Latent Attention settings (v4 Phase 41).
///
/// MLA compresses the per-token KV cache into a single low-rank latent vector
/// (`latent_dim`), reconstructing per-head keys and values at attention time
/// through small up-projection weights that are trained but never cached. A
/// small dedicated rotary slice (`rope_head_dim`) is cached alongside the
/// latent so position can be re-introduced without rotating the compressed
/// latent. See `ARCHITECTURE_V4.md` §55 for the full mechanism.
pub struct MlaConfig {
    /// Width of the compressed latent cached per token (the down-projection output).
    pub latent_dim: usize,
    /// Per-head width of the non-rotary ("nope") query/key slice.
    ///
    /// Zero derives `host_head_dim - rope_head_dim` against the host transformer.
    pub nope_head_dim: usize,
    /// Per-head width of the rotary-encoded query/key slice. Must be even.
    pub rope_head_dim: usize,
    /// Number of MLA query heads. Zero derives the host transformer head count.
    pub n_heads: usize,
    /// Per-head width of the reconstructed value. Zero derives `nope_head_dim`.
    pub value_head_dim: usize,
}

impl Default for MlaConfig {
    fn default() -> Self {
        Self {
            latent_dim: 512,
            nope_head_dim: 0,
            rope_head_dim: 16,
            n_heads: 0,
            value_head_dim: 0,
        }
    }
}

impl MlaConfig {
    /// Resolve automatic dimensions against a transformer model.
    pub fn resolve(&self, hidden_dim: usize, transformer_heads: usize) -> Result<Self> {
        let n_heads = if self.n_heads == 0 {
            transformer_heads
        } else {
            self.n_heads
        };
        let rope_head_dim = self.rope_head_dim;
        let nope_head_dim = if self.nope_head_dim == 0 {
            let host_head_dim = hidden_dim / transformer_heads;
            host_head_dim.saturating_sub(rope_head_dim).max(8)
        } else {
            self.nope_head_dim
        };
        let value_head_dim = if self.value_head_dim == 0 {
            nope_head_dim
        } else {
            self.value_head_dim
        };
        let resolved = Self {
            latent_dim: self.latent_dim,
            nope_head_dim,
            rope_head_dim,
            n_heads,
            value_head_dim,
        };
        resolved.validate()?;
        Ok(resolved)
    }

    /// Validate resolved MLA dimensions.
    pub fn validate(&self) -> Result<()> {
        if self.latent_dim == 0 {
            return Err(AarambhError::Config(
                "mla.latent_dim must be non-zero".into(),
            ));
        }
        if self.n_heads == 0 {
            return Err(AarambhError::Config(
                "mla.n_heads must resolve to a non-zero value".into(),
            ));
        }
        if self.nope_head_dim == 0 {
            return Err(AarambhError::Config(
                "mla.nope_head_dim must resolve to a non-zero value".into(),
            ));
        }
        if self.rope_head_dim == 0 || !self.rope_head_dim.is_multiple_of(2) {
            return Err(AarambhError::Config(
                "mla.rope_head_dim must be a positive even number".into(),
            ));
        }
        if self.value_head_dim == 0 {
            return Err(AarambhError::Config(
                "mla.value_head_dim must resolve to a non-zero value".into(),
            ));
        }
        Ok(())
    }

    /// Return the per-token cache width (latent + rotary slice) in elements.
    pub fn cache_width(&self) -> usize {
        self.latent_dim + self.rope_head_dim
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
/// Per-layer schedule for hybrid full, Gated DeltaNet, and LatentMLA attention.
pub struct HybridAttentionSchedule {
    /// Keep every Nth zero-based layer as full attention; other layers use Gated DeltaNet.
    pub full_attention_every_n: usize,
    /// Gated DeltaNet shape and execution settings.
    pub gated_deltanet: GatedDeltaNetConfig,
    /// Zero-based layer indices upgraded to Multi-Head Latent Attention (v4 Phase 41).
    ///
    /// Empty by default, which reproduces v3.0.0 exactly. A layer listed here
    /// takes precedence over both the `full_attention_every_n` rule and the
    /// DSA full-attention override.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mla_layers: Vec<usize>,
    /// Shared Multi-Head Latent Attention settings used by every `mla_layers` entry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mla: Option<MlaConfig>,
}

impl Default for HybridAttentionSchedule {
    fn default() -> Self {
        Self {
            full_attention_every_n: 4,
            gated_deltanet: GatedDeltaNetConfig::default(),
            mla_layers: Vec::new(),
            mla: None,
        }
    }
}

impl HybridAttentionSchedule {
    /// Return whether `layer_idx` is selected for Multi-Head Latent Attention.
    pub fn is_mla_layer(&self, layer_idx: usize) -> bool {
        self.mla_layers.contains(&layer_idx)
    }

    /// Return the token mixer selected for `layer_idx`.
    ///
    /// `LatentMLA` layers (from `mla_layers`) take precedence over the
    /// `full_attention_every_n` rule, so the DSA override applied by
    /// [`ModelConfig::attention_kind_for_layer`](crate::ModelConfig::attention_kind_for_layer)
    /// never replaces an MLA slot.
    pub fn kind_for_layer(&self, layer_idx: usize) -> AttentionKind {
        if self.is_mla_layer(layer_idx) {
            return AttentionKind::LatentMLA;
        }
        if self.full_attention_every_n > 0 && layer_idx.is_multiple_of(self.full_attention_every_n)
        {
            AttentionKind::Full
        } else {
            AttentionKind::GatedDeltaNet
        }
    }

    /// Validate this schedule and resolve its Gated DeltaNet dimensions.
    pub fn validate(
        &self,
        n_layers: usize,
        hidden_dim: usize,
        transformer_heads: usize,
    ) -> Result<GatedDeltaNetConfig> {
        if self.full_attention_every_n == 0 {
            return Err(AarambhError::Config(
                "attention_schedule.full_attention_every_n must be non-zero".into(),
            ));
        }
        let has_gated_delta =
            (0..n_layers).any(|idx| self.kind_for_layer(idx) == AttentionKind::GatedDeltaNet);
        let has_mla = (0..n_layers).any(|idx| self.kind_for_layer(idx) == AttentionKind::LatentMLA);
        if n_layers < 2 || (!has_gated_delta && !has_mla) {
            return Err(AarambhError::Config(
                "attention_schedule must select at least one Gated DeltaNet or LatentMLA layer"
                    .into(),
            ));
        }
        if has_mla && self.mla.is_none() {
            return Err(AarambhError::Config(
                "attention_schedule.mla must be set when mla_layers is non-empty".into(),
            ));
        }
        self.gated_deltanet.resolve(hidden_dim, transformer_heads)
    }

    /// Resolve the shared MLA configuration when the schedule selects MLA layers.
    ///
    /// Returns `Ok(None)` when no layer uses LatentMLA, so a v3 schedule with
    /// an empty `mla_layers` reproduces v3.0.0 exactly.
    pub fn resolved_mla(
        &self,
        n_layers: usize,
        hidden_dim: usize,
        transformer_heads: usize,
    ) -> Result<Option<MlaConfig>> {
        let has_mla = (0..n_layers).any(|idx| self.kind_for_layer(idx) == AttentionKind::LatentMLA);
        if !has_mla {
            return Ok(None);
        }
        let mla = self.mla.as_ref().ok_or_else(|| {
            AarambhError::Config(
                "attention_schedule.mla is required when mla_layers is non-empty".into(),
            )
        })?;
        mla.resolve(hidden_dim, transformer_heads).map(Some)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Decoder-only transformer model shape and numerical defaults.
pub struct ModelConfig {
    /// Number of tokenizer entries supported by the model.
    pub vocab_size: usize,
    /// Width of token embeddings and hidden states.
    pub hidden_dim: usize,
    /// Intermediate width of the feed-forward network.
    pub ffn_dim: usize,
    /// Number of transformer decoder blocks.
    pub n_layers: usize,
    /// Number of query attention heads.
    pub n_heads: usize,
    /// Number of key/value heads used by grouped-query attention.
    pub n_kv_heads: usize,
    /// Maximum context length in tokens.
    pub max_seq_len: usize,
    /// Rotary-position embedding base frequency.
    pub rope_theta: f64,
    /// Optional long-context RoPE scaling configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_scaling: Option<RopeScalingConfig>,
    /// Optional Mixture-of-Experts FFN configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moe: Option<MoeConfig>,
    /// Optional per-layer hybrid full/Gated DeltaNet attention schedule.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention_schedule: Option<HybridAttentionSchedule>,
    /// Optional learned block-sparse attention configuration. Full-attention
    /// slots in `attention_schedule` become DSA slots when this is present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dsa_config: Option<DsaConfig>,
    /// Optional multi-token prediction auxiliary heads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mtp: Option<MtpConfig>,
    /// Optional weight-only quantization-aware training configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qat: Option<QatConfig>,
    /// RMSNorm epsilon.
    pub norm_eps: f64,
    /// Whether the output head shares weights with token embeddings.
    pub tie_embeddings: bool,
    /// Declared chat-template shape version (Phase 52, `ARCHITECTURE_V4.md` §66).
    ///
    /// `None` means the checkpoint predates Phase 52 and did not declare a
    /// version; a v4.0 checkpoint records `Some(4)`. A server refuses to load a
    /// checkpoint whose declared version it does not recognize — the
    /// `aarambh_studio_tokenizer::validate_chat_template_version` function
    /// enforces this at server startup.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template_version: Option<u32>,
}

impl ModelConfig {
    /// Return the tiny smoke-test model preset.
    pub fn tiny() -> Self {
        Self {
            vocab_size: 32000,
            hidden_dim: 384,
            ffn_dim: 1024,
            n_layers: 8,
            n_heads: 6,
            n_kv_heads: 2,
            max_seq_len: 512,
            rope_theta: 10000.0,
            rope_scaling: None,
            moe: None,
            attention_schedule: None,
            dsa_config: None,
            mtp: None,
            qat: None,
            norm_eps: 1e-5,
            tie_embeddings: true,
            chat_template_version: None,
        }
    }

    /// Return the small v1 training preset.
    pub fn small() -> Self {
        Self {
            vocab_size: 32000,
            hidden_dim: 768,
            ffn_dim: 2688,
            n_layers: 12,
            n_heads: 12,
            n_kv_heads: 4,
            max_seq_len: 1024,
            rope_theta: 10000.0,
            rope_scaling: None,
            moe: None,
            attention_schedule: None,
            dsa_config: None,
            mtp: None,
            qat: None,
            norm_eps: 1e-5,
            tie_embeddings: true,
            chat_template_version: None,
        }
    }

    /// Return the medium scale-up preset.
    pub fn medium() -> Self {
        Self {
            vocab_size: 32000,
            hidden_dim: 1024,
            ffn_dim: 3392,
            n_layers: 24,
            n_heads: 16,
            n_kv_heads: 8,
            max_seq_len: 2048,
            rope_theta: 500000.0,
            rope_scaling: None,
            moe: None,
            attention_schedule: None,
            dsa_config: None,
            mtp: None,
            qat: None,
            norm_eps: 1e-5,
            tie_embeddings: true,
            chat_template_version: None,
        }
    }

    /// Return the large v1 target preset.
    pub fn large() -> Self {
        Self {
            vocab_size: 32000,
            hidden_dim: 2048,
            ffn_dim: 6656,
            n_layers: 24,
            n_heads: 32,
            n_kv_heads: 8,
            max_seq_len: 4096,
            rope_theta: 500000.0,
            rope_scaling: None,
            moe: None,
            attention_schedule: None,
            dsa_config: None,
            mtp: None,
            qat: None,
            norm_eps: 1e-5,
            tie_embeddings: true,
            chat_template_version: None,
        }
    }

    /// Return the per-head hidden width.
    pub fn head_dim(&self) -> usize {
        self.hidden_dim / self.n_heads
    }

    /// Return the token mixer selected for a layer, including DSA replacement
    /// of the schedule's full-attention slots.
    pub fn attention_kind_for_layer(&self, layer_idx: usize) -> AttentionKind {
        let scheduled = self
            .attention_schedule
            .as_ref()
            .map(|schedule| schedule.kind_for_layer(layer_idx))
            .unwrap_or(AttentionKind::Full);
        if self.dsa_config.is_some() && scheduled == AttentionKind::Full {
            AttentionKind::Sparse
        } else {
            scheduled
        }
    }

    /// Load model configuration from a JSON file.
    pub fn from_json(path: impl AsRef<Path>) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let config = serde_json::from_reader(reader)?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_model_config_json_defaults_rope_scaling_to_none() {
        let json = r#"{
            "vocab_size": 32000,
            "hidden_dim": 384,
            "ffn_dim": 1024,
            "n_layers": 8,
            "n_heads": 6,
            "n_kv_heads": 2,
            "max_seq_len": 512,
            "rope_theta": 10000.0,
            "norm_eps": 0.00001,
            "tie_embeddings": true
        }"#;
        let cfg: ModelConfig = serde_json::from_str(json).unwrap();
        assert!(cfg.rope_scaling.is_none());
        assert!(cfg.moe.is_none());
        assert!(cfg.dsa_config.is_none());
        assert!(cfg.mtp.is_none());
        assert!(cfg.qat.is_none());
    }

    #[test]
    fn qat_defaults_match_export_and_exclude_lm_head() {
        let qat = QatConfig::default();
        qat.validate().unwrap();
        assert_eq!(qat.bits, QuantBits::Int4);
        assert_eq!(qat.granularity, QuantGranularity::ExportAligned);
        assert!(qat.applies_to(QatTarget::Attention));
        assert!(qat.applies_to(QatTarget::DsaIndexer));
        assert!(!qat.applies_to(QatTarget::LmHead));
        assert_eq!(qat.effective_bits(QatTarget::DsaIndexer), QuantBits::Int8);
    }

    #[test]
    fn mtp_two_means_main_plus_one_auxiliary_head() {
        let cfg = MtpConfig::default();
        assert_eq!(cfg.num_future_tokens, 2);
        assert_eq!(cfg.auxiliary_head_count(), 1);
        cfg.validate(16).unwrap();
    }

    #[test]
    fn mtp_config_rejects_invalid_horizon_and_weight() {
        let too_short = MtpConfig {
            num_future_tokens: 1,
            ..MtpConfig::default()
        };
        assert!(too_short.validate(16).is_err());
        let too_long = MtpConfig {
            num_future_tokens: 17,
            ..MtpConfig::default()
        };
        assert!(too_long.validate(16).is_err());
        let bad_weight = MtpConfig {
            aux_loss_weight: 0.0,
            ..MtpConfig::default()
        };
        assert!(bad_weight.validate(16).is_err());
    }

    #[test]
    fn rope_scaling_validation_rejects_invalid_factor() {
        let cfg = RopeScalingConfig {
            factor: 1.0,
            original_max_seq_len: 512,
            ..RopeScalingConfig::default()
        };
        let err = cfg.validate(1024, 64).unwrap_err().to_string();
        assert!(err.contains("factor"), "{err}");
    }

    #[test]
    fn moe_config_validates_layer_selection() {
        let cfg = MoeConfig {
            expert_ffn_dim: 128,
            every_n_layers: 2,
            ..MoeConfig::default()
        };
        cfg.validate(2).unwrap();
        assert!(cfg.applies_to_layer(1));
        assert!(!cfg.applies_to_layer(0));
    }

    #[test]
    fn moe_config_rejects_top_k_larger_than_experts() {
        let cfg = MoeConfig {
            num_experts: 2,
            top_k: 3,
            expert_ffn_dim: 128,
            ..MoeConfig::default()
        };
        let err = cfg.validate(2).unwrap_err().to_string();
        assert!(err.contains("top_k"), "{err}");
    }

    #[test]
    fn old_moe_json_defaults_to_phase22_behavior() {
        let json = r#"{
            "num_experts": 8,
            "top_k": 2,
            "expert_ffn_dim": 512,
            "aux_loss_weight": 0.01,
            "every_n_layers": 2
        }"#;
        let cfg: MoeConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.fine_grained_factor, 1);
        assert_eq!(cfg.num_shared_experts, 0);
        assert_eq!(cfg.dispatch, DispatchKind::DenseMasked);
        assert_eq!(cfg.routed_expert_count().unwrap(), 8);
        assert_eq!(cfg.fine_grained_expert_dim().unwrap(), 512);
        assert_eq!(cfg.active_routed_width().unwrap(), 1024);
    }

    #[test]
    fn moe_config_dispatch_defaults_to_dense_masked() {
        let cfg = MoeConfig::default();
        assert_eq!(cfg.dispatch, DispatchKind::DenseMasked);
        assert_eq!(DispatchKind::default(), DispatchKind::DenseMasked);
    }

    #[test]
    fn moe_config_dispatch_serializes_as_snake_case() {
        let dense = MoeConfig {
            dispatch: DispatchKind::DenseMasked,
            ..MoeConfig::default()
        };
        let json = serde_json::to_string(&dense).unwrap();
        assert!(
            json.contains("\"dispatch\":\"dense_masked\""),
            "dense_masked not serialized as snake_case: {json}"
        );
        let sparse = MoeConfig {
            dispatch: DispatchKind::Sparse,
            ..MoeConfig::default()
        };
        let json = serde_json::to_string(&sparse).unwrap();
        assert!(
            json.contains("\"dispatch\":\"sparse\""),
            "sparse not serialized as snake_case: {json}"
        );
        let round_trip: MoeConfig = serde_json::from_str(r#"{"dispatch":"sparse"}"#).unwrap();
        assert_eq!(round_trip.dispatch, DispatchKind::Sparse);
    }

    #[test]
    fn fine_grained_moe_conserves_routed_capacity_and_active_width() {
        let coarse = MoeConfig {
            num_experts: 8,
            top_k: 2,
            expert_ffn_dim: 512,
            ..MoeConfig::default()
        };
        let fine = MoeConfig {
            top_k: 8,
            fine_grained_factor: 4,
            num_shared_experts: 1,
            ..coarse.clone()
        };
        assert_eq!(coarse.routed_capacity_width().unwrap(), 4096);
        assert_eq!(fine.routed_capacity_width().unwrap(), 4096);
        assert_eq!(coarse.active_routed_width().unwrap(), 1024);
        assert_eq!(fine.active_routed_width().unwrap(), 1024);
        assert_eq!(fine.routed_expert_count().unwrap(), 32);
        assert_eq!(fine.fine_grained_expert_dim().unwrap(), 128);
    }

    #[test]
    fn fine_grained_moe_requires_divisible_width() {
        let cfg = MoeConfig {
            expert_ffn_dim: 510,
            fine_grained_factor: 4,
            ..MoeConfig::default()
        };
        let err = cfg.validate(2).unwrap_err().to_string();
        assert!(err.contains("divisible"), "{err}");
    }

    #[test]
    fn dsa_replaces_only_scheduled_full_attention_layers() {
        let mut cfg = ModelConfig::tiny();
        cfg.attention_schedule = Some(HybridAttentionSchedule::default());
        cfg.dsa_config = Some(DsaConfig::default());
        assert_eq!(cfg.attention_kind_for_layer(0), AttentionKind::Sparse);
        assert_eq!(
            cfg.attention_kind_for_layer(1),
            AttentionKind::GatedDeltaNet
        );
        assert_eq!(cfg.attention_kind_for_layer(4), AttentionKind::Sparse);
    }

    #[test]
    fn schedule_with_zero_mla_layers_matches_v3_exactly() {
        // A default v3 schedule (empty mla_layers, no mla config) reproduces
        // v3.0.0 kind_for_layer exactly: Full every Nth layer, GatedDeltaNet
        // elsewhere, and resolved_mla returns None.
        let schedule = HybridAttentionSchedule::default();
        for layer in 0..8 {
            let kind = schedule.kind_for_layer(layer);
            if layer.is_multiple_of(schedule.full_attention_every_n) {
                assert_eq!(kind, AttentionKind::Full, "layer {layer}");
            } else {
                assert_eq!(kind, AttentionKind::GatedDeltaNet, "layer {layer}");
            }
        }
        assert!(schedule.resolved_mla(8, 384, 6).unwrap().is_none());
    }

    #[test]
    fn mla_layers_take_precedence_over_every_n_and_dsa_override() {
        // MLA slots must win over both the full_attention_every_n rule and the
        // DSA full-attention override applied by ModelConfig.
        let schedule = HybridAttentionSchedule {
            mla_layers: vec![0, 4],
            mla: Some(MlaConfig {
                latent_dim: 64,
                rope_head_dim: 16,
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(schedule.kind_for_layer(0), AttentionKind::LatentMLA);
        assert_eq!(schedule.kind_for_layer(1), AttentionKind::GatedDeltaNet);
        assert_eq!(schedule.kind_for_layer(4), AttentionKind::LatentMLA);

        let mut cfg = ModelConfig::tiny();
        cfg.attention_schedule = Some(schedule);
        cfg.dsa_config = Some(DsaConfig::default());
        // MLA slot is not replaced by Sparse even though dsa_config is set.
        assert_eq!(cfg.attention_kind_for_layer(0), AttentionKind::LatentMLA);
        assert_eq!(cfg.attention_kind_for_layer(4), AttentionKind::LatentMLA);
        // resolved_mla returns a resolved config (n_heads derived from the host).
        let mla = cfg
            .attention_schedule
            .as_ref()
            .unwrap()
            .resolved_mla(cfg.n_layers, cfg.hidden_dim, cfg.n_heads)
            .unwrap()
            .unwrap();
        assert_eq!(mla.n_heads, cfg.n_heads);
        assert_eq!(mla.rope_head_dim, 16);
        assert!(mla.cache_width() < 2 * cfg.n_kv_heads * cfg.head_dim());
    }

    #[test]
    fn mla_config_resolves_derived_dimensions() {
        // host: hidden=128, heads=2 -> head_dim=64.
        let mla = MlaConfig {
            latent_dim: 64,
            rope_head_dim: 16,
            ..Default::default()
        }
        .resolve(128, 2)
        .unwrap();
        assert_eq!(mla.n_heads, 2);
        assert_eq!(mla.nope_head_dim, 48); // 64 - 16
        assert_eq!(mla.value_head_dim, 48); // derives nope_head_dim
        assert_eq!(mla.cache_width(), 80); // 64 + 16
    }

    #[test]
    fn mla_config_rejects_odd_rope_head_dim() {
        let err = MlaConfig {
            latent_dim: 64,
            rope_head_dim: 15,
            ..Default::default()
        }
        .resolve(128, 2)
        .unwrap_err();
        assert!(err.to_string().contains("rope_head_dim"), "{}", err);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
/// Training hyperparameters and checkpoint cadence.
pub struct TrainConfig {
    /// Peak learning rate.
    pub lr: f64,
    /// Number of sequences per micro-batch.
    pub batch_size: usize,
    /// Number of micro-batches accumulated before an optimizer step.
    pub grad_accum_steps: usize,
    /// Maximum number of full dataset passes.
    pub max_epochs: usize,
    /// Maximum optimizer steps.
    pub max_steps: usize,
    /// Number of warmup optimizer steps.
    pub warmup_steps: usize,
    /// Final learning-rate ratio relative to the peak rate.
    pub min_lr_ratio: f64,
    /// AdamW decoupled weight decay.
    pub weight_decay: f64,
    /// AdamW first-moment coefficient.
    pub beta1: f64,
    /// AdamW second-moment coefficient.
    pub beta2: f64,
    /// AdamW numerical epsilon.
    pub epsilon: f64,
    /// Maximum global gradient norm.
    pub clip_grad_norm: f64,
    /// Checkpoint save interval in optimizer steps.
    pub save_every_n_steps: usize,
    /// Training log interval in optimizer steps.
    pub log_every_n_steps: usize,
    /// Evaluation interval in optimizer steps.
    pub eval_steps: usize,
    /// Random seed used by loaders and sampling.
    pub seed: u64,
    /// Directory where checkpoints are written.
    pub checkpoint_dir: std::path::PathBuf,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            batch_size: 2,
            grad_accum_steps: 16,
            max_epochs: 1,
            max_steps: 5000,
            warmup_steps: 200,
            min_lr_ratio: 0.1,
            weight_decay: 0.1,
            beta1: 0.9,
            beta2: 0.95,
            epsilon: 1e-8,
            clip_grad_norm: 1.0,
            save_every_n_steps: 1000,
            log_every_n_steps: 10,
            eval_steps: 500,
            seed: 42,
            checkpoint_dir: std::path::PathBuf::from("checkpoints"),
        }
    }
}
