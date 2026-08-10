//! Frozen, pretrained audio spectrogram transformer encoder.
//!
//! Mirrors the CLIP-style vision encoder (`aarambh_studio_vision::ClipVisionEncoder`)
//! in structure: a frozen, pretrained transformer that converts raw input into a
//! grid of embeddings. The only difference is the input domain — a log-mel
//! spectrogram `[batch, n_mels, time_frames]` is treated as a single-channel
//! image, split into `(patch_mel, patch_time)` patches, linearly projected to
//! `audio_d_model`, and run through standard pre-norm transformer blocks.
//!
//! Weights are loaded as SafeTensors through `candle_core`, exactly the same
//! loading path every other encoder in the project uses (v2 §24's CLIP policy):
//! no PyTorch bindings, no ONNX, no Python FFI.

use std::collections::HashMap;
use std::path::Path;

use aarambh_studio_core::{AarambhError, Result};
use candle_core::{D, DType, Device, Tensor};
use candle_nn::{
    LayerNorm, Linear, Module, VarBuilder, layer_norm, linear, linear_no_bias, ops::softmax,
};
use serde::{Deserialize, Serialize};

/// Configuration for a frozen audio spectrogram transformer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct AudioEncoderConfig {
    /// Number of mel frequency bins in the input spectrogram.
    pub n_mels: usize,
    /// Maximum number of short-time frames in the input spectrogram.
    pub max_frames: usize,
    /// Patch height in mel bins.
    pub patch_mel: usize,
    /// Patch width in time frames.
    pub patch_time: usize,
    /// Audio transformer hidden width.
    pub audio_d_model: usize,
    /// Number of transformer encoder blocks.
    pub audio_layers: usize,
    /// Number of attention heads.
    pub audio_heads: usize,
    /// MLP hidden width inside each transformer block.
    pub mlp_dim: usize,
    /// LayerNorm epsilon.
    pub norm_eps: f64,
}

impl Default for AudioEncoderConfig {
    fn default() -> Self {
        Self {
            n_mels: 80,
            max_frames: 61,
            patch_mel: 16,
            patch_time: 16,
            audio_d_model: 768,
            audio_layers: 12,
            audio_heads: 12,
            mlp_dim: 3072,
            norm_eps: 1e-5,
        }
    }
}

impl AudioEncoderConfig {
    /// Load an audio encoder configuration from JSON.
    pub fn from_json(path: impl AsRef<Path>) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let config = serde_json::from_reader(file)?;
        Ok(config)
    }

    /// Validate encoder dimensions and the patch grid.
    pub fn validate(&self) -> Result<()> {
        if self.n_mels == 0
            || self.max_frames == 0
            || self.patch_mel == 0
            || self.patch_time == 0
            || self.audio_d_model == 0
            || self.audio_layers == 0
            || self.audio_heads == 0
            || self.mlp_dim == 0
        {
            return Err(AarambhError::Config(
                "audio encoder dimensions must be non-zero".into(),
            ));
        }
        if !self.n_mels.is_multiple_of(self.patch_mel) {
            return Err(AarambhError::Config(format!(
                "n_mels {} must be divisible by patch_mel {}",
                self.n_mels, self.patch_mel
            )));
        }
        if !self.max_frames.is_multiple_of(self.patch_time) {
            return Err(AarambhError::Config(format!(
                "max_frames {} must be divisible by patch_time {}",
                self.max_frames, self.patch_time
            )));
        }
        if !self.audio_d_model.is_multiple_of(self.audio_heads) {
            return Err(AarambhError::Config(
                "audio_d_model must be divisible by audio_heads".into(),
            ));
        }
        Ok(())
    }

    /// Return the number of patches along the mel axis.
    pub fn patch_rows(&self) -> usize {
        self.n_mels / self.patch_mel
    }

    /// Return the number of patches along the time axis.
    pub fn patch_cols(&self) -> usize {
        self.max_frames / self.patch_time
    }

    /// Return the total number of non-CLS patch tokens.
    pub fn num_patches(&self) -> usize {
        self.patch_rows() * self.patch_cols()
    }

    /// Return the flattened patch width.
    pub fn patch_dim(&self) -> usize {
        self.patch_mel * self.patch_time
    }

    /// Return per-head attention width.
    pub fn head_dim(&self) -> usize {
        self.audio_d_model / self.audio_heads
    }
}

/// Frozen audio spectrogram transformer encoder.
#[derive(Debug, Clone)]
pub struct FrozenAudioEncoder {
    config: AudioEncoderConfig,
    patch_embed: Linear,
    class_embedding: Tensor,
    position_embedding: Tensor,
    pre_norm: LayerNorm,
    blocks: Vec<AudioBlock>,
    post_norm: LayerNorm,
}

impl FrozenAudioEncoder {
    /// Build the encoder from a variable builder.
    pub fn new(config: AudioEncoderConfig, vb: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        let patch_embed = linear_no_bias(
            config.patch_dim(),
            config.audio_d_model,
            vb.pp("patch_embed"),
        )?;
        let class_embedding = vb.get_with_hints(
            config.audio_d_model,
            "class_embedding",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 0.02,
            },
        )?;
        let position_embedding = vb.get_with_hints(
            (config.num_patches() + 1, config.audio_d_model),
            "position_embedding",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 0.01,
            },
        )?;
        let pre_norm = layer_norm(config.audio_d_model, config.norm_eps, vb.pp("pre_norm"))?;
        let post_norm = layer_norm(config.audio_d_model, config.norm_eps, vb.pp("post_norm"))?;
        let mut blocks = Vec::with_capacity(config.audio_layers);
        for idx in 0..config.audio_layers {
            blocks.push(AudioBlock::new(&config, vb.pp("blocks").pp(idx))?);
        }
        Ok(Self {
            config,
            patch_embed,
            class_embedding,
            position_embedding,
            pre_norm,
            blocks,
            post_norm,
        })
    }

    /// Load pretrained encoder weights from a SafeTensors checkpoint.
    pub fn load_pretrained(
        path: impl AsRef<Path>,
        config: AudioEncoderConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        // SAFETY: The checkpoint mapping is read-only and is consumed while the
        // encoder creates its owned parameter tensors.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path.as_ref()], dtype, device)? };
        Self::new(config, vb)
    }

    /// Return the encoder configuration.
    pub fn config(&self) -> &AudioEncoderConfig {
        &self.config
    }

    /// Encode `[batch, n_mels, max_frames]` spectrograms into non-CLS patch tokens.
    pub fn forward(&self, spectrograms: &Tensor) -> Result<Tensor> {
        let patches = self.patchify(spectrograms)?;
        let batch = patches.dims()[0];
        let mut x = self.patch_embed.forward(&patches)?;
        let cls = self
            .class_embedding
            .reshape((1, 1, self.config.audio_d_model))?
            .broadcast_as((batch, 1, self.config.audio_d_model))?;
        x = Tensor::cat(&[&cls, &x], 1)?;
        let pos = self.position_embedding.reshape((
            1,
            self.config.num_patches() + 1,
            self.config.audio_d_model,
        ))?;
        x = x.broadcast_add(&pos)?;
        x = self.pre_norm.forward(&x)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        x = self.post_norm.forward(&x)?;
        Ok(x.narrow(1, 1, self.config.num_patches())?)
    }

    fn patchify(&self, spectrograms: &Tensor) -> Result<Tensor> {
        let dims = spectrograms.dims();
        if dims.len() != 3 {
            return Err(AarambhError::Shape(format!(
                "spectrograms must have shape [batch, n_mels, max_frames], got {dims:?}"
            )));
        }
        let (batch, n_mels, max_frames) = (dims[0], dims[1], dims[2]);
        if n_mels != self.config.n_mels || max_frames != self.config.max_frames {
            return Err(AarambhError::Shape(format!(
                "spectrograms must be [batch, {}, {}], got {dims:?}",
                self.config.n_mels, self.config.max_frames
            )));
        }
        let rows = self.config.patch_rows();
        let cols = self.config.patch_cols();
        let patches = spectrograms
            .unfold(1, self.config.patch_mel, self.config.patch_mel)?
            .unfold(2, self.config.patch_time, self.config.patch_time)?;
        let patches = patches.permute((0, 1, 2, 3, 4))?.contiguous()?;
        Ok(patches.reshape((batch, rows * cols, self.config.patch_dim()))?)
    }

    /// Return every named trainable tensor in the encoder.
    ///
    /// Used by Phase 42 tests to assert the frozen encoder receives no
    /// gradients: callers diff this set against the gradient store after a
    /// backward pass and confirm it is empty.
    pub fn named_tensors(&self) -> HashMap<String, Tensor> {
        let mut out = HashMap::new();
        out.insert(
            "patch_embed.weight".to_string(),
            self.patch_embed.weight().clone(),
        );
        out.insert("class_embedding".to_string(), self.class_embedding.clone());
        out.insert(
            "position_embedding".to_string(),
            self.position_embedding.clone(),
        );
        out.insert(
            "pre_norm.weight".to_string(),
            self.pre_norm.weight().clone(),
        );
        out.insert(
            "pre_norm.bias".to_string(),
            self.pre_norm.bias().expect("pre_norm bias").clone(),
        );
        out.insert(
            "post_norm.weight".to_string(),
            self.post_norm.weight().clone(),
        );
        out.insert(
            "post_norm.bias".to_string(),
            self.post_norm.bias().expect("post_norm bias").clone(),
        );
        for (idx, block) in self.blocks.iter().enumerate() {
            block.collect_named(idx, &mut out);
        }
        out
    }
}

#[derive(Debug, Clone)]
struct AudioBlock {
    norm1: LayerNorm,
    attn: AudioAttention,
    norm2: LayerNorm,
    mlp: AudioMlp,
}

impl AudioBlock {
    fn new(config: &AudioEncoderConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let norm1 = layer_norm(config.audio_d_model, config.norm_eps, vb.pp("norm1"))?;
        let attn = AudioAttention::new(config, vb.pp("attn"))?;
        let norm2 = layer_norm(config.audio_d_model, config.norm_eps, vb.pp("norm2"))?;
        let mlp = AudioMlp::new(config, vb.pp("mlp"))?;
        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x = self.norm1.forward(x)?;
        let x = self.attn.forward(&x)?;
        let x = (residual + x)?;
        let residual = x.clone();
        let x = self.norm2.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        Ok((residual + x)?)
    }

    fn collect_named(&self, idx: usize, out: &mut HashMap<String, Tensor>) {
        let prefix = format!("blocks.{idx}");
        out.insert(
            format!("{prefix}.norm1.weight"),
            self.norm1.weight().clone(),
        );
        out.insert(
            format!("{prefix}.norm1.bias"),
            self.norm1.bias().expect("norm1 bias").clone(),
        );
        out.insert(
            format!("{prefix}.attn.q_proj.weight"),
            self.attn.q_proj.weight().clone(),
        );
        out.insert(
            format!("{prefix}.attn.q_proj.bias"),
            self.attn.q_proj.bias().expect("q_proj bias").clone(),
        );
        out.insert(
            format!("{prefix}.attn.k_proj.weight"),
            self.attn.k_proj.weight().clone(),
        );
        out.insert(
            format!("{prefix}.attn.k_proj.bias"),
            self.attn.k_proj.bias().expect("k_proj bias").clone(),
        );
        out.insert(
            format!("{prefix}.attn.v_proj.weight"),
            self.attn.v_proj.weight().clone(),
        );
        out.insert(
            format!("{prefix}.attn.v_proj.bias"),
            self.attn.v_proj.bias().expect("v_proj bias").clone(),
        );
        out.insert(
            format!("{prefix}.attn.out_proj.weight"),
            self.attn.out_proj.weight().clone(),
        );
        out.insert(
            format!("{prefix}.attn.out_proj.bias"),
            self.attn.out_proj.bias().expect("out_proj bias").clone(),
        );
        out.insert(
            format!("{prefix}.norm2.weight"),
            self.norm2.weight().clone(),
        );
        out.insert(
            format!("{prefix}.norm2.bias"),
            self.norm2.bias().expect("norm2 bias").clone(),
        );
        out.insert(
            format!("{prefix}.mlp.fc1.weight"),
            self.mlp.fc1.weight().clone(),
        );
        out.insert(
            format!("{prefix}.mlp.fc1.bias"),
            self.mlp.fc1.bias().expect("fc1 bias").clone(),
        );
        out.insert(
            format!("{prefix}.mlp.fc2.weight"),
            self.mlp.fc2.weight().clone(),
        );
        out.insert(
            format!("{prefix}.mlp.fc2.bias"),
            self.mlp.fc2.bias().expect("fc2 bias").clone(),
        );
    }
}

#[derive(Debug, Clone)]
struct AudioAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    heads: usize,
    head_dim: usize,
    scale: f64,
}

impl AudioAttention {
    fn new(config: &AudioEncoderConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let q_proj = linear(config.audio_d_model, config.audio_d_model, vb.pp("q_proj"))?;
        let k_proj = linear(config.audio_d_model, config.audio_d_model, vb.pp("k_proj"))?;
        let v_proj = linear(config.audio_d_model, config.audio_d_model, vb.pp("v_proj"))?;
        let out_proj = linear(
            config.audio_d_model,
            config.audio_d_model,
            vb.pp("out_proj"),
        )?;
        let head_dim = config.head_dim();
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            heads: config.audio_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        let (batch, seq_len, width) = (dims[0], dims[1], dims[2]);
        let q = self
            .q_proj
            .forward(x)?
            .reshape((batch, seq_len, self.heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((batch, seq_len, self.heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((batch, seq_len, self.heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let attn = q.matmul(&k.transpose(2, 3)?)?.affine(self.scale, 0.0)?;
        let attn = softmax(&attn, D::Minus1)?;
        let out = attn.matmul(&v)?.transpose(1, 2)?;
        let out = out.reshape((batch, seq_len, width))?;
        Ok(self.out_proj.forward(&out)?)
    }
}

#[derive(Debug, Clone)]
struct AudioMlp {
    fc1: Linear,
    fc2: Linear,
}

impl AudioMlp {
    fn new(config: &AudioEncoderConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let fc1 = linear(config.audio_d_model, config.mlp_dim, vb.pp("fc1"))?;
        let fc2 = linear(config.mlp_dim, config.audio_d_model, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?.gelu()?;
        Ok(self.fc2.forward(&x)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::{VarBuilder, VarMap};

    fn tiny_config() -> AudioEncoderConfig {
        AudioEncoderConfig {
            n_mels: 16,
            max_frames: 16,
            patch_mel: 4,
            patch_time: 4,
            audio_d_model: 16,
            audio_layers: 2,
            audio_heads: 2,
            mlp_dim: 32,
            norm_eps: 1e-5,
        }
    }

    #[test]
    fn audio_encoder_outputs_patch_tokens() {
        let device = Device::Cpu;
        let config = tiny_config();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let encoder = FrozenAudioEncoder::new(config, vb).unwrap();
        let spectrogram = Tensor::zeros((1, 16, 16), DType::F32, &device).unwrap();
        let output = encoder.forward(&spectrogram).unwrap();
        // 16 mels / 4 = 4 rows, 16 frames / 4 = 4 cols -> 16 patch tokens
        assert_eq!(output.dims(), &[1, 16, 16]);
    }

    #[test]
    fn named_tensors_cover_every_block() {
        let config = tiny_config();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let encoder = FrozenAudioEncoder::new(config, vb).unwrap();
        let names = encoder.named_tensors();
        // 7 root tensors + 16 per block * 2 blocks = 39
        assert_eq!(names.len(), 7 + 16 * 2);
        assert!(names.contains_key("blocks.0.attn.q_proj.weight"));
        assert!(names.contains_key("blocks.1.mlp.fc2.bias"));
    }
}
