use std::collections::HashMap;

use aarambh_studio_core::{AarambhError, ModelConfig, Result};
use aarambh_studio_model::AarambhModel;
use aarambh_studio_nn::RopeCache;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Embedding, Module, VarBuilder, VarMap};

use crate::lora::{LoraConfig, LoraLinear, linear_forward};

#[derive(Debug, Clone)]
/// Aarambh model with LoRA adapters attached to selected linear layers.
pub struct LoraAarambhModel {
    config: ModelConfig,
    embedding_weight: Tensor,
    blocks: Vec<LoraBlock>,
    final_norm_weight: Tensor,
    lm_head: Option<LoraLinear>,
    rope_cache: RopeCache,
    adapter_param_count: usize,
    base_param_count: usize,
    frozen_auxiliary_tensors: HashMap<String, Tensor>,
    hybrid: Option<HybridLoraModel>,
}

impl LoraAarambhModel {
    /// Build a LoRA model from base checkpoint tensors.
    pub fn from_tensors(
        config: &ModelConfig,
        tensors: &HashMap<String, Tensor>,
        lora_config: &LoraConfig,
        quantized_base: bool,
        device: &Device,
    ) -> Result<(Self, VarMap)> {
        AarambhModel::validate_config(config)?;
        if config.moe.is_some() {
            return Err(AarambhError::Config(
                "LoRA for MoE models is not supported; train the MoE base model directly or use a dense config".into(),
            ));
        }
        lora_config.validate()?;
        let varmap = VarMap::new();
        let embedding_weight = required_tensor(tensors, "embedding.weight")?;
        if config.attention_schedule.is_some() {
            let hybrid = HybridLoraModel::new(
                config,
                tensors,
                lora_config,
                &varmap,
                quantized_base,
                device,
            )?;
            let adapter_param_count = hybrid.adapter_param_count();
            let base_param_count = tensors.values().map(tensor_elem_count).sum();
            return Ok((
                Self {
                    config: config.clone(),
                    embedding_weight,
                    blocks: Vec::new(),
                    final_norm_weight: required_tensor(tensors, "final_norm.weight")?,
                    lm_head: None,
                    rope_cache: RopeCache::from_config(config, DType::F32, device)?,
                    adapter_param_count,
                    base_param_count,
                    frozen_auxiliary_tensors: HashMap::new(),
                    hybrid: Some(hybrid),
                },
                varmap,
            ));
        }
        let mut blocks = Vec::with_capacity(config.n_layers);

        for layer_idx in 0..config.n_layers {
            blocks.push(LoraBlock::new(
                layer_idx,
                config,
                tensors,
                lora_config,
                &varmap,
                quantized_base,
                device,
            )?);
        }

        let final_norm_weight = required_tensor(tensors, "final_norm.weight")?;
        let lm_head = if config.tie_embeddings {
            None
        } else {
            Some(LoraLinear::new(
                "lm_head.weight",
                required_ref(tensors, "lm_head.weight")?,
                lora_config,
                &varmap,
                quantized_base,
                device,
            )?)
        };
        let dtype = embedding_weight.dtype();
        let rope_cache = RopeCache::from_config(config, dtype, device)?;
        let adapter_param_count = adapter_param_count(&blocks, lm_head.as_ref());
        let base_param_count = tensors.values().map(tensor_elem_count).sum();
        let frozen_auxiliary_tensors = tensors
            .iter()
            .filter(|(name, _)| name.starts_with("mtp."))
            .map(|(name, tensor)| (name.clone(), tensor.detach()))
            .collect();

        let model = Self {
            config: config.clone(),
            embedding_weight,
            blocks,
            final_norm_weight,
            lm_head,
            rope_cache,
            adapter_param_count,
            base_param_count,
            frozen_auxiliary_tensors,
            hybrid: None,
        };
        Ok((model, varmap))
    }

    /// Return the model configuration.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Return the number of adapter parameters.
    pub fn adapter_param_count(&self) -> usize {
        self.adapter_param_count
    }

    /// Return the number of base model parameters.
    pub fn base_param_count(&self) -> usize {
        self.base_param_count
    }

    /// Return adapter parameters divided by base parameters.
    pub fn trainable_ratio(&self) -> f64 {
        if self.base_param_count == 0 {
            0.0
        } else {
            self.adapter_param_count as f64 / self.base_param_count as f64
        }
    }

    /// Run the training forward path with adapters enabled.
    pub fn forward_train(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.forward(token_ids, true)
    }

    /// Run the evaluation forward path with adapters enabled.
    pub fn forward_eval(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.forward(token_ids, false)
    }

    /// Embed token ids with the frozen base embedding table.
    pub fn embed_tokens(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.check_token_ids(token_ids)?;
        let embedding = Embedding::new(self.embedding_weight.clone(), self.config.hidden_dim);
        Ok(embedding.forward(token_ids)?)
    }

    /// Run the training forward path from precomputed embeddings.
    pub fn forward_embeddings_train(&self, embeddings: &Tensor) -> Result<Tensor> {
        self.forward_embeddings(embeddings, true)
    }

    /// Run the evaluation forward path from precomputed embeddings.
    pub fn forward_embeddings_eval(&self, embeddings: &Tensor) -> Result<Tensor> {
        self.forward_embeddings(embeddings, false)
    }

    /// Return checkpoint tensors with adapters merged into base weights.
    pub fn merged_tensors(&self) -> Result<HashMap<String, Tensor>> {
        if let Some(hybrid) = &self.hybrid {
            return hybrid.merged_tensors();
        }
        let mut tensors = HashMap::new();
        tensors.insert(
            "embedding.weight".to_string(),
            self.embedding_weight.detach(),
        );

        for (idx, block) in self.blocks.iter().enumerate() {
            tensors.insert(
                format!("blocks.{idx}.norm1.weight"),
                block.norm1_weight.detach(),
            );
            tensors.insert(
                format!("blocks.{idx}.attn.wq.weight"),
                block.attn.wq.merged_weight()?,
            );
            tensors.insert(
                format!("blocks.{idx}.attn.wk.weight"),
                block.attn.wk.merged_weight()?,
            );
            tensors.insert(
                format!("blocks.{idx}.attn.wv.weight"),
                block.attn.wv.merged_weight()?,
            );
            tensors.insert(
                format!("blocks.{idx}.attn.wo.weight"),
                block.attn.wo.merged_weight()?,
            );
            tensors.insert(
                format!("blocks.{idx}.norm2.weight"),
                block.norm2_weight.detach(),
            );
            tensors.insert(
                format!("blocks.{idx}.ffn.w_gate.weight"),
                block.ffn.w_gate.merged_weight()?,
            );
            tensors.insert(
                format!("blocks.{idx}.ffn.w_up.weight"),
                block.ffn.w_up.merged_weight()?,
            );
            tensors.insert(
                format!("blocks.{idx}.ffn.w_down.weight"),
                block.ffn.w_down.merged_weight()?,
            );
        }

        tensors.insert(
            "final_norm.weight".to_string(),
            self.final_norm_weight.detach(),
        );
        if let Some(lm_head) = &self.lm_head {
            tensors.insert("lm_head.weight".to_string(), lm_head.merged_weight()?);
        }
        tensors.extend(self.frozen_auxiliary_tensors.clone());
        Ok(tensors)
    }

    fn forward(&self, token_ids: &Tensor, train: bool) -> Result<Tensor> {
        let x = self.embed_tokens(token_ids)?;
        self.forward_embeddings(&x, train)
    }

    fn forward_embeddings(&self, embeddings: &Tensor, train: bool) -> Result<Tensor> {
        self.check_embeddings(embeddings)?;
        if let Some(hybrid) = &self.hybrid {
            return hybrid.forward_embeddings(embeddings, train);
        }
        let mut x = embeddings.clone();

        for block in &self.blocks {
            x = block.forward(&x, &self.rope_cache, None, train)?;
        }

        let x = candle_nn::ops::rms_norm_slow(
            &x,
            &self.final_norm_weight,
            self.config.norm_eps as f32,
        )?;
        match &self.lm_head {
            Some(lm_head) => lm_head.forward(&x, train),
            None => linear_forward(&x, &self.embedding_weight),
        }
    }

    fn check_embeddings(&self, embeddings: &Tensor) -> Result<(usize, usize)> {
        let dims = embeddings.dims();
        if dims.len() != 3 {
            return Err(AarambhError::Shape(format!(
                "embeddings must have shape [batch, seq, hidden_dim], got {dims:?}"
            )));
        }
        let batch = dims[0];
        let seq_len = dims[1];
        let hidden = dims[2];
        if batch == 0 || seq_len == 0 {
            return Err(AarambhError::Shape(
                "batch and sequence length must be non-zero".into(),
            ));
        }
        if hidden != self.config.hidden_dim {
            return Err(AarambhError::Shape(format!(
                "embedding hidden dim {hidden} does not match model hidden_dim {}",
                self.config.hidden_dim
            )));
        }
        if seq_len > self.config.max_seq_len {
            return Err(AarambhError::Shape(format!(
                "sequence length {seq_len} exceeds max_seq_len {}",
                self.config.max_seq_len
            )));
        }
        Ok((batch, seq_len))
    }

    fn check_token_ids(&self, token_ids: &Tensor) -> Result<(usize, usize)> {
        let dims = token_ids.dims();
        if dims.len() != 2 {
            return Err(AarambhError::Shape(format!(
                "token_ids must have shape [batch, seq], got {dims:?}"
            )));
        }
        let batch = dims[0];
        let seq_len = dims[1];
        if batch == 0 || seq_len == 0 {
            return Err(AarambhError::Shape(
                "batch and sequence length must be non-zero".into(),
            ));
        }
        if seq_len > self.config.max_seq_len {
            return Err(AarambhError::Shape(format!(
                "sequence length {seq_len} exceeds max_seq_len {}",
                self.config.max_seq_len
            )));
        }
        Ok((batch, seq_len))
    }
}

#[derive(Debug, Clone)]
struct HybridLoraModel {
    config: ModelConfig,
    base_tensors: HashMap<String, Tensor>,
    linears: HashMap<String, LoraLinear>,
    device: Device,
    dtype: candle_core::DType,
}

impl HybridLoraModel {
    fn new(
        config: &ModelConfig,
        tensors: &HashMap<String, Tensor>,
        lora_config: &LoraConfig,
        varmap: &VarMap,
        quantized_base: bool,
        device: &Device,
    ) -> Result<Self> {
        let mut base_tensors = HashMap::with_capacity(tensors.len());
        let mut linears = HashMap::new();
        for (name, tensor) in tensors {
            base_tensors.insert(name.clone(), tensor.detach());
            if is_projection_weight(name, tensor) {
                linears.insert(
                    name.clone(),
                    LoraLinear::new(name, tensor, lora_config, varmap, quantized_base, device)?,
                );
            }
        }
        Ok(Self {
            config: config.clone(),
            base_tensors,
            linears,
            device: device.clone(),
            dtype: required_ref(tensors, "embedding.weight")?.dtype(),
        })
    }

    fn adapter_param_count(&self) -> usize {
        self.linears
            .values()
            .map(LoraLinear::adapter_param_count)
            .sum()
    }

    fn effective_tensors(&self, train: bool) -> Result<HashMap<String, Tensor>> {
        let mut tensors = self.base_tensors.clone();
        for (name, linear) in &self.linears {
            tensors.insert(name.clone(), linear.effective_weight(train)?);
        }
        Ok(tensors)
    }

    fn forward_embeddings(&self, embeddings: &Tensor, train: bool) -> Result<Tensor> {
        let tensors = self.effective_tensors(train)?;
        let model = AarambhModel::new(
            &self.config,
            VarBuilder::from_tensors(tensors, self.dtype, &self.device),
        )?;
        if train {
            model.forward_embeddings_train(embeddings)
        } else {
            model.forward_embeddings(embeddings)
        }
    }

    fn merged_tensors(&self) -> Result<HashMap<String, Tensor>> {
        let mut tensors = self.base_tensors.clone();
        for (name, linear) in &self.linears {
            tensors.insert(name.clone(), linear.merged_weight()?);
        }
        Ok(tensors)
    }
}

fn is_projection_weight(name: &str, tensor: &Tensor) -> bool {
    !name.starts_with("mtp.")
        && tensor.rank() == 2
        && (name.contains(".attn.")
            || name.contains(".ffn.")
            || (name.contains(".deltanet.") && name.ends_with("_proj.weight"))
            || name == "lm_head.weight")
}

#[derive(Debug, Clone)]
struct LoraBlock {
    norm1_weight: Tensor,
    attn: LoraAttention,
    norm2_weight: Tensor,
    ffn: LoraFfn,
    norm_eps: f32,
}

impl LoraBlock {
    fn new(
        layer_idx: usize,
        config: &ModelConfig,
        tensors: &HashMap<String, Tensor>,
        lora_config: &LoraConfig,
        varmap: &VarMap,
        quantized_base: bool,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("blocks.{layer_idx}");
        Ok(Self {
            norm1_weight: required_tensor(tensors, &format!("{prefix}.norm1.weight"))?,
            attn: LoraAttention::new(
                layer_idx,
                config,
                tensors,
                lora_config,
                varmap,
                quantized_base,
                device,
            )?,
            norm2_weight: required_tensor(tensors, &format!("{prefix}.norm2.weight"))?,
            ffn: LoraFfn::new(
                layer_idx,
                tensors,
                lora_config,
                varmap,
                quantized_base,
                device,
            )?,
            norm_eps: config.norm_eps as f32,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        rope: &RopeCache,
        mask: Option<&Tensor>,
        train: bool,
    ) -> Result<Tensor> {
        let residual = x.clone();
        let hidden = candle_nn::ops::rms_norm_slow(x, &self.norm1_weight, self.norm_eps)?;
        let hidden = self.attn.forward(&hidden, rope, mask, train)?;
        let x = (residual + hidden)?;

        let residual = x.clone();
        let hidden = candle_nn::ops::rms_norm_slow(&x, &self.norm2_weight, self.norm_eps)?;
        let hidden = self.ffn.forward(&hidden, train)?;
        Ok((residual + hidden)?)
    }
}

#[derive(Debug, Clone)]
struct LoraAttention {
    wq: LoraLinear,
    wk: LoraLinear,
    wv: LoraLinear,
    wo: LoraLinear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl LoraAttention {
    fn new(
        layer_idx: usize,
        config: &ModelConfig,
        tensors: &HashMap<String, Tensor>,
        lora_config: &LoraConfig,
        varmap: &VarMap,
        quantized_base: bool,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("blocks.{layer_idx}.attn");
        let head_dim = config.head_dim();
        Ok(Self {
            wq: make_linear(
                tensors,
                &format!("{prefix}.wq.weight"),
                lora_config,
                varmap,
                quantized_base,
                device,
            )?,
            wk: make_linear(
                tensors,
                &format!("{prefix}.wk.weight"),
                lora_config,
                varmap,
                quantized_base,
                device,
            )?,
            wv: make_linear(
                tensors,
                &format!("{prefix}.wv.weight"),
                lora_config,
                varmap,
                quantized_base,
                device,
            )?,
            wo: make_linear(
                tensors,
                &format!("{prefix}.wo.weight"),
                lora_config,
                varmap,
                quantized_base,
                device,
            )?,
            n_heads: config.n_heads,
            n_kv_heads: config.n_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        rope: &RopeCache,
        mask: Option<&Tensor>,
        train: bool,
    ) -> Result<Tensor> {
        let dims = x.dims();
        let b = dims[0];
        let seq_len = dims[1];

        let q = self.wq.forward(x, train)?;
        let k = self.wk.forward(x, train)?;
        let v = self.wv.forward(x, train)?;

        let q = q.reshape((b, seq_len, self.n_heads, self.head_dim))?;
        let k = k.reshape((b, seq_len, self.n_kv_heads, self.head_dim))?;
        let v = v.reshape((b, seq_len, self.n_kv_heads, self.head_dim))?;
        let (q, k) = rope.apply(&q, &k, 0)?;

        let n_repeats = self.n_heads / self.n_kv_heads;
        let k = repeat_heads(&k, n_repeats)?;
        let v = repeat_heads(&v, n_repeats)?;

        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;
        let out = match (mask, train) {
            (Some(mask), _) => aarambh_studio_kernel::dispatch::attention_forward_candle(
                &q,
                &k,
                &v,
                Some(mask),
                self.scale,
            )?,
            (None, true) => aarambh_studio_kernel::dispatch::attention_forward_train_causal(
                &q, &k, &v, self.scale,
            )?,
            (None, false) => {
                aarambh_studio_kernel::dispatch::attention_forward_causal(&q, &k, &v, self.scale)?
            }
        };

        let out = out.transpose(1, 2)?;
        let out = out.reshape((b, seq_len, self.n_heads * self.head_dim))?;
        self.wo.forward(&out, train)
    }
}

#[derive(Debug, Clone)]
struct LoraFfn {
    w_gate: LoraLinear,
    w_up: LoraLinear,
    w_down: LoraLinear,
}

impl LoraFfn {
    fn new(
        layer_idx: usize,
        tensors: &HashMap<String, Tensor>,
        lora_config: &LoraConfig,
        varmap: &VarMap,
        quantized_base: bool,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("blocks.{layer_idx}.ffn");
        Ok(Self {
            w_gate: make_linear(
                tensors,
                &format!("{prefix}.w_gate.weight"),
                lora_config,
                varmap,
                quantized_base,
                device,
            )?,
            w_up: make_linear(
                tensors,
                &format!("{prefix}.w_up.weight"),
                lora_config,
                varmap,
                quantized_base,
                device,
            )?,
            w_down: make_linear(
                tensors,
                &format!("{prefix}.w_down.weight"),
                lora_config,
                varmap,
                quantized_base,
                device,
            )?,
        })
    }

    fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.w_gate.forward(x, train)?)?;
        let up = self.w_up.forward(x, train)?;
        let hidden = (gate * up)?;
        self.w_down.forward(&hidden, train)
    }
}

fn make_linear(
    tensors: &HashMap<String, Tensor>,
    name: &str,
    lora_config: &LoraConfig,
    varmap: &VarMap,
    quantized_base: bool,
    device: &Device,
) -> Result<LoraLinear> {
    LoraLinear::new(
        name,
        required_ref(tensors, name)?,
        lora_config,
        varmap,
        quantized_base,
        device,
    )
}

fn required_tensor(tensors: &HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    Ok(required_ref(tensors, name)?.detach())
}

fn required_ref<'a>(tensors: &'a HashMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
    tensors
        .get(name)
        .ok_or_else(|| AarambhError::Checkpoint(format!("missing tensor {name}")))
}

fn repeat_heads(x: &Tensor, n_repeats: usize) -> Result<Tensor> {
    if n_repeats == 1 {
        return Ok(x.clone());
    }
    let dims = x.dims();
    let b = dims[0];
    let seq = dims[1];
    let n_kv = dims[2];
    let head_dim = dims[3];
    let x = x.unsqueeze(2)?;
    let x = x.expand((b, seq, n_repeats, n_kv, head_dim))?;
    Ok(x.reshape((b, seq, n_kv * n_repeats, head_dim))?
        .contiguous()?)
}

fn adapter_param_count(blocks: &[LoraBlock], lm_head: Option<&LoraLinear>) -> usize {
    let mut count = 0;
    for block in blocks {
        count += block.attn.wq.adapter_param_count();
        count += block.attn.wk.adapter_param_count();
        count += block.attn.wv.adapter_param_count();
        count += block.attn.wo.adapter_param_count();
        count += block.ffn.w_gate.adapter_param_count();
        count += block.ffn.w_up.adapter_param_count();
        count += block.ffn.w_down.adapter_param_count();
    }
    if let Some(lm_head) = lm_head {
        count += lm_head.adapter_param_count();
    }
    count
}

fn tensor_elem_count(tensor: &Tensor) -> usize {
    tensor.dims().iter().product()
}

#[cfg(test)]
mod tests {
    use super::*;
    use aarambh_studio_core::{GatedDeltaNetConfig, HybridAttentionSchedule, MoeConfig, MtpConfig};
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn lora_model_trainable_ratio_is_small() {
        let device = Device::Cpu;
        let config = ModelConfig {
            vocab_size: 32,
            hidden_dim: 64,
            ffn_dim: 128,
            n_layers: 1,
            n_heads: 1,
            n_kv_heads: 1,
            max_seq_len: 8,
            rope_theta: 10000.0,
            rope_scaling: None,
            moe: None,
            attention_schedule: None,
            dsa_config: None,
            mtp: Some(MtpConfig::default()),
            qat: None,
            norm_eps: 1e-5,
            tie_embeddings: true,
            chat_template_version: None,
        };
        let base_varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&base_varmap, DType::F32, &device);
        let base = AarambhModel::new(&config, vb).unwrap();
        let lora = LoraConfig {
            rank: 2,
            alpha: 4.0,
            dropout: 0.0,
            ..Default::default()
        };
        let (model, _varmap) =
            LoraAarambhModel::from_tensors(&config, &base.named_tensors(), &lora, false, &device)
                .unwrap();
        assert!(model.adapter_param_count() > 0);
        assert!(model.trainable_ratio() < 0.2);
        assert!(
            model
                .merged_tensors()
                .unwrap()
                .contains_key("mtp.heads.0.refine.attn.wq.weight")
        );
    }

    #[test]
    fn lora_model_backward_reaches_adapter_params() {
        let device = Device::Cpu;
        let config = ModelConfig {
            vocab_size: 32,
            hidden_dim: 64,
            ffn_dim: 128,
            n_layers: 1,
            n_heads: 1,
            n_kv_heads: 1,
            max_seq_len: 8,
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
        };
        let base_varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&base_varmap, DType::F32, &device);
        let base = AarambhModel::new(&config, vb).unwrap();
        let lora = LoraConfig {
            rank: 2,
            alpha: 4.0,
            dropout: 0.0,
            ..Default::default()
        };
        let (model, varmap) =
            LoraAarambhModel::from_tensors(&config, &base.named_tensors(), &lora, false, &device)
                .unwrap();
        let ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
        let loss = model.forward_train(&ids).unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        let data = varmap.data().lock().unwrap();
        assert!(
            data.values()
                .any(|var| grads.get(var.as_tensor()).is_some())
        );
    }

    #[test]
    fn hybrid_lora_backward_reaches_deltanet_adapters() {
        let device = Device::Cpu;
        let config = ModelConfig {
            vocab_size: 32,
            hidden_dim: 64,
            ffn_dim: 128,
            n_layers: 2,
            n_heads: 1,
            n_kv_heads: 1,
            max_seq_len: 8,
            rope_theta: 10000.0,
            rope_scaling: None,
            moe: None,
            attention_schedule: Some(HybridAttentionSchedule {
                full_attention_every_n: 2,
                gated_deltanet: GatedDeltaNetConfig {
                    n_heads: 1,
                    key_head_dim: 16,
                    value_head_dim: 32,
                    conv_kernel_size: 4,
                    chunk_size: 16,
                },
                mla_layers: Vec::new(),
                mla: None,
            }),
            dsa_config: None,
            mtp: None,
            qat: None,
            norm_eps: 1e-5,
            tie_embeddings: true,
            chat_template_version: None,
        };
        let base_varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&base_varmap, DType::F32, &device);
        let base = AarambhModel::new(&config, vb).unwrap();
        let lora = LoraConfig {
            rank: 2,
            alpha: 4.0,
            dropout: 0.0,
            ..Default::default()
        };
        let (model, varmap) =
            LoraAarambhModel::from_tensors(&config, &base.named_tensors(), &lora, false, &device)
                .unwrap();
        let ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
        let loss = model
            .forward_train(&ids)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap();
        let grads = loss.backward().unwrap();
        let data = varmap.data().lock().unwrap();
        let deltanet_gradients = data
            .iter()
            .filter(|(name, _)| name.contains("blocks.1.deltanet"))
            .map(|(name, var)| (name.clone(), grads.get(var.as_tensor()).is_some()))
            .collect::<Vec<_>>();
        assert!(
            deltanet_gradients.iter().any(|(name, present)| {
                name.contains("blocks.1.deltanet") && name.ends_with(".lora_b") && *present
            }),
            "{deltanet_gradients:?}"
        );
    }

    #[test]
    fn lora_model_rejects_moe_config() {
        let device = Device::Cpu;
        let config = ModelConfig {
            vocab_size: 32,
            hidden_dim: 64,
            ffn_dim: 128,
            n_layers: 2,
            n_heads: 1,
            n_kv_heads: 1,
            max_seq_len: 8,
            rope_theta: 10000.0,
            rope_scaling: None,
            moe: Some(MoeConfig {
                num_experts: 2,
                top_k: 1,
                expert_ffn_dim: 64,
                aux_loss_weight: 0.01,
                every_n_layers: 2,
                ..MoeConfig::default()
            }),
            attention_schedule: None,
            dsa_config: None,
            mtp: None,
            qat: None,
            norm_eps: 1e-5,
            tie_embeddings: true,
            chat_template_version: None,
        };
        let base_varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&base_varmap, DType::F32, &device);
        let base = AarambhModel::new(&config, vb).unwrap();
        let lora = LoraConfig {
            rank: 2,
            alpha: 4.0,
            dropout: 0.0,
            ..Default::default()
        };
        let err = match LoraAarambhModel::from_tensors(
            &config,
            &base.named_tensors(),
            &lora,
            false,
            &device,
        ) {
            Ok(_) => panic!("MoE LoRA construction unexpectedly succeeded"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("MoE"), "{err}");
    }
}
