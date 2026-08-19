use std::collections::HashMap;

use aarambh_studio_core::{AarambhError, ModelConfig, Result};
use aarambh_studio_model::AarambhModel;
use aarambh_studio_nn::RopeCache;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Embedding, Init, Module, VarBuilder, VarMap};

use crate::lora::{BaseLinear, LoraConfig, adapter_tensor_name, linear_forward};

const DORA_EPS: f64 = 1e-6;

/// DoRA adapter hyperparameters and target-module selection.
pub type DoraConfig = LoraConfig;

#[derive(Debug, Clone)]
/// Linear layer with weight-decomposed low-rank adaptation.
pub struct DoraLinear {
    name: String,
    base: BaseLinear,
    magnitude: Option<Tensor>,
    direction_lora_a: Option<Tensor>,
    direction_lora_b: Option<Tensor>,
    scale: f64,
    dropout: f32,
    device: Device,
}

impl DoraLinear {
    /// Create a DoRA-wrapped linear layer.
    pub fn new(
        name: impl Into<String>,
        base_weight: &Tensor,
        config: &DoraConfig,
        varmap: &VarMap,
        quantized_base: bool,
        device: &Device,
    ) -> Result<Self> {
        let name = name.into();
        let dims = base_weight.dims();
        if dims.len() != 2 {
            return Err(AarambhError::Shape(format!(
                "DoRA linear {name} expected 2D base weight, got {dims:?}"
            )));
        }
        config.validate()?;
        let base = BaseLinear::from_tensor(base_weight, quantized_base, config.group_size)?;
        let should_adapt = config.targets_weight(&name);
        let (magnitude, direction_lora_a, direction_lora_b) = if should_adapt {
            let out_dim = dims[0];
            let in_dim = dims[1];
            let magnitude_name = adapter_tensor_name(&name, "magnitude");
            let a_name = adapter_tensor_name(&name, "direction_lora_a");
            let b_name = adapter_tensor_name(&name, "direction_lora_b");
            let magnitude = varmap.get(
                (out_dim,),
                &magnitude_name,
                Init::Const(1.0),
                DType::F32,
                device,
            )?;
            let initial_magnitude = row_norm(base_weight)?.to_dtype(DType::F32)?;
            let mut shared = varmap.clone();
            shared.set_one(&magnitude_name, &initial_magnitude)?;
            let a = varmap.get(
                (config.rank, in_dim),
                &a_name,
                Init::Randn {
                    mean: 0.0,
                    stdev: 0.01,
                },
                DType::F32,
                device,
            )?;
            let b = varmap.get(
                (out_dim, config.rank),
                &b_name,
                Init::Const(0.0),
                DType::F32,
                device,
            )?;
            (Some(magnitude), Some(a), Some(b))
        } else {
            (None, None, None)
        };

        Ok(Self {
            name,
            base,
            magnitude,
            direction_lora_a,
            direction_lora_b,
            scale: config.alpha / config.rank as f64,
            dropout: config.dropout,
            device: device.clone(),
        })
    }

    /// Create a DoRA layer intended to load an existing adapter.
    pub fn from_adapter(
        name: impl Into<String>,
        base_weight: &Tensor,
        config: &DoraConfig,
        varmap: &VarMap,
        quantized_base: bool,
        device: &Device,
    ) -> Result<Self> {
        Self::new(name, base_weight, config, varmap, quantized_base, device)
    }

    /// Return the checkpoint weight name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return true when this layer owns adapter tensors.
    pub fn has_adapter(&self) -> bool {
        self.magnitude.is_some()
            && self.direction_lora_a.is_some()
            && self.direction_lora_b.is_some()
    }

    /// Run the DoRA projection.
    pub fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let weight = self.effective_weight(train)?;
        linear_forward(x, &weight)
    }

    /// Return the base weight with the DoRA adapter merged.
    pub fn merged_weight(&self) -> Result<Tensor> {
        Ok(self.effective_weight(false)?.detach())
    }

    /// Return the number of trainable adapter parameters.
    pub fn adapter_param_count(&self) -> usize {
        let mut count = 0;
        if let Some(magnitude) = &self.magnitude {
            count += magnitude.elem_count();
        }
        if let Some(a) = &self.direction_lora_a {
            count += a.elem_count();
        }
        if let Some(b) = &self.direction_lora_b {
            count += b.elem_count();
        }
        count
    }

    pub(crate) fn effective_weight(&self, train: bool) -> Result<Tensor> {
        let base = self.base.weight(&self.device)?;
        let base_dtype = base.dtype();
        let (Some(magnitude), Some(direction_lora_a), Some(direction_lora_b)) = (
            &self.magnitude,
            &self.direction_lora_a,
            &self.direction_lora_b,
        ) else {
            return Ok(base);
        };
        let delta = direction_lora_b
            .matmul(direction_lora_a)?
            .affine(self.scale, 0.0)?;
        let delta = if train && self.dropout > 0.0 {
            candle_nn::ops::dropout(&delta, self.dropout)?
        } else {
            delta
        };
        let source = (base.to_dtype(DType::F32)? + delta)?;
        let norm = row_norm_keepdim(&source)?;
        let direction = source.broadcast_div(&norm)?;
        let magnitude = magnitude.reshape((magnitude.elem_count(), 1))?;
        Ok(direction.broadcast_mul(&magnitude)?.to_dtype(base_dtype)?)
    }
}

#[derive(Debug, Clone)]
/// Aarambh model with DoRA adapters attached to selected linear layers.
pub struct DoraAarambhModel {
    config: ModelConfig,
    embedding_weight: Tensor,
    blocks: Vec<DoraBlock>,
    final_norm_weight: Tensor,
    lm_head: Option<DoraLinear>,
    rope_cache: RopeCache,
    adapter_param_count: usize,
    base_param_count: usize,
    frozen_auxiliary_tensors: HashMap<String, Tensor>,
    hybrid: Option<HybridDoraModel>,
}

impl DoraAarambhModel {
    /// Build a DoRA model from base checkpoint tensors.
    pub fn from_tensors(
        config: &ModelConfig,
        tensors: &HashMap<String, Tensor>,
        dora_config: &DoraConfig,
        quantized_base: bool,
        device: &Device,
    ) -> Result<(Self, VarMap)> {
        AarambhModel::validate_config(config)?;
        if config.moe.is_some() {
            return Err(AarambhError::Config(
                "DoRA for MoE models is not supported; train the MoE base model directly or use a dense config".into(),
            ));
        }
        dora_config.validate()?;
        let varmap = VarMap::new();
        let embedding_weight = required_tensor(tensors, "embedding.weight")?;
        if config.attention_schedule.is_some() {
            let hybrid = HybridDoraModel::new(
                config,
                tensors,
                dora_config,
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
            blocks.push(DoraBlock::new(
                layer_idx,
                config,
                tensors,
                dora_config,
                &varmap,
                quantized_base,
                device,
            )?);
        }

        let final_norm_weight = required_tensor(tensors, "final_norm.weight")?;
        let lm_head = if config.tie_embeddings {
            None
        } else {
            Some(DoraLinear::new(
                "lm_head.weight",
                required_ref(tensors, "lm_head.weight")?,
                dora_config,
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

    /// Convert token ids into frozen base token embeddings.
    pub fn embed_tokens(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.check_token_ids(token_ids)?;
        let embedding = Embedding::new(self.embedding_weight.clone(), self.config.hidden_dim);
        Ok(embedding.forward(token_ids)?)
    }

    /// Run the training forward path over precomputed token embeddings.
    pub fn forward_embeddings_train(&self, embeddings: &Tensor) -> Result<Tensor> {
        self.forward_embeddings(embeddings, true)
    }

    /// Run the evaluation forward path over precomputed token embeddings.
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
        self.check_token_ids(token_ids)?;
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
        let hidden_dim = dims[2];
        if batch == 0 || seq_len == 0 {
            return Err(AarambhError::Shape(
                "batch and sequence length must be non-zero".into(),
            ));
        }
        if hidden_dim != self.config.hidden_dim {
            return Err(AarambhError::Shape(format!(
                "embedding hidden_dim {hidden_dim} does not match model hidden_dim {}",
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
struct HybridDoraModel {
    config: ModelConfig,
    base_tensors: HashMap<String, Tensor>,
    linears: HashMap<String, DoraLinear>,
    device: Device,
    dtype: DType,
}

impl HybridDoraModel {
    fn new(
        config: &ModelConfig,
        tensors: &HashMap<String, Tensor>,
        dora_config: &DoraConfig,
        varmap: &VarMap,
        quantized_base: bool,
        device: &Device,
    ) -> Result<Self> {
        let mut base_tensors = HashMap::with_capacity(tensors.len());
        let mut linears = HashMap::new();
        for (name, tensor) in tensors {
            base_tensors.insert(name.clone(), tensor.detach());
            if is_dora_projection_weight(name, tensor) {
                linears.insert(
                    name.clone(),
                    DoraLinear::new(name, tensor, dora_config, varmap, quantized_base, device)?,
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
            .map(DoraLinear::adapter_param_count)
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
        let model = AarambhModel::new(
            &self.config,
            VarBuilder::from_tensors(self.effective_tensors(train)?, self.dtype, &self.device),
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

fn is_dora_projection_weight(name: &str, tensor: &Tensor) -> bool {
    !name.starts_with("mtp.")
        && tensor.rank() == 2
        && (name.contains(".attn.")
            || name.contains(".ffn.")
            || (name.contains(".deltanet.") && name.ends_with("_proj.weight"))
            || name == "lm_head.weight")
}

#[derive(Debug, Clone)]
struct DoraBlock {
    norm1_weight: Tensor,
    attn: DoraAttention,
    norm2_weight: Tensor,
    ffn: DoraFfn,
    norm_eps: f32,
}

impl DoraBlock {
    fn new(
        layer_idx: usize,
        config: &ModelConfig,
        tensors: &HashMap<String, Tensor>,
        dora_config: &DoraConfig,
        varmap: &VarMap,
        quantized_base: bool,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("blocks.{layer_idx}");
        Ok(Self {
            norm1_weight: required_tensor(tensors, &format!("{prefix}.norm1.weight"))?,
            attn: DoraAttention::new(
                layer_idx,
                config,
                tensors,
                dora_config,
                varmap,
                quantized_base,
                device,
            )?,
            norm2_weight: required_tensor(tensors, &format!("{prefix}.norm2.weight"))?,
            ffn: DoraFfn::new(
                layer_idx,
                tensors,
                dora_config,
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
struct DoraAttention {
    wq: DoraLinear,
    wk: DoraLinear,
    wv: DoraLinear,
    wo: DoraLinear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl DoraAttention {
    fn new(
        layer_idx: usize,
        config: &ModelConfig,
        tensors: &HashMap<String, Tensor>,
        dora_config: &DoraConfig,
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
                dora_config,
                varmap,
                quantized_base,
                device,
            )?,
            wk: make_linear(
                tensors,
                &format!("{prefix}.wk.weight"),
                dora_config,
                varmap,
                quantized_base,
                device,
            )?,
            wv: make_linear(
                tensors,
                &format!("{prefix}.wv.weight"),
                dora_config,
                varmap,
                quantized_base,
                device,
            )?,
            wo: make_linear(
                tensors,
                &format!("{prefix}.wo.weight"),
                dora_config,
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
struct DoraFfn {
    w_gate: DoraLinear,
    w_up: DoraLinear,
    w_down: DoraLinear,
}

impl DoraFfn {
    fn new(
        layer_idx: usize,
        tensors: &HashMap<String, Tensor>,
        dora_config: &DoraConfig,
        varmap: &VarMap,
        quantized_base: bool,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("blocks.{layer_idx}.ffn");
        Ok(Self {
            w_gate: make_linear(
                tensors,
                &format!("{prefix}.w_gate.weight"),
                dora_config,
                varmap,
                quantized_base,
                device,
            )?,
            w_up: make_linear(
                tensors,
                &format!("{prefix}.w_up.weight"),
                dora_config,
                varmap,
                quantized_base,
                device,
            )?,
            w_down: make_linear(
                tensors,
                &format!("{prefix}.w_down.weight"),
                dora_config,
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
    dora_config: &DoraConfig,
    varmap: &VarMap,
    quantized_base: bool,
    device: &Device,
) -> Result<DoraLinear> {
    DoraLinear::new(
        name,
        required_ref(tensors, name)?,
        dora_config,
        varmap,
        quantized_base,
        device,
    )
}

fn row_norm(weight: &Tensor) -> Result<Tensor> {
    Ok(row_norm_keepdim(weight)?.squeeze(1)?)
}

fn row_norm_keepdim(weight: &Tensor) -> Result<Tensor> {
    Ok((weight.sqr()?.sum_keepdim(1)? + DORA_EPS)?.sqrt()?)
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

fn adapter_param_count(blocks: &[DoraBlock], lm_head: Option<&DoraLinear>) -> usize {
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
    use aarambh_studio_core::{GatedDeltaNetConfig, HybridAttentionSchedule, MtpConfig};
    use candle_nn::VarBuilder;

    fn tiny_config() -> ModelConfig {
        ModelConfig {
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
        }
    }

    #[test]
    fn zero_direction_update_matches_base_forward() {
        let device = Device::Cpu;
        let weight = Tensor::from_vec(vec![1f32, 2., 3., 4.], (2, 2), &device).unwrap();
        let x = Tensor::from_vec(vec![10f32, 100.], (1, 2), &device).unwrap();
        let varmap = VarMap::new();
        let config = DoraConfig {
            rank: 1,
            alpha: 1.0,
            dropout: 0.0,
            target_modules: vec!["w".into()],
            ..Default::default()
        };
        let dora = DoraLinear::new("w.weight", &weight, &config, &varmap, false, &device).unwrap();
        let base = linear_forward(&x, &weight)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let out = dora.forward(&x, false).unwrap().to_vec2::<f32>().unwrap();
        for (lhs, rhs) in out.iter().flatten().zip(base.iter().flatten()) {
            assert!((lhs - rhs).abs() < 1e-4, "lhs={lhs} rhs={rhs}");
        }
    }

    #[test]
    fn dora_trainable_params_include_magnitude_and_direction_lora() {
        let device = Device::Cpu;
        let config = tiny_config();
        let base_varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&base_varmap, DType::F32, &device);
        let base = AarambhModel::new(&config, vb).unwrap();
        let dora = DoraConfig {
            rank: 2,
            alpha: 4.0,
            dropout: 0.0,
            ..Default::default()
        };
        let (model, _varmap) =
            DoraAarambhModel::from_tensors(&config, &base.named_tensors(), &dora, false, &device)
                .unwrap();
        assert!(model.adapter_param_count() > 0);
        assert!(model.trainable_ratio() < 0.25);
    }

    #[test]
    fn dora_model_backward_reaches_magnitude_and_direction_params() {
        let device = Device::Cpu;
        let config = tiny_config();
        let base_varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&base_varmap, DType::F32, &device);
        let base = AarambhModel::new(&config, vb).unwrap();
        let dora = DoraConfig {
            rank: 2,
            alpha: 4.0,
            dropout: 0.0,
            ..Default::default()
        };
        let (model, varmap) =
            DoraAarambhModel::from_tensors(&config, &base.named_tensors(), &dora, false, &device)
                .unwrap();
        let ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
        let loss = model.forward_train(&ids).unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        let data = varmap.data().lock().unwrap();
        assert!(data.iter().any(
            |(name, var)| name.ends_with(".magnitude") && grads.get(var.as_tensor()).is_some()
        ));
        assert!(
            data.iter()
                .any(|(name, var)| name.ends_with(".direction_lora_b")
                    && grads.get(var.as_tensor()).is_some())
        );
    }

    #[test]
    fn hybrid_dora_backward_reaches_deltanet_adapters() {
        let device = Device::Cpu;
        let mut config = tiny_config();
        config.n_layers = 2;
        config.attention_schedule = Some(HybridAttentionSchedule {
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
        });
        let base_varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&base_varmap, DType::F32, &device);
        let base = AarambhModel::new(&config, vb).unwrap();
        let dora = DoraConfig {
            rank: 2,
            alpha: 4.0,
            dropout: 0.0,
            ..Default::default()
        };
        let (model, varmap) =
            DoraAarambhModel::from_tensors(&config, &base.named_tensors(), &dora, false, &device)
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
                name.contains("blocks.1.deltanet")
                    && name.ends_with(".direction_lora_b")
                    && *present
            }),
            "{deltanet_gradients:?}"
        );
    }

    #[test]
    fn qdora_dequantises_packed_int4_base_before_forward() {
        let device = Device::Cpu;
        let config = tiny_config();
        let base_varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&base_varmap, DType::F32, &device);
        let base = AarambhModel::new(&config, vb).unwrap();
        let dora = DoraConfig {
            rank: 2,
            alpha: 4.0,
            dropout: 0.0,
            ..Default::default()
        };
        let (model, _varmap) =
            DoraAarambhModel::from_tensors(&config, &base.named_tensors(), &dora, true, &device)
                .unwrap();
        let ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
        let logits = model.forward_eval(&ids).unwrap();
        assert_eq!(logits.dims(), &[1, 4, 32]);
        assert!(
            logits
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
                .is_finite()
        );
    }

    #[test]
    fn dora_merge_produces_valid_normal_checkpoint_tensors() {
        let device = Device::Cpu;
        let mut config = tiny_config();
        config.mtp = Some(MtpConfig::default());
        let base_varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&base_varmap, DType::F32, &device);
        let base = AarambhModel::new(&config, vb).unwrap();
        let dora = DoraConfig {
            rank: 2,
            alpha: 4.0,
            dropout: 0.0,
            ..Default::default()
        };
        let (model, _varmap) =
            DoraAarambhModel::from_tensors(&config, &base.named_tensors(), &dora, false, &device)
                .unwrap();
        let merged = model.merged_tensors().unwrap();
        assert!(merged.contains_key("embedding.weight"));
        assert!(merged.contains_key("blocks.0.attn.wq.weight"));
        assert!(merged.contains_key("final_norm.weight"));
        assert!(merged.contains_key("mtp.heads.0.refine.attn.wq.weight"));
    }
}
