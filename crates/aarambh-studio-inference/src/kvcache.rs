use aarambh_studio_model::AarambhModel;
use aarambh_studio_nn::HybridKvCache;
use candle_core::Result;

#[derive(Debug, Clone)]
/// Multi-layer KV cache used by the inference engine.
pub struct KvCache {
    layers: Vec<HybridKvCache>,
}

impl KvCache {
    /// Create a cache with `n_layers` empty layer caches.
    pub fn new(n_layers: usize) -> Self {
        Self {
            layers: (0..n_layers)
                .map(|_| HybridKvCache::Full(aarambh_studio_nn::KVCache::new()))
                .collect(),
        }
    }

    /// Create a cache sized for a model.
    pub fn for_model(model: &AarambhModel) -> Self {
        Self {
            layers: model.empty_kv_cache(),
        }
    }

    /// Create a model-sized cache with fixed sequence capacity per layer.
    pub fn for_model_with_capacity(model: &AarambhModel, capacity: usize) -> Self {
        Self {
            layers: model.empty_kv_cache_with_capacity(capacity),
        }
    }

    /// Return mutable layer caches.
    pub fn layers_mut(&mut self) -> &mut [HybridKvCache] {
        &mut self.layers
    }

    /// Clear all layer caches.
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }

    /// Roll every layer back to a previously committed sequence length.
    pub fn truncate(&mut self, new_len: usize) -> Result<()> {
        for layer in &mut self.layers {
            layer.truncate(new_len)?;
        }
        debug_assert!(self.layers.iter().all(|layer| layer.seq_len() == new_len));
        Ok(())
    }

    /// Capture an exact transaction snapshot for speculative cache rollback.
    pub fn snapshot(&self) -> Self {
        self.clone()
    }

    /// Restore an exact transaction snapshot.
    pub fn restore(&mut self, snapshot: Self) {
        *self = snapshot;
    }

    /// Return cached sequence length from the first layer.
    pub fn seqlen(&self) -> usize {
        self.layers.first().map(HybridKvCache::seq_len).unwrap_or(0)
    }

    /// Return the number of layer caches.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Return true when no layers are cached.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::KvCache as InferenceKvCache;
    use aarambh_studio_core::{GatedDeltaNetConfig, HybridAttentionSchedule, ModelConfig};
    use aarambh_studio_model::AarambhModel;
    use aarambh_studio_nn::KVCache;
    use candle_core::{Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn kvcache_seqlen_grows_each_step() {
        let device = Device::Cpu;
        let mut cache = KVCache::new();
        let k1 = Tensor::zeros((1, 1, 2, 64), candle_core::DType::F32, &device).unwrap();
        let v1 = Tensor::zeros((1, 1, 2, 64), candle_core::DType::F32, &device).unwrap();
        cache.update(&k1, &v1).unwrap();
        assert_eq!(cache.seq_len(), 1);

        let k2 = Tensor::zeros((1, 1, 2, 64), candle_core::DType::F32, &device).unwrap();
        let v2 = Tensor::zeros((1, 1, 2, 64), candle_core::DType::F32, &device).unwrap();
        cache.update(&k2, &v2).unwrap();
        assert_eq!(cache.seq_len(), 2);
    }

    #[test]
    fn hybrid_snapshot_restores_recurrent_state_length() {
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
        let vars = VarMap::new();
        let model = AarambhModel::new(
            &config,
            VarBuilder::from_varmap(&vars, candle_core::DType::F32, &device),
        )
        .unwrap();
        let mut cache = InferenceKvCache::for_model_with_capacity(&model, 4);
        let first = Tensor::from_vec(vec![1u32], (1, 1), &device).unwrap();
        model
            .forward_with_cache(&first, 0, cache.layers_mut())
            .unwrap();
        let snapshot = cache.snapshot();
        let second = Tensor::from_vec(vec![2u32], (1, 1), &device).unwrap();
        model
            .forward_with_cache(&second, 1, cache.layers_mut())
            .unwrap();
        assert_eq!(cache.seqlen(), 2);
        cache.restore(snapshot);
        assert_eq!(cache.seqlen(), 1);
        assert_eq!(cache.layers[1].as_linear().unwrap().seq_len(), 1);
    }
}
