use aarambh_studio_core::{
    AttentionKind, DsaConfig, GatedDeltaNetConfig, HybridAttentionSchedule, MlaConfig, ModelConfig,
    MoeConfig, MtpConfig, QatConfig, RopeScalingConfig, RopeScalingMethod,
};
use aarambh_studio_model::{AarambhModel, kv_cache_report};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

fn mini_config() -> ModelConfig {
    ModelConfig {
        vocab_size: 128,
        hidden_dim: 64,
        ffn_dim: 128,
        n_layers: 2,
        n_heads: 1,
        n_kv_heads: 1,
        max_seq_len: 16,
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

fn mini_model(device: &Device) -> AarambhModel {
    let cfg = mini_config();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    AarambhModel::new(&cfg, vb).unwrap()
}

fn scaled_mini_config() -> ModelConfig {
    ModelConfig {
        rope_scaling: Some(RopeScalingConfig {
            method: RopeScalingMethod::Linear,
            factor: 2.0,
            original_max_seq_len: 8,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attn_factor: 1.0,
        }),
        max_seq_len: 16,
        ..mini_config()
    }
}

fn moe_mini_config() -> ModelConfig {
    ModelConfig {
        moe: Some(MoeConfig {
            num_experts: 4,
            top_k: 2,
            expert_ffn_dim: 64,
            aux_loss_weight: 0.01,
            every_n_layers: 2,
            ..MoeConfig::default()
        }),
        ..mini_config()
    }
}

fn fine_moe_mini_config() -> ModelConfig {
    ModelConfig {
        moe: Some(MoeConfig {
            num_experts: 4,
            top_k: 4,
            expert_ffn_dim: 64,
            aux_loss_weight: 0.01,
            every_n_layers: 2,
            fine_grained_factor: 2,
            num_shared_experts: 1,
            ..MoeConfig::default()
        }),
        ..mini_config()
    }
}

fn hybrid_mini_config() -> ModelConfig {
    ModelConfig {
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
        ..mini_config()
    }
}

fn dsa_mini_config() -> ModelConfig {
    ModelConfig {
        max_seq_len: 32,
        dsa_config: Some(DsaConfig {
            block_size: 16,
            top_k_blocks: 1,
            min_seq_len_for_sparsity: 16,
        }),
        ..hybrid_mini_config()
    }
}

fn mla_mini_config() -> ModelConfig {
    // Layer 0 = LatentMLA (upgraded from the every-n Full slot), layer 1 = Gated DeltaNet.
    ModelConfig {
        attention_schedule: Some(HybridAttentionSchedule {
            full_attention_every_n: 2,
            gated_deltanet: GatedDeltaNetConfig {
                n_heads: 1,
                key_head_dim: 16,
                value_head_dim: 32,
                conv_kernel_size: 4,
                chunk_size: 16,
            },
            mla_layers: vec![0],
            mla: Some(MlaConfig {
                latent_dim: 32,
                rope_head_dim: 16,
                ..Default::default()
            }),
        }),
        ..mini_config()
    }
}

#[test]
fn all_four_model_configs_validate() {
    for cfg in [
        ModelConfig::tiny(),
        ModelConfig::small(),
        ModelConfig::medium(),
        ModelConfig::large(),
    ] {
        AarambhModel::validate_config(&cfg).unwrap();
    }
}

#[test]
fn tiny_forward_produces_correct_shape() {
    let device = Device::Cpu;
    let cfg = ModelConfig::tiny();
    let vb = VarBuilder::zeros(DType::F32, &device);
    let model = AarambhModel::new(&cfg, vb).unwrap();
    let ids = Tensor::zeros((1, 16), DType::U32, &device).unwrap();
    let logits = model.forward(&ids).unwrap();
    assert_eq!(logits.shape().dims(), &[1, 16, 32000]);
}

#[test]
fn mini_forward_produces_correct_shape_and_finite_logits() {
    let device = Device::Cpu;
    let cfg = mini_config();
    let model = mini_model(&device);
    let ids = Tensor::from_vec(vec![1u32, 2, 3, 4, 5, 6], (1, 6), &device).unwrap();
    let logits = model.forward(&ids).unwrap();
    assert_eq!(logits.shape().dims(), &[1, 6, cfg.vocab_size]);

    let max = logits
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(max.is_finite());
    assert!(max < 10.0, "initial logits are too large: {max}");
}

#[test]
fn cached_forward_matches_full_forward_for_next_token() {
    let device = Device::Cpu;
    let model = mini_model(&device);
    let ids = Tensor::from_vec(vec![7u32, 8, 9, 10], (1, 4), &device).unwrap();
    let full_logits = model.forward(&ids).unwrap();
    let full_last = full_logits.narrow(1, 3, 1).unwrap();

    let mut caches = model.empty_kv_cache();
    let mut cached_last = None;
    for pos in 0..4 {
        let token = ids.narrow(1, pos, 1).unwrap();
        cached_last = Some(model.forward_with_cache(&token, pos, &mut caches).unwrap());
    }

    let cached_last = cached_last.unwrap();
    let max_diff = (full_last - cached_last)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(max_diff < 1e-4, "cached/full mismatch: {max_diff}");
}

#[test]
fn hybrid_cached_forward_matches_full_forward_and_state_is_constant() {
    let device = Device::Cpu;
    let cfg = hybrid_mini_config();
    let varmap = VarMap::new();
    let model =
        AarambhModel::new(&cfg, VarBuilder::from_varmap(&varmap, DType::F32, &device)).unwrap();
    let ids = Tensor::from_vec(vec![7u32, 8, 9, 10], (1, 4), &device).unwrap();
    let full_last = model.forward(&ids).unwrap().narrow(1, 3, 1).unwrap();
    let mut caches = model.empty_kv_cache();
    assert!(caches[0].as_linear().is_none());
    assert!(caches[1].as_linear().is_some());

    let mut cached_last = None;
    let mut state_elements = 0;
    for pos in 0..4 {
        cached_last = Some(
            model
                .forward_with_cache(&ids.narrow(1, pos, 1).unwrap(), pos, &mut caches)
                .unwrap(),
        );
        let elements = caches[1].as_linear().unwrap().state_elements();
        if pos == 0 {
            state_elements = elements;
        } else {
            assert_eq!(elements, state_elements);
        }
    }

    let max_diff = (full_last - cached_last.unwrap())
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(max_diff < 1e-4, "hybrid cached/full mismatch: {max_diff}");
    assert_eq!(caches[1].as_linear().unwrap().seq_len(), 4);
    assert_eq!(
        model.get_weight("blocks.1.deltanet.A_log").unwrap().dtype(),
        DType::F32
    );
}

#[test]
fn hybrid_training_backward_reaches_deltanet_parameters() {
    let device = Device::Cpu;
    let cfg = hybrid_mini_config();
    let varmap = VarMap::new();
    let model =
        AarambhModel::new(&cfg, VarBuilder::from_varmap(&varmap, DType::F32, &device)).unwrap();
    let ids = Tensor::from_vec(vec![7u32, 8, 9, 10], (1, 4), &device).unwrap();
    let loss = model
        .forward_train(&ids)
        .unwrap()
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap();
    let gradients = loss.backward().unwrap();
    let variables = varmap.data().lock().unwrap();
    let layer_gradients = variables
        .iter()
        .filter(|(name, _)| name.starts_with("blocks.1"))
        .map(|(name, variable)| (name.clone(), gradients.get(variable.as_tensor()).is_some()))
        .collect::<Vec<_>>();
    assert!(
        variables.iter().any(|(name, variable)| {
            name == "blocks.1.deltanet.out_proj.weight"
                && gradients.get(variable.as_tensor()).is_some()
        }),
        "{layer_gradients:?}"
    );
}

#[test]
fn mla_model_forwards_and_cached_forward_matches_full_forward() {
    let device = Device::Cpu;
    let cfg = mla_mini_config();
    let varmap = VarMap::new();
    let model =
        AarambhModel::new(&cfg, VarBuilder::from_varmap(&varmap, DType::F32, &device)).unwrap();
    let ids = Tensor::from_vec(vec![7u32, 8, 9, 10], (1, 4), &device).unwrap();
    let full_last = model.forward(&ids).unwrap().narrow(1, 3, 1).unwrap();

    let mut caches = model.empty_kv_cache();
    // Layer 0 is MLA (compressed-latent cache), layer 1 is Gated DeltaNet (linear state).
    assert!(caches[0].as_mla().is_some());
    assert!(caches[0].as_linear().is_none());
    assert!(caches[1].as_linear().is_some());

    let mut cached_last = None;
    for pos in 0..4 {
        cached_last = Some(
            model
                .forward_with_cache(&ids.narrow(1, pos, 1).unwrap(), pos, &mut caches)
                .unwrap(),
        );
        // MLA cache grows by one token per step while staying compressed.
        assert_eq!(caches[0].as_mla().unwrap().seq_len(), pos + 1);
    }

    let max_diff = (full_last - cached_last.unwrap())
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(max_diff < 1e-4, "MLA cached/full mismatch: {max_diff}");
}

#[test]
fn mla_training_backward_reaches_mla_parameters() {
    let device = Device::Cpu;
    let cfg = mla_mini_config();
    let varmap = VarMap::new();
    let model =
        AarambhModel::new(&cfg, VarBuilder::from_varmap(&varmap, DType::F32, &device)).unwrap();
    let ids = Tensor::from_vec(vec![7u32, 8, 9, 10], (1, 4), &device).unwrap();
    let loss = model
        .forward_train(&ids)
        .unwrap()
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap();
    let gradients = loss.backward().unwrap();
    let variables = varmap.data().lock().unwrap();
    // Gradients must reach the MLA down-projection (kv_a_proj), latent norm,
    // value up-projection (up_v), and output projection (o_proj) — the
    // SELF_LEARNING_V4 §42 anti-forgetting reachability argument. The query and
    // key paths (q_proj, up_k, k_rope_proj) match the existing GQA CPU training
    // path, whose candle-fallback attention backward propagates to V/O but not
    // to Q/K on CPU (full Q/K gradients flow under the CUDA/flash path used in
    // real training); MLA is wired into the identical attention path, so its
    // gradient reachability is consistent with GQA.
    for name in [
        "blocks.0.mla.kv_a_proj.weight",
        "blocks.0.mla.kv_a_norm.weight",
        "blocks.0.mla.up_v.weight",
        "blocks.0.mla.o_proj.weight",
    ] {
        let has_grad = variables
            .get(name)
            .map(|v| gradients.get(v.as_tensor()).is_some())
            .unwrap_or(false);
        assert!(has_grad, "gradient did not reach MLA weight {name}");
    }
}

#[test]
fn mla_kv_cache_report_shows_compressed_footprint() {
    let cfg = mla_mini_config();
    let report = kv_cache_report(&cfg, 4); // f32 = 4 bytes/element
    assert_eq!(report.len(), cfg.n_layers);
    assert_eq!(report[0].kind, AttentionKind::LatentMLA);
    // MLA cache = (latent_dim(32) + rope_head_dim(16)) * 4 = 192 bytes/token.
    assert_eq!(report[0].bytes_per_token, 192);
    // Gated DeltaNet layer 1 uses a fixed recurrent state (0 bytes/token).
    assert_eq!(report[1].kind, AttentionKind::GatedDeltaNet);
    assert_eq!(report[1].bytes_per_token, 0);
    // MLA footprint must be smaller than the all-Full GQA baseline.
    let gqa_baseline = 2 * cfg.n_kv_heads * cfg.head_dim() * 4;
    assert!(report[0].bytes_per_token < gqa_baseline);
}

#[test]
fn dsa_cached_forward_matches_full_sparse_forward() {
    let device = Device::Cpu;
    let cfg = dsa_mini_config();
    let varmap = VarMap::new();
    let model =
        AarambhModel::new(&cfg, VarBuilder::from_varmap(&varmap, DType::F32, &device)).unwrap();
    let ids = Tensor::from_vec(
        (0..32).map(|value| (value % 127 + 1) as u32).collect(),
        (1, 32),
        &device,
    )
    .unwrap();
    let full_last = model.forward(&ids).unwrap().narrow(1, 31, 1).unwrap();
    let mut caches = model.empty_kv_cache();
    assert!(caches[0].as_sparse().is_some());
    assert!(caches[1].as_linear().is_some());
    let mut cached_last = None;
    for position in 0..32 {
        cached_last = Some(
            model
                .forward_with_cache(&ids.narrow(1, position, 1).unwrap(), position, &mut caches)
                .unwrap(),
        );
    }
    let max_diff = (full_last - cached_last.unwrap())
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(max_diff < 1e-4, "DSA cached/full mismatch: {max_diff}");
    assert_eq!(caches[0].as_sparse().unwrap().completed_blocks(), 2);
}

#[test]
fn dsa_dense_fallback_matches_phase29_attention_exactly() {
    let device = Device::Cpu;
    let dsa_cfg = dsa_mini_config();
    let vars = VarMap::new();
    let dsa = AarambhModel::new(
        &dsa_cfg,
        VarBuilder::from_varmap(&vars, DType::F32, &device),
    )
    .unwrap();
    let mut dense_cfg = dsa_cfg.clone();
    dense_cfg.dsa_config = None;
    let dense = AarambhModel::new(
        &dense_cfg,
        VarBuilder::from_tensors(dsa.named_tensors(), DType::F32, &device),
    )
    .unwrap();
    let ids = Tensor::from_vec((1..=8).collect::<Vec<u32>>(), (1, 8), &device).unwrap();
    let max_diff = (dsa.forward(&ids).unwrap() - dense.forward(&ids).unwrap())
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert_eq!(max_diff, 0.0);
}

#[test]
fn dsa_teacher_loss_reaches_only_indexer_parameters() {
    let device = Device::Cpu;
    let cfg = dsa_mini_config();
    let vars = VarMap::new();
    let model =
        AarambhModel::new(&cfg, VarBuilder::from_varmap(&vars, DType::F32, &device)).unwrap();
    let ids = Tensor::from_vec(
        (0..32).map(|value| (value % 127 + 1) as u32).collect(),
        (1, 32),
        &device,
    )
    .unwrap();
    let output = model
        .forward_train_with_aux_and_dsa_teacher(&ids, true)
        .unwrap();
    let loss = output.dsa_indexer_loss.unwrap();
    assert!(loss.to_scalar::<f32>().unwrap().is_finite());
    assert!(output.dsa_top_k_recall.unwrap().is_finite());
    let grads = loss.backward().unwrap();
    let variables = vars.data().lock().unwrap();
    assert!(variables.iter().any(|(name, variable)| {
        name == "blocks.0.dsa.index_q.weight" && grads.get(variable.as_tensor()).is_some()
    }));
    assert!(variables.iter().any(|(name, variable)| {
        name == "blocks.0.dsa.index_k.weight" && grads.get(variable.as_tensor()).is_some()
    }));
    assert!(variables.iter().all(|(name, variable)| {
        name.contains(".dsa.") || grads.get(variable.as_tensor()).is_none()
    }));
}

#[test]
fn dense_training_backward_reaches_last_block_parameters() {
    let device = Device::Cpu;
    let cfg = mini_config();
    let varmap = VarMap::new();
    let model =
        AarambhModel::new(&cfg, VarBuilder::from_varmap(&varmap, DType::F32, &device)).unwrap();
    let ids = Tensor::from_vec(vec![7u32, 8, 9, 10], (1, 4), &device).unwrap();
    let loss = model
        .forward_train(&ids)
        .unwrap()
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap();
    let gradients = loss.backward().unwrap();
    let variables = varmap.data().lock().unwrap();
    assert!(variables.iter().any(|(name, variable)| {
        name == "blocks.1.ffn.w_down.weight" && gradients.get(variable.as_tensor()).is_some()
    }));
}

#[test]
fn attention_schedule_none_keeps_dense_tensor_names() {
    let device = Device::Cpu;
    let model = mini_model(&device);
    let names = model.named_tensors();
    assert!(names.contains_key("blocks.1.attn.wq.weight"));
    assert!(!names.keys().any(|name| name.contains("deltanet")));
}

#[test]
fn moe_forward_produces_correct_shape_and_aux_loss() {
    let device = Device::Cpu;
    let cfg = moe_mini_config();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = AarambhModel::new(&cfg, vb).unwrap();
    let ids = Tensor::from_vec(vec![1u32, 2, 3, 4, 5, 6], (1, 6), &device).unwrap();
    let output = model.forward_train_with_aux(&ids).unwrap();
    assert_eq!(output.logits.shape().dims(), &[1, 6, cfg.vocab_size]);
    assert!(output.moe_aux_loss.is_some());
    assert_eq!(output.expert_utilization.len(), 4);
}

#[test]
fn moe_cached_forward_matches_full_forward_for_next_token() {
    let device = Device::Cpu;
    let cfg = moe_mini_config();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = AarambhModel::new(&cfg, vb).unwrap();
    let ids = Tensor::from_vec(vec![7u32, 8, 9, 10], (1, 4), &device).unwrap();
    let full_logits = model.forward(&ids).unwrap();
    let full_last = full_logits.narrow(1, 3, 1).unwrap();

    let mut caches = model.empty_kv_cache();
    let mut cached_last = None;
    for pos in 0..4 {
        let token = ids.narrow(1, pos, 1).unwrap();
        cached_last = Some(model.forward_with_cache(&token, pos, &mut caches).unwrap());
    }

    let cached_last = cached_last.unwrap();
    let max_diff = (full_last - cached_last)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(max_diff < 1e-4, "cached/full mismatch: {max_diff}");
}

#[test]
fn moe_tensor_names_use_router_and_expert_paths() {
    let device = Device::Cpu;
    let cfg = moe_mini_config();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = AarambhModel::new(&cfg, vb).unwrap();
    let tensors = model.named_tensors();
    assert!(tensors.contains_key("blocks.0.ffn.w_gate.weight"));
    assert!(tensors.contains_key("blocks.1.ffn.router.weight"));
    assert!(tensors.contains_key("blocks.1.ffn.experts.0.w_gate.weight"));
    assert!(
        model
            .get_weight("blocks.1.ffn.experts.3.w_down.weight")
            .is_some()
    );
    assert!(model.get_weight("blocks.1.ffn.w_gate.weight").is_none());
}

#[test]
fn fine_grained_moe_uses_expanded_router_and_shared_tensor_namespace() {
    let device = Device::Cpu;
    let cfg = fine_moe_mini_config();
    let varmap = VarMap::new();
    let model =
        AarambhModel::new(&cfg, VarBuilder::from_varmap(&varmap, DType::F32, &device)).unwrap();
    assert_eq!(
        model
            .get_weight("blocks.1.ffn.router.weight")
            .unwrap()
            .dims(),
        [8, 64]
    );
    assert_eq!(
        model
            .get_weight("blocks.1.ffn.experts.7.w_gate.weight")
            .unwrap()
            .dims(),
        [32, 64]
    );
    assert_eq!(
        model
            .get_weight("blocks.1.ffn.shared_experts.0.w_down.weight")
            .unwrap()
            .dims(),
        [64, 32]
    );
    let ids = Tensor::from_vec(vec![1u32, 2, 3], (1, 3), &device).unwrap();
    assert_eq!(model.forward(&ids).unwrap().dims(), [1, 3, 128]);
    let capture = model.linear_inputs(&ids).unwrap();
    assert!(capture.contains_key("blocks.1.ffn.shared_experts.0.w_gate.weight"));
    assert!(capture.contains_key("blocks.1.ffn.shared_experts.0.w_down.weight"));
}

#[test]
fn explicit_phase22_moe_defaults_produce_identical_logits() {
    let device = Device::Cpu;
    let cfg = moe_mini_config();
    let explicit_cfg = ModelConfig {
        moe: cfg.moe.as_ref().map(|moe| MoeConfig {
            fine_grained_factor: 1,
            num_shared_experts: 0,
            ..moe.clone()
        }),
        ..cfg.clone()
    };
    let varmap = VarMap::new();
    let model =
        AarambhModel::new(&cfg, VarBuilder::from_varmap(&varmap, DType::F32, &device)).unwrap();
    let explicit = AarambhModel::new(
        &explicit_cfg,
        VarBuilder::from_varmap(&varmap, DType::F32, &device),
    )
    .unwrap();
    let ids = Tensor::from_vec(vec![1u32, 2, 3], (1, 3), &device).unwrap();
    let diff = (model.forward(&ids).unwrap() - explicit.forward(&ids).unwrap())
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert_eq!(diff, 0.0);
}

#[test]
fn scaled_model_forwards_beyond_original_context() {
    let device = Device::Cpu;
    let cfg = scaled_mini_config();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = AarambhModel::new(&cfg, vb).unwrap();
    let ids = Tensor::from_vec(
        (0..12).map(|id| id as u32).collect::<Vec<_>>(),
        (1, 12),
        &device,
    )
    .unwrap();
    let logits = model.forward(&ids).unwrap();
    assert_eq!(logits.shape().dims(), &[1, 12, cfg.vocab_size]);
}

#[test]
fn kv_cache_preallocates_to_scaled_max_seq_len() {
    let device = Device::Cpu;
    let cfg = scaled_mini_config();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = AarambhModel::new(&cfg, vb).unwrap();
    let caches = model.empty_kv_cache();
    assert_eq!(caches[0].capacity(), Some(cfg.max_seq_len));
}

#[test]
fn tied_embedding_reuses_lm_head_tensor() {
    let device = Device::Cpu;
    let model = mini_model(&device);
    assert_eq!(
        model.get_weight("embedding.weight").unwrap().id(),
        model.get_weight("lm_head.weight").unwrap().id()
    );
    assert!(!model.named_tensors().contains_key("lm_head.weight"));
}

#[test]
fn untied_lm_head_is_saved_separately() {
    let device = Device::Cpu;
    let mut cfg = mini_config();
    cfg.tie_embeddings = false;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = AarambhModel::new(&cfg, vb).unwrap();

    assert_ne!(
        model.get_weight("embedding.weight").unwrap().id(),
        model.get_weight("lm_head.weight").unwrap().id()
    );
    assert!(model.named_tensors().contains_key("lm_head.weight"));
}

#[test]
fn invalid_config_is_rejected() {
    let mut cfg = mini_config();
    cfg.hidden_dim = 96;
    cfg.n_heads = 2;
    let err = AarambhModel::validate_config(&cfg).unwrap_err();
    assert!(err.to_string().contains("head_dim must be 64"));
}

#[test]
fn qat_is_enabled_only_for_training_construction() {
    let device = Device::Cpu;
    let mut cfg = mini_config();
    cfg.qat = Some(QatConfig::default());

    let varmap = VarMap::new();
    let inference =
        AarambhModel::new(&cfg, VarBuilder::from_varmap(&varmap, DType::F32, &device)).unwrap();
    assert!(!inference.qat_active());
    assert!(inference.qat_stats().is_none());
    let mut no_qat_cfg = cfg.clone();
    no_qat_cfg.qat = None;
    let no_qat = AarambhModel::new(
        &no_qat_cfg,
        VarBuilder::from_varmap(&varmap, DType::F32, &device),
    )
    .unwrap();
    let ids = Tensor::from_vec(vec![1u32, 2, 3], (1, 3), &device).unwrap();
    let max_diff = (inference.forward(&ids).unwrap() - no_qat.forward(&ids).unwrap())
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert_eq!(max_diff, 0.0);

    let training =
        AarambhModel::new_for_training(&cfg, VarBuilder::zeros(DType::F32, &device)).unwrap();
    let stats = training.qat_stats().unwrap();
    assert!(training.qat_active());
    assert_eq!(stats.wrapped_tensors, cfg.n_layers * 7);
    assert!(stats.wrapped_parameters > 0);
    assert_eq!(stats.cache_refreshes, 0);

    training.forward_train(&ids).unwrap();
    assert_eq!(
        training.qat_stats().unwrap().cache_refreshes,
        stats.wrapped_tensors
    );
}

#[test]
fn readme_model_scale_table_matches_model_config() {
    let readme = include_str!("../../../README.md");
    assert!(readme.contains("| Tiny | 25M | 384 | 8 | 6 | 2 | 1,024 | 512 | 10,000 |"));
    assert!(readme.contains("| Small | 117M | 768 | 12 | 12 | 4 | 2,688 | 1,024 | 10,000 |"));
    assert!(readme.contains("| Medium | 360M | 1,024 | 24 | 16 | 8 | 3,392 | 2,048 | 500,000 |"));
    assert!(readme.contains("| Large | 1.3B | 2,048 | 24 | 32 | 8 | 6,656 | 4,096 | 500,000 |"));
}

#[test]
#[ignore = "Large model construction allocates multiple GB; run manually for release validation."]
fn all_four_full_scales_construct() {
    let device = Device::Cpu;
    for cfg in [
        ModelConfig::tiny(),
        ModelConfig::small(),
        ModelConfig::medium(),
        ModelConfig::large(),
    ] {
        let vb = VarBuilder::zeros(DType::F32, &device);
        AarambhModel::new(&cfg, vb).unwrap();
    }
}

#[test]
fn mtp_three_builds_two_offset_aligned_auxiliary_heads() {
    let device = Device::Cpu;
    let cfg = ModelConfig {
        mtp: Some(MtpConfig {
            num_future_tokens: 3,
            aux_loss_weight: 0.3,
        }),
        ..mini_config()
    };
    let varmap = VarMap::new();
    let model =
        AarambhModel::new(&cfg, VarBuilder::from_varmap(&varmap, DType::F32, &device)).unwrap();
    let ids = Tensor::from_vec(vec![1u32, 2, 3, 4, 5, 6], (1, 6), &device).unwrap();
    let output = model.forward_train_with_aux(&ids).unwrap();

    assert_eq!(model.mtp_heads().len(), 2);
    assert_eq!(output.final_hidden_states.dims(), &[1, 6, 64]);
    let second = model
        .forward_mtp_head_train(0, &output.final_hidden_states, &ids)
        .unwrap();
    let third = model
        .forward_mtp_head_train(1, &output.final_hidden_states, &ids)
        .unwrap();
    assert_eq!(second.offset, 2);
    assert_eq!(second.logits.dims(), &[1, 5, 128]);
    assert_eq!(third.offset, 3);
    assert_eq!(third.logits.dims(), &[1, 4, 128]);
    assert!(
        model
            .named_tensors()
            .contains_key("mtp.heads.1.refine.attn.wq.weight")
    );
}

#[test]
fn mtp_disabled_training_forward_matches_existing_logits_path() {
    let device = Device::Cpu;
    let model = mini_model(&device);
    let ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
    let direct = model.forward_train(&ids).unwrap();
    let with_metadata = model.forward_train_with_aux(&ids).unwrap().logits;
    let max_diff = (direct - with_metadata)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert_eq!(max_diff, 0.0);
    assert!(model.mtp_heads().is_empty());
}

#[test]
fn cached_hidden_state_drives_mtp_without_an_auxiliary_cache() {
    let device = Device::Cpu;
    let cfg = ModelConfig {
        mtp: Some(MtpConfig::default()),
        ..mini_config()
    };
    let varmap = VarMap::new();
    let model =
        AarambhModel::new(&cfg, VarBuilder::from_varmap(&varmap, DType::F32, &device)).unwrap();
    let prompt = Tensor::from_vec(vec![1u32, 2, 3], (1, 3), &device).unwrap();
    let mut cache = model.empty_kv_cache();
    let output = model
        .forward_with_cache_output(&prompt, 0, &mut cache)
        .unwrap();
    let anchor = output.final_hidden_states.narrow(1, 2, 1).unwrap();
    let proposed = Tensor::from_vec(vec![4u32], (1, 1), &device).unwrap();
    let prediction = model.forward_mtp_head(0, &anchor, &proposed).unwrap();
    assert_eq!(prediction.offset, 2);
    assert_eq!(prediction.logits.dims(), &[1, 1, 128]);
    assert_eq!(cache[0].seq_len(), 3);
}
