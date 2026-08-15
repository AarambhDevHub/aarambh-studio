use std::time::{SystemTime, UNIX_EPOCH};

use aarambh_studio_core::{
    Configurable, DsaConfig, GatedDeltaNetConfig, HybridAttentionSchedule, MlaConfig, ModelConfig,
    MoeConfig, MtpConfig,
};
use aarambh_studio_model::AarambhModel;
use aarambh_studio_weights::{
    GgufFormat, MoeRetrofitOptions, load_gguf, load_model, load_retrofit_into_varmap,
    load_retrofit_into_varmap_with_moe, save_gguf, save_model,
};
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

fn mtp_mini_config() -> ModelConfig {
    ModelConfig {
        mtp: Some(MtpConfig {
            num_future_tokens: 3,
            aux_loss_weight: 0.3,
        }),
        ..mini_config()
    }
}

fn temp_safetensors_path() -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "aarambh-studio-model-{}-{nanos}.safetensors",
        std::process::id()
    ))
}

fn temp_gguf_path() -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "aarambh-studio-model-{}-{nanos}.gguf",
        std::process::id()
    ))
}

#[test]
fn safetensors_roundtrip_preserves_weights_and_logits() {
    let device = Device::Cpu;
    let cfg = mini_config();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = AarambhModel::new(&cfg, vb).unwrap();

    let path = temp_safetensors_path();
    save_model(&model, &path).unwrap();
    let loaded = load_model(&path, &cfg, &device).unwrap();
    let _ = std::fs::remove_file(&path);

    let w1 = model.get_weight("blocks.0.attn.wq.weight").unwrap();
    let w2 = loaded.get_weight("blocks.0.attn.wq.weight").unwrap();
    let weight_diff = (w1 - w2)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(weight_diff < 1e-6, "weight diff: {weight_diff}");

    let ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
    let logits1 = model.forward(&ids).unwrap();
    let logits2 = loaded.forward(&ids).unwrap();
    let logits_diff = (logits1 - logits2)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(logits_diff < 1e-6, "logits diff: {logits_diff}");
}

#[test]
fn mtp_safetensors_roundtrip_preserves_all_auxiliary_heads() {
    let device = Device::Cpu;
    let cfg = mtp_mini_config();
    let vars = VarMap::new();
    let model =
        AarambhModel::new(&cfg, VarBuilder::from_varmap(&vars, DType::F32, &device)).unwrap();
    let path = temp_safetensors_path();
    save_model(&model, &path).unwrap();
    let loaded = load_model(&path, &cfg, &device).unwrap();
    let _ = std::fs::remove_file(path);

    assert_eq!(loaded.mtp_heads().len(), 2);
    let name = "mtp.heads.1.refine.ffn.w_down.weight";
    let diff = (model.get_weight(name).unwrap() - loaded.get_weight(name).unwrap())
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(diff < 1e-6, "MTP SafeTensors weight diff: {diff}");
}

#[test]
fn gguf_save_load_roundtrip_produces_logits() {
    let device = Device::Cpu;
    let cfg = mini_config();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = AarambhModel::new(&cfg, vb).unwrap();

    let path = temp_gguf_path();
    save_gguf(&model, GgufFormat::Q4KM, &path).unwrap();
    let loaded = load_gguf(&path, &device).unwrap();
    let _ = std::fs::remove_file(&path);

    let ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
    let logits = loaded.forward(&ids).unwrap();
    assert_eq!(logits.shape().dims(), &[1, 4, cfg.vocab_size]);
    let max = logits
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(max.is_finite());
}

#[test]
fn mtp_gguf_roundtrip_preserves_config_and_heads() {
    let device = Device::Cpu;
    let cfg = mtp_mini_config();
    let vars = VarMap::new();
    let model =
        AarambhModel::new(&cfg, VarBuilder::from_varmap(&vars, DType::F32, &device)).unwrap();
    let path = temp_gguf_path();
    save_gguf(&model, GgufFormat::Q4KM, &path).unwrap();
    let loaded = load_gguf(&path, &device).unwrap();
    let _ = std::fs::remove_file(path);

    assert_eq!(loaded.config().mtp, cfg.mtp);
    assert_eq!(loaded.mtp_heads().len(), 2);
    assert!(
        loaded
            .get_weight("mtp.heads.0.refine.attn.wq.weight")
            .is_some()
    );
}

#[test]
fn dense_retrofit_initializes_complete_mtp_heads_without_changing_main_logits() {
    let device = Device::Cpu;
    let dense_cfg = mini_config();
    let dense_vars = VarMap::new();
    let dense = AarambhModel::new(
        &dense_cfg,
        VarBuilder::from_varmap(&dense_vars, DType::F32, &device),
    )
    .unwrap();
    let path = temp_safetensors_path();
    save_model(&dense, &path).unwrap();

    let mtp_cfg = mtp_mini_config();
    let mut mtp_vars = VarMap::new();
    let mtp_model = AarambhModel::new(
        &mtp_cfg,
        VarBuilder::from_varmap(&mtp_vars, DType::F32, &device),
    )
    .unwrap();
    let expected_mtp_tensors = mtp_model
        .named_tensors()
        .keys()
        .filter(|name| name.starts_with("mtp."))
        .count();
    let report =
        load_retrofit_into_varmap(&path, &mtp_cfg, &mut mtp_vars, &device, DType::F32).unwrap();
    let _ = std::fs::remove_file(path);
    assert_eq!(report.initialized_mtp_tensors, expected_mtp_tensors);

    let ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
    let diff = (dense.forward(&ids).unwrap() - mtp_model.forward(&ids).unwrap())
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(diff < 1e-6, "MTP retrofit changed trunk logits by {diff}");
}

#[test]
fn retrofit_rejects_partial_mtp_tensor_sets() {
    let device = Device::Cpu;
    let cfg = mtp_mini_config();
    let source_vars = VarMap::new();
    let source = AarambhModel::new(
        &cfg,
        VarBuilder::from_varmap(&source_vars, DType::F32, &device),
    )
    .unwrap();
    let path = temp_safetensors_path();
    let mut tensors = source.named_tensors();
    tensors.remove("mtp.heads.0.trunk_norm.weight");
    candle_core::safetensors::save(&tensors, &path).unwrap();

    let mut target_vars = VarMap::new();
    let _target = AarambhModel::new(
        &cfg,
        VarBuilder::from_varmap(&target_vars, DType::F32, &device),
    )
    .unwrap();
    let result = load_retrofit_into_varmap(&path, &cfg, &mut target_vars, &device, DType::F32);
    let _ = std::fs::remove_file(path);
    assert!(result.is_err());
}

#[test]
fn moe_safetensors_roundtrip_preserves_logits() {
    let device = Device::Cpu;
    let cfg = moe_mini_config();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = AarambhModel::new(&cfg, vb).unwrap();

    let path = temp_safetensors_path();
    save_model(&model, &path).unwrap();
    let loaded = load_model(&path, &cfg, &device).unwrap();
    let _ = std::fs::remove_file(&path);

    assert!(
        loaded
            .get_weight("blocks.1.ffn.experts.0.w_gate.weight")
            .is_some()
    );
    let ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
    let logits1 = model.forward(&ids).unwrap();
    let logits2 = loaded.forward(&ids).unwrap();
    let logits_diff = (logits1 - logits2)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(logits_diff < 1e-6, "logits diff: {logits_diff}");
}

#[test]
fn moe_gguf_roundtrip_produces_logits() {
    let device = Device::Cpu;
    let cfg = moe_mini_config();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = AarambhModel::new(&cfg, vb).unwrap();

    let path = temp_gguf_path();
    save_gguf(&model, GgufFormat::Q4KM, &path).unwrap();
    let loaded = load_gguf(&path, &device).unwrap();
    let _ = std::fs::remove_file(&path);

    let ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
    let logits = loaded.forward(&ids).unwrap();
    assert_eq!(logits.shape().dims(), &[1, 4, cfg.vocab_size]);
    assert!(
        loaded
            .get_weight("blocks.1.ffn.experts.0.w_gate.weight")
            .is_some()
    );
}

#[test]
fn fine_grained_moe_gguf_roundtrip_preserves_shared_experts() {
    let device = Device::Cpu;
    let cfg = fine_moe_mini_config();
    let varmap = VarMap::new();
    let model =
        AarambhModel::new(&cfg, VarBuilder::from_varmap(&varmap, DType::F32, &device)).unwrap();
    let path = temp_gguf_path();
    save_gguf(&model, GgufFormat::Q4KM, &path).unwrap();
    let loaded = load_gguf(&path, &device).unwrap();
    let _ = std::fs::remove_file(path);
    let moe = loaded.config().moe.as_ref().unwrap();
    assert_eq!(moe.fine_grained_factor, 2);
    assert_eq!(moe.num_shared_experts, 1);
    assert!(
        loaded
            .get_weight("blocks.1.ffn.shared_experts.0.w_gate.weight")
            .is_some()
    );
    let ids = Tensor::from_vec(vec![1u32, 2, 3], (1, 3), &device).unwrap();
    assert_eq!(loaded.forward(&ids).unwrap().dims(), [1, 3, 128]);
}

#[test]
fn coarse_moe_retrofit_preserves_function_and_zero_starts_shared_output() {
    let device = Device::Cpu;
    let source_cfg = moe_mini_config();
    let source_vars = VarMap::new();
    let source = AarambhModel::new(
        &source_cfg,
        VarBuilder::from_varmap(&source_vars, DType::F32, &device),
    )
    .unwrap();
    let path = temp_safetensors_path();
    save_model(&source, &path).unwrap();

    let target_cfg = fine_moe_mini_config();
    let mut target_vars = VarMap::new();
    let target = AarambhModel::new(
        &target_cfg,
        VarBuilder::from_varmap(&target_vars, DType::F32, &device),
    )
    .unwrap();
    let report = load_retrofit_into_varmap_with_moe(
        &path,
        &target_cfg,
        &mut target_vars,
        &device,
        DType::F32,
        Some(MoeRetrofitOptions { source_top_k: 2 }),
    )
    .unwrap();
    let _ = std::fs::remove_file(path);
    assert_eq!(report.expanded_moe_router_tensors, 1);
    assert_eq!(report.sharded_moe_expert_tensors, 24);
    assert_eq!(report.initialized_shared_expert_tensors, 3);

    let shared_down = target
        .get_weight("blocks.1.ffn.shared_experts.0.w_down.weight")
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert_eq!(shared_down, 0.0);

    let ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
    let diff = (source.forward(&ids).unwrap() - target.forward(&ids).unwrap())
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(
        diff < 1e-5,
        "coarse-to-fine retrofit logits differ by {diff}"
    );
}

#[test]
fn retrofit_load_preserves_full_layers_and_initializes_deltanet() {
    let device = Device::Cpu;
    let dense_cfg = mini_config();
    let dense_vars = VarMap::new();
    let dense = AarambhModel::new(
        &dense_cfg,
        VarBuilder::from_varmap(&dense_vars, DType::F32, &device),
    )
    .unwrap();
    let path = temp_safetensors_path();
    save_model(&dense, &path).unwrap();

    let hybrid_cfg = hybrid_mini_config();
    let mut hybrid_vars = VarMap::new();
    let hybrid = AarambhModel::new(
        &hybrid_cfg,
        VarBuilder::from_varmap(&hybrid_vars, DType::F32, &device),
    )
    .unwrap();
    let report =
        load_retrofit_into_varmap(&path, &hybrid_cfg, &mut hybrid_vars, &device, DType::F32)
            .unwrap();
    let _ = std::fs::remove_file(&path);

    assert!(report.loaded_tensors > 0);
    assert_eq!(report.initialized_deltanet_tensors, 13);
    let source = dense.get_weight("blocks.0.attn.wq.weight").unwrap();
    let loaded = hybrid.get_weight("blocks.0.attn.wq.weight").unwrap();
    let diff = (source - loaded)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    assert!(diff < 1e-6, "retrofit full-layer weight diff: {diff}");
    assert!(
        hybrid
            .get_weight("blocks.1.deltanet.q_proj.weight")
            .is_some()
    );
    assert!(hybrid.get_weight("blocks.1.attn.wq.weight").is_none());
}

#[test]
fn partial_checkpoint_load_preserves_non_mla_layer_weights_exactly() {
    let device = Device::Cpu;
    // Source: a dense all-Full v3 checkpoint.
    let dense_cfg = mini_config();
    let dense_vars = VarMap::new();
    let dense = AarambhModel::new(
        &dense_cfg,
        VarBuilder::from_varmap(&dense_vars, DType::F32, &device),
    )
    .unwrap();
    let path = temp_safetensors_path();
    save_model(&dense, &path).unwrap();

    // Target: layer 0 = LatentMLA, layer 1 = Gated DeltaNet.
    let mla_cfg = mla_mini_config();
    let mut mla_vars = VarMap::new();
    let mla = AarambhModel::new(
        &mla_cfg,
        VarBuilder::from_varmap(&mla_vars, DType::F32, &device),
    )
    .unwrap();
    let report =
        load_retrofit_into_varmap(&path, &mla_cfg, &mut mla_vars, &device, DType::F32).unwrap();
    let _ = std::fs::remove_file(&path);

    // Shared (non-MLA, non-deltanet) tensors load bit-exactly from the source.
    assert!(report.loaded_tensors > 0, "no shared tensors loaded");
    for name in [
        "embedding.weight",
        "blocks.0.norm1.weight",
        "blocks.0.ffn.w_gate.weight",
        "blocks.1.norm2.weight",
        "final_norm.weight",
    ] {
        let source = dense.get_weight(name).unwrap();
        let loaded = mla.get_weight(name).unwrap();
        let diff = (source - loaded)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff < 1e-6,
            "non-MLA tensor {name} changed during retrofit: {diff}"
        );
    }
    // MLA layer 0 tensors were freshly initialized (7 weights).
    assert_eq!(
        report.initialized_mla_tensors, 7,
        "expected 7 MLA tensors initialized"
    );
    // Gated DeltaNet layer 1 tensors were freshly initialized (13 weights).
    assert_eq!(report.initialized_deltanet_tensors, 13);
    // The MLA layer is present in the retrofitted model.
    assert!(mla.get_weight("blocks.0.mla.q_proj.weight").is_some());
    assert!(mla.get_weight("blocks.0.mla.kv_a_proj.weight").is_some());
    assert!(mla.get_weight("blocks.0.mla.k_rope_proj.weight").is_some());
    // The original Full-attention tensors are gone (replaced by MLA).
    assert!(mla.get_weight("blocks.0.attn.wq.weight").is_none());
    // The retrofitted model still forwards end to end.
    let ids = Tensor::from_vec(vec![1u32, 2, 3], (1, 3), &device).unwrap();
    assert_eq!(mla.forward(&ids).unwrap().dims(), [1, 3, 128]);
}

#[test]
fn hybrid_gguf_roundtrip_keeps_float_recurrent_parameters() {
    let device = Device::Cpu;
    let cfg = hybrid_mini_config();
    let varmap = VarMap::new();
    let model =
        AarambhModel::new(&cfg, VarBuilder::from_varmap(&varmap, DType::F32, &device)).unwrap();
    let path = temp_gguf_path();
    save_gguf(&model, GgufFormat::Q4KM, &path).unwrap();
    let loaded = load_gguf(&path, &device).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(
        loaded
            .get_weight("blocks.1.deltanet.A_log")
            .unwrap()
            .dtype(),
        DType::F32
    );
    let ids = Tensor::from_vec(vec![1u32, 2, 3], (1, 3), &device).unwrap();
    assert_eq!(loaded.forward(&ids).unwrap().dims(), [1, 3, 128]);
}

#[test]
fn phase29_retrofit_initializes_only_dsa_indexer_tensors() {
    let device = Device::Cpu;
    let source_cfg = hybrid_mini_config();
    let source_vars = VarMap::new();
    let source = AarambhModel::new(
        &source_cfg,
        VarBuilder::from_varmap(&source_vars, DType::F32, &device),
    )
    .unwrap();
    let path = temp_safetensors_path();
    save_model(&source, &path).unwrap();

    let dsa_cfg = dsa_mini_config();
    let mut dsa_vars = VarMap::new();
    let dsa = AarambhModel::new(
        &dsa_cfg,
        VarBuilder::from_varmap(&dsa_vars, DType::F32, &device),
    )
    .unwrap();
    let report =
        load_retrofit_into_varmap(&path, &dsa_cfg, &mut dsa_vars, &device, DType::F32).unwrap();
    let _ = std::fs::remove_file(path);
    assert_eq!(report.initialized_deltanet_tensors, 0);
    assert_eq!(report.initialized_dsa_tensors, 2);
    assert!(dsa.get_weight("blocks.0.dsa.index_q.weight").is_some());
    let diff = (source.get_weight("blocks.0.attn.wq.weight").unwrap()
        - dsa.get_weight("blocks.0.attn.wq.weight").unwrap())
    .unwrap()
    .abs()
    .unwrap()
    .max_all()
    .unwrap()
    .to_scalar::<f32>()
    .unwrap();
    assert!(diff < 1e-6, "DSA retrofit attention diff: {diff}");
}

#[test]
fn dsa_gguf_roundtrip_preserves_config_and_indexers() {
    let device = Device::Cpu;
    let cfg = dsa_mini_config();
    let vars = VarMap::new();
    let model =
        AarambhModel::new(&cfg, VarBuilder::from_varmap(&vars, DType::F32, &device)).unwrap();
    let path = temp_gguf_path();
    save_gguf(&model, GgufFormat::Q4KM, &path).unwrap();
    let loaded = load_gguf(&path, &device).unwrap();
    let _ = std::fs::remove_file(path);
    assert_eq!(loaded.config().dsa_config, cfg.dsa_config);
    assert!(loaded.get_weight("blocks.0.dsa.index_q.weight").is_some());
    let ids = Tensor::from_vec(
        (0..32).map(|value| (value % 127 + 1) as u32).collect(),
        (1, 32),
        &device,
    )
    .unwrap();
    assert_eq!(loaded.forward(&ids).unwrap().dims(), [1, 32, 128]);
}
