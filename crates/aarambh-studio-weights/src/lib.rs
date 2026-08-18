//! Weight loading, saving, conversion, and GGUF serialization helpers.
#![deny(missing_docs)]

use std::path::Path;

/// HuggingFace conversion helpers.
pub mod convert;
/// GGUF checkpoint reader and writer.
pub mod gguf;
/// Model merging and weight averaging (Phase 50).
pub mod merge;
/// Vocabulary-row migration helpers.
pub mod vocab;

use aarambh_studio_core::{ModelConfig, Result};
use aarambh_studio_model::AarambhModel;
pub use aarambh_studio_quant::GgufFormat;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

pub use convert::{HfArch, convert_hf, convert_hf_tensors, convert_hf_with_arch};
pub use gguf::{load_gguf, load_gguf_tensors, load_gguf_with_dtype, save_gguf};
pub use merge::{
    DEFAULT_DENSITY, MergeConfig, MergeMethod, MergeReport, SLERP_PARALLEL_EPSILON,
    merge_models_from_paths,
};
pub use vocab::{VocabularyExpansion, VocabularyExpansionReport, expand_safetensors_vocabulary};

#[derive(Debug, Clone, PartialEq, Eq)]
/// Summary of tensors copied and initialized during a hybrid retrofit.
pub struct RetrofitLoadReport {
    /// Number of existing checkpoint tensors copied into the hybrid model.
    pub loaded_tensors: usize,
    /// Number of new Gated DeltaNet tensors left at their fresh initialization.
    pub initialized_deltanet_tensors: usize,
    /// Number of new DSA indexer tensors left at their fresh initialization.
    pub initialized_dsa_tensors: usize,
    /// Number of new Multi-Head Latent Attention tensors left at their fresh
    /// initialization (v4 Phase 41).
    pub initialized_mla_tensors: usize,
    /// Number of coarse router tensors expanded across fine-grained children.
    pub expanded_moe_router_tensors: usize,
    /// Number of coarse expert tensors sharded into fine-grained children.
    pub sharded_moe_expert_tensors: usize,
    /// Number of new shared-expert tensors initialized by the retrofit.
    pub initialized_shared_expert_tensors: usize,
    /// Number of new MTP tensors initialized by the retrofit.
    pub initialized_mtp_tensors: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Options for a function-preserving coarse-to-fine MoE retrofit.
pub struct MoeRetrofitOptions {
    /// Number of experts selected by the source coarse router.
    pub source_top_k: usize,
}

/// Save an Aarambh model as a safetensors checkpoint.
pub fn save_model(model: &AarambhModel, path: impl AsRef<Path>) -> Result<()> {
    candle_core::safetensors::save(&model.named_tensors(), path.as_ref())?;
    Ok(())
}

/// Load a safetensors checkpoint as an Aarambh model using f32 parameters.
pub fn load_model(
    path: impl AsRef<Path>,
    cfg: &ModelConfig,
    device: &Device,
) -> Result<AarambhModel> {
    load_model_with_dtype(path, cfg, device, DType::F32)
}

/// Load a safetensors checkpoint as an Aarambh model using the requested dtype.
pub fn load_model_with_dtype(
    path: impl AsRef<Path>,
    cfg: &ModelConfig,
    device: &Device,
    dtype: DType,
) -> Result<AarambhModel> {
    let path = path.as_ref();
    // SAFETY: Aarambh only reads the checkpoint mapping while constructing owned
    // tensors, and never mutates checkpoint files during this load operation.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], dtype, device)? };
    AarambhModel::new(cfg, vb)
}

/// Copy an exact SafeTensors model into an initialized training variable map.
///
/// Source and target tensor names and shapes must match exactly. This is used
/// when QAT starts from a floating-point checkpoint without changing model
/// architecture.
pub fn load_exact_into_varmap(
    path: impl AsRef<Path>,
    varmap: &mut VarMap,
    device: &Device,
    dtype: DType,
) -> Result<usize> {
    let source = candle_core::safetensors::load(path.as_ref(), device)?;
    let variables = varmap.data().lock().map_err(|_| {
        aarambh_studio_core::AarambhError::Checkpoint("training variable map lock poisoned".into())
    })?;
    let missing = variables
        .keys()
        .filter(|name| !source.contains_key(name.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    let unexpected = source
        .keys()
        .filter(|name| !variables.contains_key(name.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    if !missing.is_empty() || !unexpected.is_empty() {
        return Err(aarambh_studio_core::AarambhError::Checkpoint(format!(
            "exact checkpoint tensor mismatch: missing={missing:?} unexpected={unexpected:?}"
        )));
    }

    for (name, variable) in variables.iter() {
        let value = source
            .get(name)
            .expect("exact tensor name sets were validated");
        if value.dims() != variable.dims() {
            return Err(aarambh_studio_core::AarambhError::Checkpoint(format!(
                "exact checkpoint tensor {name} shape {:?} does not match {:?}",
                value.dims(),
                variable.dims()
            )));
        }
        let target_dtype = if name.ends_with(".A_log") || name.ends_with(".dt_bias") {
            DType::F32
        } else {
            dtype
        };
        variable.set(&value.to_dtype(target_dtype)?)?;
    }
    Ok(variables.len())
}

/// Copy a dense SafeTensors checkpoint into an initialized hybrid-model variable map.
///
/// All embedding, normalization, FFN/MoE, output-head, and scheduled full-attention
/// parameters must exist and match shape. New `deltanet`, `dsa`, and complete
/// `mtp` parameter sets may be absent.
pub fn load_retrofit_into_varmap(
    path: impl AsRef<Path>,
    cfg: &ModelConfig,
    varmap: &mut VarMap,
    device: &Device,
    dtype: DType,
) -> Result<RetrofitLoadReport> {
    load_retrofit_into_varmap_with_moe(path, cfg, varmap, device, dtype, None)
}

/// Copy a checkpoint into a hybrid model and optionally expand a coarse MoE pool.
pub fn load_retrofit_into_varmap_with_moe(
    path: impl AsRef<Path>,
    cfg: &ModelConfig,
    varmap: &mut VarMap,
    device: &Device,
    dtype: DType,
    moe_options: Option<MoeRetrofitOptions>,
) -> Result<RetrofitLoadReport> {
    if cfg.attention_schedule.is_none() && cfg.mtp.is_none() && moe_options.is_none() {
        return Err(aarambh_studio_core::AarambhError::Config(
            "retrofit loading requires model.attention_schedule, model.mtp, or moe_retrofit options"
                .into(),
        ));
    }
    if let Some(options) = moe_options {
        validate_moe_retrofit(cfg, options)?;
    }
    let source = candle_core::safetensors::load(path.as_ref(), device)?;
    let mut loaded_tensors = 0usize;
    let mut initialized_deltanet_tensors = 0usize;
    let mut initialized_dsa_tensors = 0usize;
    let mut initialized_mla_tensors = 0usize;
    let mut expanded_moe_router_tensors = 0usize;
    let mut sharded_moe_expert_tensors = 0usize;
    let mut initialized_shared_expert_tensors = 0usize;
    let mut initialized_mtp_tensors = 0usize;
    let variables = varmap.data().lock().unwrap();
    let target_mtp_names = variables
        .keys()
        .filter(|name| name.starts_with("mtp."))
        .collect::<Vec<_>>();
    let present_mtp_tensors = target_mtp_names
        .iter()
        .filter(|name| source.contains_key(name.as_str()))
        .count();
    if present_mtp_tensors > 0 && present_mtp_tensors != target_mtp_names.len() {
        return Err(aarambh_studio_core::AarambhError::Checkpoint(format!(
            "retrofit source contains a partial MTP tensor set: found {present_mtp_tensors} of {} required tensors",
            target_mtp_names.len()
        )));
    }
    let initialize_mtp = !target_mtp_names.is_empty() && present_mtp_tensors == 0;
    for (name, variable) in variables.iter() {
        if moe_options.is_some() {
            if name.ends_with(".ffn.router.weight") {
                let source_router = source.get(name).ok_or_else(|| {
                    aarambh_studio_core::AarambhError::Checkpoint(format!(
                        "MoE retrofit source is missing coarse router tensor {name}"
                    ))
                })?;
                let expanded = expand_router(source_router, variable.as_tensor(), cfg, name)?;
                variable.set(&expanded.to_dtype(dtype)?)?;
                expanded_moe_router_tensors += 1;
                continue;
            }
            if let Some((source_name, child_idx, projection)) = coarse_expert_source(name, cfg)? {
                let source_weight = source.get(&source_name).ok_or_else(|| {
                    aarambh_studio_core::AarambhError::Checkpoint(format!(
                        "MoE retrofit source is missing coarse expert tensor {source_name} for {name}"
                    ))
                })?;
                let sharded = shard_expert_weight(
                    source_weight,
                    variable.as_tensor(),
                    cfg,
                    child_idx,
                    projection,
                    name,
                )?;
                variable.set(&sharded.to_dtype(dtype)?)?;
                sharded_moe_expert_tensors += 1;
                continue;
            }
            if name.contains(".ffn.shared_experts.") && !source.contains_key(name) {
                if name.ends_with(".w_down.weight") {
                    variable.set(&Tensor::zeros(
                        variable.shape(),
                        variable.dtype(),
                        variable.device(),
                    )?)?;
                }
                initialized_shared_expert_tensors += 1;
                continue;
            }
        }
        match source.get(name) {
            Some(value) => {
                if value.dims() != variable.dims() {
                    return Err(aarambh_studio_core::AarambhError::Checkpoint(format!(
                        "retrofit tensor {name} shape {:?} does not match hybrid shape {:?}",
                        value.dims(),
                        variable.dims()
                    )));
                }
                let target_dtype = if name.ends_with(".A_log") || name.ends_with(".dt_bias") {
                    DType::F32
                } else {
                    dtype
                };
                variable.set(&value.to_dtype(target_dtype)?)?;
                loaded_tensors += 1;
            }
            None if name.contains(".deltanet.") => {
                initialized_deltanet_tensors += 1;
            }
            None if name.contains(".dsa.") => {
                initialized_dsa_tensors += 1;
            }
            None if name.contains(".mla.") => {
                initialized_mla_tensors += 1;
            }
            None if name.starts_with("mtp.") && initialize_mtp => {
                initialized_mtp_tensors += 1;
            }
            None => {
                return Err(aarambh_studio_core::AarambhError::Checkpoint(format!(
                    "retrofit source is missing required tensor {name}"
                )));
            }
        }
    }
    drop(variables);
    Ok(RetrofitLoadReport {
        loaded_tensors,
        initialized_deltanet_tensors,
        initialized_dsa_tensors,
        initialized_mla_tensors,
        expanded_moe_router_tensors,
        sharded_moe_expert_tensors,
        initialized_shared_expert_tensors,
        initialized_mtp_tensors,
    })
}

fn validate_moe_retrofit(cfg: &ModelConfig, options: MoeRetrofitOptions) -> Result<()> {
    let moe = cfg.moe.as_ref().ok_or_else(|| {
        aarambh_studio_core::AarambhError::Config(
            "moe_retrofit requires model.moe to be configured".into(),
        )
    })?;
    if moe.fine_grained_factor <= 1 {
        return Err(aarambh_studio_core::AarambhError::Config(
            "moe_retrofit requires fine_grained_factor greater than one".into(),
        ));
    }
    if options.source_top_k == 0 {
        return Err(aarambh_studio_core::AarambhError::Config(
            "moe_retrofit.source_top_k must be non-zero".into(),
        ));
    }
    let expected_top_k = options
        .source_top_k
        .checked_mul(moe.fine_grained_factor)
        .ok_or_else(|| {
            aarambh_studio_core::AarambhError::Config(
                "moe_retrofit source_top_k scaling overflows usize".into(),
            )
        })?;
    if moe.top_k != expected_top_k {
        return Err(aarambh_studio_core::AarambhError::Config(format!(
            "function-preserving MoE retrofit requires target top_k={expected_top_k}, got {}",
            moe.top_k
        )));
    }
    Ok(())
}

fn expand_router(
    source: &Tensor,
    target: &Tensor,
    cfg: &ModelConfig,
    name: &str,
) -> Result<Tensor> {
    let moe = cfg.moe.as_ref().expect("validated MoE retrofit config");
    let factor = moe.fine_grained_factor;
    let (source_experts, source_hidden) = source.dims2()?;
    let (target_experts, target_hidden) = target.dims2()?;
    if source_experts != moe.num_experts
        || target_experts != moe.routed_expert_count()?
        || source_hidden != target_hidden
    {
        return Err(aarambh_studio_core::AarambhError::Checkpoint(format!(
            "cannot expand router {name}: source {:?}, target {:?}, expected {} coarse groups and factor {factor}",
            source.dims(),
            target.dims(),
            moe.num_experts
        )));
    }
    Ok(source
        .unsqueeze(1)?
        .broadcast_as((source_experts, factor, source_hidden))?
        .contiguous()?
        .reshape((target_experts, target_hidden))?)
}

fn coarse_expert_source(
    target_name: &str,
    cfg: &ModelConfig,
) -> Result<Option<(String, usize, &'static str)>> {
    let Some((prefix, suffix)) = target_name.split_once(".ffn.experts.") else {
        return Ok(None);
    };
    let Some((fine_idx, projection)) = suffix.split_once('.') else {
        return Ok(None);
    };
    let projection = match projection {
        "w_gate.weight" => "w_gate",
        "w_up.weight" => "w_up",
        "w_down.weight" => "w_down",
        _ => return Ok(None),
    };
    let fine_idx = fine_idx.parse::<usize>().map_err(|err| {
        aarambh_studio_core::AarambhError::Checkpoint(format!(
            "invalid fine expert index in {target_name}: {err}"
        ))
    })?;
    let moe = cfg.moe.as_ref().expect("validated MoE retrofit config");
    if fine_idx >= moe.routed_expert_count()? {
        return Err(aarambh_studio_core::AarambhError::Checkpoint(format!(
            "fine expert index {fine_idx} exceeds configured routed pool"
        )));
    }
    let coarse_idx = fine_idx / moe.fine_grained_factor;
    let child_idx = fine_idx % moe.fine_grained_factor;
    Ok(Some((
        format!("{prefix}.ffn.experts.{coarse_idx}.{projection}.weight"),
        child_idx,
        projection,
    )))
}

fn shard_expert_weight(
    source: &Tensor,
    target: &Tensor,
    cfg: &ModelConfig,
    child_idx: usize,
    projection: &str,
    target_name: &str,
) -> Result<Tensor> {
    let moe = cfg.moe.as_ref().expect("validated MoE retrofit config");
    let fine_dim = moe.fine_grained_expert_dim()?;
    let start = child_idx.checked_mul(fine_dim).ok_or_else(|| {
        aarambh_studio_core::AarambhError::Checkpoint(
            "fine expert channel offset overflows usize".into(),
        )
    })?;
    let shard = match projection {
        "w_gate" | "w_up" => source.narrow(0, start, fine_dim)?,
        "w_down" => source
            .narrow(1, start, fine_dim)?
            .affine(moe.fine_grained_factor as f64, 0.0)?,
        _ => unreachable!("projection was validated"),
    };
    if shard.dims() != target.dims() {
        return Err(aarambh_studio_core::AarambhError::Checkpoint(format!(
            "cannot shard expert tensor {target_name}: source {:?} produced {:?}, target {:?}",
            source.dims(),
            shard.dims(),
            target.dims()
        )));
    }
    Ok(shard)
}

/// Load either a safetensors or GGUF checkpoint using f32 parameters.
pub fn load_any_model(
    path: impl AsRef<Path>,
    cfg: &ModelConfig,
    device: &Device,
) -> Result<AarambhModel> {
    load_any_model_with_dtype(path, cfg, device, DType::F32)
}

/// Load either a safetensors or GGUF checkpoint using the requested dtype.
pub fn load_any_model_with_dtype(
    path: impl AsRef<Path>,
    cfg: &ModelConfig,
    device: &Device,
    dtype: DType,
) -> Result<AarambhModel> {
    let path = path.as_ref();
    if path.extension().and_then(|ext| ext.to_str()) == Some("gguf") {
        load_gguf_with_dtype(path, device, dtype)
    } else {
        load_model_with_dtype(path, cfg, device, dtype)
    }
}
