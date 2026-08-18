//! Model merging and weight averaging — Phase 50.
//!
//! Combines two or more architecturally-compatible Aarambh checkpoints into a
//! single SafeTensors file using one of five standard algorithms:
//!
//! - [`MergeMethod::Linear`] — weighted averaging (Model Soups)
//! - [`MergeMethod::Slerp`] — spherical linear interpolation
//! - [`MergeMethod::TaskArithmetic`] — combine task vectors (`delta = tuned − base`)
//! - [`MergeMethod::Ties`] — TIES-Merging (trim, elect sign, disjoint merge)
//! - [`MergeMethod::Dare`] — DARE (drop and rescale)
//!
//! # Layout
//!
//! This module is a single file: the public surface is [`MergeMethod`],
//! [`MergeConfig`], [`MergeReport`], and [`merge_models_from_paths`]. Each
//! algorithm is a private `merge_*` function operating on raw
//! `HashMap<String, Tensor>` maps loaded from disk via
//! `candle_core::safetensors::load`.
//!
//! # Honesty boundary
//!
//! - All input checkpoints must share an identical tensor-name set, per-tensor
//!   shape, and per-tensor dtype. Mismatches are rejected **before** any output
//!   byte is written — mirroring the "never silently produce garbage" discipline
//!   every other checkpoint operation in this project holds
//!   (see `ARCHITECTURE_V4.md` §64).
//! - MoE, MLA, MTP, and hybrid-attention checkpoints merge transparently:
//!   merging operates on raw name/shape-matched tensor maps, so every tensor —
//!   expert weights, router weights, MLA projections, MTP heads — is treated
//!   identically. There is no architecture-specific special-casing and no
//!   `reject_*` guard, because none is needed at the tensor level.
//! - All arithmetic is performed in `f32` regardless of the input dtype, and
//!   the output is always written as `f32`. This matches `load_model`'s default
//!   precision and the standard practice (e.g. mergekit) of merging in fp32.
//! - A merged checkpoint's downstream eval-harness score is **reported**, never
//!   assumed improved. [`MergeReport`] carries only structural facts (tensor
//!   counts, fallback counts); any quality claim is measured separately by the
//!   `eval` command against the merged artifact.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use aarambh_studio_core::{AarambhError, ModelConfig, Result};
use candle_core::{DType, Device, Tensor};

/// Per-tensor SLERP numerical guard. When the cosine of the angle between two
/// flattened tensors exceeds this value, the vectors are treated as parallel
/// and SLERP degrades gracefully to linear interpolation (avoiding division by
/// `sin(θ) ≈ 0`).
pub const SLERP_PARALLEL_EPSILON: f64 = 1.0 - 1e-6;

/// Default keep-density for TIES trimming and DARE drop-and-rescale when the
/// caller does not specify one. 0.5 matches the mergekit default.
pub const DEFAULT_DENSITY: f64 = 0.5;

/// The merge algorithm to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MergeMethod {
    /// Weighted linear averaging of N checkpoints (Model Soups).
    Linear,
    /// Spherical linear interpolation between checkpoints.
    Slerp,
    /// Task-vector arithmetic: `out = base + Σ scaleᵢ·(Mᵢ − base)`.
    TaskArithmetic,
    /// TIES-Merging: trim, elect sign, disjoint merge of task vectors.
    Ties,
    /// DARE: drop-and-rescale task vectors before linear combination.
    Dare,
}

impl MergeMethod {
    /// Whether this method consumes `--base` + `--deltas` (task-vector family)
    /// rather than `--inputs` (interpolation family).
    pub fn is_task_vector_family(self) -> bool {
        matches!(
            self,
            MergeMethod::TaskArithmetic | MergeMethod::Ties | MergeMethod::Dare
        )
    }
}

/// Configuration for a single merge operation.
///
/// `weights` is used by [`MergeMethod::Linear`] and [`MergeMethod::Slerp`];
/// `scales` and `density` are used by the task-vector family
/// ([`MergeMethod::TaskArithmetic`], [`MergeMethod::Ties`], [`MergeMethod::Dare`]).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MergeConfig {
    /// The merge algorithm.
    pub method: MergeMethod,
    /// Per-input interpolation weights (linear/slerp). Need not sum to one —
    /// they are normalized internally.
    pub weights: Vec<f64>,
    /// Per-delta scaling factors (task-arithmetic/ties/dare).
    pub scales: Vec<f64>,
    /// Fraction of each task vector retained: for TIES, the top-`density`
    /// magnitude entries are kept and the rest trimmed; for DARE, each
    /// parameter is kept with probability `density`. Must be in `(0.0, 1.0]`.
    pub density: f64,
    /// Whether to rescale surviving TIES/DARE entries so the merged delta's
    /// expected magnitude is preserved. Recommended default `true`.
    pub normalize: bool,
    /// Fixed seed for DARE's deterministic drop mask (so merges are
    /// reproducible). DARE never depends on system randomness.
    pub seed: u64,
}

impl MergeConfig {
    /// Build a config with default density (`0.5`), normalization on, and a
    /// fixed seed of `0`.
    pub fn new(method: MergeMethod) -> Self {
        Self {
            method,
            weights: Vec::new(),
            scales: Vec::new(),
            density: DEFAULT_DENSITY,
            normalize: true,
            seed: 0,
        }
    }
}

/// Structural report returned by a successful merge. Contains no quality
/// claim — only counts that let an operator audit what happened.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MergeReport {
    /// The algorithm that was applied.
    pub method: MergeMethod,
    /// Number of input checkpoints consumed (inputs for linear/slerp;
    /// deltas for the task-vector family).
    pub input_count: usize,
    /// Number of distinct tensors written.
    pub tensor_count: usize,
    /// Path of the SafeTensors file that was written.
    pub output_path: PathBuf,
    /// Number of tensors for which SLERP fell back to linear interpolation
    /// because the two flattened vectors were near-parallel. Always `0` for
    /// non-SLERP methods.
    pub slerp_linear_fallback_count: usize,
    /// Number of tensors for which TIES resolved a non-trivial sign conflict
    /// (mixed positive/negative deltas at the same position). Always `0` for
    /// non-TIES methods.
    pub ties_resolved_tensors: usize,
    /// Fraction of task-vector parameters dropped by DARE (averaged across all
    /// deltas and tensors). Always `0.0` for non-DARE methods.
    pub dare_dropped_fraction: f64,
}

/// Merge compatible Aarambh checkpoints into one SafeTensors file.
///
/// - For [`MergeMethod::Linear`] and [`MergeMethod::Slerp`], `inputs` carries
///   the N checkpoints to interpolate; `base` and `deltas` are ignored.
/// - For the task-vector family, `base` is the shared base checkpoint and
///   `deltas` are the independently-tuned checkpoints whose task vector is
///   `delta − base`; `inputs` is ignored.
///
/// `config` (the model config) is used only for documentation/compatibility
/// today — the actual merge operates on raw tensor maps, so any two checkpoints
/// with matching tensor names/shapes merge regardless of the config object.
/// It is accepted so the call site mirrors the adapter-merge helper in the
/// `aarambh-studio-finetune` crate (`merge_adapter_from_paths`) and so future
/// config-aware validation can be added without an API break.
#[allow(clippy::too_many_arguments)]
pub fn merge_models_from_paths(
    config: &ModelConfig,
    inputs: &[PathBuf],
    base: Option<&Path>,
    deltas: &[PathBuf],
    output: impl AsRef<Path>,
    merge: &MergeConfig,
) -> Result<MergeReport> {
    let _ = config; // reserved for future config-aware validation; see doc comment.
    let output = output.as_ref();

    // Resolve which path set applies and validate counts up front.
    let (active_paths, active_coeffs) = if merge.method.is_task_vector_family() {
        let base = base.ok_or_else(|| {
            AarambhError::Config(format!("merge method {:?} requires --base", merge.method))
        })?;
        if deltas.is_empty() {
            return Err(AarambhError::Config(format!(
                "merge method {:?} requires at least one --deltas checkpoint",
                merge.method
            )));
        }
        if merge.scales.len() != deltas.len() {
            return Err(AarambhError::Config(format!(
                "merge method {:?} expects {} scales, got {}",
                merge.method,
                deltas.len(),
                merge.scales.len()
            )));
        }
        // active_paths for the task-vector family = [base, delta1, delta2, ...]
        let mut paths = Vec::with_capacity(deltas.len() + 1);
        paths.push(base.to_path_buf());
        paths.extend(deltas.iter().cloned());
        (paths, merge.scales.clone())
    } else {
        if inputs.len() < 2 {
            return Err(AarambhError::Config(format!(
                "merge method {:?} requires at least two --inputs checkpoints, got {}",
                merge.method,
                inputs.len()
            )));
        }
        if merge.weights.len() != inputs.len() {
            return Err(AarambhError::Config(format!(
                "merge method {:?} expects {} weights, got {}",
                merge.method,
                inputs.len(),
                merge.weights.len()
            )));
        }
        if merge.weights.iter().any(|w| *w < 0.0) {
            return Err(AarambhError::Config(format!(
                "merge method {:?} requires non-negative weights, got {:?}",
                merge.method, merge.weights
            )));
        }
        if merge.weights.iter().sum::<f64>() == 0.0 {
            return Err(AarambhError::Config(format!(
                "merge method {:?} requires weights that sum to a positive value, got {:?}",
                merge.method, merge.weights
            )));
        }
        (inputs.to_vec(), merge.weights.clone())
    };

    if (merge.method == MergeMethod::Ties || merge.method == MergeMethod::Dare)
        && (!merge.density.is_finite() || merge.density <= 0.0 || merge.density > 1.0)
    {
        return Err(AarambhError::Config(format!(
            "merge method {:?} requires density in (0.0, 1.0], got {}",
            merge.method, merge.density
        )));
    }

    // Load every active checkpoint as a raw tensor map on CPU. All math is f32.
    let mut maps: Vec<HashMap<String, Tensor>> = Vec::with_capacity(active_paths.len());
    for path in &active_paths {
        let map = candle_core::safetensors::load(path, &Device::Cpu).map_err(|err| {
            AarambhError::Checkpoint(format!(
                "failed to load checkpoint {}: {err}",
                path.display()
            ))
        })?;
        maps.push(map);
    }

    // Hard validation: identical tensor-name sets, shapes, dtypes — BEFORE any
    // arithmetic touches a single tensor.
    validate_tensor_maps(&maps, &active_paths)?;

    // Dispatch to the chosen algorithm. All algorithms return the merged map
    // plus a small bag of method-specific audit counts.
    let (merged, audit) = match merge.method {
        MergeMethod::Linear => (merge_linear(&maps, &active_coeffs)?, MethodAudit::default()),
        MergeMethod::Slerp => merge_slerp(&maps, &active_coeffs)?,
        MergeMethod::TaskArithmetic => (
            merge_task_arithmetic(&maps, &active_coeffs)?,
            MethodAudit::default(),
        ),
        MergeMethod::Ties => merge_ties(&maps, &active_coeffs, merge.density, merge.normalize)?,
        MergeMethod::Dare => merge_dare(&maps, &active_coeffs, merge.density, merge.seed)?,
    };

    // Sort keys for deterministic output (mirrors gguf.rs write ordering).
    let sorted: Vec<(String, Tensor)> = {
        let mut entries: Vec<(String, Tensor)> = merged.into_iter().collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        entries
    };
    let merged_sorted: HashMap<String, Tensor> = sorted.into_iter().collect();

    // Write output, creating parent directories as needed.
    if let Some(parent) = output.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    candle_core::safetensors::save(&merged_sorted, output).map_err(|err| {
        AarambhError::Checkpoint(format!(
            "failed to write merged checkpoint to {}: {err}",
            output.display()
        ))
    })?;

    Ok(MergeReport {
        method: merge.method,
        input_count: if merge.method.is_task_vector_family() {
            deltas.len()
        } else {
            inputs.len()
        },
        tensor_count: merged_sorted.len(),
        output_path: output.to_path_buf(),
        slerp_linear_fallback_count: audit.slerp_linear_fallback_count,
        ties_resolved_tensors: audit.ties_resolved_tensors,
        dare_dropped_fraction: audit.dare_dropped_fraction,
    })
}

/// Method-specific audit counters accumulated during arithmetic.
#[derive(Default)]
struct MethodAudit {
    /// SLERP tensors that fell back to linear (parallel vectors).
    slerp_linear_fallback_count: usize,
    /// TIES tensors with a non-trivial sign conflict.
    ties_resolved_tensors: usize,
    /// DARE average fraction of dropped parameters.
    dare_dropped_fraction: f64,
}

/// Validate that every loaded tensor map shares the same key set and that each
/// shared key has identical shape and dtype across all maps.
fn validate_tensor_maps(maps: &[HashMap<String, Tensor>], paths: &[PathBuf]) -> Result<()> {
    if maps.is_empty() {
        return Err(AarambhError::Checkpoint(
            "cannot validate an empty checkpoint set".into(),
        ));
    }
    let first = &maps[0];
    let first_keys: std::collections::BTreeSet<&String> = first.keys().collect();
    for (idx, map) in maps.iter().enumerate().skip(1) {
        let keys: std::collections::BTreeSet<&String> = map.keys().collect();
        if keys != first_keys {
            let missing_in_second: Vec<&String> = first_keys.difference(&keys).copied().collect();
            let extra_in_second: Vec<&String> = keys.difference(&first_keys).copied().collect();
            return Err(AarambhError::Checkpoint(format!(
                "tensor-name mismatch between {} and {}: missing in second={:?} extra in second={:?}",
                paths[0].display(),
                paths[idx].display(),
                missing_in_second,
                extra_in_second
            )));
        }
    }
    // Same names — now verify shapes and dtypes for each shared tensor.
    for name in first_keys {
        let reference = &first[name];
        let ref_shape = reference.dims();
        let ref_dtype = reference.dtype();
        for map in maps.iter().skip(1) {
            let candidate = &map[name];
            if candidate.dims() != ref_shape {
                return Err(AarambhError::Shape(format!(
                    "tensor {name} shape mismatch: {:?} vs {:?}",
                    ref_shape,
                    candidate.dims()
                )));
            }
            if candidate.dtype() != ref_dtype {
                return Err(AarambhError::Shape(format!(
                    "tensor {name} dtype mismatch: {ref_dtype:?} vs {:?}",
                    candidate.dtype()
                )));
            }
        }
    }
    Ok(())
}

/// Convert a tensor to f32 for arithmetic. f32 inputs are returned unchanged.
fn to_f32(t: &Tensor) -> Result<Tensor> {
    if t.dtype() == DType::F32 {
        Ok(t.clone())
    } else {
        Ok(t.to_dtype(DType::F32)?)
    }
}

/// Weighted linear average (Model Soups): `out = Σ wᵢ·Mᵢ`, weights normalized.
fn merge_linear(
    maps: &[HashMap<String, Tensor>],
    weights: &[f64],
) -> Result<HashMap<String, Tensor>> {
    let total: f64 = weights.iter().sum();
    let normalized: Vec<f64> = weights.iter().map(|w| w / total).collect();
    let reference = &maps[0];
    let mut out = HashMap::with_capacity(reference.len());
    for (name, tensor) in reference {
        let acc = to_f32(tensor)?;
        let mut sum = acc.affine(normalized[0], 0.0)?;
        for (map, w) in maps.iter().skip(1).zip(normalized.iter().skip(1)) {
            let t = to_f32(&map[name])?;
            sum = sum.add(&t.affine(*w, 0.0)?)?;
        }
        out.insert(name.clone(), sum);
    }
    Ok(out)
}

/// Pairwise chained spherical linear interpolation across N checkpoints.
///
/// For two inputs this is the textbook SLERP. For N inputs the result is a
/// left-to-right fold where each step's interpolation weight is the current
/// input's weight divided by the cumulative weight up to and including it.
fn merge_slerp(
    maps: &[HashMap<String, Tensor>],
    weights: &[f64],
) -> Result<(HashMap<String, Tensor>, MethodAudit)> {
    let mut audit = MethodAudit::default();
    let reference = &maps[0];
    let mut out = HashMap::with_capacity(reference.len());
    // cumulative weights for the fold: step i interpolates between the running
    // accumulator and maps[i] with t = w_i / (Σ_{j<=i} w_j).
    let mut cumulative = 0.0f64;
    let mut cum_weights: Vec<f64> = Vec::with_capacity(weights.len());
    for w in weights {
        cumulative += *w;
        cum_weights.push(cumulative);
    }
    for (name, tensor) in reference {
        let mut acc = to_f32(tensor)?;
        for i in 1..maps.len() {
            let t = weights[i] / cum_weights[i];
            let next = to_f32(&maps[i][name])?;
            let (blended, fell_back) = slerp_pair(&acc, &next, t)?;
            if fell_back {
                audit.slerp_linear_fallback_count += 1;
            }
            acc = blended;
        }
        out.insert(name.clone(), acc);
    }
    Ok((out, audit))
}

/// Spherical linear interpolation between two flattened tensors.
///
/// Returns `(result, fell_back_to_linear)`. Falls back to linear when the two
/// vectors are (near-)parallel, which avoids division by `sin(θ) ≈ 0` and is
/// the standard numerical guard used by mergekit.
fn slerp_pair(a: &Tensor, b: &Tensor, t: f64) -> Result<(Tensor, bool)> {
    let a_flat = a.flatten_all()?;
    let b_flat = b.flatten_all()?;
    let dot = a_flat.mul(&b_flat)?.sum_all()?.to_scalar::<f32>()? as f64;
    let norm_a = a_flat.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
    let norm_b = b_flat.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
    let denom = (norm_a * norm_b).sqrt();
    if denom == 0.0 {
        // One of the tensors is all zeros — linear is the only sensible path.
        let out = linear_blend(&a_flat, &b_flat, t)?;
        return Ok((out.reshape(a.dims())?, true));
    }
    let cos_theta = (dot / denom).clamp(-1.0, 1.0);
    if !(-SLERP_PARALLEL_EPSILON..=SLERP_PARALLEL_EPSILON).contains(&cos_theta) {
        // Near-parallel (or anti-parallel): SLERP is numerically unstable;
        // fall back to linear interpolation. This is the documented guard.
        let out = linear_blend(&a_flat, &b_flat, t)?;
        return Ok((out.reshape(a.dims())?, true));
    }
    let theta = cos_theta.acos();
    let sin_theta = theta.sin();
    let coeff_a = ((1.0 - t) * theta).sin() / sin_theta;
    let coeff_b = (t * theta).sin() / sin_theta;
    let blended = a_flat
        .affine(coeff_a, 0.0)?
        .add(&b_flat.affine(coeff_b, 0.0)?)?;
    Ok((blended.reshape(a.dims())?, false))
}

/// Linear blend `out = (1−t)·a + t·b` on already-flattened tensors.
fn linear_blend(a: &Tensor, b: &Tensor, t: f64) -> Result<Tensor> {
    Ok(a.affine(1.0 - t, 0.0)?.add(&b.affine(t, 0.0)?)?)
}

/// Task-vector arithmetic: `out = base + Σ sᵢ·(Mᵢ − base)`.
///
/// `maps[0]` is the base; `maps[1..]` are the tuned deltas. `scales` aligns
/// with the deltas (`maps[1..]`).
fn merge_task_arithmetic(
    maps: &[HashMap<String, Tensor>],
    scales: &[f64],
) -> Result<HashMap<String, Tensor>> {
    let base = &maps[0];
    let mut out = HashMap::with_capacity(base.len());
    for (name, base_t) in base {
        let base_f = to_f32(base_t)?;
        // Start from the base.
        let mut acc = base_f.clone();
        for (map, scale) in maps.iter().skip(1).zip(scales.iter()) {
            let tuned = to_f32(&map[name])?;
            let delta = tuned.sub(&base_f)?;
            acc = acc.add(&delta.affine(*scale, 0.0)?)?;
        }
        out.insert(name.clone(), acc);
    }
    Ok(out)
}

/// TIES-Merging: trim → elect sign → disjoint merge, applied to task vectors.
///
/// For each tensor:
/// 1. Compute per-delta task vector `δᵢ = Mᵢ − base`.
/// 2. Trim each `δᵢ` to its top-`density` magnitude entries (zero the rest).
/// 3. Elect a sign per position by weighted majority of surviving deltas.
/// 4. Disjoint-merge: average only the deltas whose sign matches the elected
///    sign; positions with no agreement contribute zero.
/// 5. Optionally rescale by `1/density` to preserve expected magnitude.
fn merge_ties(
    maps: &[HashMap<String, Tensor>],
    scales: &[f64],
    density: f64,
    normalize: bool,
) -> Result<(HashMap<String, Tensor>, MethodAudit)> {
    let mut audit = MethodAudit::default();
    let base = &maps[0];
    let mut out = HashMap::with_capacity(base.len());
    for (name, base_t) in base {
        let base_f = to_f32(base_t)?;
        let flat_shape = base_f.dims().to_vec();
        let base_flat = base_f.flatten_all()?;
        // Per-delta trimmed task vectors, flattened.
        let mut trimmed_deltas: Vec<Tensor> = Vec::with_capacity(scales.len());
        for map in maps.iter().skip(1) {
            let tuned = to_f32(&map[name])?;
            let delta = tuned.sub(&base_f)?.flatten_all()?;
            let trimmed = trim_magnitude(&delta, density)?;
            trimmed_deltas.push(trimmed);
        }
        let (merged_delta, had_conflict) = ties_elect_and_disjoint_merge(&trimmed_deltas, scales)?;
        if had_conflict {
            audit.ties_resolved_tensors += 1;
        }
        let final_delta = if normalize {
            merged_delta.affine(1.0 / density, 0.0)?
        } else {
            merged_delta
        };
        let out_flat = base_flat.add(&final_delta)?;
        out.insert(name.clone(), out_flat.reshape(flat_shape.as_slice())?);
    }
    Ok((out, audit))
}

/// Zero out all but the top-`density` fraction of entries by absolute value.
///
/// `density = 1.0` keeps everything; `density = 0.5` keeps the largest-magnitude
/// half. Implemented by materializing the tensor to a Vec, sorting indices by
/// magnitude, and rebuilding a masked tensor — correct and simple for the
/// offline, CPU-only merge path this module targets.
fn trim_magnitude(delta: &Tensor, density: f64) -> Result<Tensor> {
    if density >= 1.0 {
        return Ok(delta.clone());
    }
    let numel = delta.elem_count();
    if numel == 0 {
        return Ok(delta.clone());
    }
    let keep = ((numel as f64) * density).round() as usize;
    let keep = keep.clamp(1, numel);
    let values = delta.to_vec1::<f32>()?;
    let mut indices: Vec<usize> = (0..numel).collect();
    indices.sort_by(|&a, &b| {
        values[b]
            .abs()
            .partial_cmp(&values[a].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut kept = vec![0.0f32; numel];
    for &idx in indices.iter().take(keep) {
        kept[idx] = values[idx];
    }
    Ok(Tensor::from_slice(&kept, (numel,), delta.device())?)
}

/// Elect a sign per position by weighted majority over the surviving (non-zero)
/// deltas, then disjoint-merge only the deltas whose sign agrees. Returns
/// `(merged_delta_flat, had_sign_conflict)`.
fn ties_elect_and_disjoint_merge(deltas: &[Tensor], scales: &[f64]) -> Result<(Tensor, bool)> {
    let n = deltas.len();
    if n == 0 {
        return Ok((Tensor::zeros((1,), DType::F32, &Device::Cpu)?, false));
    }
    let numel = deltas[0].elem_count();
    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(n);
    for d in deltas {
        rows.push(d.to_vec1::<f32>()?);
    }
    let mut out = vec![0.0f32; numel];
    let mut had_conflict = false;
    for i in 0..numel {
        let mut pos_weight = 0.0f64;
        let mut neg_weight = 0.0f64;
        let mut pos_sum = 0.0f64;
        let mut neg_sum = 0.0f64;
        for (row, scale) in rows.iter().zip(scales.iter()) {
            let v = row[i] as f64;
            if v > 0.0 {
                pos_weight += *scale;
                pos_sum += v * *scale;
            } else if v < 0.0 {
                neg_weight += *scale;
                neg_sum += v * *scale;
            }
        }
        if pos_weight == 0.0 && neg_weight == 0.0 {
            continue; // no surviving deltas at this position
        }
        if pos_weight > 0.0 && neg_weight > 0.0 {
            had_conflict = true; // mixed signs -> a real TIES resolution
        }
        if pos_weight >= neg_weight && pos_weight > 0.0 {
            out[i] = (pos_sum / pos_weight) as f32;
        } else if neg_weight > 0.0 {
            out[i] = (neg_sum / neg_weight) as f32;
        }
    }
    Ok((
        Tensor::from_slice(&out, (numel,), &Device::Cpu)?,
        had_conflict,
    ))
}

/// DARE: drop-and-rescale each task vector with a deterministic Bernoulli mask,
/// then linearly combine the surviving (rescaled) deltas and add to the base.
///
/// The drop mask is generated by a tiny seeded xorshift PRNG so a merge is
/// fully reproducible — DARE never touches system randomness.
fn merge_dare(
    maps: &[HashMap<String, Tensor>],
    scales: &[f64],
    density: f64,
    seed: u64,
) -> Result<(HashMap<String, Tensor>, MethodAudit)> {
    let mut audit = MethodAudit::default();
    let base = &maps[0];
    let mut out = HashMap::with_capacity(base.len());
    let mut total_dropped = 0u64;
    let mut total_params = 0u64;
    for (name, base_t) in base {
        let base_f = to_f32(base_t)?;
        let flat_shape = base_f.dims().to_vec();
        let base_flat = base_f.flatten_all()?;
        let numel = base_f.elem_count();
        let mut acc = base_flat.clone();
        for (map, scale) in maps.iter().skip(1).zip(scales.iter()) {
            let tuned = to_f32(&map[name])?;
            let delta = tuned.sub(&base_f)?.flatten_all()?;
            let (dared, dropped) = dare_drop_and_rescale(&delta, density, seed)?;
            total_dropped += dropped;
            total_params += numel as u64;
            acc = acc.add(&dared.affine(*scale, 0.0)?)?;
        }
        out.insert(name.clone(), acc.reshape(flat_shape.as_slice())?);
    }
    audit.dare_dropped_fraction = if total_params > 0 {
        total_dropped as f64 / total_params as f64
    } else {
        0.0
    };
    Ok((out, audit))
}

/// Apply DARE's drop-and-rescale to a flattened delta.
///
/// Each parameter is kept with probability `density`; surviving parameters are
/// rescaled by `1/density` so the delta's expected magnitude is preserved.
/// The mask is derived from a seeded xorshift PRNG (deterministic, no `rand`
/// dependency). Returns `(rescaled_delta, dropped_count)`.
fn dare_drop_and_rescale(delta: &Tensor, density: f64, seed: u64) -> Result<(Tensor, u64)> {
    let numel = delta.elem_count();
    if numel == 0 {
        return Ok((delta.clone(), 0));
    }
    let values = delta.to_vec1::<f32>()?;
    let mut out = vec![0.0f32; numel];
    let mut rng = XorShift64::new(seed.wrapping_add(numel as u64));
    let mut dropped = 0u64;
    let rescale = 1.0 / density;
    for (i, v) in values.iter().enumerate() {
        let r = rng.next_double();
        if r < density {
            out[i] = *v * rescale as f32;
        } else {
            dropped += 1;
        }
    }
    Ok((Tensor::from_slice(&out, (numel,), delta.device())?, dropped))
}

/// A tiny seeded xorshift64 PRNG. Deterministic, allocation-free, and
/// dependency-free — sufficient for DARE's reproducible Bernoulli masks.
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    /// Create a PRNG. A zero seed is remapped to a fixed nonzero constant to
    /// avoid the degenerate all-zero state.
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    /// Next raw 64-bit value.
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Next value in `[0.0, 1.0)`.
    fn next_double(&mut self) -> f64 {
        // Use the top 53 bits for a full-precision double.
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xorshift_is_deterministic_and_in_unit_interval() {
        let mut a = XorShift64::new(42);
        let mut b = XorShift64::new(42);
        for _ in 0..1000 {
            let va = a.next_double();
            let vb = b.next_double();
            assert_eq!(va.to_bits(), vb.to_bits(), "xorshift must be deterministic");
            assert!((0.0..1.0).contains(&va), "value must be in [0,1): got {va}");
        }
    }

    #[test]
    fn xorshift_zero_seed_is_remapped() {
        // A zero seed must not produce the degenerate all-zero stream.
        let mut rng = XorShift64::new(0);
        let v = rng.next_u64();
        assert_ne!(v, 0, "zero seed must be remapped to a nonzero state");
    }

    #[test]
    fn method_classification_matches_task_vector_family() {
        assert!(!MergeMethod::Linear.is_task_vector_family());
        assert!(!MergeMethod::Slerp.is_task_vector_family());
        assert!(MergeMethod::TaskArithmetic.is_task_vector_family());
        assert!(MergeMethod::Ties.is_task_vector_family());
        assert!(MergeMethod::Dare.is_task_vector_family());
    }

    #[test]
    fn merge_config_default_density_is_half() {
        let cfg = MergeConfig::new(MergeMethod::Linear);
        assert!((cfg.density - 0.5).abs() < 1e-12);
        assert!(cfg.normalize);
        assert_eq!(cfg.seed, 0);
    }
}
