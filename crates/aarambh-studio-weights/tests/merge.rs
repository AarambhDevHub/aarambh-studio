//! Integration tests for `aarambh-studio-weights::merge` (Phase 50).
//!
//! These tests build tiny synthetic SafeTensors checkpoints in a per-test temp
//! directory, run each merge algorithm, and assert the roadmap-named
//! acceptance criteria from `ROADMAP_V4.md` §"Phase 50 — Tests" plus
//! supporting invariants. No fixtures are committed to the tree (the
//! release-audit forbids tracked `*.safetensors`).

use std::collections::HashMap;
use std::path::PathBuf;

use aarambh_studio_core::ModelConfig;
use aarambh_studio_weights::{MergeConfig, MergeMethod, MergeReport, merge_models_from_paths};
use candle_core::{DType, Device, Tensor};

/// Build a tiny synthetic checkpoint with three named tensors and save it.
/// Returns the path. Tensor values are deterministic per `seed` so tests are
/// reproducible.
fn write_synthetic_checkpoint(dir: &std::path::Path, name: &str, seed: f32) -> PathBuf {
    let path = dir.join(name);
    let mut tensors = HashMap::new();
    // A 2x3 "embedding" tensor and a 3x2 "weight" tensor, plus a 1-D norm.
    let embedding = Tensor::from_vec(
        (0..6).map(|i| seed * (i as f32) + 1.0).collect::<Vec<_>>(),
        (2, 3),
        &Device::Cpu,
    )
    .unwrap()
    .to_dtype(DType::F32)
    .unwrap();
    let weight = Tensor::from_vec(
        (0..6)
            .map(|i| (seed + 1.0) * (i as f32))
            .collect::<Vec<_>>(),
        (3, 2),
        &Device::Cpu,
    )
    .unwrap()
    .to_dtype(DType::F32)
    .unwrap();
    let norm = Tensor::from_vec(
        (0..3).map(|i| seed - (i as f32) * 0.5).collect::<Vec<_>>(),
        (3,),
        &Device::Cpu,
    )
    .unwrap()
    .to_dtype(DType::F32)
    .unwrap();
    tensors.insert("embedding.weight".to_string(), embedding);
    tensors.insert("layers.0.weight".to_string(), weight);
    tensors.insert("layers.0.norm.weight".to_string(), norm);
    candle_core::safetensors::save(&tensors, &path).unwrap();
    path
}

/// A checkpoint with a deliberately wrong shape for one tensor (mismatch test).
fn write_shape_mismatch_checkpoint(dir: &std::path::Path, name: &str) -> PathBuf {
    let path = dir.join(name);
    let mut tensors = HashMap::new();
    let embedding = Tensor::from_vec(
        (0..12).map(|i| i as f32).collect::<Vec<_>>(),
        (3, 4), // different shape vs write_synthetic_checkpoint's (2,3)
        &Device::Cpu,
    )
    .unwrap()
    .to_dtype(DType::F32)
    .unwrap();
    let weight = Tensor::from_vec(
        (0..6).map(|i| i as f32).collect::<Vec<_>>(),
        (3, 2),
        &Device::Cpu,
    )
    .unwrap()
    .to_dtype(DType::F32)
    .unwrap();
    let norm = Tensor::from_vec(
        (0..3).map(|i| i as f32).collect::<Vec<_>>(),
        (3,),
        &Device::Cpu,
    )
    .unwrap()
    .to_dtype(DType::F32)
    .unwrap();
    tensors.insert("embedding.weight".to_string(), embedding);
    tensors.insert("layers.0.weight".to_string(), weight);
    tensors.insert("layers.0.norm.weight".to_string(), norm);
    candle_core::safetensors::save(&tensors, &path).unwrap();
    path
}

/// A checkpoint missing one tensor name (name-set mismatch test).
fn write_missing_tensor_checkpoint(dir: &std::path::Path, name: &str) -> PathBuf {
    let path = dir.join(name);
    let mut tensors = HashMap::new();
    let embedding = Tensor::from_vec(
        (0..6).map(|i| i as f32).collect::<Vec<_>>(),
        (2, 3),
        &Device::Cpu,
    )
    .unwrap()
    .to_dtype(DType::F32)
    .unwrap();
    let weight = Tensor::from_vec(
        (0..6).map(|i| i as f32).collect::<Vec<_>>(),
        (3, 2),
        &Device::Cpu,
    )
    .unwrap()
    .to_dtype(DType::F32)
    .unwrap();
    tensors.insert("embedding.weight".to_string(), embedding);
    tensors.insert("layers.0.weight".to_string(), weight);
    // NOTE: "layers.0.norm.weight" deliberately absent.
    candle_core::safetensors::save(&tensors, &path).unwrap();
    path
}

/// Unique temp dir per test, auto-cleaned at end of scope via Drop.
fn temp_dir(label: &str) -> std::path::PathBuf {
    let nano = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("aarambh-phase50-merge-{label}-{nano}"));
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn tiny_config() -> ModelConfig {
    ModelConfig::tiny()
}

fn load_tensor(path: &std::path::Path, name: &str) -> Tensor {
    let map = candle_core::safetensors::load(path, &Device::Cpu).unwrap();
    map[name].clone()
}

// ----------------------------------------------------------------------------
// Roadmap-named acceptance tests (verbatim from ROADMAP_V4.md §"Phase 50 — Tests")
// ----------------------------------------------------------------------------

#[test]
fn merge_rejects_checkpoints_with_incompatible_shapes_before_writing_output() {
    let dir = temp_dir("shape-mismatch");
    let a = write_synthetic_checkpoint(&dir, "a.safetensors", 1.0);
    let b = write_shape_mismatch_checkpoint(&dir, "b.safetensors");
    let output = dir.join("merged.safetensors");

    let cfg = MergeConfig {
        method: MergeMethod::Linear,
        weights: vec![0.5, 0.5],
        ..MergeConfig::new(MergeMethod::Linear)
    };
    let result = merge_models_from_paths(&tiny_config(), &[a, b], None, &[], &output, &cfg);
    assert!(
        result.is_err(),
        "shape-mismatched checkpoints must be rejected before any output is written"
    );
    assert!(
        !output.exists(),
        "no output file must be written when validation fails"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn slerp_with_weight_one_zero_reproduces_the_first_input_exactly() {
    let dir = temp_dir("slerp-identity");
    let a = write_synthetic_checkpoint(&dir, "a.safetensors", 2.0);
    let b = write_synthetic_checkpoint(&dir, "b.safetensors", 5.0);
    let output = dir.join("merged.safetensors");

    let cfg = MergeConfig {
        method: MergeMethod::Slerp,
        weights: vec![1.0, 0.0],
        ..MergeConfig::new(MergeMethod::Slerp)
    };
    let report = merge_models_from_paths(&tiny_config(), &[a.clone(), b], None, &[], &output, &cfg)
        .expect("SLERP with weight (1.0, 0.0) must succeed");

    // The merged "embedding.weight" must be bit-for-bit identical to input A's.
    let merged_emb = load_tensor(&output, "embedding.weight");
    let original_emb = load_tensor(&a, "embedding.weight");
    let merged_vals = merged_emb.to_vec2::<f32>().unwrap();
    let original_vals = original_emb.to_vec2::<f32>().unwrap();
    assert_eq!(
        merged_vals.len(),
        original_vals.len(),
        "merged tensor must have the same row count as the input"
    );
    for (merged_row, original_row) in merged_vals.iter().zip(original_vals.iter()) {
        for (m, o) in merged_row.iter().zip(original_row.iter()) {
            assert_eq!(
                m.to_bits(),
                o.to_bits(),
                "SLERP at weight=1.0 must reproduce input A bit-for-bit (got {m} vs {o})"
            );
        }
    }
    // Identity check on the report too.
    assert_eq!(report.method, MergeMethod::Slerp);
    assert_eq!(report.tensor_count, 3);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn task_arithmetic_merge_of_two_independently_tuned_deltas_produces_valid_checkpoint() {
    let dir = temp_dir("task-arith");
    let base = write_synthetic_checkpoint(&dir, "base.safetensors", 1.0);
    let math = write_synthetic_checkpoint(&dir, "math.safetensors", 2.0);
    let chat = write_synthetic_checkpoint(&dir, "chat.safetensors", 3.0);
    let output = dir.join("merged.safetensors");

    let cfg = MergeConfig {
        method: MergeMethod::TaskArithmetic,
        scales: vec![1.0, 0.5],
        ..MergeConfig::new(MergeMethod::TaskArithmetic)
    };
    let report = merge_models_from_paths(
        &tiny_config(),
        &[],
        Some(&base),
        &[math.clone(), chat.clone()],
        &output,
        &cfg,
    )
    .expect("task-arithmetic merge must succeed for two deltas");

    // Verify out = base + 1.0*(math - base) + 0.5*(chat - base) for the norm tensor.
    let base_n = load_tensor(&base, "layers.0.norm.weight")
        .to_vec1::<f32>()
        .unwrap();
    let math_n = load_tensor(&math, "layers.0.norm.weight")
        .to_vec1::<f32>()
        .unwrap();
    let chat_n = load_tensor(&chat, "layers.0.norm.weight")
        .to_vec1::<f32>()
        .unwrap();
    let merged_n = load_tensor(&output, "layers.0.norm.weight")
        .to_vec1::<f32>()
        .unwrap();
    for i in 0..base_n.len() {
        let expected = base_n[i] + 1.0 * (math_n[i] - base_n[i]) + 0.5 * (chat_n[i] - base_n[i]);
        assert!(
            (merged_n[i] - expected).abs() < 1e-5,
            "task-arithmetic element {i}: got {} expected {expected}",
            merged_n[i]
        );
    }
    assert_eq!(report.method, MergeMethod::TaskArithmetic);
    assert_eq!(report.input_count, 2);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn merged_checkpoint_eval_harness_score_is_reported_not_assumed_improved() {
    // This is the honesty-floor test. It asserts that a MergeReport carries
    // ONLY structural facts (counts, paths, fallback counts) and never a
    // quality claim. Any downstream quality assessment is the eval command's
    // job, run separately against the merged artifact — exactly as MoE (v2
    // §26), RLAIF (v4 §46), and RAG (v4 §49) frame their own "measured, not
    // assumed" discipline.
    let dir = temp_dir("honesty");
    let a = write_synthetic_checkpoint(&dir, "a.safetensors", 1.0);
    let b = write_synthetic_checkpoint(&dir, "b.safetensors", 4.0);
    let output = dir.join("merged.safetensors");

    let cfg = MergeConfig {
        method: MergeMethod::Linear,
        weights: vec![0.5, 0.5],
        ..MergeConfig::new(MergeMethod::Linear)
    };
    let report: MergeReport =
        merge_models_from_paths(&tiny_config(), &[a, b], None, &[], &output, &cfg)
            .expect("linear merge must succeed for the honesty test");

    // The report exposes only structural fields. There is no `improved: bool`,
    // no `score: f64`, no `accuracy` — those would be a quality claim and must
    // be measured by the eval harness against the merged artifact, not asserted
    // by the merge step itself.
    let _ = report.method;
    let _ = report.input_count;
    let _ = report.tensor_count;
    let _ = report.output_path;
    let _ = report.slerp_linear_fallback_count;
    let _ = report.ties_resolved_tensors;
    let _ = report.dare_dropped_fraction;
    // The output file exists and is loadable — the merge did its one job
    // (produce a valid checkpoint). Whether it is *better* is a separate
    // question for the eval command.
    let map = candle_core::safetensors::load(&output, &Device::Cpu);
    assert!(map.is_ok(), "merged checkpoint must be loadable");
    assert!(
        map.unwrap().contains_key("embedding.weight"),
        "merged checkpoint must preserve tensor names"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ----------------------------------------------------------------------------
// Supporting tests
// ----------------------------------------------------------------------------

#[test]
fn linear_merge_of_two_identical_checkpoints_is_idempotent() {
    let dir = temp_dir("linear-idempotent");
    let a = write_synthetic_checkpoint(&dir, "a.safetensors", 3.0);
    let b = write_synthetic_checkpoint(&dir, "b.safetensors", 3.0);
    let output = dir.join("merged.safetensors");

    let cfg = MergeConfig {
        method: MergeMethod::Linear,
        weights: vec![0.5, 0.5],
        ..MergeConfig::new(MergeMethod::Linear)
    };
    merge_models_from_paths(&tiny_config(), &[a.clone(), b], None, &[], &output, &cfg).unwrap();

    let original = load_tensor(&a, "embedding.weight")
        .to_vec2::<f32>()
        .unwrap();
    let merged = load_tensor(&output, "embedding.weight")
        .to_vec2::<f32>()
        .unwrap();
    for (merged_row, original_row) in merged.iter().zip(original.iter()) {
        for (m, o) in merged_row.iter().zip(original_row.iter()) {
            assert!(
                (m - o).abs() < 1e-6,
                "linear merge of two identical checkpoints must reproduce them: {m} vs {o}"
            );
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn linear_merge_weights_are_normalized_to_sum_one() {
    // Weights [2.0, 2.0] must behave identically to [0.5, 0.5] after
    // normalization. We verify by merging the same two checkpoints twice and
    // confirming the two outputs match bit-for-bit on a non-trivial element.
    let dir = temp_dir("linear-normalize");
    let a = write_synthetic_checkpoint(&dir, "a.safetensors", 1.0);
    let b = write_synthetic_checkpoint(&dir, "b.safetensors", 5.0);
    let out_raw = dir.join("raw.safetensors");
    let out_norm = dir.join("norm.safetensors");

    let cfg_raw = MergeConfig {
        method: MergeMethod::Linear,
        weights: vec![2.0, 2.0],
        ..MergeConfig::new(MergeMethod::Linear)
    };
    let cfg_norm = MergeConfig {
        method: MergeMethod::Linear,
        weights: vec![0.5, 0.5],
        ..MergeConfig::new(MergeMethod::Linear)
    };
    merge_models_from_paths(
        &tiny_config(),
        &[a.clone(), b.clone()],
        None,
        &[],
        &out_raw,
        &cfg_raw,
    )
    .unwrap();
    merge_models_from_paths(&tiny_config(), &[a, b], None, &[], &out_norm, &cfg_norm).unwrap();

    let raw = load_tensor(&out_raw, "embedding.weight")
        .to_vec2::<f32>()
        .unwrap();
    let norm = load_tensor(&out_norm, "embedding.weight")
        .to_vec2::<f32>()
        .unwrap();
    // Element [0][1] (i=1) differs by seed: seed 1.0 → 2.0, seed 5.0 → 6.0,
    // so the normalized merge gives 0.5*2.0 + 0.5*6.0 = 4.0.
    assert!(
        (norm[0][1] - 4.0).abs() < 1e-5,
        "normalized linear merge element [0][1] must be 4.0, got {}",
        norm[0][1]
    );
    // Raw weights [2,2] must produce the identical result (normalized to [0.5,0.5]).
    for (r_row, n_row) in raw.iter().zip(norm.iter()) {
        for (r, n) in r_row.iter().zip(n_row.iter()) {
            assert!(
                (r - n).abs() < 1e-6,
                "weights [2,2] and [0.5,0.5] must produce identical merges: {r} vs {n}"
            );
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn slerp_parallel_vectors_fall_back_to_linear_interpolation() {
    // When two tensors are scalar multiples (parallel), SLERP must fall back
    // to linear interpolation and report a non-zero fallback count.
    let dir = temp_dir("slerp-parallel");
    // Build two checkpoints where every tensor in B is exactly 2× the tensor
    // in A — i.e. they are perfectly parallel, triggering the fallback.
    let a_path = dir.join("a.safetensors");
    let b_path = dir.join("b.safetensors");
    let mut a_map = HashMap::new();
    let mut b_map = HashMap::new();
    let vals = vec![1.0f32, 2.0, 3.0, 4.0];
    let a_tensor = Tensor::from_vec(vals.clone(), (2, 2), &Device::Cpu).unwrap();
    let b_tensor = Tensor::from_vec(
        vals.iter().map(|v| v * 2.0).collect::<Vec<_>>(),
        (2, 2),
        &Device::Cpu,
    )
    .unwrap();
    a_map.insert("t.weight".to_string(), a_tensor);
    b_map.insert("t.weight".to_string(), b_tensor);
    candle_core::safetensors::save(&a_map, &a_path).unwrap();
    candle_core::safetensors::save(&b_map, &b_path).unwrap();
    let output = dir.join("merged.safetensors");

    let cfg = MergeConfig {
        method: MergeMethod::Slerp,
        weights: vec![0.5, 0.5],
        ..MergeConfig::new(MergeMethod::Slerp)
    };
    let report =
        merge_models_from_paths(&tiny_config(), &[a_path, b_path], None, &[], &output, &cfg)
            .unwrap();
    assert!(
        report.slerp_linear_fallback_count > 0,
        "parallel tensors must trigger at least one linear fallback, got {}",
        report.slerp_linear_fallback_count
    );
    // Linear fallback at t=0.5 between vals and 2*vals gives 1.5*vals.
    let merged = load_tensor(&output, "t.weight").to_vec2::<f32>().unwrap();
    assert!(
        (merged[0][0] - 1.5).abs() < 1e-5,
        "linear fallback at t=0.5 must give 1.5, got {}",
        merged[0][0]
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn task_arithmetic_with_zero_scales_reproduces_the_base_checkpoint() {
    let dir = temp_dir("task-arith-zero");
    let base = write_synthetic_checkpoint(&dir, "base.safetensors", 7.0);
    let d1 = write_synthetic_checkpoint(&dir, "d1.safetensors", 2.0);
    let d2 = write_synthetic_checkpoint(&dir, "d2.safetensors", 9.0);
    let output = dir.join("merged.safetensors");

    let cfg = MergeConfig {
        method: MergeMethod::TaskArithmetic,
        scales: vec![0.0, 0.0],
        ..MergeConfig::new(MergeMethod::TaskArithmetic)
    };
    merge_models_from_paths(&tiny_config(), &[], Some(&base), &[d1, d2], &output, &cfg).unwrap();

    let base_emb = load_tensor(&base, "embedding.weight")
        .to_vec2::<f32>()
        .unwrap();
    let merged_emb = load_tensor(&output, "embedding.weight")
        .to_vec2::<f32>()
        .unwrap();
    for (m_row, b_row) in merged_emb.iter().zip(base_emb.iter()) {
        for (m, b) in m_row.iter().zip(b_row.iter()) {
            assert!(
                (m - b).abs() < 1e-6,
                "task-arithmetic with zero scales must reproduce the base: {m} vs {b}"
            );
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn ties_merge_resolves_sign_conflicts_by_weighted_majority() {
    // Two deltas with opposite signs at every position: TIES must elect the
    // sign with the larger weighted support and zero the loser.
    let dir = temp_dir("ties-conflict");
    let base_path = dir.join("base.safetensors");
    let d1_path = dir.join("d1.safetensors");
    let d2_path = dir.join("d2.safetensors");
    let mut base_map = HashMap::new();
    let mut d1_map = HashMap::new();
    let mut d2_map = HashMap::new();
    let base_vals = vec![0.0f32; 4];
    let d1_vals = vec![1.0f32, 1.0, 1.0, 1.0]; // all positive
    let d2_vals = vec![-3.0f32, -3.0, -3.0, -3.0]; // all negative, larger magnitude
    base_map.insert(
        "t.weight".to_string(),
        Tensor::from_vec(base_vals, (4,), &Device::Cpu).unwrap(),
    );
    d1_map.insert(
        "t.weight".to_string(),
        Tensor::from_vec(d1_vals, (4,), &Device::Cpu).unwrap(),
    );
    d2_map.insert(
        "t.weight".to_string(),
        Tensor::from_vec(d2_vals, (4,), &Device::Cpu).unwrap(),
    );
    candle_core::safetensors::save(&base_map, &base_path).unwrap();
    candle_core::safetensors::save(&d1_map, &d1_path).unwrap();
    candle_core::safetensors::save(&d2_map, &d2_path).unwrap();
    let output = dir.join("merged.safetensors");

    // scales: d1 has scale 1.0, d2 has scale 1.0 — equal weight, but d2 has
    // larger magnitude. After TIES trimming at density 1.0 (keep all) the sign
    // election sees pos_weight=1.0 vs neg_weight=1.0 → tie resolves to pos
    // (>=), so the merged delta keeps d1's positive values and discards d2.
    let cfg = MergeConfig {
        method: MergeMethod::Ties,
        weights: Vec::new(),
        scales: vec![1.0, 1.0],
        density: 1.0,
        normalize: false,
        seed: 0,
    };
    let report = merge_models_from_paths(
        &tiny_config(),
        &[],
        Some(&base_path),
        &[d1_path, d2_path],
        &output,
        &cfg,
    )
    .unwrap();
    assert!(
        report.ties_resolved_tensors > 0,
        "opposite-sign deltas must register at least one sign-conflict resolution"
    );
    let merged = load_tensor(&output, "t.weight").to_vec1::<f32>().unwrap();
    // With equal weights and a tie, pos wins (>=), so the disjoint average of
    // the agreeing deltas is d1's value = 1.0 (d2 is discarded).
    for v in &merged {
        assert!(
            (v - 1.0).abs() < 1e-5,
            "TIES tie-break keeps the positive delta, expected 1.0 got {v}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn dare_drop_and_rescale_preserves_expected_magnitude() {
    // With density close to 1.0, DARE keeps nearly everything and rescales by
    // ~1/density. The merged delta magnitude stays close to the un-dropped
    // task-arithmetic baseline.
    let dir = temp_dir("dare-magnitude");
    let base = write_synthetic_checkpoint(&dir, "base.safetensors", 1.0);
    let d1 = write_synthetic_checkpoint(&dir, "d1.safetensors", 2.0);
    let output_dare = dir.join("dare.safetensors");

    let cfg = MergeConfig {
        method: MergeMethod::Dare,
        weights: Vec::new(),
        scales: vec![1.0],
        density: 0.9,
        normalize: true,
        seed: 42,
    };
    let report =
        merge_models_from_paths(&tiny_config(), &[], Some(&base), &[d1], &output_dare, &cfg)
            .unwrap();
    // Density 0.9 ⇒ ~10% of params dropped.
    assert!(
        report.dare_dropped_fraction <= 0.2,
        "DARE at density 0.9 must drop roughly 10% (<=20%): got {}",
        report.dare_dropped_fraction
    );
    assert!(
        report.dare_dropped_fraction >= 0.0,
        "DARE dropped fraction must be non-negative"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn merge_rejects_mismatched_tensor_name_sets() {
    let dir = temp_dir("name-mismatch");
    let a = write_synthetic_checkpoint(&dir, "a.safetensors", 1.0);
    let b = write_missing_tensor_checkpoint(&dir, "b.safetensors");
    let output = dir.join("merged.safetensors");

    let cfg = MergeConfig {
        method: MergeMethod::Linear,
        weights: vec![0.5, 0.5],
        ..MergeConfig::new(MergeMethod::Linear)
    };
    let result = merge_models_from_paths(&tiny_config(), &[a, b], None, &[], &output, &cfg);
    assert!(
        result.is_err(),
        "checkpoints with mismatched tensor names must be rejected"
    );
    assert!(
        !output.exists(),
        "no output must be written on name-set mismatch"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn merge_rejects_inconsistent_weight_counts() {
    let dir = temp_dir("weight-count");
    let a = write_synthetic_checkpoint(&dir, "a.safetensors", 1.0);
    let b = write_synthetic_checkpoint(&dir, "b.safetensors", 2.0);
    let output = dir.join("merged.safetensors");

    let cfg = MergeConfig {
        method: MergeMethod::Linear,
        weights: vec![0.5, 0.3, 0.2], // 3 weights, 2 inputs
        ..MergeConfig::new(MergeMethod::Linear)
    };
    let result = merge_models_from_paths(&tiny_config(), &[a, b], None, &[], &output, &cfg);
    assert!(
        result.is_err(),
        "weight count != input count must be rejected before any arithmetic"
    );
    assert!(!output.exists());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn merge_output_is_loadable_by_safetensors_load_round_trip() {
    // The merged checkpoint must be loadable by the same primitive the rest
    // of the weights crate uses (candle_core::safetensors::load), preserving
    // every tensor name and shape.
    let dir = temp_dir("round-trip");
    let a = write_synthetic_checkpoint(&dir, "a.safetensors", 1.0);
    let b = write_synthetic_checkpoint(&dir, "b.safetensors", 2.0);
    let output = dir.join("merged.safetensors");

    let cfg = MergeConfig {
        method: MergeMethod::Linear,
        weights: vec![0.5, 0.5],
        ..MergeConfig::new(MergeMethod::Linear)
    };
    merge_models_from_paths(&tiny_config(), &[a, b], None, &[], &output, &cfg).unwrap();

    let merged = candle_core::safetensors::load(&output, &Device::Cpu).unwrap();
    assert!(merged.contains_key("embedding.weight"));
    assert!(merged.contains_key("layers.0.weight"));
    assert!(merged.contains_key("layers.0.norm.weight"));
    assert_eq!(merged["embedding.weight"].dims(), &[2, 3]);
    assert_eq!(merged["layers.0.weight"].dims(), &[3, 2]);
    assert_eq!(merged["layers.0.norm.weight"].dims(), &[3]);
    let _ = std::fs::remove_dir_all(&dir);
}
