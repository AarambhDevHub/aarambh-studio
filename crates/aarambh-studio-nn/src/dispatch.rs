use aarambh_studio_core::DispatchKind;
use candle_core::{D, DType, Device, Result, Tensor};

use crate::ffn::SwiGluFfn;

/// Combine per-expert outputs using dense masked expert weights.
///
/// This is the v2 §26 / v3 §40 dispatch contract: every routed expert
/// computes on every token and the router weights mask the non-selected
/// contributions. Kept as the [`DispatchKind::DenseMasked`] reference and as
/// the CPU fallback for [`DispatchKind::Sparse`] (see
/// [`effective_dispatch_kind`]).
pub fn dense_weighted_dispatch(
    expert_outputs: &[Tensor],
    dispatch_weights: &Tensor,
) -> Result<Tensor> {
    if expert_outputs.is_empty() {
        candle_core::bail!("dense_weighted_dispatch requires at least one expert output");
    }
    let first_dims = expert_outputs[0].dims().to_vec();
    if first_dims.len() != 3 {
        candle_core::bail!(
            "expert outputs must have shape [batch, seq, hidden], got {:?}",
            first_dims
        );
    }
    for output in expert_outputs.iter().skip(1) {
        if output.dims() != first_dims {
            candle_core::bail!(
                "all expert outputs must have the same shape, expected {:?}, got {:?}",
                first_dims,
                output.dims()
            );
        }
    }

    let expected_weights = [first_dims[0], first_dims[1], expert_outputs.len()];
    if dispatch_weights.dims() != expected_weights {
        candle_core::bail!(
            "dispatch weights must have shape {:?}, got {:?}",
            expected_weights,
            dispatch_weights.dims()
        );
    }

    let stacked = Tensor::stack(expert_outputs, 2)?;
    let weights = dispatch_weights
        .to_dtype(stacked.dtype())?
        .unsqueeze(D::Minus1)?;
    stacked.broadcast_mul(&weights)?.sum(2)
}

/// Resolve the dispatch strategy that actually runs for a configured kind on
/// a given device.
///
/// [`DispatchKind::Sparse`] only activates on a CUDA device — the real
/// throughput win lives in the grouped-GEMM path executed through candle's
/// CUDA backend (cuBLAS matmuls per expert token group). On every other
/// device the CPU path continues to use [`DispatchKind::DenseMasked`]
/// *regardless of configuration*, documented plainly as "GPU only pays off"
/// rather than silently downgrading the request. This honesty discipline
/// mirrors v2 §29's speculative-decoding speed-claim methodology.
pub fn effective_dispatch_kind(configured: DispatchKind, device: &Device) -> DispatchKind {
    match configured {
        DispatchKind::DenseMasked => DispatchKind::DenseMasked,
        DispatchKind::Sparse => {
            if device.is_cuda() {
                DispatchKind::Sparse
            } else {
                DispatchKind::DenseMasked
            }
        }
    }
}

/// Run the routed-expert portion of a Mixture-of-Experts layer using sparse
/// grouped dispatch (v4 Phase 43).
///
/// Each token is routed to its `top_k` selected experts (given by `indices`)
/// with per-assignment softmax `weights`. Tokens are grouped by router
/// assignment into per-expert contiguous batches; each expert's SwiGLU
/// feed-forward matmul then executes *only* on its assigned token group —
/// not the full sequence — and the weighted results are scattered back into
/// the original token order. This is numerically equivalent to
/// [`dense_weighted_dispatch`] (same tokens, same weights, same per-expert
/// reduction order) but skips all non-routed expert computation.
///
/// The implementation is fully differentiable through candle's `gather`,
/// `index_select`, and `index_add` ops, so router logits, expert
/// parameters, and input activations all receive correct gradients. The
/// permutation that groups tokens by expert is computed with
/// `arg_sort_last_dim` (a no-grad index operation), which is correct because
/// discrete routing assignments carry no gradient.
///
/// `train` selects each expert's training forward path (`forward_train`)
/// when true, otherwise the inference path (`forward`). Shared experts are
/// not handled here — the caller adds them afterwards, unchanged by the
/// dispatch kind.
pub fn sparse_grouped_dispatch(
    x: &Tensor,
    experts: &[SwiGluFfn],
    indices: &Tensor,
    weights: &Tensor,
    train: bool,
) -> Result<Tensor> {
    let x_dims = x.dims();
    if x_dims.len() != 3 {
        candle_core::bail!(
            "sparse_grouped_dispatch input must have shape [batch, seq, hidden], got {:?}",
            x_dims
        );
    }
    let (batch, seq_len, hidden) = (x_dims[0], x_dims[1], x_dims[2]);
    let idx_dims = indices.dims();
    if idx_dims.len() != 3 || idx_dims[0] != batch || idx_dims[1] != seq_len {
        candle_core::bail!(
            "sparse_grouped_dispatch indices must have shape [batch, seq, top_k] matching input, got {:?}",
            idx_dims
        );
    }
    let top_k = idx_dims[2];
    if experts.is_empty() {
        candle_core::bail!("sparse_grouped_dispatch requires at least one expert");
    }
    if weights.dims() != idx_dims {
        candle_core::bail!(
            "sparse_grouped_dispatch weights must match indices shape {:?}, got {:?}",
            idx_dims,
            weights.dims()
        );
    }

    let device = x.device();
    let num_experts = experts.len();
    let num_tokens = batch * seq_len;
    let num_assignments = num_tokens * top_k;
    if num_assignments == 0 {
        return Tensor::zeros((batch, seq_len, hidden), x.dtype(), device);
    }

    // Flatten tokens and router assignments into rank-1 tensors.
    let x_flat = x.contiguous()?.reshape((num_tokens, hidden))?;
    let indices_flat = indices.to_dtype(DType::U32)?.contiguous()?.flatten_all()?;
    let weights_flat = weights.to_dtype(DType::F32)?.contiguous()?.flatten_all()?;

    // token_ids[i] = i / top_k — the flattened token index for assignment i.
    let token_ids = Tensor::arange(0u32, num_tokens as u32, device)?
        .unsqueeze(1)?
        .broadcast_as((num_tokens, top_k))?
        .contiguous()?
        .flatten_all()?;

    // Group assignments by expert id (ascending) so each expert's token
    // group is contiguous. The sort order is a discrete permutation with no
    // gradient; the gathered weights remain differentiable.
    let sort_order = indices_flat.arg_sort_last_dim(true)?;
    let grouped_token_ids = token_ids.gather(&sort_order, 0)?.contiguous()?;
    let grouped_weights = weights_flat.gather(&sort_order, 0)?.contiguous()?;
    let grouped_expert_ids = indices_flat
        .gather(&sort_order, 0)?
        .to_dtype(DType::U32)?
        .to_vec1::<u32>()?;

    // Per-expert (start, count) boundaries, scanned on the host. The expert
    // ids are sorted ascending, so expert e occupies a contiguous range.
    let mut counts = vec![0usize; num_experts];
    for &expert_id in &grouped_expert_ids {
        let expert_id = expert_id as usize;
        if expert_id < num_experts {
            counts[expert_id] += 1;
        }
    }

    let out_dtype = x.dtype();
    let mut routed = Tensor::zeros((num_tokens, hidden), out_dtype, device)?;
    let mut start = 0usize;
    for (expert_idx, &count) in counts.iter().enumerate() {
        if count == 0 {
            // No token routed to this expert — skip it entirely. This is the
            // core of the sparse win: the expert's matmul never runs.
            continue;
        }
        let group_token_ids = grouped_token_ids.narrow(0, start, count)?.contiguous()?;
        let group_weights = grouped_weights.narrow(0, start, count)?;
        let expert_input = x_flat.index_select(&group_token_ids, 0)?;
        let expert_output = if train {
            experts[expert_idx].forward_train(&expert_input)?
        } else {
            experts[expert_idx].forward(&expert_input)?
        };
        let group_weights = group_weights.to_dtype(expert_output.dtype())?;
        let weighted = expert_output.broadcast_mul(&group_weights.unsqueeze(1)?)?;
        routed = routed.index_add(&group_token_ids, &weighted, 0)?;
        start += count;
    }

    routed.reshape((batch, seq_len, hidden))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Module};
    use candle_nn::{Init, VarBuilder, VarMap, linear_no_bias};

    fn test_expert(vb: VarBuilder<'_>, hidden: usize, intermediate: usize) -> SwiGluFfn {
        SwiGluFfn::new(
            linear_no_bias(hidden, intermediate, vb.pp("w_gate")).unwrap(),
            linear_no_bias(hidden, intermediate, vb.pp("w_up")).unwrap(),
            linear_no_bias(intermediate, hidden, vb.pp("w_down")).unwrap(),
        )
    }

    fn build_experts(
        vb: VarBuilder<'_>,
        hidden: usize,
        intermediate: usize,
        num_experts: usize,
    ) -> Vec<SwiGluFfn> {
        (0..num_experts)
            .map(|idx| test_expert(vb.pp(idx), hidden, intermediate))
            .collect()
    }

    fn max_abs_diff(left: &Tensor, right: &Tensor) -> f32 {
        (left - right)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .fold(0.0f32, |acc, value| acc.max(value))
    }

    #[test]
    fn dense_dispatch_weighted_sum_matches_shape() {
        let device = Device::Cpu;
        let a = Tensor::ones((1, 2, 4), DType::F32, &device).unwrap();
        let b = a.affine(2.0, 0.0).unwrap();
        let weights = Tensor::from_vec(vec![1.0f32, 0.0, 0.25, 0.75], (1, 2, 2), &device).unwrap();
        let out = dense_weighted_dispatch(&[a, b], &weights).unwrap();
        assert_eq!(out.dims(), &[1, 2, 4]);
        let values = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(values[..4].iter().all(|value| (*value - 1.0).abs() < 1e-6));
        assert!(values[4..].iter().all(|value| (*value - 1.75).abs() < 1e-6));
    }

    #[test]
    fn effective_dispatch_kind_falls_back_to_dense_masked_off_cuda() {
        assert_eq!(
            effective_dispatch_kind(DispatchKind::DenseMasked, &Device::Cpu),
            DispatchKind::DenseMasked
        );
        assert_eq!(
            effective_dispatch_kind(DispatchKind::Sparse, &Device::Cpu),
            DispatchKind::DenseMasked
        );
    }

    #[test]
    fn sparse_dispatch_output_matches_dense_masked_reference_within_tolerance() {
        // Correctness first, exactly like v2 §26 shipped DenseMasked first —
        // Sparse must reproduce the same numbers, just faster.
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let hidden = 8;
        let intermediate = 8;
        let num_experts = 3;
        let top_k = 1;
        let experts = build_experts(vb.pp("experts"), hidden, intermediate, num_experts);
        let router = linear_no_bias(hidden, num_experts, vb.pp("router")).unwrap();
        let x = vb
            .get_with_hints(
                (2, 3, hidden),
                "x",
                Init::Randn {
                    mean: 0.0,
                    stdev: 0.1,
                },
            )
            .unwrap();

        let logits = router.forward(&x).unwrap();
        let gating = crate::moe::top_k_gating(&logits, top_k).unwrap();

        // Dense reference: every expert on every token, masked by dispatch weights.
        let expert_outputs = experts
            .iter()
            .map(|expert| expert.forward(&x).unwrap())
            .collect::<Vec<_>>();
        let dense = dense_weighted_dispatch(&expert_outputs, &gating.dispatch_weights).unwrap();

        let sparse =
            sparse_grouped_dispatch(&x, &experts, &gating.indices, &gating.weights, false).unwrap();
        assert_eq!(sparse.dims(), dense.dims());
        let diff = max_abs_diff(&dense, &sparse);
        assert!(diff < 1e-5, "sparse vs dense max diff was {diff}");
    }

    #[test]
    fn sparse_dispatch_supports_top_k_greater_than_one() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let hidden = 8;
        let intermediate = 8;
        let num_experts = 4;
        let top_k = 2;
        let experts = build_experts(vb.pp("experts"), hidden, intermediate, num_experts);
        let router = linear_no_bias(hidden, num_experts, vb.pp("router")).unwrap();
        let x = vb
            .get_with_hints(
                (2, 4, hidden),
                "x",
                Init::Randn {
                    mean: 0.0,
                    stdev: 0.1,
                },
            )
            .unwrap();

        let logits = router.forward(&x).unwrap();
        let gating = crate::moe::top_k_gating(&logits, top_k).unwrap();
        let expert_outputs = experts
            .iter()
            .map(|expert| expert.forward(&x).unwrap())
            .collect::<Vec<_>>();
        let dense = dense_weighted_dispatch(&expert_outputs, &gating.dispatch_weights).unwrap();
        let sparse =
            sparse_grouped_dispatch(&x, &experts, &gating.indices, &gating.weights, false).unwrap();
        assert_eq!(sparse.dims(), dense.dims());
        let diff = max_abs_diff(&dense, &sparse);
        assert!(diff < 1e-5, "top_k=2 sparse vs dense max diff was {diff}");
    }

    #[test]
    fn sparse_dispatch_backward_reaches_router_and_expert_weights() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let hidden = 6;
        let intermediate = 6;
        let num_experts = 2;
        let top_k = 1;
        let experts = build_experts(vb.pp("experts"), hidden, intermediate, num_experts);
        let router = linear_no_bias(hidden, num_experts, vb.pp("router")).unwrap();
        let x = vb
            .get_with_hints(
                (1, 2, hidden),
                "x",
                Init::Randn {
                    mean: 0.0,
                    stdev: 0.1,
                },
            )
            .unwrap();

        let logits = router.forward(&x).unwrap();
        let gating = crate::moe::top_k_gating(&logits, top_k).unwrap();
        let sparse =
            sparse_grouped_dispatch(&x, &experts, &gating.indices, &gating.weights, true).unwrap();
        let loss = sparse.sum_all().unwrap();
        let gradients = loss.backward().unwrap();
        let variables = varmap.data().lock().unwrap();

        // Router logits receive gradients through the gathered weights.
        assert!(
            gradients
                .get(variables["router.weight"].as_tensor())
                .is_some(),
            "router weight received no gradient"
        );
        // At least one routed expert's down projection receives gradients.
        assert!(
            gradients
                .get(variables["experts.0.w_down.weight"].as_tensor())
                .is_some()
                || gradients
                    .get(variables["experts.1.w_down.weight"].as_tensor())
                    .is_some(),
            "no expert down projection received a gradient"
        );
    }

    #[test]
    fn sparse_dispatch_empty_expert_group_is_skipped() {
        // An expert with no routed tokens must not error and must match dense.
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let hidden = 4;
        let intermediate = 4;
        let num_experts = 3;
        let experts = build_experts(vb.pp("experts"), hidden, intermediate, num_experts);
        let x = vb
            .get_with_hints(
                (1, 2, hidden),
                "x",
                Init::Randn {
                    mean: 0.0,
                    stdev: 0.1,
                },
            )
            .unwrap();

        // Route every token to expert 0 only — experts 1 and 2 get no tokens.
        let indices = Tensor::zeros((1, 2, 1), DType::U32, &device).unwrap();
        let weights = Tensor::ones((1, 2, 1), DType::F32, &device).unwrap();
        let sparse = sparse_grouped_dispatch(&x, &experts, &indices, &weights, false).unwrap();

        let expert0 = experts[0].forward(&x).unwrap();
        let diff = max_abs_diff(&expert0, &sparse);
        assert!(
            diff < 1e-5,
            "empty-group sparse vs expert 0 diff was {diff}"
        );
    }

    #[test]
    fn sparse_dispatch_matches_dense_with_shared_expert_summed_separately() {
        // Shared experts are added outside the sparse dispatch; verify the
        // combined routed+sparse output still matches dense-with-shared.
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let hidden = 8;
        let intermediate = 8;
        let num_experts = 2;
        let top_k = 1;
        let experts = build_experts(vb.pp("experts"), hidden, intermediate, num_experts);
        let shared = test_expert(vb.pp("shared"), hidden, intermediate);
        let router = linear_no_bias(hidden, num_experts, vb.pp("router")).unwrap();
        let x = vb
            .get_with_hints(
                (2, 3, hidden),
                "x",
                Init::Randn {
                    mean: 0.0,
                    stdev: 0.1,
                },
            )
            .unwrap();

        let logits = router.forward(&x).unwrap();
        let gating = crate::moe::top_k_gating(&logits, top_k).unwrap();
        let expert_outputs = experts
            .iter()
            .map(|expert| expert.forward(&x).unwrap())
            .collect::<Vec<_>>();
        let dense_routed =
            dense_weighted_dispatch(&expert_outputs, &gating.dispatch_weights).unwrap();
        let dense_total = (dense_routed + shared.forward(&x).unwrap()).unwrap();

        let sparse_routed =
            sparse_grouped_dispatch(&x, &experts, &gating.indices, &gating.weights, false).unwrap();
        let sparse_total = (sparse_routed + shared.forward(&x).unwrap()).unwrap();

        let diff = max_abs_diff(&dense_total, &sparse_total);
        assert!(diff < 1e-5, "sparse+shared vs dense+shared diff was {diff}");
    }
}
