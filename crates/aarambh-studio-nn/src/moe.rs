use std::collections::HashMap;

use aarambh_studio_core::{DispatchKind, MoeConfig};
use aarambh_studio_quant::QatLinear;
use candle_core::{D, DType, Result, Tensor};

use crate::dispatch::{effective_dispatch_kind, sparse_grouped_dispatch};
use crate::ffn::SwiGluFfn;

#[derive(Debug)]
/// Router output after selecting top-k experts per token.
pub struct GatingOutput {
    /// Selected expert indices with shape `[batch, seq, top_k]`.
    pub indices: Tensor,
    /// Selected expert probabilities with shape `[batch, seq, top_k]`.
    pub weights: Tensor,
    /// Dense expert dispatch weights with shape `[batch, seq, num_experts]`.
    pub dispatch_weights: Tensor,
    /// Differentiable load-balancing auxiliary loss.
    pub aux_loss: Tensor,
    /// Per-expert selected-token fraction, normalized to sum to 1.0.
    pub expert_utilization: Vec<f32>,
}

#[derive(Debug, Default)]
/// Aggregated MoE metadata collected during a model forward pass.
pub struct MoeForwardStats {
    aux_losses: Vec<Tensor>,
    expert_utilization: Vec<f32>,
    routed_experts_by_layer: Vec<Vec<usize>>,
}

impl MoeForwardStats {
    /// Add one MoE layer's auxiliary loss and expert-utilization summary.
    pub fn record(
        &mut self,
        aux_loss: Tensor,
        expert_utilization: &[f32],
        top_k: usize,
    ) -> Result<()> {
        if self.expert_utilization.is_empty() {
            self.expert_utilization = vec![0.0; expert_utilization.len()];
        }
        if self.expert_utilization.len() != expert_utilization.len() {
            candle_core::bail!("cannot aggregate expert utilization with different expert counts");
        }
        for (dst, src) in self
            .expert_utilization
            .iter_mut()
            .zip(expert_utilization.iter())
        {
            *dst += *src;
        }
        let mut ranked = expert_utilization
            .iter()
            .copied()
            .enumerate()
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        let mut routed = ranked
            .into_iter()
            .take(top_k)
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        routed.sort_unstable();
        self.routed_experts_by_layer.push(routed);
        self.aux_losses.push(aux_loss);
        Ok(())
    }

    /// Return the average auxiliary loss across recorded MoE layers.
    pub fn aux_loss(&self) -> Result<Option<Tensor>> {
        let Some((first, rest)) = self.aux_losses.split_first() else {
            return Ok(None);
        };
        let mut sum = first.clone();
        for loss in rest {
            sum = (&sum + loss)?;
        }
        Ok(Some(sum.affine(1.0 / self.aux_losses.len() as f64, 0.0)?))
    }

    /// Return average selected-token fraction per expert.
    pub fn expert_utilization(&self) -> Vec<f32> {
        if self.aux_losses.is_empty() {
            return Vec::new();
        }
        self.expert_utilization
            .iter()
            .map(|value| *value / self.aux_losses.len() as f32)
            .collect()
    }

    /// Return true when no MoE layers have recorded stats.
    pub fn is_empty(&self) -> bool {
        self.aux_losses.is_empty()
    }

    /// Return sorted top-routed expert sets for each recorded MoE layer.
    pub fn routed_experts_by_layer(&self) -> Vec<Vec<usize>> {
        self.routed_experts_by_layer.clone()
    }
}

#[derive(Debug, Clone)]
/// Always-active SwiGLU experts whose outputs are summed for every token.
pub struct SharedExpertPath {
    experts: Vec<SwiGluFfn>,
}

impl SharedExpertPath {
    /// Create an always-active shared expert path.
    pub fn new(experts: Vec<SwiGluFfn>) -> Self {
        Self { experts }
    }

    /// Create an empty shared expert path for Phase 22 compatibility.
    pub fn empty() -> Self {
        Self::new(Vec::new())
    }

    /// Run all shared experts and sum their outputs.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_inner(x, false, 0, None)
    }

    /// Run all shared experts through the differentiable training path.
    pub fn forward_train(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_inner(x, true, 0, None)
    }

    /// Run all shared experts while recording quantisation calibration inputs.
    pub fn forward_with_capture(
        &self,
        x: &Tensor,
        layer_idx: usize,
        capture: &mut HashMap<String, Tensor>,
    ) -> Result<Tensor> {
        self.forward_inner(x, false, layer_idx, Some(capture))
    }

    /// Return the shared expert layers.
    pub fn experts(&self) -> &[SwiGluFfn] {
        &self.experts
    }

    fn forward_inner(
        &self,
        x: &Tensor,
        train: bool,
        layer_idx: usize,
        mut capture: Option<&mut HashMap<String, Tensor>>,
    ) -> Result<Tensor> {
        let mut output = None;
        for (expert_idx, expert) in self.experts.iter().enumerate() {
            let expert_output = if let Some(capture) = capture.as_deref_mut() {
                forward_expert_with_capture(
                    expert,
                    x,
                    &format!("blocks.{layer_idx}.ffn.shared_experts.{expert_idx}"),
                    capture,
                )?
            } else if train {
                expert.forward_train(x)?
            } else {
                expert.forward(x)?
            };
            output = Some(match output {
                Some(accumulator) => (accumulator + expert_output)?,
                None => expert_output,
            });
        }
        output.ok_or_else(|| candle_core::Error::msg("shared expert path is empty"))
    }
}

#[derive(Debug, Clone)]
/// Router followed by fine-grained routed experts and optional shared experts.
pub struct MoeFfn {
    config: MoeConfig,
    router: QatLinear,
    experts: Vec<SwiGluFfn>,
    shared_experts: SharedExpertPath,
}

impl MoeFfn {
    /// Create a Phase 22-compatible MoE layer without shared experts.
    pub fn new(config: MoeConfig, router: impl Into<QatLinear>, experts: Vec<SwiGluFfn>) -> Self {
        Self::new_with_shared(config, router, experts, SharedExpertPath::empty())
    }

    /// Create a fine-grained MoE layer with an explicit shared expert path.
    pub fn new_with_shared(
        config: MoeConfig,
        router: impl Into<QatLinear>,
        experts: Vec<SwiGluFfn>,
        shared_experts: SharedExpertPath,
    ) -> Self {
        Self {
            config,
            router: router.into(),
            experts,
            shared_experts,
        }
    }

    /// Run inference through top-k expert routing.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_inner(x, false, None)
    }

    /// Run training through top-k expert routing.
    pub fn forward_train(&self, x: &Tensor, stats: Option<&mut MoeForwardStats>) -> Result<Tensor> {
        self.forward_inner(x, true, stats)
    }

    /// Run the MoE layer while recording calibration activations.
    pub fn forward_with_capture(
        &self,
        x: &Tensor,
        layer_idx: usize,
        capture: &mut HashMap<String, Tensor>,
    ) -> Result<Tensor> {
        capture.insert(format!("blocks.{layer_idx}.ffn.router.weight"), x.clone());
        self.forward_inner_with_expert_capture(x, false, layer_idx, capture, None)
    }

    /// Return the router weight tensor.
    pub fn router_weight(&self) -> &Tensor {
        self.router.weight()
    }

    /// Return the expert layers.
    pub fn experts(&self) -> &[SwiGluFfn] {
        &self.experts
    }

    /// Return the always-active shared expert path.
    pub fn shared_experts(&self) -> &SharedExpertPath {
        &self.shared_experts
    }

    /// Return the validated MoE configuration.
    pub fn config(&self) -> &MoeConfig {
        &self.config
    }

    /// Return the configured routed-expert dispatch strategy (v4 Phase 43).
    ///
    /// Note that the *effective* dispatch kind may differ at runtime — see
    /// [`effective_dispatch_kind`], which falls back to
    /// [`DispatchKind::DenseMasked`] on non-CUDA devices.
    pub fn dispatch_kind(&self) -> DispatchKind {
        self.config.dispatch
    }

    fn forward_inner(
        &self,
        x: &Tensor,
        train: bool,
        stats: Option<&mut MoeForwardStats>,
    ) -> Result<Tensor> {
        self.forward_inner_with_expert_capture(x, train, 0, &mut HashMap::new(), stats)
    }

    fn forward_inner_with_expert_capture(
        &self,
        x: &Tensor,
        train: bool,
        layer_idx: usize,
        capture: &mut HashMap<String, Tensor>,
        stats: Option<&mut MoeForwardStats>,
    ) -> Result<Tensor> {
        let routed_experts = self
            .config
            .routed_expert_count()
            .map_err(|err| candle_core::Error::msg(err.to_string()))?;
        if self.experts.len() != routed_experts {
            candle_core::bail!(
                "MoE expert count mismatch: config has {}, layer has {}",
                routed_experts,
                self.experts.len()
            );
        }
        if self.shared_experts.experts().len() != self.config.num_shared_experts {
            candle_core::bail!(
                "MoE shared expert count mismatch: config has {}, layer has {}",
                self.config.num_shared_experts,
                self.shared_experts.experts().len()
            );
        }
        let logits = self.router.forward(x)?;
        let gating = top_k_gating(&logits, self.config.top_k)?;
        if let Some(stats) = stats {
            stats.record(
                gating.aux_loss.clone(),
                &gating.expert_utilization,
                self.config.top_k,
            )?;
        }

        // Phase 43: select the routed-expert dispatch path. Sparse grouped
        // dispatch only activates on CUDA (see `effective_dispatch_kind`);
        // the CPU path keeps DenseMasked regardless of configuration,
        // documented plainly as "GPU only pays off." Calibration capture
        // always uses the dense reference so QAT observes full per-expert
        // activation distributions. The load-balancing auxiliary loss above
        // is computed in `top_k_gating` before dispatch, so it is identical
        // for both kinds — Sparse changes the compute path only.
        let effective = effective_dispatch_kind(self.config.dispatch, x.device());
        let use_dense = effective == DispatchKind::DenseMasked || !capture.is_empty();
        let mut output = if use_dense {
            let mut routed_output = None;
            for (expert_idx, expert) in self.experts.iter().enumerate() {
                let expert_output = if capture.is_empty() {
                    if train {
                        expert.forward_train(x)?
                    } else {
                        expert.forward(x)?
                    }
                } else {
                    forward_expert_with_capture(
                        expert,
                        x,
                        &format!("blocks.{layer_idx}.ffn.experts.{expert_idx}"),
                        capture,
                    )?
                };
                let weight = gating
                    .dispatch_weights
                    .narrow(D::Minus1, expert_idx, 1)?
                    .to_dtype(expert_output.dtype())?;
                let weighted = expert_output.broadcast_mul(&weight)?;
                routed_output = Some(match routed_output {
                    Some(accumulator) => (accumulator + weighted)?,
                    None => weighted,
                });
            }
            routed_output.ok_or_else(|| candle_core::Error::msg("MoE has no experts"))?
        } else {
            sparse_grouped_dispatch(x, &self.experts, &gating.indices, &gating.weights, train)?
        };
        if !self.shared_experts.experts().is_empty() {
            let shared = if capture.is_empty() {
                if train {
                    self.shared_experts.forward_train(x)?
                } else {
                    self.shared_experts.forward(x)?
                }
            } else {
                self.shared_experts
                    .forward_with_capture(x, layer_idx, capture)?
            };
            output = (output + shared)?;
        }
        Ok(output)
    }
}

fn forward_expert_with_capture(
    expert: &SwiGluFfn,
    x: &Tensor,
    prefix: &str,
    capture: &mut HashMap<String, Tensor>,
) -> Result<Tensor> {
    capture.insert(format!("{prefix}.w_gate.weight"), x.clone());
    capture.insert(format!("{prefix}.w_up.weight"), x.clone());
    let gate = expert.w_gate_forward(x)?;
    let up = expert.w_up_forward(x)?;
    let hidden = aarambh_studio_kernel::fused_ffn::fused_swiglu(&gate, &up).or_else(|_| {
        let gate = candle_nn::ops::silu(&gate)?;
        gate * up
    })?;
    capture.insert(format!("{prefix}.w_down.weight"), hidden.clone());
    expert.w_down_forward(&hidden)
}

/// Select top-k experts and produce dense dispatch weights.
pub fn top_k_gating(logits: &Tensor, top_k: usize) -> Result<GatingOutput> {
    let dims = logits.dims();
    if dims.len() != 3 {
        candle_core::bail!(
            "router logits must have shape [batch, seq, num_experts], got {:?}",
            dims
        );
    }
    let (batch, seq_len, num_experts) = (dims[0], dims[1], dims[2]);
    if top_k == 0 || top_k > num_experts {
        candle_core::bail!("top_k must be in 1..={num_experts}, got {top_k}");
    }

    let logits_f32 = logits.to_dtype(DType::F32)?.contiguous()?;
    let sorted_indices = logits_f32.arg_sort_last_dim(false)?;
    let indices = sorted_indices.narrow(D::Minus1, 0, top_k)?.contiguous()?;
    let selected_logits = logits_f32.gather(&indices, D::Minus1)?;
    let weights = candle_nn::ops::softmax(&selected_logits, D::Minus1)?;

    let mut expert_weights = Vec::with_capacity(num_experts);
    let mut expert_masks = Vec::with_capacity(num_experts);
    for expert_idx in 0..num_experts {
        let mask = indices.eq(expert_idx as u32)?.to_dtype(DType::F32)?;
        expert_weights.push(weights.broadcast_mul(&mask)?.sum_keepdim(D::Minus1)?);
        expert_masks.push(mask.sum_keepdim(D::Minus1)?);
    }
    let dispatch_weights = Tensor::cat(&expert_weights.iter().collect::<Vec<_>>(), D::Minus1)?;
    let selected_mask = Tensor::cat(&expert_masks.iter().collect::<Vec<_>>(), D::Minus1)?;

    let router_probs = candle_nn::ops::softmax(&logits_f32, D::Minus1)?;
    let token_count = (batch * seq_len) as f64;
    let router_prob_mean = router_probs.sum((0, 1))?.affine(1.0 / token_count, 0.0)?;
    let dispatch_fraction = selected_mask
        .sum((0, 1))?
        .affine(1.0 / (token_count * top_k as f64), 0.0)?;
    let aux_loss = load_balancing_loss_from_stats(&router_prob_mean, &dispatch_fraction)?;
    let expert_utilization = dispatch_fraction.to_vec1::<f32>()?;

    Ok(GatingOutput {
        indices,
        weights,
        dispatch_weights,
        aux_loss,
        expert_utilization,
    })
}

/// Compute shifted Switch-style load-balancing loss from per-expert means.
pub fn load_balancing_loss_from_stats(
    router_prob_mean: &Tensor,
    dispatch_fraction: &Tensor,
) -> Result<Tensor> {
    if router_prob_mean.dims().len() != 1 || dispatch_fraction.dims() != router_prob_mean.dims() {
        candle_core::bail!(
            "router probability and dispatch stats must be same rank-1 shape, got {:?} and {:?}",
            router_prob_mean.dims(),
            dispatch_fraction.dims()
        );
    }
    let num_experts = router_prob_mean.dims()[0] as f64;
    (router_prob_mean * dispatch_fraction)?
        .sum_all()?
        .affine(num_experts, -1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense_weighted_dispatch;
    use candle_core::{DType, Device};
    use candle_nn::{Init, VarBuilder, VarMap, linear_no_bias};

    fn test_expert(vb: VarBuilder<'_>, hidden: usize, intermediate: usize) -> SwiGluFfn {
        SwiGluFfn::new(
            linear_no_bias(hidden, intermediate, vb.pp("w_gate")).unwrap(),
            linear_no_bias(hidden, intermediate, vb.pp("w_up")).unwrap(),
            linear_no_bias(intermediate, hidden, vb.pp("w_down")).unwrap(),
        )
    }

    #[test]
    fn top_k_gating_selects_correct_number_of_experts_per_token() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(
            vec![0.0f32, 3.0, 1.0, 2.0, 9.0, 8.0, 7.0, 6.0],
            (1, 2, 4),
            &device,
        )
        .unwrap();
        let gating = top_k_gating(&logits, 2).unwrap();
        assert_eq!(gating.indices.dims(), &[1, 2, 2]);
        assert_eq!(gating.weights.dims(), &[1, 2, 2]);
        assert_eq!(gating.dispatch_weights.dims(), &[1, 2, 4]);
        let sums = gating
            .weights
            .sum(D::Minus1)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        for row in sums.into_iter().flatten() {
            assert!((row - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn top_one_gating_backward_reaches_router_logits() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let logits = vb
            .get_with_hints(
                (1, 4, 2),
                "logits",
                Init::Randn {
                    mean: 0.0,
                    stdev: 0.1,
                },
            )
            .unwrap();
        let gating = top_k_gating(&logits, 1).unwrap();
        let expert_values = Tensor::from_vec(vec![1.0f32, 2.0], (1, 1, 2), &device).unwrap();
        let loss = gating
            .dispatch_weights
            .broadcast_mul(&expert_values)
            .unwrap()
            .sum_all()
            .unwrap();
        let gradients = loss.backward().unwrap();
        let variables = varmap.data().lock().unwrap();
        assert!(gradients.get(variables["logits"].as_tensor()).is_some());
    }

    #[test]
    fn load_balancing_loss_is_zero_at_perfectly_uniform_routing() {
        let device = Device::Cpu;
        let probs = Tensor::from_vec(vec![0.25f32; 4], (4,), &device).unwrap();
        let dispatch = Tensor::from_vec(vec![0.25f32; 4], (4,), &device).unwrap();
        let loss = load_balancing_loss_from_stats(&probs, &dispatch)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(loss.abs() < 1e-6, "loss was {loss}");
    }

    #[test]
    fn moe_ffn_output_shape_matches_dense_ffn_output_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let cfg = MoeConfig {
            num_experts: 2,
            top_k: 1,
            expert_ffn_dim: 16,
            aux_loss_weight: 0.01,
            every_n_layers: 1,
            ..MoeConfig::default()
        };
        let router = linear_no_bias(8, 2, vb.pp("router")).unwrap();
        let experts = (0..2)
            .map(|idx| {
                let expert_vb = vb.pp("experts").pp(idx);
                SwiGluFfn::new(
                    linear_no_bias(8, 16, expert_vb.pp("w_gate")).unwrap(),
                    linear_no_bias(8, 16, expert_vb.pp("w_up")).unwrap(),
                    linear_no_bias(16, 8, expert_vb.pp("w_down")).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let moe = MoeFfn::new(cfg, router, experts);
        let x = vb
            .get_with_hints(
                (2, 3, 8),
                "x",
                Init::Randn {
                    mean: 0.0,
                    stdev: 0.02,
                },
            )
            .unwrap();
        let out = moe.forward(&x).unwrap();
        assert_eq!(out.dims(), &[2, 3, 8]);
    }

    #[test]
    fn shared_expert_output_is_added_for_every_token_and_excluded_from_balancing() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let routed = vec![
            test_expert(vb.pp("experts").pp(0), 8, 8),
            test_expert(vb.pp("experts").pp(1), 8, 8),
        ];
        let shared = test_expert(vb.pp("shared_experts").pp(0), 8, 8);
        let router = linear_no_bias(8, 2, vb.pp("router")).unwrap();
        let base_config = MoeConfig {
            num_experts: 2,
            top_k: 1,
            expert_ffn_dim: 8,
            every_n_layers: 1,
            ..MoeConfig::default()
        };
        let base = MoeFfn::new(base_config.clone(), router.clone(), routed.clone());
        let with_shared = MoeFfn::new_with_shared(
            MoeConfig {
                num_shared_experts: 1,
                ..base_config
            },
            router,
            routed,
            SharedExpertPath::new(vec![shared.clone()]),
        );
        let x = vb
            .get_with_hints(
                (2, 3, 8),
                "x",
                Init::Randn {
                    mean: 0.0,
                    stdev: 0.1,
                },
            )
            .unwrap();
        let mut base_stats = MoeForwardStats::default();
        let mut shared_stats = MoeForwardStats::default();
        let base_output = base.forward_train(&x, Some(&mut base_stats)).unwrap();
        let shared_output = with_shared
            .forward_train(&x, Some(&mut shared_stats))
            .unwrap();
        let expected_delta = shared.forward_train(&x).unwrap();
        let delta_error = ((shared_output - base_output).unwrap() - expected_delta)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(delta_error < 1e-5, "shared output error: {delta_error}");

        assert_eq!(
            base_stats.expert_utilization(),
            shared_stats.expert_utilization()
        );
        let aux_error = (base_stats.aux_loss().unwrap().unwrap()
            - shared_stats.aux_loss().unwrap().unwrap())
        .unwrap()
        .abs()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
        assert!(aux_error < 1e-6, "shared expert changed aux loss");
    }

    #[test]
    fn shared_expert_training_backward_reaches_shared_parameters() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let config = MoeConfig {
            num_experts: 2,
            top_k: 1,
            expert_ffn_dim: 8,
            every_n_layers: 1,
            num_shared_experts: 1,
            ..MoeConfig::default()
        };
        let moe = MoeFfn::new_with_shared(
            config,
            linear_no_bias(8, 2, vb.pp("router")).unwrap(),
            vec![
                test_expert(vb.pp("experts").pp(0), 8, 8),
                test_expert(vb.pp("experts").pp(1), 8, 8),
            ],
            SharedExpertPath::new(vec![test_expert(vb.pp("shared_experts").pp(0), 8, 8)]),
        );
        let x = vb
            .get_with_hints(
                (1, 2, 8),
                "x",
                Init::Randn {
                    mean: 0.0,
                    stdev: 0.1,
                },
            )
            .unwrap();
        let loss = moe.forward_train(&x, None).unwrap().sum_all().unwrap();
        let gradients = loss.backward().unwrap();
        let variables = varmap.data().lock().unwrap();
        assert!(
            gradients
                .get(variables["shared_experts.0.w_down.weight"].as_tensor())
                .is_some()
        );
    }

    #[test]
    fn dispatch_kind_dense_masked_is_bit_identical_to_v2_v3_behaviour() {
        // The default (DenseMasked) MoeFfn must reproduce the v2/v3 dense
        // formula exactly: every expert on every token, masked by the router
        // dispatch weights, plus shared experts. Compared against the
        // standalone `dense_weighted_dispatch` reference utility.
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let config = MoeConfig {
            num_experts: 2,
            top_k: 1,
            expert_ffn_dim: 8,
            every_n_layers: 1,
            num_shared_experts: 1,
            ..MoeConfig::default()
        };
        assert_eq!(config.dispatch, DispatchKind::DenseMasked);
        let experts = vec![
            test_expert(vb.pp("experts").pp(0), 8, 8),
            test_expert(vb.pp("experts").pp(1), 8, 8),
        ];
        let shared = SharedExpertPath::new(vec![test_expert(vb.pp("shared_experts").pp(0), 8, 8)]);
        let moe = MoeFfn::new_with_shared(
            config,
            linear_no_bias(8, 2, vb.pp("router")).unwrap(),
            experts,
            shared,
        );
        let x = vb
            .get_with_hints(
                (2, 3, 8),
                "x",
                Init::Randn {
                    mean: 0.0,
                    stdev: 0.1,
                },
            )
            .unwrap();

        let moe_output = moe.forward(&x).unwrap();
        // Reference: dense masked dispatch of the routed experts + shared.
        let logits = moe.router.forward(&x).unwrap();
        let gating = top_k_gating(&logits, 1).unwrap();
        let expert_outputs = moe
            .experts
            .iter()
            .map(|expert| expert.forward(&x).unwrap())
            .collect::<Vec<_>>();
        let routed_ref =
            dense_weighted_dispatch(&expert_outputs, &gating.dispatch_weights).unwrap();
        let shared_ref = moe.shared_experts.forward(&x).unwrap();
        let reference = (routed_ref + shared_ref).unwrap();
        let diff = ((moe_output - reference)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap())
        .to_scalar::<f32>()
        .unwrap();
        assert!(
            diff == 0.0,
            "DenseMasked output drifted from v2/v3 by {diff}"
        );
    }

    #[test]
    fn load_balancing_loss_value_is_unaffected_by_dispatch_kind() {
        // The auxiliary loss is computed in `top_k_gating` before dispatch,
        // so it must be identical regardless of the configured dispatch kind.
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let base = MoeConfig {
            num_experts: 3,
            top_k: 2,
            expert_ffn_dim: 8,
            every_n_layers: 1,
            ..MoeConfig::default()
        };
        let dense_config = MoeConfig {
            dispatch: DispatchKind::DenseMasked,
            ..base.clone()
        };
        let sparse_config = MoeConfig {
            dispatch: DispatchKind::Sparse,
            ..base
        };
        let build = |config: MoeConfig| {
            MoeFfn::new_with_shared(
                config,
                linear_no_bias(8, 3, vb.pp("router")).unwrap(),
                (0..3)
                    .map(|idx| test_expert(vb.pp("experts").pp(idx), 8, 8))
                    .collect::<Vec<_>>(),
                SharedExpertPath::empty(),
            )
        };
        let dense_moe = build(dense_config);
        let sparse_moe = build(sparse_config);
        let x = vb
            .get_with_hints(
                (2, 4, 8),
                "x",
                Init::Randn {
                    mean: 0.0,
                    stdev: 0.1,
                },
            )
            .unwrap();
        let mut dense_stats = MoeForwardStats::default();
        let mut sparse_stats = MoeForwardStats::default();
        dense_moe.forward_train(&x, Some(&mut dense_stats)).unwrap();
        sparse_moe
            .forward_train(&x, Some(&mut sparse_stats))
            .unwrap();
        let dense_aux = dense_stats.aux_loss().unwrap().unwrap();
        let sparse_aux = sparse_stats.aux_loss().unwrap().unwrap();
        let aux_diff = ((dense_aux - sparse_aux)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap())
        .to_scalar::<f32>()
        .unwrap();
        assert!(
            aux_diff < 1e-6,
            "aux loss changed with dispatch kind: {aux_diff}"
        );
        assert_eq!(
            dense_stats.expert_utilization(),
            sparse_stats.expert_utilization()
        );
    }

    #[test]
    fn sparse_configured_moe_falls_back_to_dense_masked_on_cpu() {
        // On CPU, a Sparse-configured MoeFfn must run the DenseMasked path
        // (the documented "GPU only pays off" policy) and still produce
        // output equivalent to a DenseMasked-configured MoeFfn.
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let base = MoeConfig {
            num_experts: 2,
            top_k: 1,
            expert_ffn_dim: 8,
            every_n_layers: 1,
            ..MoeConfig::default()
        };
        let build = |dispatch| {
            MoeFfn::new_with_shared(
                MoeConfig {
                    dispatch,
                    ..base.clone()
                },
                linear_no_bias(8, 2, vb.pp("router")).unwrap(),
                vec![
                    test_expert(vb.pp("experts").pp(0), 8, 8),
                    test_expert(vb.pp("experts").pp(1), 8, 8),
                ],
                SharedExpertPath::empty(),
            )
        };
        let dense_moe = build(DispatchKind::DenseMasked);
        let sparse_moe = build(DispatchKind::Sparse);
        assert_eq!(sparse_moe.dispatch_kind(), DispatchKind::Sparse);
        assert_eq!(
            effective_dispatch_kind(sparse_moe.dispatch_kind(), &Device::Cpu),
            DispatchKind::DenseMasked
        );
        let x = vb
            .get_with_hints(
                (2, 3, 8),
                "x",
                Init::Randn {
                    mean: 0.0,
                    stdev: 0.1,
                },
            )
            .unwrap();
        let dense_out = dense_moe.forward(&x).unwrap();
        let sparse_out = sparse_moe.forward(&x).unwrap();
        let diff = ((dense_out - sparse_out)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap())
        .to_scalar::<f32>()
        .unwrap();
        assert!(
            diff == 0.0,
            "CPU Sparse fallback drifted from DenseMasked by {diff}"
        );
    }

    #[test]
    fn effective_dispatch_kind_uses_sparse_on_cuda() {
        // Wall-clock throughput is a CUDA-only claim. Skip honestly on CPU,
        // mirroring v2 §29's speculative-decoding speed-claim discipline.
        let device = Device::cuda_if_available(0).unwrap();
        if !device.is_cuda() {
            return;
        }
        assert_eq!(
            effective_dispatch_kind(DispatchKind::Sparse, &device),
            DispatchKind::Sparse
        );
        assert_eq!(
            effective_dispatch_kind(DispatchKind::DenseMasked, &device),
            DispatchKind::DenseMasked
        );
    }

    #[test]
    fn sparse_dispatch_cuda_throughput_exceeds_dense_masked_at_kaggle_gpu_scale() {
        // Wall-clock, not a correctness gate — same honesty discipline v2 §29
        // used for speculative decoding's speed claim. Skipped on CPU.
        let device = Device::cuda_if_available(0).unwrap();
        if !device.is_cuda() {
            return;
        }
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let num_experts = 16;
        let top_k = 2;
        let hidden = 256;
        let intermediate = 256;
        let base = MoeConfig {
            num_experts,
            top_k,
            expert_ffn_dim: intermediate,
            every_n_layers: 1,
            ..MoeConfig::default()
        };
        let build = |dispatch| {
            MoeFfn::new_with_shared(
                MoeConfig {
                    dispatch,
                    ..base.clone()
                },
                linear_no_bias(hidden, num_experts, vb.pp("router")).unwrap(),
                (0..num_experts)
                    .map(|idx| test_expert(vb.pp("experts").pp(idx), hidden, intermediate))
                    .collect::<Vec<_>>(),
                SharedExpertPath::empty(),
            )
        };
        let dense_moe = build(DispatchKind::DenseMasked);
        let sparse_moe = build(DispatchKind::Sparse);
        let x = vb
            .get_with_hints(
                (4, 128, hidden),
                "x",
                Init::Randn {
                    mean: 0.0,
                    stdev: 0.1,
                },
            )
            .unwrap();

        // Warm up both paths so the first kernel compile does not skew timing.
        for _ in 0..3 {
            dense_moe.forward(&x).unwrap();
            sparse_moe.forward(&x).unwrap();
        }

        let iterations = 20;
        let dense_start = std::time::Instant::now();
        for _ in 0..iterations {
            dense_moe.forward(&x).unwrap();
        }
        let dense_elapsed = dense_start.elapsed();

        let sparse_start = std::time::Instant::now();
        for _ in 0..iterations {
            sparse_moe.forward(&x).unwrap();
        }
        let sparse_elapsed = sparse_start.elapsed();

        assert!(
            sparse_elapsed < dense_elapsed,
            "sparse dispatch ({sparse_elapsed:?}) was not faster than dense masked ({dense_elapsed:?}) at Kaggle GPU scale"
        );
    }
}
