use std::path::Path;
use std::time::Instant;

use aarambh_studio_core::{AarambhError, Configurable, ModelConfig, Result, TrainConfig};
use aarambh_studio_data::{Batch, DataLoader};
use aarambh_studio_model::AarambhModel;
use candle_core::DType;
use candle_core::backprop::GradStore;
use candle_nn::{VarBuilder, VarMap};

use crate::checkpoint::{CheckpointManager, TrainState};
use crate::config::DsaTrainingConfig;
use crate::distributed::DistributedContext;
use crate::loss::cross_entropy_loss;
use crate::mtp_loss::{combine_mtp_losses, mtp_head_loss};
use crate::observer::{TrainingObserver, TrainingObserverEvent, TrainingObserverSnapshot};
use crate::optim::{AdamW, AdamWConfig, GradMap, clip_gradients};
use crate::schedule::CosineScheduleWithWarmup;

#[derive(Debug, Clone)]
/// Scalar training metric for one MTP future-token offset.
pub struct MtpHeadMetric {
    /// Future-token offset represented by the loss.
    pub offset: usize,
    /// Unscaled cross-entropy loss for this head.
    pub loss: f64,
}

#[derive(Debug, Clone)]
/// Metrics emitted after a training micro-step.
pub struct TrainingMetrics {
    /// Current optimizer step.
    pub step: usize,
    /// Unscaled total loss for this batch.
    pub loss: f64,
    /// Unscaled cross-entropy loss for this batch.
    pub ce_loss: f64,
    /// Mean unscaled MTP auxiliary loss before weighting.
    pub mtp_aux_loss: Option<f64>,
    /// Individual MTP losses in increasing future-token offset order.
    pub mtp_head_losses: Vec<MtpHeadMetric>,
    /// Unscaled MoE auxiliary loss for this batch.
    pub moe_aux_loss: Option<f64>,
    /// Periodic DSA indexer teacher loss.
    pub dsa_indexer_loss: Option<f64>,
    /// DSA indexer top-k recall against the dense teacher.
    pub dsa_top_k_recall: Option<f32>,
    /// Selected sparse blocks across this micro-step.
    pub dsa_selected_blocks: usize,
    /// Selected K/V token rows across this micro-step.
    pub dsa_selected_tokens: usize,
    /// Number of DSA layers that used dense fallback.
    pub dsa_dense_fallbacks: usize,
    /// Average per-expert utilization for this batch.
    pub expert_utilization: Vec<f32>,
    /// Exponential of loss.
    pub perplexity: f64,
    /// Learning rate used or scheduled for this step.
    pub lr: f64,
    /// Gradient norm when an optimizer step occurred.
    pub grad_norm: Option<f64>,
    /// Whether this micro-step performed an optimizer update.
    pub did_optimizer_step: bool,
}

/// Owns model, optimizer, loaders, checkpoints, and training state.
pub struct Trainer {
    model: AarambhModel,
    varmap: VarMap,
    optimizer: AdamW,
    schedule: CosineScheduleWithWarmup,
    checkpoint: CheckpointManager,
    distributed: Option<DistributedContext>,
    train_loader: DataLoader,
    val_loader: Option<DataLoader>,
    train_config: TrainConfig,
    dsa_training_config: DsaTrainingConfig,
    device: candle_core::Device,
    dtype: DType,
    state: TrainState,
    pending_grads: GradMap,
    last_loss: Option<f64>,
    last_ce_loss: Option<f64>,
    last_mtp_aux_loss: Option<f64>,
    last_mtp_head_losses: Vec<MtpHeadMetric>,
    last_moe_aux_loss: Option<f64>,
    last_dsa_indexer_loss: Option<f64>,
    last_dsa_top_k_recall: Option<f32>,
    last_dsa_selected_blocks: usize,
    last_dsa_selected_tokens: usize,
    last_dsa_dense_fallbacks: usize,
    last_expert_utilization: Vec<f32>,
    tokens_since_log: usize,
    last_log_at: Instant,
    observer: Option<Box<dyn TrainingObserver>>,
    observer_finished: bool,
}

impl Trainer {
    /// Create a trainer from model config, training config, loaders, device, and dtype.
    pub fn new(
        model_config: ModelConfig,
        train_config: TrainConfig,
        train_loader: DataLoader,
        val_loader: Option<DataLoader>,
        device: candle_core::Device,
        dtype: DType,
    ) -> Result<Self> {
        Self::new_with_distributed(
            model_config,
            train_config,
            train_loader,
            val_loader,
            device,
            dtype,
            None,
        )
    }

    /// Create a trainer with an optional distributed data-parallel context.
    pub fn new_with_distributed(
        model_config: ModelConfig,
        train_config: TrainConfig,
        train_loader: DataLoader,
        val_loader: Option<DataLoader>,
        device: candle_core::Device,
        dtype: DType,
        distributed: Option<DistributedContext>,
    ) -> Result<Self> {
        if train_config.grad_accum_steps == 0 {
            return Err(AarambhError::Config(
                "grad_accum_steps must be greater than zero".into(),
            ));
        }
        if train_config.max_steps == 0 {
            return Err(AarambhError::Config(
                "max_steps must be greater than zero".into(),
            ));
        }
        if train_loader.is_empty() {
            return Err(AarambhError::Config(
                "training dataloader has no full batches".into(),
            ));
        }

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
        let model = AarambhModel::new_for_training(&model_config, vb)?;
        let optimizer = AdamW::from_varmap(&varmap, AdamWConfig::from(&train_config))?;
        let schedule = CosineScheduleWithWarmup::from_train_config(&train_config);
        let checkpoint = CheckpointManager::new(train_config.checkpoint_dir.clone());

        Ok(Self {
            model,
            varmap,
            optimizer,
            schedule,
            checkpoint,
            distributed,
            train_loader,
            val_loader,
            train_config,
            dsa_training_config: DsaTrainingConfig::default(),
            device,
            dtype,
            state: TrainState {
                qat: model_config.qat.clone(),
                ..TrainState::default()
            },
            pending_grads: GradMap::new(),
            last_loss: None,
            last_ce_loss: None,
            last_mtp_aux_loss: None,
            last_mtp_head_losses: Vec::new(),
            last_moe_aux_loss: None,
            last_dsa_indexer_loss: None,
            last_dsa_top_k_recall: None,
            last_dsa_selected_blocks: 0,
            last_dsa_selected_tokens: 0,
            last_dsa_dense_fallbacks: 0,
            last_expert_utilization: Vec::new(),
            tokens_since_log: 0,
            last_log_at: Instant::now(),
            observer: None,
            observer_finished: false,
        })
    }

    /// Return current training state.
    pub fn state(&self) -> &TrainState {
        &self.state
    }

    /// Return the model.
    pub fn model(&self) -> &AarambhModel {
        &self.model
    }

    /// Return the variable map.
    pub fn varmap(&self) -> &VarMap {
        &self.varmap
    }

    /// Return the optimizer.
    pub fn optimizer(&self) -> &AdamW {
        &self.optimizer
    }

    /// Return true when this trainer owns rank-0 side effects.
    pub fn is_rank0(&self) -> bool {
        self.distributed.as_ref().is_none_or(|ctx| ctx.is_rank0())
    }

    /// Replace the DSA teacher cadence and loss scaling for this run.
    pub fn set_dsa_training_config(&mut self, config: DsaTrainingConfig) {
        self.dsa_training_config = config;
    }

    /// Install a read-only live training observer.
    pub fn set_observer(&mut self, observer: Box<dyn TrainingObserver>) {
        self.observer = Some(observer);
        self.observer_finished = false;
    }

    /// Run the installed observer against the initial model state.
    pub fn observe_start(&mut self) -> Result<()> {
        self.run_observer(TrainingObserverEvent::Start)
    }

    /// Run the installed observer against the final model state once.
    pub fn finish_observer(&mut self) -> Result<()> {
        if self.observer_finished {
            return Ok(());
        }
        self.run_observer(TrainingObserverEvent::Finish)?;
        self.observer_finished = true;
        Ok(())
    }

    /// Load the latest checkpoint if one exists.
    pub fn load_latest_checkpoint(&mut self) -> Result<bool> {
        match self
            .checkpoint
            .load_latest(&mut self.varmap, &mut self.optimizer, &self.device)?
        {
            Some(state) => {
                if state.qat != self.model.config().qat {
                    return Err(AarambhError::Checkpoint(format!(
                        "QAT resume policy mismatch: checkpoint={:?} configured={:?}",
                        state.qat,
                        self.model.config().qat
                    )));
                }
                self.state = state;
                self.model.advance_qat_generation();
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Load compatible v2 weights while retaining fresh Gated DeltaNet parameters.
    pub fn load_retrofit_checkpoint(
        &mut self,
        path: impl AsRef<Path>,
        dtype: DType,
    ) -> Result<aarambh_studio_weights::RetrofitLoadReport> {
        self.load_retrofit_checkpoint_with_moe(path, dtype, None)
    }

    /// Load compatible weights and optionally expand a coarse MoE checkpoint.
    pub fn load_retrofit_checkpoint_with_moe(
        &mut self,
        path: impl AsRef<Path>,
        dtype: DType,
        moe_options: Option<aarambh_studio_weights::MoeRetrofitOptions>,
    ) -> Result<aarambh_studio_weights::RetrofitLoadReport> {
        let report = aarambh_studio_weights::load_retrofit_into_varmap_with_moe(
            path,
            self.model.config(),
            &mut self.varmap,
            &self.device,
            dtype,
            moe_options,
        )?;
        self.optimizer = AdamW::from_varmap(&self.varmap, AdamWConfig::from(&self.train_config))?;
        self.model.advance_qat_generation();
        Ok(report)
    }

    /// Load a model-only SafeTensors checkpoint with exact names and shapes.
    pub fn load_exact_model_checkpoint(
        &mut self,
        path: impl AsRef<Path>,
        dtype: DType,
    ) -> Result<usize> {
        let loaded = aarambh_studio_weights::load_exact_into_varmap(
            path,
            &mut self.varmap,
            &self.device,
            dtype,
        )?;
        self.optimizer = AdamW::from_varmap(&self.varmap, AdamWConfig::from(&self.train_config))?;
        self.model.advance_qat_generation();
        Ok(loaded)
    }

    /// Replace train/validation loaders while preserving model, optimizer, and schedule state.
    pub fn replace_loaders(&mut self, train_loader: DataLoader, val_loader: Option<DataLoader>) {
        self.train_loader = train_loader;
        self.val_loader = val_loader;
    }

    /// Run one training micro-step.
    pub fn train_step(&mut self, batch: Batch) -> Result<TrainingMetrics> {
        let token_count = batch.input_ids.elem_count();
        let collect_dsa_teacher = self.model.config().dsa_config.is_some()
            && self
                .state
                .step
                .is_multiple_of(self.dsa_training_config.teacher_every_n_steps);
        let output = self
            .model
            .forward_train_with_aux_and_dsa_teacher(&batch.input_ids, collect_dsa_teacher)?;
        let ce_loss = cross_entropy_loss(&output.logits, &batch.labels, &batch.attention_mask)?;
        let mut mtp_head_losses = Vec::with_capacity(self.model.mtp_heads().len());
        for head_index in 0..self.model.mtp_heads().len() {
            let prediction = self.model.forward_mtp_head_train(
                head_index,
                &output.final_hidden_states,
                &batch.input_ids,
            )?;
            mtp_head_losses.push(mtp_head_loss(
                prediction,
                &batch.labels,
                &batch.attention_mask,
            )?);
        }
        let mtp_weight = self
            .model
            .config()
            .mtp
            .as_ref()
            .map(|mtp| mtp.aux_loss_weight)
            .unwrap_or(0.0);
        let mtp_loss = combine_mtp_losses(ce_loss, mtp_head_losses, mtp_weight)?;
        let moe_aux_weight = self
            .model
            .config()
            .moe
            .as_ref()
            .map(|moe| moe.aux_loss_weight)
            .unwrap_or(0.0);
        let mut loss = match &output.moe_aux_loss {
            Some(aux_loss) if moe_aux_weight > 0.0 => {
                (&mtp_loss.total_loss + &aux_loss.affine(moe_aux_weight, 0.0)?)?
            }
            _ => mtp_loss.total_loss.clone(),
        };
        if let Some(indexer_loss) = &output.dsa_indexer_loss
            && self.dsa_training_config.indexer_loss_weight > 0.0
        {
            loss = (&loss
                + &indexer_loss.affine(self.dsa_training_config.indexer_loss_weight, 0.0)?)?;
        }
        let ce_loss_value = mtp_loss.main_loss.to_scalar::<f32>()? as f64;
        let mtp_aux_loss_value = mtp_loss
            .auxiliary_loss
            .as_ref()
            .map(|loss| loss.to_scalar::<f32>().map(|value| value as f64))
            .transpose()?;
        let mtp_head_loss_values = mtp_loss
            .head_losses
            .iter()
            .map(|head| {
                Ok(MtpHeadMetric {
                    offset: head.offset,
                    loss: head.loss.to_scalar::<f32>()? as f64,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let moe_aux_loss_value = output
            .moe_aux_loss
            .as_ref()
            .map(|loss| loss.to_scalar::<f32>().map(|value| value as f64))
            .transpose()?;
        let dsa_indexer_loss_value = output
            .dsa_indexer_loss
            .as_ref()
            .map(|loss| loss.to_scalar::<f32>().map(|value| value as f64))
            .transpose()?;
        let loss_value = loss.to_scalar::<f32>()? as f64;
        if !loss_value.is_finite() || !ce_loss_value.is_finite() {
            return Err(AarambhError::Config(format!(
                "non-finite training loss: total={loss_value} ce={ce_loss_value}"
            )));
        }
        if let Some(aux) = moe_aux_loss_value
            && !aux.is_finite()
        {
            return Err(AarambhError::Config(format!(
                "non-finite MoE auxiliary loss: {aux}"
            )));
        }
        if mtp_aux_loss_value.is_some_and(|value| !value.is_finite())
            || mtp_head_loss_values
                .iter()
                .any(|metric| !metric.loss.is_finite())
        {
            return Err(AarambhError::Config("non-finite MTP auxiliary loss".into()));
        }

        let scaled_loss = loss.affine(1.0 / self.train_config.grad_accum_steps as f64, 0.0)?;
        let grads = scaled_loss.backward()?;
        self.accumulate_gradients(&grads)?;
        self.state.micro_step += 1;
        self.state.train_loss = Some(loss_value);
        self.last_loss = Some(loss_value);
        self.last_ce_loss = Some(ce_loss_value);
        self.last_mtp_aux_loss = mtp_aux_loss_value;
        self.last_mtp_head_losses = mtp_head_loss_values.clone();
        self.last_moe_aux_loss = moe_aux_loss_value;
        self.last_dsa_indexer_loss = dsa_indexer_loss_value;
        self.last_dsa_top_k_recall = output.dsa_top_k_recall;
        self.last_dsa_selected_blocks = output.dsa_stats.selected_blocks;
        self.last_dsa_selected_tokens = output.dsa_stats.selected_tokens;
        self.last_dsa_dense_fallbacks = output.dsa_stats.dense_fallbacks;
        self.last_expert_utilization = output.expert_utilization.clone();
        self.tokens_since_log += token_count;

        let should_step = self
            .state
            .micro_step
            .is_multiple_of(self.train_config.grad_accum_steps);
        if should_step {
            let (lr, grad_norm) = self.optimizer_step()?;
            Ok(TrainingMetrics {
                step: self.state.step,
                loss: loss_value,
                ce_loss: ce_loss_value,
                mtp_aux_loss: mtp_aux_loss_value,
                mtp_head_losses: mtp_head_loss_values,
                moe_aux_loss: moe_aux_loss_value,
                dsa_indexer_loss: dsa_indexer_loss_value,
                dsa_top_k_recall: output.dsa_top_k_recall,
                dsa_selected_blocks: output.dsa_stats.selected_blocks,
                dsa_selected_tokens: output.dsa_stats.selected_tokens,
                dsa_dense_fallbacks: output.dsa_stats.dense_fallbacks,
                expert_utilization: output.expert_utilization,
                perplexity: ce_loss_value.exp(),
                lr,
                grad_norm: Some(grad_norm),
                did_optimizer_step: true,
            })
        } else {
            Ok(TrainingMetrics {
                step: self.state.step,
                loss: loss_value,
                ce_loss: ce_loss_value,
                mtp_aux_loss: mtp_aux_loss_value,
                mtp_head_losses: mtp_head_loss_values,
                moe_aux_loss: moe_aux_loss_value,
                dsa_indexer_loss: dsa_indexer_loss_value,
                dsa_top_k_recall: output.dsa_top_k_recall,
                dsa_selected_blocks: output.dsa_stats.selected_blocks,
                dsa_selected_tokens: output.dsa_stats.selected_tokens,
                dsa_dense_fallbacks: output.dsa_stats.dense_fallbacks,
                expert_utilization: output.expert_utilization,
                perplexity: ce_loss_value.exp(),
                lr: self.schedule.lr_at_step(self.state.step),
                grad_norm: None,
                did_optimizer_step: false,
            })
        }
    }

    /// Train until the current epoch or max-step boundary completes.
    pub fn train_epoch(&mut self) -> Result<()> {
        let _ = self.train_epoch_until(self.train_config.max_steps)?;
        Ok(())
    }

    /// Train until a target optimizer step or epoch boundary is reached.
    pub fn train_epoch_until(&mut self, target_step: usize) -> Result<bool> {
        self.train_loader.reset();
        while self.state.step < target_step {
            let Some(batch) = self.train_loader.next() else {
                self.flush_pending_step()?;
                self.state.epoch += 1;
                return Ok(true);
            };
            let metrics = self.train_step(batch?)?;
            if metrics.did_optimizer_step {
                self.after_optimizer_step(&metrics)?;
            }
        }
        self.flush_pending_step()?;
        Ok(false)
    }

    /// Run the full training loop and save a final checkpoint.
    pub fn train(&mut self) -> Result<()> {
        self.train_until(self.train_config.max_steps)?;
        self.finish_observer()?;
        self.save_checkpoint()
    }

    /// Run training until `target_step` without resetting model or optimizer state.
    pub fn train_until(&mut self, target_step: usize) -> Result<()> {
        while self.state.epoch < self.train_config.max_epochs && self.state.step < target_step {
            let completed_epoch = self.train_epoch_until(target_step)?;
            if !completed_epoch {
                break;
            }
        }
        Ok(())
    }

    /// Save a checkpoint for the current training state.
    pub fn save_checkpoint(&mut self) -> Result<()> {
        if !self.is_rank0() {
            return Ok(());
        }
        self.checkpoint
            .save(&self.varmap, &self.optimizer, &self.state)?;
        Ok(())
    }

    /// Evaluate the validation loader when present.
    pub fn validate(&mut self) -> Result<Option<f64>> {
        if !self.is_rank0() {
            return Ok(None);
        }
        let Some(loader) = self.val_loader.as_mut() else {
            return Ok(None);
        };
        loader.reset();

        let mut total = 0f64;
        let mut batches = 0usize;
        for batch in loader.by_ref() {
            let batch = batch?;
            let logits = self.model.forward_train(&batch.input_ids)?;
            let loss = cross_entropy_loss(&logits, &batch.labels, &batch.attention_mask)?;
            total += loss.to_scalar::<f32>()? as f64;
            batches += 1;
        }

        if batches == 0 {
            return Ok(None);
        }
        let loss = total / batches as f64;
        self.state.val_loss = Some(loss);
        Ok(Some(loss))
    }

    fn accumulate_gradients(&mut self, grads: &GradStore) -> Result<()> {
        let mut updates = Vec::new();
        for param in self.optimizer.parameters() {
            let Some(grad) = grads.get(param.tensor()) else {
                continue;
            };
            let grad = grad.detach();
            let next = match self.pending_grads.get(param.name()) {
                Some(existing) => ((existing + &grad)?).detach(),
                None => grad,
            };
            updates.push((param.name().to_string(), next));
        }

        if updates.is_empty() {
            return Err(AarambhError::Config(
                "backward produced no parameter gradients".into(),
            ));
        }

        for (name, grad) in updates {
            self.pending_grads.insert(name, grad);
        }
        Ok(())
    }

    fn optimizer_step(&mut self) -> Result<(f64, f64)> {
        let lr = self.schedule.lr_at_step(self.state.step);
        if let Some(distributed) = &self.distributed {
            distributed.all_reduce_gradients(&mut self.pending_grads)?;
        }
        let grad_norm = clip_gradients(&mut self.pending_grads, self.train_config.clip_grad_norm)?;
        self.optimizer.step(&self.pending_grads, lr)?;
        self.pending_grads.clear();
        self.state.step += 1;
        self.model.advance_qat_generation();
        Ok((lr, grad_norm))
    }

    fn flush_pending_step(&mut self) -> Result<()> {
        if self.pending_grads.is_empty() || self.state.step >= self.train_config.max_steps {
            return Ok(());
        }
        let (lr, grad_norm) = self.optimizer_step()?;
        let loss = self.last_loss.unwrap_or(0.0);
        let ce_loss = self.last_ce_loss.unwrap_or(loss);
        let metrics = TrainingMetrics {
            step: self.state.step,
            loss,
            ce_loss,
            mtp_aux_loss: self.last_mtp_aux_loss,
            mtp_head_losses: self.last_mtp_head_losses.clone(),
            moe_aux_loss: self.last_moe_aux_loss,
            dsa_indexer_loss: self.last_dsa_indexer_loss,
            dsa_top_k_recall: self.last_dsa_top_k_recall,
            dsa_selected_blocks: self.last_dsa_selected_blocks,
            dsa_selected_tokens: self.last_dsa_selected_tokens,
            dsa_dense_fallbacks: self.last_dsa_dense_fallbacks,
            expert_utilization: self.last_expert_utilization.clone(),
            perplexity: ce_loss.exp(),
            lr,
            grad_norm: Some(grad_norm),
            did_optimizer_step: true,
        };
        self.after_optimizer_step(&metrics)
    }

    fn after_optimizer_step(&mut self, metrics: &TrainingMetrics) -> Result<()> {
        if !self.is_rank0() {
            return self.run_observer(TrainingObserverEvent::OptimizerStep);
        }
        if self.train_config.log_every_n_steps > 0
            && metrics
                .step
                .is_multiple_of(self.train_config.log_every_n_steps)
        {
            let grad_norm = metrics.grad_norm.unwrap_or(0.0);
            let tok_s = self.tokens_per_second_since_last_log();
            if let Some(moe_aux_loss) = metrics.moe_aux_loss {
                let moe = self
                    .model
                    .config()
                    .moe
                    .as_ref()
                    .expect("MoE auxiliary loss requires MoE config");
                let routed_experts = moe.routed_expert_count()?;
                let fine_dim = moe.fine_grained_expert_dim()?;
                let active_width = moe.active_routed_width()?;
                let (util_min, util_max, dead_experts) =
                    utilization_summary(&metrics.expert_utilization);
                println!(
                    "step={} loss={:.4} ce_loss={:.4} moe_aux={:.6} ppl={:.2} lr={:.6} grad_norm={:.4} routed_experts={} active_routed={} shared_experts={} fine_dim={} active_width={} util_min={:.3} util_max={:.3} dead_experts={} expert_util=[{}] tok/s={:.2}",
                    metrics.step,
                    metrics.loss,
                    metrics.ce_loss,
                    moe_aux_loss,
                    metrics.perplexity,
                    metrics.lr,
                    grad_norm,
                    routed_experts,
                    moe.top_k,
                    moe.num_shared_experts,
                    fine_dim,
                    active_width,
                    util_min,
                    util_max,
                    dead_experts,
                    format_expert_utilization(&metrics.expert_utilization),
                    tok_s
                );
            } else {
                println!(
                    "step={} loss={:.4} ppl={:.2} lr={:.6} grad_norm={:.4} tok/s={:.2}",
                    metrics.step, metrics.loss, metrics.perplexity, metrics.lr, grad_norm, tok_s
                );
            }
            if self.model.config().dsa_config.is_some() {
                println!(
                    "dsa step={} indexer_loss={} top_k_recall={} selected_blocks={} selected_tokens={} dense_fallbacks={}",
                    metrics.step,
                    metrics
                        .dsa_indexer_loss
                        .map(|value| format!("{value:.6}"))
                        .unwrap_or_else(|| "-".to_string()),
                    metrics
                        .dsa_top_k_recall
                        .map(|value| format!("{value:.3}"))
                        .unwrap_or_else(|| "-".to_string()),
                    metrics.dsa_selected_blocks,
                    metrics.dsa_selected_tokens,
                    metrics.dsa_dense_fallbacks,
                );
            }
            if let Some(mtp_aux_loss) = metrics.mtp_aux_loss {
                println!(
                    "mtp step={} aux_loss={:.6} weight={:.3} heads=[{}]",
                    metrics.step,
                    mtp_aux_loss,
                    self.model
                        .config()
                        .mtp
                        .as_ref()
                        .expect("MTP metrics require MTP config")
                        .aux_loss_weight,
                    format_mtp_losses(&metrics.mtp_head_losses),
                );
            }
            if let Some(qat) = self.model.qat_stats() {
                let config = self
                    .model
                    .config()
                    .qat
                    .as_ref()
                    .expect("active QAT has a model policy");
                println!(
                    "qat step={} bits={} granularity={:?} tensors={} parameters={} generation={} cache_refreshes={}",
                    metrics.step,
                    config.bits.bits(),
                    config.granularity,
                    qat.wrapped_tensors,
                    qat.wrapped_parameters,
                    qat.generation,
                    qat.cache_refreshes,
                );
            }
        }

        if self.train_config.eval_steps > 0
            && metrics.step.is_multiple_of(self.train_config.eval_steps)
            && let Some(val_loss) = self.validate()?
        {
            let improved = self.state.best_val_loss.is_none_or(|best| val_loss < best);
            if improved {
                self.state.best_val_loss = Some(val_loss);
                self.checkpoint
                    .save_best(&self.varmap, &self.optimizer, &self.state)?;
            }
            println!(
                "eval step={} val_loss={:.4} val_ppl={:.2}",
                metrics.step,
                val_loss,
                val_loss.exp()
            );
        }

        if self.train_config.save_every_n_steps > 0
            && metrics
                .step
                .is_multiple_of(self.train_config.save_every_n_steps)
        {
            self.checkpoint
                .save(&self.varmap, &self.optimizer, &self.state)?;
        }
        self.run_observer(TrainingObserverEvent::OptimizerStep)?;
        Ok(())
    }

    fn run_observer(&mut self, event: TrainingObserverEvent) -> Result<()> {
        let step = self.state.step;
        let should_observe = self
            .observer
            .as_ref()
            .is_some_and(|observer| observer.should_observe(event, step));
        if !should_observe {
            return Ok(());
        }
        if let Some(distributed) = &self.distributed {
            distributed.barrier()?;
        }
        let mut observer_error = None;
        if self.is_rank0() {
            let mut observer = self.observer.take().expect("observer checked above");
            let result = observer.observe(TrainingObserverSnapshot {
                event,
                step,
                model: &self.model,
                device: &self.device,
                dtype: self.dtype,
            });
            self.observer = Some(observer);
            observer_error = result.err();
        }
        if let Some(distributed) = &self.distributed {
            if distributed.any_rank_failed(observer_error.is_some())? {
                return Err(observer_error.unwrap_or_else(|| {
                    AarambhError::Config("rank-0 training observer failed".into())
                }));
            }
        } else if let Some(error) = observer_error {
            return Err(error);
        }
        Ok(())
    }

    fn tokens_per_second_since_last_log(&mut self) -> f64 {
        let elapsed = self.last_log_at.elapsed().as_secs_f64();
        let tokens = self.tokens_since_log;
        self.tokens_since_log = 0;
        self.last_log_at = Instant::now();
        if elapsed > 0.0 {
            tokens as f64 / elapsed
        } else {
            0.0
        }
    }
}

fn format_expert_utilization(values: &[f32]) -> String {
    values
        .iter()
        .map(|value| format!("{value:.3}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn format_mtp_losses(values: &[MtpHeadMetric]) -> String {
    values
        .iter()
        .map(|metric| format!("t+{}:{:.6}", metric.offset, metric.loss))
        .collect::<Vec<_>>()
        .join(",")
}

fn utilization_summary(values: &[f32]) -> (f32, f32, usize) {
    if values.is_empty() {
        return (0.0, 0.0, 0);
    }
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let dead = values.iter().filter(|value| **value <= 1e-6).count();
    (min, max, dead)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aarambh_studio_core::{
        Device as AarambhDevice, DsaConfig, GatedDeltaNetConfig, HybridAttentionSchedule,
        MoeConfig, MtpConfig, QatConfig, QuantBits, TokenizerLike,
    };
    use aarambh_studio_data::dataset::PlaintextDataset;
    use std::collections::HashMap;

    struct CharTokenizer {
        ids: HashMap<char, u32>,
    }

    impl TokenizerLike for CharTokenizer {
        fn encode(&self, text: &str) -> Result<Vec<u32>> {
            Ok(text
                .chars()
                .filter_map(|c| self.ids.get(&c).copied())
                .collect())
        }

        fn decode(&self, ids: &[u32]) -> Result<String> {
            let rev = self
                .ids
                .iter()
                .map(|(k, v)| (*v, *k))
                .collect::<HashMap<_, _>>();
            Ok(ids.iter().filter_map(|id| rev.get(id)).collect())
        }

        fn vocab_size(&self) -> usize {
            self.ids.len()
        }

        fn eos_token_id(&self) -> u32 {
            0
        }

        fn bos_token_id(&self) -> Option<u32> {
            None
        }
    }

    #[test]
    fn tiny_training_loss_decreases() {
        let tokenizer = CharTokenizer {
            ids: HashMap::from([('a', 0), ('b', 1), ('c', 2), ('d', 3)]),
        };
        let dataset = PlaintextDataset::from_lines(vec!["abcdabcdabcdabcdabcdabcd".into()]);
        let device = AarambhDevice::Cpu;
        let train_loader = DataLoader::new(&dataset, &tokenizer, 1, 4, false, device.clone());
        let candle_device = device.to_candle().unwrap();
        let model_config = ModelConfig {
            vocab_size: tokenizer.vocab_size(),
            hidden_dim: 64,
            ffn_dim: 128,
            n_layers: 1,
            n_heads: 1,
            n_kv_heads: 1,
            max_seq_len: 4,
            rope_theta: 10000.0,
            rope_scaling: None,
            moe: None,
            attention_schedule: None,
            dsa_config: None,
            mtp: None,
            qat: None,
            norm_eps: 1e-5,
            tie_embeddings: true,
        };
        let train_config = TrainConfig {
            lr: 1e-2,
            batch_size: 1,
            grad_accum_steps: 1,
            max_epochs: 1,
            max_steps: 4,
            warmup_steps: 0,
            min_lr_ratio: 0.1,
            weight_decay: 0.0,
            beta1: 0.9,
            beta2: 0.95,
            epsilon: 1e-8,
            clip_grad_norm: 1.0,
            save_every_n_steps: 0,
            log_every_n_steps: 0,
            eval_steps: 0,
            seed: 42,
            checkpoint_dir: std::env::temp_dir().join("aarambh_train_loss_decreases"),
        };

        let mut trainer = Trainer::new(
            model_config,
            train_config,
            train_loader,
            None,
            candle_device,
            DType::F32,
        )
        .unwrap();
        let mut eval_loader = DataLoader::new(&dataset, &tokenizer, 1, 4, false, device);
        let eval_batch = eval_loader.next().unwrap().unwrap();
        let first = cross_entropy_loss(
            &trainer.model.forward_train(&eval_batch.input_ids).unwrap(),
            &eval_batch.labels,
            &eval_batch.attention_mask,
        )
        .unwrap()
        .to_scalar::<f32>()
        .unwrap() as f64;
        trainer.train().unwrap();
        let last = cross_entropy_loss(
            &trainer.model.forward_train(&eval_batch.input_ids).unwrap(),
            &eval_batch.labels,
            &eval_batch.attention_mask,
        )
        .unwrap()
        .to_scalar::<f32>()
        .unwrap() as f64;
        assert!(
            last < first,
            "loss did not decrease: first={first}, last={last}"
        );
    }

    #[test]
    fn fine_grained_moe_two_step_training_reports_routed_utilization() {
        let tokenizer = CharTokenizer {
            ids: HashMap::from([('a', 0), ('b', 1), ('c', 2), ('d', 3)]),
        };
        let dataset = PlaintextDataset::from_lines(vec!["abcdabcdabcdabcdabcdabcd".into()]);
        let device = AarambhDevice::Cpu;
        let train_loader = DataLoader::new(&dataset, &tokenizer, 1, 4, false, device.clone());
        let mut batch_loader = DataLoader::new(&dataset, &tokenizer, 1, 4, false, device.clone());
        let candle_device = device.to_candle().unwrap();
        let model_config = ModelConfig {
            vocab_size: tokenizer.vocab_size(),
            hidden_dim: 64,
            ffn_dim: 128,
            n_layers: 2,
            n_heads: 1,
            n_kv_heads: 1,
            max_seq_len: 4,
            rope_theta: 10000.0,
            rope_scaling: None,
            moe: Some(MoeConfig {
                num_experts: 2,
                top_k: 2,
                expert_ffn_dim: 64,
                aux_loss_weight: 0.01,
                every_n_layers: 2,
                fine_grained_factor: 2,
                num_shared_experts: 1,
                ..MoeConfig::default()
            }),
            attention_schedule: None,
            dsa_config: None,
            mtp: None,
            qat: None,
            norm_eps: 1e-5,
            tie_embeddings: true,
        };
        let train_config = TrainConfig {
            lr: 1e-3,
            batch_size: 1,
            grad_accum_steps: 1,
            max_epochs: 1,
            max_steps: 2,
            warmup_steps: 0,
            min_lr_ratio: 1.0,
            weight_decay: 0.0,
            beta1: 0.9,
            beta2: 0.95,
            epsilon: 1e-8,
            clip_grad_norm: 1.0,
            save_every_n_steps: 0,
            log_every_n_steps: 0,
            eval_steps: 0,
            seed: 42,
            checkpoint_dir: std::env::temp_dir().join("aarambh_moe_train_step"),
        };

        let mut trainer = Trainer::new(
            model_config,
            train_config,
            train_loader,
            None,
            candle_device,
            DType::F32,
        )
        .unwrap();
        for expected_step in 1..=2 {
            let metrics = trainer
                .train_step(batch_loader.next().unwrap().unwrap())
                .unwrap();
            assert_eq!(metrics.step, expected_step);
            assert!(metrics.moe_aux_loss.is_some());
            assert_eq!(metrics.expert_utilization.len(), 4);
            let util_sum = metrics.expert_utilization.iter().sum::<f32>();
            assert!((util_sum - 1.0).abs() < 1e-5, "util_sum={util_sum}");
        }
    }

    #[test]
    fn mtp_two_step_training_updates_auxiliary_heads() {
        let tokenizer = CharTokenizer {
            ids: HashMap::from([('a', 0), ('b', 1), ('c', 2), ('d', 3)]),
        };
        let dataset = PlaintextDataset::from_lines(vec!["abcdabcdabcdabcdabcdabcd".into()]);
        let device = AarambhDevice::Cpu;
        let train_loader = DataLoader::new(&dataset, &tokenizer, 1, 4, false, device.clone());
        let mut batch_loader = DataLoader::new(&dataset, &tokenizer, 1, 4, false, device.clone());
        let model_config = ModelConfig {
            vocab_size: tokenizer.vocab_size(),
            hidden_dim: 64,
            ffn_dim: 128,
            n_layers: 1,
            n_heads: 1,
            n_kv_heads: 1,
            max_seq_len: 4,
            rope_theta: 10000.0,
            rope_scaling: None,
            moe: None,
            attention_schedule: None,
            dsa_config: None,
            mtp: Some(MtpConfig {
                num_future_tokens: 3,
                aux_loss_weight: 0.3,
            }),
            qat: None,
            norm_eps: 1e-5,
            tie_embeddings: true,
        };
        let train_config = TrainConfig {
            lr: 1e-3,
            batch_size: 1,
            grad_accum_steps: 1,
            max_epochs: 1,
            max_steps: 2,
            warmup_steps: 0,
            min_lr_ratio: 1.0,
            weight_decay: 0.0,
            beta1: 0.9,
            beta2: 0.95,
            epsilon: 1e-8,
            clip_grad_norm: 1.0,
            save_every_n_steps: 0,
            log_every_n_steps: 0,
            eval_steps: 0,
            seed: 42,
            checkpoint_dir: std::env::temp_dir().join("aarambh_mtp_train_step"),
        };
        let mut trainer = Trainer::new(
            model_config,
            train_config,
            train_loader,
            None,
            device.to_candle().unwrap(),
            DType::F32,
        )
        .unwrap();
        let before = trainer
            .model()
            .named_tensors()
            .into_iter()
            .filter(|(name, _)| name.starts_with("mtp.heads.0."))
            .map(|(name, tensor)| {
                let values = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                (name, values)
            })
            .collect::<HashMap<_, _>>();
        for expected_step in 1..=2 {
            let metrics = trainer
                .train_step(batch_loader.next().unwrap().unwrap())
                .unwrap();
            assert_eq!(metrics.step, expected_step);
            assert!(metrics.mtp_aux_loss.is_some());
            assert_eq!(metrics.mtp_head_losses.len(), 2);
            assert!(
                metrics
                    .mtp_head_losses
                    .iter()
                    .all(|loss| loss.loss.is_finite())
            );
        }
        let changed = trainer
            .model()
            .named_tensors()
            .into_iter()
            .filter(|(name, _)| name.starts_with("mtp.heads.0."))
            .any(|(name, tensor)| {
                let after = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                before[&name]
                    .iter()
                    .zip(after)
                    .any(|(before, after)| (before - after).abs() > 0.0)
            });
        assert!(changed, "MTP auxiliary head did not update");
    }

    #[test]
    fn dsa_two_step_smoke_alternates_teacher_and_sparse_steps() {
        let tokenizer = CharTokenizer {
            ids: HashMap::from([('a', 0), ('b', 1), ('c', 2), ('d', 3)]),
        };
        let dataset = PlaintextDataset::from_lines(vec!["abcd".repeat(40)]);
        let device = AarambhDevice::Cpu;
        let train_loader = DataLoader::new(&dataset, &tokenizer, 1, 32, false, device.clone());
        let mut batch_loader = DataLoader::new(&dataset, &tokenizer, 1, 32, false, device.clone());
        let model_config = ModelConfig {
            vocab_size: tokenizer.vocab_size(),
            hidden_dim: 64,
            ffn_dim: 128,
            n_layers: 2,
            n_heads: 1,
            n_kv_heads: 1,
            max_seq_len: 32,
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
            dsa_config: Some(DsaConfig {
                block_size: 16,
                top_k_blocks: 1,
                min_seq_len_for_sparsity: 16,
            }),
            mtp: None,
            qat: None,
            norm_eps: 1e-5,
            tie_embeddings: true,
        };
        let train_config = TrainConfig {
            lr: 1e-3,
            batch_size: 1,
            grad_accum_steps: 1,
            max_epochs: 1,
            max_steps: 2,
            warmup_steps: 0,
            min_lr_ratio: 1.0,
            weight_decay: 0.0,
            beta1: 0.9,
            beta2: 0.95,
            epsilon: 1e-8,
            clip_grad_norm: 1.0,
            save_every_n_steps: 0,
            log_every_n_steps: 0,
            eval_steps: 0,
            seed: 42,
            checkpoint_dir: std::env::temp_dir().join("aarambh_dsa_train_smoke"),
        };
        let mut trainer = Trainer::new(
            model_config,
            train_config,
            train_loader,
            None,
            device.to_candle().unwrap(),
            DType::F32,
        )
        .unwrap();
        trainer.set_dsa_training_config(DsaTrainingConfig {
            teacher_every_n_steps: 8,
            indexer_loss_weight: 1.0,
        });
        let first = trainer
            .train_step(batch_loader.next().unwrap().unwrap())
            .unwrap();
        assert!(first.dsa_indexer_loss.is_some());
        assert!(first.dsa_selected_blocks > 0);
        batch_loader.reset();
        let second = trainer
            .train_step(batch_loader.next().unwrap().unwrap())
            .unwrap();
        assert!(second.dsa_indexer_loss.is_none());
        assert!(second.loss.is_finite());
    }

    #[test]
    fn qat_two_step_smoke_updates_weights_and_cache_generation() {
        let tokenizer = CharTokenizer {
            ids: HashMap::from([('a', 0), ('b', 1), ('c', 2), ('d', 3)]),
        };
        let dataset = PlaintextDataset::from_lines(vec!["abcdabcdabcdabcdabcdabcd".into()]);
        let device = AarambhDevice::Cpu;
        let train_loader = DataLoader::new(&dataset, &tokenizer, 1, 4, false, device.clone());
        let mut batches = DataLoader::new(&dataset, &tokenizer, 1, 4, false, device.clone());
        let model_config = ModelConfig {
            vocab_size: tokenizer.vocab_size(),
            hidden_dim: 64,
            ffn_dim: 128,
            n_layers: 1,
            n_heads: 1,
            n_kv_heads: 1,
            max_seq_len: 4,
            rope_theta: 10000.0,
            rope_scaling: None,
            moe: None,
            attention_schedule: None,
            dsa_config: None,
            mtp: None,
            qat: Some(QatConfig::default()),
            norm_eps: 1e-5,
            tie_embeddings: true,
        };
        let checkpoint_dir =
            std::env::temp_dir().join(format!("aarambh_qat_smoke_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
        let train_config = TrainConfig {
            lr: 1e-3,
            batch_size: 1,
            grad_accum_steps: 2,
            max_epochs: 1,
            max_steps: 2,
            warmup_steps: 0,
            min_lr_ratio: 1.0,
            weight_decay: 0.0,
            beta1: 0.9,
            beta2: 0.95,
            epsilon: 1e-8,
            clip_grad_norm: 1.0,
            save_every_n_steps: 0,
            log_every_n_steps: 0,
            eval_steps: 0,
            seed: 42,
            checkpoint_dir: checkpoint_dir.clone(),
        };
        let mut mismatch_model_config = model_config.clone();
        let mismatch_train_config = train_config.clone();
        let mut trainer = Trainer::new(
            model_config,
            train_config,
            train_loader,
            None,
            device.to_candle().unwrap(),
            DType::F32,
        )
        .unwrap();
        let before = trainer
            .model()
            .named_tensors()
            .into_iter()
            .filter(|(name, _)| name.contains(".attn.") || name.contains(".ffn."))
            .map(|(name, tensor)| {
                (
                    name,
                    tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                )
            })
            .collect::<HashMap<_, _>>();

        for expected_step in 1..=2 {
            let accumulated = trainer
                .train_step(batches.next().unwrap().unwrap())
                .unwrap();
            assert!(!accumulated.did_optimizer_step);
            assert_eq!(accumulated.step, expected_step - 1);
            assert!(accumulated.loss.is_finite());
            assert_eq!(
                trainer.model().qat_stats().unwrap().cache_refreshes,
                expected_step * 7
            );

            let stepped = trainer
                .train_step(batches.next().unwrap().unwrap())
                .unwrap();
            assert!(stepped.did_optimizer_step);
            assert_eq!(stepped.step, expected_step);
            assert!(stepped.loss.is_finite());
            assert_eq!(
                trainer.model().qat_stats().unwrap().cache_refreshes,
                expected_step * 7
            );
        }
        let stats = trainer.model().qat_stats().unwrap();
        assert_eq!(stats.generation, 2);
        assert_eq!(stats.wrapped_tensors, 7);
        assert_eq!(stats.cache_refreshes, 14);
        let changed = trainer
            .model()
            .named_tensors()
            .into_iter()
            .filter(|(name, _)| before.contains_key(name))
            .any(|(name, tensor)| {
                let after = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                before[&name]
                    .iter()
                    .zip(after)
                    .any(|(before, after)| (before - after).abs() > 0.0)
            });
        assert!(
            changed,
            "no QAT-covered attention or FFN projection updated"
        );

        trainer.save_checkpoint().unwrap();
        mismatch_model_config.qat.as_mut().unwrap().bits = QuantBits::Int8;
        let mismatch_loader = DataLoader::new(&dataset, &tokenizer, 1, 4, false, device.clone());
        let mut mismatch = Trainer::new(
            mismatch_model_config,
            mismatch_train_config,
            mismatch_loader,
            None,
            device.to_candle().unwrap(),
            DType::F32,
        )
        .unwrap();
        let error = mismatch.load_latest_checkpoint().unwrap_err().to_string();
        assert!(error.contains("QAT resume policy mismatch"), "{error}");
        let _ = std::fs::remove_dir_all(checkpoint_dir);
    }
}
