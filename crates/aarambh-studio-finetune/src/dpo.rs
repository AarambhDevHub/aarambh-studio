use std::fs;
use std::path::{Path, PathBuf};

use aarambh_studio_core::{AarambhError, Device, ModelConfig, Result, TokenizerLike, TrainConfig};
use aarambh_studio_tokenizer::{BpeTokenizer, ENDOFTEXT_ID, PAD_ID};
use aarambh_studio_train::optim::clip_gradients;
use aarambh_studio_train::{AdamW, AdamWConfig, CosineScheduleWithWarmup, GradMap, TrainState};
use aarambh_studio_weights::load_any_model;
use candle_core::backprop::GradStore;
use candle_core::{DType, Tensor};
use candle_nn::VarMap;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

use crate::adapter::{AdapterMetadata, AdapterMethod, save_adapter};
use crate::dora::DoraAarambhModel;
use crate::lora::LoraConfig;
use crate::sft::{ChatTemplate, build_loss_mask};

/// DPO objective and tokenization settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DpoConfig {
    /// Preference-temperature coefficient from the DPO objective.
    pub beta: f64,
    /// Whether to omit the frozen reference-policy log-ratio.
    pub reference_free: bool,
    /// Optional maximum number of prompt tokens retained per pair.
    pub max_prompt_tokens: Option<usize>,
    /// Optional maximum number of completion tokens retained per response.
    pub max_completion_tokens: Option<usize>,
}

impl Default for DpoConfig {
    fn default() -> Self {
        Self {
            beta: 0.1,
            reference_free: false,
            max_prompt_tokens: None,
            max_completion_tokens: None,
        }
    }
}

impl DpoConfig {
    /// Validate DPO objective and token limits.
    pub fn validate(&self) -> Result<()> {
        if !self.beta.is_finite() || self.beta <= 0.0 {
            return Err(AarambhError::Config(
                "DPO beta must be finite and greater than zero".into(),
            ));
        }
        if self.max_prompt_tokens == Some(0) {
            return Err(AarambhError::Config(
                "DPO max_prompt_tokens must be greater than zero".into(),
            ));
        }
        if self.max_completion_tokens == Some(0) {
            return Err(AarambhError::Config(
                "DPO max_completion_tokens must be greater than zero".into(),
            ));
        }
        Ok(())
    }
}

/// Complete configuration for one DPO or QDPO run.
#[derive(Debug, Clone)]
pub struct DpoRunConfig {
    /// Base model architecture.
    pub model_config: ModelConfig,
    /// Optimizer, scheduling, and checkpoint settings.
    pub train_config: TrainConfig,
    /// DPO objective and sequence limits.
    pub dpo_config: DpoConfig,
    /// Trainable policy base checkpoint.
    pub base_model_path: PathBuf,
    /// Optional frozen reference checkpoint; defaults to the policy base.
    pub reference_model_path: Option<PathBuf>,
    /// Tokenizer JSON path.
    pub tokenizer_path: PathBuf,
    /// Preference-pair JSONL path.
    pub data_path: PathBuf,
    /// Adapter output directory.
    pub output_dir: PathBuf,
    /// DoRA low-rank adapter configuration.
    pub dora_config: LoraConfig,
    /// Logical training device.
    pub device: Device,
    /// Whether to quantize the frozen base for QDPO.
    pub qdpo: bool,
    /// Whether to reshuffle preference pairs each epoch.
    pub shuffle: bool,
}

/// One prompt with a preferred and dispreferred response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DpoExample {
    /// Prompt shared by both responses.
    pub prompt: String,
    /// Preferred response.
    pub chosen: String,
    /// Dispreferred response.
    pub rejected: String,
}

#[derive(Debug, Clone)]
struct DpoSequence {
    input_ids: Vec<u32>,
    labels: Vec<u32>,
    response_mask: Vec<u32>,
}

#[derive(Debug, Clone)]
struct TokenizedDpoExample {
    chosen: DpoSequence,
    rejected: DpoSequence,
    reference_logps: Option<(f32, f32)>,
}

/// Tokenized in-memory DPO preference dataset.
#[derive(Debug, Clone)]
pub struct DpoDataset {
    examples: Vec<TokenizedDpoExample>,
}

impl DpoDataset {
    /// Load and tokenize canonical `{prompt, chosen, rejected}` JSONL records.
    pub fn from_jsonl(
        path: impl AsRef<Path>,
        tokenizer: &dyn TokenizerLike,
        max_seq_len: usize,
        config: &DpoConfig,
    ) -> Result<Self> {
        let content = fs::read_to_string(path.as_ref())?;
        let mut examples = Vec::new();
        for (line_idx, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let example: DpoExample = serde_json::from_str(line).map_err(|err| {
                AarambhError::Config(format!("invalid DPO JSONL at line {}: {err}", line_idx + 1))
            })?;
            examples.push(example);
        }
        Self::from_examples(&examples, tokenizer, max_seq_len, config).map_err(|err| {
            AarambhError::Config(format!(
                "DPO dataset {} is invalid: {err}",
                path.as_ref().display()
            ))
        })
    }

    /// Tokenize already loaded preference examples.
    pub fn from_examples(
        examples: &[DpoExample],
        tokenizer: &dyn TokenizerLike,
        max_seq_len: usize,
        config: &DpoConfig,
    ) -> Result<Self> {
        config.validate()?;
        if max_seq_len == 0 {
            return Err(AarambhError::Config("max_seq_len must be non-zero".into()));
        }
        if examples.is_empty() {
            return Err(AarambhError::Config(
                "DPO dataset must contain at least one preference pair".into(),
            ));
        }

        let template = ChatTemplate;
        let mut tokenized = Vec::with_capacity(examples.len());
        for (index, example) in examples.iter().enumerate() {
            validate_example(example, index + 1)?;
            let prefix = template.prefix(&example.prompt, None);
            let prefix_ids = tokenizer.encode(&prefix)?;
            let chosen_ids = tokenizer.encode(&template.target(&example.chosen))?;
            let rejected_ids = tokenizer.encode(&template.target(&example.rejected))?;
            let (chosen, rejected) =
                encode_pair(prefix_ids, chosen_ids, rejected_ids, max_seq_len, config)?;
            tokenized.push(TokenizedDpoExample {
                chosen,
                rejected,
                reference_logps: None,
            });
        }
        Ok(Self {
            examples: tokenized,
        })
    }

    /// Return the number of preference pairs.
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Return true when no preference pairs are available.
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    fn set_reference_logps(&mut self, values: Vec<(f32, f32)>) -> Result<()> {
        if values.len() != self.examples.len() {
            return Err(AarambhError::Shape(format!(
                "reference log-prob count {} does not match DPO dataset length {}",
                values.len(),
                self.examples.len()
            )));
        }
        for (example, value) in self.examples.iter_mut().zip(values) {
            example.reference_logps = Some(value);
        }
        Ok(())
    }
}

/// One dynamically padded DPO mini-batch.
#[derive(Debug)]
pub struct DpoBatch {
    /// Chosen rows followed by rejected rows, shape `[2 * batch, sequence]`.
    pub input_ids: Tensor,
    /// Next-token labels with the same shape as `input_ids`.
    pub labels: Tensor,
    /// Completion-only scoring mask.
    pub response_mask: Tensor,
    /// Cached frozen-reference chosen sequence log-probabilities.
    pub reference_chosen_logps: Option<Tensor>,
    /// Cached frozen-reference rejected sequence log-probabilities.
    pub reference_rejected_logps: Option<Tensor>,
    /// Number of preference pairs represented by this batch.
    pub batch_size: usize,
    example_indices: Vec<usize>,
}

/// Deterministic mini-batch loader for DPO pairs.
pub struct DpoDataLoader {
    examples: Vec<TokenizedDpoExample>,
    order: Vec<usize>,
    batch_size: usize,
    shuffle: bool,
    rng: StdRng,
    pos: usize,
    device: Device,
}

impl DpoDataLoader {
    /// Create a dynamically padded DPO data loader.
    pub fn new(
        dataset: &DpoDataset,
        batch_size: usize,
        shuffle: bool,
        seed: u64,
        device: Device,
    ) -> Result<Self> {
        if batch_size == 0 {
            return Err(AarambhError::Config("batch_size must be non-zero".into()));
        }
        if dataset.is_empty() {
            return Err(AarambhError::Config("DPO dataset is empty".into()));
        }
        Ok(Self {
            examples: dataset.examples.clone(),
            order: (0..dataset.len()).collect(),
            batch_size,
            shuffle,
            rng: StdRng::seed_from_u64(seed),
            pos: 0,
            device,
        })
    }

    /// Reset iteration and reshuffle pair indices when enabled.
    pub fn reset(&mut self) {
        self.pos = 0;
        if self.shuffle {
            self.order.shuffle(&mut self.rng);
        }
    }

    /// Return the number of batches, including a final partial batch.
    pub fn len(&self) -> usize {
        self.examples.len().div_ceil(self.batch_size)
    }

    /// Return true when no batches are available.
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }
}

impl Iterator for DpoDataLoader {
    type Item = Result<DpoBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.order.len() {
            return None;
        }
        let end = (self.pos + self.batch_size).min(self.order.len());
        let indices = self.order[self.pos..end].to_vec();
        self.pos = end;
        Some(batch_to_tensors(&self.examples, &indices, &self.device))
    }
}

/// Metrics emitted for one DPO micro-step.
#[derive(Debug, Clone)]
pub struct DpoMetrics {
    /// Completed optimizer step.
    pub step: usize,
    /// Mean DPO loss.
    pub loss: f64,
    /// Mean implicit chosen-minus-rejected reward margin.
    pub reward_margin: f64,
    /// Fraction of pairs whose chosen implicit reward is larger.
    pub reward_accuracy: f64,
    /// Mean chosen policy sequence log-probability.
    pub chosen_logp: f64,
    /// Mean rejected policy sequence log-probability.
    pub rejected_logp: f64,
    /// Learning rate for the current optimizer boundary.
    pub lr: f64,
    /// Gradient norm when an optimizer update occurred.
    pub grad_norm: Option<f64>,
    /// Whether this micro-step performed an optimizer update.
    pub did_optimizer_step: bool,
}

/// Serializable metadata written next to DPO adapter weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DpoSaveMetadata {
    /// DPO objective configuration.
    pub dpo: DpoConfig,
    /// Training configuration.
    pub train: TrainConfig,
    /// Reference checkpoint path, or `None` for reference-free DPO.
    pub reference_model: Option<String>,
    /// Whether the policy used a quantized QDoRA base.
    pub qdpo: bool,
}

/// Adapter-only DoRA/QDoRA trainer for Direct Preference Optimization.
pub struct DpoTrainer {
    model: DoraAarambhModel,
    varmap: VarMap,
    optimizer: AdamW,
    schedule: CosineScheduleWithWarmup,
    /// Crate-visible so the Phase 46 RLAIF integration test (in `rlaif.rs`)
    /// can pull one batch and prove RLAIF-generated pairs feed through the
    /// unmodified DPO `train_step`. Not part of the public API.
    pub(crate) train_loader: DpoDataLoader,
    dpo_config: DpoConfig,
    train_config: TrainConfig,
    output_dir: PathBuf,
    metadata: AdapterMetadata,
    save_metadata: DpoSaveMetadata,
    state: TrainState,
    pending_grads: GradMap,
    last_metrics: Option<DpoMetrics>,
}

impl DpoTrainer {
    /// Construct a DPO trainer from a policy, cached dataset, and adapter state.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: DoraAarambhModel,
        varmap: VarMap,
        train_loader: DpoDataLoader,
        dpo_config: DpoConfig,
        train_config: TrainConfig,
        output_dir: impl Into<PathBuf>,
        metadata: AdapterMetadata,
        save_metadata: DpoSaveMetadata,
    ) -> Result<Self> {
        dpo_config.validate()?;
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
            return Err(AarambhError::Config("DPO dataloader is empty".into()));
        }
        let optimizer = AdamW::from_varmap(&varmap, AdamWConfig::from(&train_config))?;
        if optimizer.parameters().is_empty() {
            return Err(AarambhError::Config(
                "DoRA target_modules produced zero DPO trainable tensors".into(),
            ));
        }
        let schedule = CosineScheduleWithWarmup::from_train_config(&train_config);
        Ok(Self {
            model,
            varmap,
            optimizer,
            schedule,
            train_loader,
            dpo_config,
            train_config,
            output_dir: output_dir.into(),
            metadata,
            save_metadata,
            state: TrainState::default(),
            pending_grads: GradMap::new(),
            last_metrics: None,
        })
    }

    /// Return the trainable DoRA policy.
    pub fn model(&self) -> &DoraAarambhModel {
        &self.model
    }

    /// Return trainable adapter variables.
    pub fn varmap(&self) -> &VarMap {
        &self.varmap
    }

    /// Return current training state.
    pub fn state(&self) -> &TrainState {
        &self.state
    }

    /// Run one DPO micro-step.
    pub fn train_step(&mut self, batch: DpoBatch) -> Result<DpoMetrics> {
        let logits = self.model.forward_train(&batch.input_ids)?;
        let sequence_logps = sequence_log_probs(&logits, &batch.labels, &batch.response_mask)?;
        let chosen = sequence_logps.narrow(0, 0, batch.batch_size)?;
        let rejected = sequence_logps.narrow(0, batch.batch_size, batch.batch_size)?;
        let reference = if self.dpo_config.reference_free {
            (None, None)
        } else {
            (
                batch.reference_chosen_logps.as_ref(),
                batch.reference_rejected_logps.as_ref(),
            )
        };
        if !self.dpo_config.reference_free && (reference.0.is_none() || reference.1.is_none()) {
            return Err(AarambhError::Config(
                "standard DPO batch is missing precomputed reference log-probabilities".into(),
            ));
        }
        let loss = dpo_loss(
            &chosen,
            &rejected,
            reference.0,
            reference.1,
            self.dpo_config.beta,
        )?;
        let loss_value = loss.to_scalar::<f32>()? as f64;
        if !loss_value.is_finite() {
            return Err(AarambhError::Config(format!(
                "non-finite DPO loss: {loss_value}"
            )));
        }

        let metrics_values = dpo_metric_values(
            &chosen,
            &rejected,
            reference.0,
            reference.1,
            self.dpo_config.beta,
        )?;
        let scaled_loss = loss.affine(1.0 / self.train_config.grad_accum_steps as f64, 0.0)?;
        let grads = scaled_loss.backward()?;
        self.accumulate_gradients(&grads)?;
        self.state.micro_step += 1;
        self.state.train_loss = Some(loss_value);

        let should_step = self
            .state
            .micro_step
            .is_multiple_of(self.train_config.grad_accum_steps);
        let (lr, grad_norm, did_optimizer_step) = if should_step {
            let (lr, grad_norm) = self.optimizer_step()?;
            (lr, Some(grad_norm), true)
        } else {
            (self.schedule.lr_at_step(self.state.step), None, false)
        };
        let metrics = DpoMetrics {
            step: self.state.step,
            loss: loss_value,
            reward_margin: metrics_values.0,
            reward_accuracy: metrics_values.1,
            chosen_logp: metrics_values.2,
            rejected_logp: metrics_values.3,
            lr,
            grad_norm,
            did_optimizer_step,
        };
        self.last_metrics = Some(metrics.clone());
        Ok(metrics)
    }

    /// Train until the current epoch or max-step boundary.
    pub fn train_epoch(&mut self) -> Result<()> {
        self.train_loader.reset();
        while self.state.step < self.train_config.max_steps {
            let Some(batch) = self.train_loader.next() else {
                break;
            };
            let metrics = self.train_step(batch?)?;
            if metrics.did_optimizer_step {
                self.after_optimizer_step(&metrics)?;
            }
        }
        self.flush_pending_step()?;
        self.state.epoch += 1;
        Ok(())
    }

    /// Run the complete DPO training loop and save final artifacts.
    pub fn train(&mut self) -> Result<()> {
        while self.state.epoch < self.train_config.max_epochs
            && self.state.step < self.train_config.max_steps
        {
            self.train_epoch()?;
        }
        self.save_final()
    }

    /// Save final adapter, DPO metadata, and training state.
    pub fn save_final(&self) -> Result<()> {
        save_dpo_artifacts(
            &self.varmap,
            &self.metadata,
            &self.save_metadata,
            &self.state,
            &self.output_dir,
        )
    }

    fn save_step(&self) -> Result<()> {
        let dir = self
            .output_dir
            .join("checkpoints")
            .join(format!("step_{:06}", self.state.step));
        save_dpo_artifacts(
            &self.varmap,
            &self.metadata,
            &self.save_metadata,
            &self.state,
            dir,
        )
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
                "DPO backward produced no DoRA parameter gradients".into(),
            ));
        }
        for (name, grad) in updates {
            self.pending_grads.insert(name, grad);
        }
        Ok(())
    }

    fn optimizer_step(&mut self) -> Result<(f64, f64)> {
        let lr = self.schedule.lr_at_step(self.state.step);
        let grad_norm = clip_gradients(&mut self.pending_grads, self.train_config.clip_grad_norm)?;
        self.optimizer.step(&self.pending_grads, lr)?;
        self.pending_grads.clear();
        self.state.step += 1;
        Ok((lr, grad_norm))
    }

    fn flush_pending_step(&mut self) -> Result<()> {
        if self.pending_grads.is_empty() || self.state.step >= self.train_config.max_steps {
            return Ok(());
        }
        let (lr, grad_norm) = self.optimizer_step()?;
        let mut metrics = self.last_metrics.clone().unwrap_or(DpoMetrics {
            step: self.state.step,
            loss: 0.0,
            reward_margin: 0.0,
            reward_accuracy: 0.0,
            chosen_logp: 0.0,
            rejected_logp: 0.0,
            lr,
            grad_norm: Some(grad_norm),
            did_optimizer_step: true,
        });
        metrics.step = self.state.step;
        metrics.lr = lr;
        metrics.grad_norm = Some(grad_norm);
        metrics.did_optimizer_step = true;
        self.after_optimizer_step(&metrics)
    }

    fn after_optimizer_step(&self, metrics: &DpoMetrics) -> Result<()> {
        if self.train_config.log_every_n_steps > 0
            && metrics
                .step
                .is_multiple_of(self.train_config.log_every_n_steps)
        {
            println!(
                "dpo step={} loss={:.4} reward_margin={:.4} reward_acc={:.3} chosen_logp={:.2} rejected_logp={:.2} lr={:.6} grad_norm={:.4}",
                metrics.step,
                metrics.loss,
                metrics.reward_margin,
                metrics.reward_accuracy,
                metrics.chosen_logp,
                metrics.rejected_logp,
                metrics.lr,
                metrics.grad_norm.unwrap_or(0.0),
            );
        }
        if self.train_config.save_every_n_steps > 0
            && metrics
                .step
                .is_multiple_of(self.train_config.save_every_n_steps)
        {
            self.save_step()?;
        }
        Ok(())
    }
}

/// Build and run a DoRA/QDoRA DPO trainer.
pub fn run_dpo_from_config(config: DpoRunConfig) -> Result<()> {
    config.dora_config.validate()?;
    config.dpo_config.validate()?;
    if config.dpo_config.reference_free && config.reference_model_path.is_some() {
        return Err(AarambhError::Config(
            "--reference-free cannot be combined with an explicit reference model".into(),
        ));
    }
    let candle_device = config.device.to_candle()?;
    let tokenizer = BpeTokenizer::from_pretrained(&config.tokenizer_path)?;
    tokenizer.validate_special_tokens()?;
    let mut model_config = config.model_config.clone();
    model_config.vocab_size = tokenizer.vocab_size();
    if model_config.moe.is_some() {
        return Err(AarambhError::Config(
            "DPO DoRA/QDoRA adapter training for MoE models is not supported; use a dense config"
                .into(),
        ));
    }

    let mut dataset = DpoDataset::from_jsonl(
        &config.data_path,
        &tokenizer,
        model_config.max_seq_len,
        &config.dpo_config,
    )?;
    let reference_path = (!config.dpo_config.reference_free).then(|| {
        config
            .reference_model_path
            .clone()
            .unwrap_or_else(|| config.base_model_path.clone())
    });

    let same_reference = reference_path
        .as_ref()
        .is_some_and(|path| path == &config.base_model_path);
    if let Some(reference_path) = reference_path.as_ref().filter(|_| !same_reference) {
        eprintln!("precomputing DPO reference log-probabilities");
        let reference = load_any_model(reference_path, &model_config, &candle_device)?;
        let logps = precompute_reference_logps(
            &dataset,
            config.train_config.batch_size,
            config.train_config.seed,
            config.device.clone(),
            |input| reference.forward_train(input),
        )?;
        dataset.set_reference_logps(logps)?;
        drop(reference);
    }

    let base = load_any_model(&config.base_model_path, &model_config, &candle_device)?;
    let base_tensors = base.named_tensors();
    drop(base);
    let (model, varmap) = DoraAarambhModel::from_tensors(
        &model_config,
        &base_tensors,
        &config.dora_config,
        config.qdpo,
        &candle_device,
    )?;
    eprintln!(
        "dpo adapter params: {} / {} ({:.3}%)",
        model.adapter_param_count(),
        model.base_param_count(),
        model.trainable_ratio() * 100.0
    );

    if same_reference {
        eprintln!("precomputing DPO reference log-probabilities from the initial policy");
        let logps = precompute_reference_logps(
            &dataset,
            config.train_config.batch_size,
            config.train_config.seed,
            config.device.clone(),
            |input| model.forward_eval(input),
        )?;
        dataset.set_reference_logps(logps)?;
    }

    let loader = DpoDataLoader::new(
        &dataset,
        config.train_config.batch_size,
        config.shuffle,
        config.train_config.seed,
        config.device.clone(),
    )?;
    let metadata = AdapterMetadata::new_with_method(
        model_config,
        config.dora_config.clone(),
        Some(config.base_model_path.display().to_string()),
        config.qdpo,
        AdapterMethod::Dora,
    );
    let save_metadata = DpoSaveMetadata {
        dpo: config.dpo_config.clone(),
        train: config.train_config.clone(),
        reference_model: reference_path.map(|path| path.display().to_string()),
        qdpo: config.qdpo,
    };
    let mut trainer = DpoTrainer::new(
        model,
        varmap,
        loader,
        config.dpo_config,
        config.train_config,
        config.output_dir,
        metadata,
        save_metadata,
    )?;
    trainer.train()
}

/// Sum completion-token log-probabilities for every sequence in a batch.
pub fn sequence_log_probs(
    logits: &Tensor,
    labels: &Tensor,
    response_mask: &Tensor,
) -> Result<Tensor> {
    let dims = logits.dims();
    if dims.len() != 3 {
        return Err(AarambhError::Shape(format!(
            "DPO logits must have shape [batch, seq, vocab], got {dims:?}"
        )));
    }
    let batch = dims[0];
    let seq = dims[1];
    let vocab = dims[2];
    if labels.dims() != [batch, seq] || response_mask.dims() != [batch, seq] {
        return Err(AarambhError::Shape(format!(
            "DPO labels and response_mask must have shape [{batch}, {seq}], got {:?} and {:?}",
            labels.dims(),
            response_mask.dims()
        )));
    }
    let flat_logits = logits.to_dtype(DType::F32)?.reshape((batch * seq, vocab))?;
    let flat_labels = labels.reshape((batch * seq, 1))?;
    let selected = candle_nn::ops::log_softmax(&flat_logits, 1)?
        .gather(&flat_labels, 1)?
        .reshape((batch, seq))?;
    let mask = response_mask.to_dtype(DType::F32)?;
    Ok((selected * mask)?.sum(1)?)
}

/// Compute the standard pairwise DPO loss, or reference-free loss when refs are absent.
pub fn dpo_loss(
    policy_chosen_logps: &Tensor,
    policy_rejected_logps: &Tensor,
    reference_chosen_logps: Option<&Tensor>,
    reference_rejected_logps: Option<&Tensor>,
    beta: f64,
) -> Result<Tensor> {
    if !beta.is_finite() || beta <= 0.0 {
        return Err(AarambhError::Config(
            "DPO beta must be finite and greater than zero".into(),
        ));
    }
    let shape = policy_chosen_logps.dims();
    if shape.len() != 1 || policy_rejected_logps.dims() != shape {
        return Err(AarambhError::Shape(
            "DPO policy log-probabilities must be matching rank-1 tensors".into(),
        ));
    }
    if shape[0] == 0 {
        return Err(AarambhError::Shape(
            "DPO loss requires at least one preference pair".into(),
        ));
    }
    let reference_ratio = match (reference_chosen_logps, reference_rejected_logps) {
        (Some(chosen), Some(rejected)) => {
            if chosen.dims() != shape || rejected.dims() != shape {
                return Err(AarambhError::Shape(
                    "DPO reference log-probabilities must match policy shapes".into(),
                ));
            }
            (chosen.detach() - &rejected.detach())?
        }
        (None, None) => Tensor::zeros(shape, DType::F32, policy_chosen_logps.device())?,
        _ => {
            return Err(AarambhError::Shape(
                "DPO reference chosen and rejected tensors must both be present or absent".into(),
            ));
        }
    };
    let policy_ratio = (policy_chosen_logps - policy_rejected_logps)?;
    let logits = (policy_ratio - reference_ratio)?.affine(beta, 0.0)?;

    // -log(sigmoid(logits)) expressed as a stable two-class log-softmax.
    let zeros = Tensor::zeros(logits.shape(), DType::F32, logits.device())?;
    let classes = Tensor::stack(&[&zeros, &logits.to_dtype(DType::F32)?], 1)?;
    let chosen_log_prob = candle_nn::ops::log_softmax(&classes, 1)?.narrow(1, 1, 1)?;
    Ok(chosen_log_prob
        .sum_all()?
        .affine(-1.0 / shape[0] as f64, 0.0)?)
}

fn precompute_reference_logps<F>(
    dataset: &DpoDataset,
    batch_size: usize,
    seed: u64,
    device: Device,
    mut forward: F,
) -> Result<Vec<(f32, f32)>>
where
    F: FnMut(&Tensor) -> Result<Tensor>,
{
    let mut loader = DpoDataLoader::new(dataset, batch_size, false, seed, device)?;
    let mut values = vec![(0.0, 0.0); dataset.len()];
    for batch in &mut loader {
        let batch = batch?;
        let logits = forward(&batch.input_ids)?.detach();
        let logps = sequence_log_probs(&logits, &batch.labels, &batch.response_mask)?.detach();
        let chosen = logps.narrow(0, 0, batch.batch_size)?.to_vec1::<f32>()?;
        let rejected = logps
            .narrow(0, batch.batch_size, batch.batch_size)?
            .to_vec1::<f32>()?;
        for ((index, chosen), rejected) in batch
            .example_indices
            .iter()
            .copied()
            .zip(chosen)
            .zip(rejected)
        {
            values[index] = (chosen, rejected);
        }
    }
    Ok(values)
}

fn validate_example(example: &DpoExample, line: usize) -> Result<()> {
    if example.prompt.trim().is_empty()
        || example.chosen.trim().is_empty()
        || example.rejected.trim().is_empty()
    {
        return Err(AarambhError::Config(format!(
            "DPO pair {line} contains an empty prompt, chosen, or rejected field"
        )));
    }
    if example.chosen.trim() == example.rejected.trim() {
        return Err(AarambhError::Config(format!(
            "DPO pair {line} has identical chosen and rejected responses"
        )));
    }
    Ok(())
}

fn encode_pair(
    mut prefix_ids: Vec<u32>,
    mut chosen_ids: Vec<u32>,
    mut rejected_ids: Vec<u32>,
    max_seq_len: usize,
    config: &DpoConfig,
) -> Result<(DpoSequence, DpoSequence)> {
    if prefix_ids.is_empty() {
        return Err(AarambhError::Config(
            "DPO prompt encoded to zero tokens".into(),
        ));
    }
    if chosen_ids.is_empty() || rejected_ids.is_empty() {
        return Err(AarambhError::Config(
            "DPO chosen and rejected responses must encode to at least one token".into(),
        ));
    }

    let max_tokens = max_seq_len + 1;
    let configured_prompt_limit = config.max_prompt_tokens.unwrap_or(max_tokens - 1);
    let prompt_limit = configured_prompt_limit.min(max_tokens - 1);
    truncate_prompt(&mut prefix_ids, prompt_limit);
    let available_completion = max_tokens - prefix_ids.len();
    let completion_limit = config
        .max_completion_tokens
        .unwrap_or(available_completion)
        .min(available_completion);
    truncate_completion(&mut chosen_ids, completion_limit);
    truncate_completion(&mut rejected_ids, completion_limit);

    let prefix_len = prefix_ids.len();
    let chosen = make_sequence(&prefix_ids, &chosen_ids, prefix_len)?;
    let rejected = make_sequence(&prefix_ids, &rejected_ids, prefix_len)?;
    Ok((chosen, rejected))
}

fn truncate_prompt(ids: &mut Vec<u32>, limit: usize) {
    if ids.len() <= limit {
        return;
    }
    if limit == 1 {
        ids.truncate(1);
        return;
    }
    let first = ids[0];
    let tail_start = ids.len() - (limit - 1);
    let mut truncated = Vec::with_capacity(limit);
    truncated.push(first);
    truncated.extend_from_slice(&ids[tail_start..]);
    *ids = truncated;
}

fn truncate_completion(ids: &mut Vec<u32>, limit: usize) {
    ids.truncate(limit.max(1));
    if let Some(last) = ids.last_mut() {
        *last = ENDOFTEXT_ID;
    }
}

fn make_sequence(prefix: &[u32], target: &[u32], prefix_len: usize) -> Result<DpoSequence> {
    let mut ids = Vec::with_capacity(prefix.len() + target.len());
    ids.extend_from_slice(prefix);
    ids.extend_from_slice(target);
    if ids.len() < 2 {
        return Err(AarambhError::Config(
            "DPO sequence must contain at least two tokens".into(),
        ));
    }
    let input_ids = ids[..ids.len() - 1].to_vec();
    let labels = ids[1..].to_vec();
    let response_mask = build_loss_mask(prefix_len, ids.len());
    if response_mask.iter().all(|value| *value == 0) {
        return Err(AarambhError::Config(
            "DPO sequence contains no scoreable response tokens".into(),
        ));
    }
    Ok(DpoSequence {
        input_ids,
        labels,
        response_mask,
    })
}

fn batch_to_tensors(
    examples: &[TokenizedDpoExample],
    indices: &[usize],
    device: &Device,
) -> Result<DpoBatch> {
    let selected = indices
        .iter()
        .map(|index| &examples[*index])
        .collect::<Vec<_>>();
    let max_len = selected
        .iter()
        .flat_map(|example| [&example.chosen, &example.rejected])
        .map(|sequence| sequence.input_ids.len())
        .max()
        .ok_or_else(|| AarambhError::Config("cannot batch zero DPO examples".into()))?;
    let rows = selected.len() * 2;
    let mut input_ids = Vec::with_capacity(rows * max_len);
    let mut labels = Vec::with_capacity(rows * max_len);
    let mut response_mask = Vec::with_capacity(rows * max_len);
    for sequence in selected
        .iter()
        .map(|example| &example.chosen)
        .chain(selected.iter().map(|example| &example.rejected))
    {
        push_padded(&mut input_ids, &sequence.input_ids, max_len, PAD_ID);
        push_padded(&mut labels, &sequence.labels, max_len, PAD_ID);
        push_padded(&mut response_mask, &sequence.response_mask, max_len, 0);
    }
    let candle_device = device.to_candle()?;
    let refs = selected
        .iter()
        .map(|example| example.reference_logps)
        .collect::<Option<Vec<_>>>();
    let (reference_chosen_logps, reference_rejected_logps) = match refs {
        Some(refs) => {
            let chosen = refs.iter().map(|pair| pair.0).collect::<Vec<_>>();
            let rejected = refs.iter().map(|pair| pair.1).collect::<Vec<_>>();
            (
                Some(Tensor::from_vec(chosen, (selected.len(),), &candle_device)?),
                Some(Tensor::from_vec(
                    rejected,
                    (selected.len(),),
                    &candle_device,
                )?),
            )
        }
        None => (None, None),
    };
    Ok(DpoBatch {
        input_ids: Tensor::from_vec(input_ids, (rows, max_len), &candle_device)?,
        labels: Tensor::from_vec(labels, (rows, max_len), &candle_device)?,
        response_mask: Tensor::from_vec(response_mask, (rows, max_len), &candle_device)?,
        reference_chosen_logps,
        reference_rejected_logps,
        batch_size: selected.len(),
        example_indices: indices.to_vec(),
    })
}

fn push_padded(dst: &mut Vec<u32>, values: &[u32], max_len: usize, pad: u32) {
    dst.extend_from_slice(values);
    dst.extend(std::iter::repeat_n(pad, max_len - values.len()));
}

fn dpo_metric_values(
    policy_chosen: &Tensor,
    policy_rejected: &Tensor,
    reference_chosen: Option<&Tensor>,
    reference_rejected: Option<&Tensor>,
    beta: f64,
) -> Result<(f64, f64, f64, f64)> {
    let chosen = policy_chosen.to_vec1::<f32>()?;
    let rejected = policy_rejected.to_vec1::<f32>()?;
    let ref_chosen = reference_chosen
        .map(Tensor::to_vec1::<f32>)
        .transpose()?
        .unwrap_or_else(|| vec![0.0; chosen.len()]);
    let ref_rejected = reference_rejected
        .map(Tensor::to_vec1::<f32>)
        .transpose()?
        .unwrap_or_else(|| vec![0.0; chosen.len()]);
    let mut margin = 0.0;
    let mut accurate = 0usize;
    for (((chosen, rejected), ref_chosen), ref_rejected) in chosen
        .iter()
        .zip(&rejected)
        .zip(&ref_chosen)
        .zip(&ref_rejected)
    {
        let chosen_reward = beta * (*chosen as f64 - *ref_chosen as f64);
        let rejected_reward = beta * (*rejected as f64 - *ref_rejected as f64);
        margin += chosen_reward - rejected_reward;
        accurate += usize::from(chosen_reward > rejected_reward);
    }
    let count = chosen.len() as f64;
    Ok((
        margin / count,
        accurate as f64 / count,
        chosen.iter().map(|value| *value as f64).sum::<f64>() / count,
        rejected.iter().map(|value| *value as f64).sum::<f64>() / count,
    ))
}

fn save_dpo_artifacts(
    varmap: &VarMap,
    metadata: &AdapterMetadata,
    dpo: &DpoSaveMetadata,
    state: &TrainState,
    dir: impl AsRef<Path>,
) -> Result<()> {
    let dir = dir.as_ref();
    save_adapter(varmap, metadata, dir)?;
    write_json(dir.join("dpo_config.json"), dpo)?;
    write_json(dir.join("train_state.json"), state)
}

fn write_json(path: impl AsRef<Path>, value: &impl Serialize) -> Result<()> {
    let file = fs::File::create(path.as_ref())?;
    serde_json::to_writer_pretty(file, value).map_err(AarambhError::Json)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use aarambh_studio_model::AarambhModel;
    use candle_core::Device as CandleDevice;
    use candle_nn::VarBuilder;

    struct NumericTokenizer;

    impl TokenizerLike for NumericTokenizer {
        fn encode(&self, text: &str) -> Result<Vec<u32>> {
            Ok(text.bytes().map(|byte| byte as u32).collect())
        }

        fn decode(&self, ids: &[u32]) -> Result<String> {
            Ok(ids.iter().map(|id| char::from_u32(*id).unwrap()).collect())
        }

        fn vocab_size(&self) -> usize {
            256
        }

        fn bos_token_id(&self) -> Option<u32> {
            None
        }

        fn eos_token_id(&self) -> u32 {
            ENDOFTEXT_ID
        }
    }

    fn examples() -> Vec<DpoExample> {
        vec![
            DpoExample {
                prompt: "Be helpful".into(),
                chosen: "Certainly".into(),
                rejected: "No".into(),
            },
            DpoExample {
                prompt: "Say hello".into(),
                chosen: "Hello".into(),
                rejected: "Leave".into(),
            },
        ]
    }

    fn tiny_model_config() -> ModelConfig {
        ModelConfig {
            vocab_size: 32,
            hidden_dim: 64,
            ffn_dim: 128,
            n_layers: 1,
            n_heads: 1,
            n_kv_heads: 1,
            max_seq_len: 8,
            rope_theta: 10_000.0,
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

    #[test]
    fn dpo_dataset_rejects_identical_responses() {
        let bad = [DpoExample {
            prompt: "p".into(),
            chosen: "same".into(),
            rejected: "same".into(),
        }];
        let err = DpoDataset::from_examples(&bad, &NumericTokenizer, 128, &DpoConfig::default())
            .unwrap_err()
            .to_string();
        assert!(err.contains("identical"), "{err}");
    }

    #[test]
    fn chosen_and_rejected_share_identical_prompt_tokens() {
        let dataset = DpoDataset::from_examples(
            &examples()[..1],
            &NumericTokenizer,
            128,
            &DpoConfig::default(),
        )
        .unwrap();
        let pair = &dataset.examples[0];
        let chosen_start = pair
            .chosen
            .response_mask
            .iter()
            .position(|value| *value == 1)
            .unwrap();
        let rejected_start = pair
            .rejected
            .response_mask
            .iter()
            .position(|value| *value == 1)
            .unwrap();
        assert_eq!(chosen_start, rejected_start);
        assert_eq!(
            &pair.chosen.input_ids[..chosen_start],
            &pair.rejected.input_ids[..rejected_start]
        );
    }

    #[test]
    fn dpo_loader_uses_dynamic_padding_and_keeps_pair_order() {
        let dataset =
            DpoDataset::from_examples(&examples(), &NumericTokenizer, 128, &DpoConfig::default())
                .unwrap();
        let mut loader = DpoDataLoader::new(&dataset, 2, false, 42, Device::Cpu).unwrap();
        let batch = loader.next().unwrap().unwrap();
        assert_eq!(batch.input_ids.dims()[0], 4);
        assert_eq!(batch.batch_size, 2);
        assert_eq!(batch.example_indices, vec![0, 1]);
    }

    #[test]
    fn sequence_log_probs_ignore_prompt_and_padding() {
        let logits = Tensor::from_vec(
            vec![
                0.0f32, 0.0, 0.0, 0.0, // prompt position
                0.0, 2.0, 0.0, 0.0, // scored target 1
                0.0, 0.0, 2.0, 0.0, // scored target 2
                100.0, -100.0, -100.0, -100.0, // masked padding
            ],
            (1, 4, 4),
            &CandleDevice::Cpu,
        )
        .unwrap();
        let labels = Tensor::from_vec(vec![3u32, 1, 2, 3], (1, 4), &CandleDevice::Cpu).unwrap();
        let mask = Tensor::from_vec(vec![0u32, 1, 1, 0], (1, 4), &CandleDevice::Cpu).unwrap();
        let score = sequence_log_probs(&logits, &labels, &mask)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        let expected = 2.0 * (2.0f32.exp() / (2.0f32.exp() + 3.0)).ln();
        assert!((score - expected).abs() < 1e-5, "{score} vs {expected}");
    }

    #[test]
    fn dpo_loss_decreases_when_chosen_relative_logprob_increases() {
        let rejected = Tensor::new(&[-2.0f32], &CandleDevice::Cpu).unwrap();
        let low = Tensor::new(&[-3.0f32], &CandleDevice::Cpu).unwrap();
        let high = Tensor::new(&[-1.0f32], &CandleDevice::Cpu).unwrap();
        let low_loss = dpo_loss(&low, &rejected, None, None, 0.1)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let high_loss = dpo_loss(&high, &rejected, None, None, 0.1)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(high_loss < low_loss);
    }

    #[test]
    fn reference_free_matches_zero_reference_ratio() {
        let chosen = Tensor::new(&[-1.0f32, -2.0], &CandleDevice::Cpu).unwrap();
        let rejected = Tensor::new(&[-3.0f32, -2.5], &CandleDevice::Cpu).unwrap();
        let zeros = Tensor::zeros(2, DType::F32, &CandleDevice::Cpu).unwrap();
        let free = dpo_loss(&chosen, &rejected, None, None, 0.2)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let standard = dpo_loss(&chosen, &rejected, Some(&zeros), Some(&zeros), 0.2)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!((free - standard).abs() < 1e-6);
    }

    #[test]
    fn extreme_dpo_logits_remain_finite() {
        let chosen = Tensor::new(&[10_000.0f32, -10_000.0], &CandleDevice::Cpu).unwrap();
        let rejected = Tensor::new(&[-10_000.0f32, 10_000.0], &CandleDevice::Cpu).unwrap();
        let loss = dpo_loss(&chosen, &rejected, None, None, 0.1)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(loss.is_finite());
    }

    #[test]
    fn truncation_preserves_eos_and_response_score() {
        let config = DpoConfig {
            max_prompt_tokens: Some(8),
            max_completion_tokens: Some(3),
            ..DpoConfig::default()
        };
        let dataset =
            DpoDataset::from_examples(&examples()[..1], &NumericTokenizer, 16, &config).unwrap();
        let chosen = &dataset.examples[0].chosen;
        assert_eq!(*chosen.labels.last().unwrap(), ENDOFTEXT_ID);
        assert!(chosen.response_mask.contains(&1));
    }

    #[test]
    fn dpo_trainer_updates_only_dora_adapter_variables() {
        let device = CandleDevice::Cpu;
        let model_config = tiny_model_config();
        let base_varmap = VarMap::new();
        let base = AarambhModel::new(
            &model_config,
            VarBuilder::from_varmap(&base_varmap, DType::F32, &device),
        )
        .unwrap();
        let dora_config = LoraConfig {
            rank: 2,
            alpha: 4.0,
            dropout: 0.0,
            ..LoraConfig::default()
        };
        let (model, varmap) = DoraAarambhModel::from_tensors(
            &model_config,
            &base.named_tensors(),
            &dora_config,
            false,
            &device,
        )
        .unwrap();
        let pair = TokenizedDpoExample {
            chosen: DpoSequence {
                input_ids: vec![1, 2, 3, 4],
                labels: vec![2, 3, 4, 5],
                response_mask: vec![0, 1, 1, 1],
            },
            rejected: DpoSequence {
                input_ids: vec![1, 2, 6, 7],
                labels: vec![2, 6, 7, 8],
                response_mask: vec![0, 1, 1, 1],
            },
            reference_logps: None,
        };
        let dataset = DpoDataset {
            examples: vec![pair],
        };
        let loader = DpoDataLoader::new(&dataset, 1, false, 42, Device::Cpu).unwrap();
        let mut train_config = TrainConfig {
            batch_size: 1,
            grad_accum_steps: 1,
            max_steps: 1,
            max_epochs: 1,
            warmup_steps: 0,
            save_every_n_steps: 0,
            log_every_n_steps: 0,
            ..TrainConfig::default()
        };
        train_config.checkpoint_dir = std::env::temp_dir().join("aarambh-dpo-test");
        let metadata = AdapterMetadata::new_with_method(
            model_config,
            dora_config,
            None,
            false,
            AdapterMethod::Dora,
        );
        let save_metadata = DpoSaveMetadata {
            dpo: DpoConfig {
                reference_free: true,
                ..DpoConfig::default()
            },
            train: train_config.clone(),
            reference_model: None,
            qdpo: false,
        };
        let mut trainer = DpoTrainer::new(
            model,
            varmap,
            loader,
            save_metadata.dpo.clone(),
            train_config,
            std::env::temp_dir().join("aarambh-dpo-test"),
            metadata,
            save_metadata,
        )
        .unwrap();
        let names = trainer
            .varmap()
            .data()
            .lock()
            .unwrap()
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        assert!(names.iter().all(|name| {
            name.ends_with(".magnitude")
                || name.ends_with(".direction_lora_a")
                || name.ends_with(".direction_lora_b")
        }));
        let batch = trainer.train_loader.next().unwrap().unwrap();
        let metrics = trainer.train_step(batch).unwrap();
        assert!(metrics.loss.is_finite());
        assert!(metrics.did_optimizer_step);
        assert!(metrics.grad_norm.unwrap().is_finite());
    }
}
