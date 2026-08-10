use std::path::{Path, PathBuf};
use std::time::Instant;

use aarambh_studio_audio::MelSpectrogramConfig;
use aarambh_studio_core::{AarambhError, ModelConfig, Result, TokenizerLike, TrainConfig};
use aarambh_studio_model::AarambhModel;
use aarambh_studio_tokenizer::{
    BpeTokenizer, ENDOFTEXT_ID, IMAGE, IMAGE_END, IMAGE_END_ID, IMAGE_ID,
};
use aarambh_studio_vision::{
    ClipVisionEncoder, FrameSamplingStrategy, ImagePreprocessor, LayoutEncodingKind,
    ProjectorConfig, TemporalEncodingKind, VisionEncoderConfig, VisionPreprocessConfig,
    VisionProjector, interleave_image_tokens,
};
use candle_core::backprop::GradStore;
use candle_core::{DType, Tensor};
use candle_nn::{VarBuilder, VarMap};
use serde::{Deserialize, Serialize};

use crate::checkpoint::{CheckpointManager, TrainState};
use crate::config::TrainingRunConfig;
use crate::loss::cross_entropy_loss;
use crate::optim::{AdamW, AdamWConfig, GradMap, clip_gradients};
use crate::schedule::CosineScheduleWithWarmup;

/// Vision training mode and data paths used by Phase 19.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct VisionTrainingConfig {
    /// Vision mode: `projector_pretrain` or `vlm_instruction`.
    pub mode: String,
    /// Frozen base language-model checkpoint.
    pub base_model_path: PathBuf,
    /// JSON config for the frozen CLIP-style encoder.
    pub clip_config_path: PathBuf,
    /// SafeTensors checkpoint for the frozen CLIP-style encoder.
    pub clip_weights_path: PathBuf,
    /// Optional projector checkpoint used by image inference.
    pub projector_path: Option<PathBuf>,
    /// JSONL caption data with `image` or `image_path` plus `caption`.
    pub caption_jsonl: PathBuf,
    /// Root directory used to resolve relative image paths.
    pub image_root: PathBuf,
    /// Projector hidden-width multiplier.
    pub projector_hidden_mult: usize,
    /// Maximum caption tokens, including the end-of-text target.
    pub max_caption_tokens: usize,
    /// Optional cap for local smoke runs.
    pub max_samples: Option<usize>,
    /// Optional native video understanding configuration.
    pub video: Option<VideoTrainingConfig>,
    /// Optional native document understanding configuration.
    pub document: Option<DocumentTrainingConfig>,
    /// Optional native audio understanding configuration (Phase 42).
    pub audio: Option<AudioTrainingConfig>,
}

impl Default for VisionTrainingConfig {
    fn default() -> Self {
        Self {
            mode: "disabled".to_string(),
            base_model_path: PathBuf::new(),
            clip_config_path: PathBuf::new(),
            clip_weights_path: PathBuf::new(),
            projector_path: None,
            caption_jsonl: PathBuf::new(),
            image_root: PathBuf::from("."),
            projector_hidden_mult: 4,
            max_caption_tokens: 128,
            max_samples: None,
            video: None,
            document: None,
            audio: None,
        }
    }
}

impl VisionTrainingConfig {
    /// Validate required fields for projector pretraining.
    pub fn validate(&self) -> Result<()> {
        if let Some(video) = &self.video {
            video.validate()?;
        }
        if let Some(document) = &self.document {
            document.validate()?;
        }
        if let Some(audio) = &self.audio {
            audio.validate()?;
        }
        match self.mode.as_str() {
            "disabled" | "" => Ok(()),
            "projector_pretrain" => {
                if self.base_model_path.as_os_str().is_empty() {
                    return Err(AarambhError::Config(
                        "vision.base_model_path is required for projector_pretrain".into(),
                    ));
                }
                if self.clip_config_path.as_os_str().is_empty()
                    || self.clip_weights_path.as_os_str().is_empty()
                {
                    return Err(AarambhError::Config(
                        "vision.clip_config_path and vision.clip_weights_path are required".into(),
                    ));
                }
                if self.caption_jsonl.as_os_str().is_empty() {
                    return Err(AarambhError::Config(
                        "vision.caption_jsonl is required for projector_pretrain".into(),
                    ));
                }
                if self.projector_hidden_mult == 0 || self.max_caption_tokens == 0 {
                    return Err(AarambhError::Config(
                        "vision projector_hidden_mult and max_caption_tokens must be non-zero"
                            .into(),
                    ));
                }
                Ok(())
            }
            "vlm_instruction" => {
                if self.base_model_path.as_os_str().is_empty() {
                    return Err(AarambhError::Config(
                        "vision.base_model_path is required for vlm_instruction".into(),
                    ));
                }
                if self.clip_config_path.as_os_str().is_empty()
                    || self.clip_weights_path.as_os_str().is_empty()
                {
                    return Err(AarambhError::Config(
                        "vision.clip_config_path and vision.clip_weights_path are required".into(),
                    ));
                }
                if self.projector_path.is_none() {
                    return Err(AarambhError::Config(
                        "vision.projector_path is required for vlm_instruction".into(),
                    ));
                }
                if self.projector_hidden_mult == 0 || self.max_caption_tokens == 0 {
                    return Err(AarambhError::Config(
                        "vision projector_hidden_mult and max_caption_tokens must be non-zero"
                            .into(),
                    ));
                }
                Ok(())
            }
            other => Err(AarambhError::Config(format!(
                "unsupported vision.mode '{other}', expected disabled|projector_pretrain|vlm_instruction"
            ))),
        }
    }
}

/// Native document rasterization, layout projection, batching, and cache configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct DocumentTrainingConfig {
    /// Root directory used to resolve relative document and page-image paths.
    pub document_root: PathBuf,
    /// PDF rendering resolution in dots per inch.
    pub target_dpi: u32,
    /// Maximum number of pages consumed from one document.
    pub max_pages_per_document: usize,
    /// Maximum decoded or rendered pixels accepted for one page.
    pub max_page_pixels: usize,
    /// Maximum number of pages passed through CLIP in one forward call.
    pub encoder_page_batch_size: usize,
    /// Number of detached pre-projector document features cached in memory.
    pub feature_cache_entries: usize,
    /// Learned or sinusoidal row/column layout positions.
    pub layout_encoding: LayoutEncodingKind,
    /// Optional learned layout checkpoint for inference or continued training.
    pub layout_path: Option<PathBuf>,
}

impl Default for DocumentTrainingConfig {
    fn default() -> Self {
        Self {
            document_root: PathBuf::from("."),
            target_dpi: 150,
            max_pages_per_document: 16,
            max_page_pixels: 32_000_000,
            encoder_page_batch_size: 4,
            feature_cache_entries: 8,
            layout_encoding: LayoutEncodingKind::Learned,
            layout_path: None,
        }
    }
}

impl DocumentTrainingConfig {
    /// Validate document resource limits and layout configuration.
    pub fn validate(&self) -> Result<()> {
        if self.target_dpi == 0
            || self.max_pages_per_document == 0
            || self.max_page_pixels == 0
            || self.encoder_page_batch_size == 0
        {
            return Err(AarambhError::Config(
                "vision.document target_dpi, page limits, max_page_pixels, and encoder_page_batch_size must be non-zero".into(),
            ));
        }
        Ok(())
    }
}

/// Native audio decoding, mel-spectrogram, and cache configuration (Phase 42).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct AudioTrainingConfig {
    /// Root directory used to resolve relative audio paths.
    pub audio_root: PathBuf,
    /// JSON config for the frozen audio spectrogram transformer.
    pub encoder_config_path: PathBuf,
    /// SafeTensors checkpoint for the frozen audio spectrogram transformer.
    pub encoder_weights_path: PathBuf,
    /// Mel-spectrogram extraction parameters.
    pub mel: MelSpectrogramConfig,
    /// Maximum number of clips passed through the audio encoder in one forward call.
    pub encoder_batch_size: usize,
    /// Number of detached pre-projector audio features cached in memory.
    pub feature_cache_entries: usize,
}

impl Default for AudioTrainingConfig {
    fn default() -> Self {
        Self {
            audio_root: PathBuf::from("."),
            encoder_config_path: PathBuf::new(),
            encoder_weights_path: PathBuf::new(),
            mel: MelSpectrogramConfig::default(),
            encoder_batch_size: 4,
            feature_cache_entries: 16,
        }
    }
}

impl AudioTrainingConfig {
    /// Validate audio resource bounds and mel configuration.
    pub fn validate(&self) -> Result<()> {
        if self.encoder_batch_size == 0 || self.feature_cache_entries == 0 {
            return Err(AarambhError::Config(
                "vision.audio encoder_batch_size and feature_cache_entries must be non-zero".into(),
            ));
        }
        self.mel.validate()?;
        Ok(())
    }
}

/// Native video decoding, sampling, temporal fusion, and cache configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct VideoTrainingConfig {
    /// Root directory used to resolve relative video paths.
    pub video_root: PathBuf,
    /// Number of source frames sampled per example.
    pub frame_count: usize,
    /// Maximum accepted frame count.
    pub max_frame_count: usize,
    /// Uniform or scene-aware frame selection.
    pub sampling: FrameSamplingStrategy,
    /// Minimum source-frame distance between scene-aware choices.
    pub scene_min_gap: usize,
    /// Learned or sinusoidal temporal frame positions.
    pub temporal_encoding: TemporalEncodingKind,
    /// Optional learned temporal checkpoint for inference or continued training.
    pub temporal_path: Option<PathBuf>,
    /// Maximum number of frames passed through CLIP in one forward call.
    pub encoder_frame_batch_size: usize,
    /// Number of detached pre-projector video features cached in memory.
    pub feature_cache_entries: usize,
}

impl Default for VideoTrainingConfig {
    fn default() -> Self {
        Self {
            video_root: PathBuf::from("."),
            frame_count: 8,
            max_frame_count: 8,
            sampling: FrameSamplingStrategy::Uniform,
            scene_min_gap: 8,
            temporal_encoding: TemporalEncodingKind::Learned,
            temporal_path: None,
            encoder_frame_batch_size: 8,
            feature_cache_entries: 16,
        }
    }
}

impl VideoTrainingConfig {
    /// Validate video resource bounds and temporal configuration.
    pub fn validate(&self) -> Result<()> {
        if self.frame_count == 0 || self.max_frame_count == 0 || self.encoder_frame_batch_size == 0
        {
            return Err(AarambhError::Config(
                "vision.video frame counts and encoder_frame_batch_size must be non-zero".into(),
            ));
        }
        if self.frame_count > self.max_frame_count {
            return Err(AarambhError::Config(format!(
                "vision.video.frame_count {} exceeds max_frame_count {}",
                self.frame_count, self.max_frame_count
            )));
        }
        if self.sampling == FrameSamplingStrategy::SceneAware && self.scene_min_gap == 0 {
            return Err(AarambhError::Config(
                "vision.video.scene_min_gap must be non-zero".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct CaptionExample {
    image: Option<PathBuf>,
    image_path: Option<PathBuf>,
    caption: String,
}

/// Run frozen-encoder, frozen-LLM projector pretraining.
pub fn run_projector_pretrain(config: &TrainingRunConfig) -> Result<()> {
    let vision = config
        .vision
        .as_ref()
        .ok_or_else(|| AarambhError::Config("vision config is required".into()))?;
    let device_selector = config.device()?;
    let dtype = config.dtype_for_device(&device_selector)?.to_candle();
    let candle_device = device_selector.to_candle()?;
    let tokenizer_path = config.tokenizer_path.as_ref().ok_or_else(|| {
        AarambhError::Config("tokenizer_path is required for vision training".into())
    })?;
    let tokenizer = BpeTokenizer::from_pretrained(tokenizer_path)?;
    tokenizer.validate_vision_special_tokens()?;
    let mut model_config = config.model.clone();
    model_config.vocab_size = tokenizer.vocab_size();
    let llm = aarambh_studio_weights::load_any_model_with_dtype(
        &vision.base_model_path,
        &model_config,
        &candle_device,
        dtype,
    )?;
    let encoder_config = VisionEncoderConfig::from_json(&vision.clip_config_path)?;
    let encoder = ClipVisionEncoder::load_pretrained(
        &vision.clip_weights_path,
        encoder_config.clone(),
        &candle_device,
        dtype,
    )?;
    let preprocess = ImagePreprocessor::new(VisionPreprocessConfig {
        image_size: encoder_config.image_size,
        ..VisionPreprocessConfig::default()
    })?;
    let examples = load_caption_examples(vision)?;
    if examples.is_empty() {
        return Err(AarambhError::Config(format!(
            "{} contains no caption examples",
            vision.caption_jsonl.display()
        )));
    }

    let mut trainer = ProjectorTrainer::new(
        &model_config,
        &config.train,
        llm,
        encoder,
        preprocess,
        tokenizer,
        vision.clone(),
        examples,
        &candle_device,
        dtype,
    )?;
    if config.resume && trainer.load_latest_checkpoint()? {
        println!(
            "resumed projector checkpoint at step={}",
            trainer.state.step
        );
    }
    trainer.train()
}

fn load_caption_examples(config: &VisionTrainingConfig) -> Result<Vec<CaptionExample>> {
    let content = std::fs::read_to_string(&config.caption_jsonl)?;
    let mut examples = Vec::new();
    for (line_idx, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let example = serde_json::from_str::<CaptionExample>(line).map_err(|err| {
            AarambhError::Config(format!(
                "failed to parse {} line {}: {err}",
                config.caption_jsonl.display(),
                line_idx + 1
            ))
        })?;
        examples.push(example);
        if config.max_samples.is_some_and(|max| examples.len() >= max) {
            break;
        }
    }
    Ok(examples)
}

struct ProjectorTrainer {
    llm: AarambhModel,
    encoder: ClipVisionEncoder,
    preprocess: ImagePreprocessor,
    projector: VisionProjector,
    projector_varmap: VarMap,
    optimizer: AdamW,
    schedule: CosineScheduleWithWarmup,
    checkpoint: CheckpointManager,
    tokenizer: BpeTokenizer,
    vision_config: VisionTrainingConfig,
    train_config: TrainConfig,
    examples: Vec<CaptionExample>,
    state: TrainState,
    pending_grads: GradMap,
    device: candle_core::Device,
    last_loss: Option<f64>,
    samples_since_log: usize,
    last_log_at: Instant,
}

#[allow(clippy::too_many_arguments)]
impl ProjectorTrainer {
    fn new(
        model_config: &ModelConfig,
        train_config: &TrainConfig,
        llm: AarambhModel,
        encoder: ClipVisionEncoder,
        preprocess: ImagePreprocessor,
        tokenizer: BpeTokenizer,
        vision_config: VisionTrainingConfig,
        examples: Vec<CaptionExample>,
        device: &candle_core::Device,
        dtype: DType,
    ) -> Result<Self> {
        if train_config.batch_size == 0 || train_config.grad_accum_steps == 0 {
            return Err(AarambhError::Config(
                "batch_size and grad_accum_steps must be greater than zero".into(),
            ));
        }
        if train_config.max_steps == 0 {
            return Err(AarambhError::Config(
                "max_steps must be greater than zero".into(),
            ));
        }
        let projector_varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&projector_varmap, dtype, device);
        let projector_config = ProjectorConfig {
            vit_d_model: encoder.config().vit_d_model,
            llm_d_model: model_config.hidden_dim,
            hidden_mult: vision_config.projector_hidden_mult,
        };
        let projector = VisionProjector::new(projector_config, vb)?;
        let optimizer = AdamW::from_varmap(&projector_varmap, AdamWConfig::from(train_config))?;
        Ok(Self {
            llm,
            encoder,
            preprocess,
            projector,
            projector_varmap,
            optimizer,
            schedule: CosineScheduleWithWarmup::from_train_config(train_config),
            checkpoint: CheckpointManager::new(train_config.checkpoint_dir.clone()),
            tokenizer,
            vision_config,
            train_config: train_config.clone(),
            examples,
            state: TrainState::default(),
            pending_grads: GradMap::new(),
            device: device.clone(),
            last_loss: None,
            samples_since_log: 0,
            last_log_at: Instant::now(),
        })
    }

    fn load_latest_checkpoint(&mut self) -> Result<bool> {
        match self.checkpoint.load_latest(
            &mut self.projector_varmap,
            &mut self.optimizer,
            &self.device,
        )? {
            Some(state) => {
                self.state = state;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    fn train(&mut self) -> Result<()> {
        let examples_per_step = self.train_config.batch_size * self.train_config.grad_accum_steps;
        let mut example_idx = 0usize;
        while self.state.epoch < self.train_config.max_epochs
            && self.state.step < self.train_config.max_steps
        {
            if example_idx >= self.examples.len() {
                example_idx = 0;
                self.state.epoch += 1;
                continue;
            }
            let loss = self.example_loss(&self.examples[example_idx])?;
            example_idx += 1;
            let loss_value = loss.to_scalar::<f32>()? as f64;
            if !loss_value.is_finite() {
                return Err(AarambhError::Config(format!(
                    "non-finite projector loss: {loss_value}"
                )));
            }
            let scaled = loss.affine(1.0 / examples_per_step as f64, 0.0)?;
            let grads = scaled.backward()?;
            self.accumulate_gradients(&grads)?;
            self.state.micro_step += 1;
            self.state.train_loss = Some(loss_value);
            self.last_loss = Some(loss_value);
            self.samples_since_log += 1;

            if self.state.micro_step.is_multiple_of(examples_per_step) {
                self.optimizer_step()?;
            }
        }
        if !self.pending_grads.is_empty() && self.state.step < self.train_config.max_steps {
            self.optimizer_step()?;
        }
        self.checkpoint
            .save(&self.projector_varmap, &self.optimizer, &self.state)?;
        Ok(())
    }

    fn example_loss(&self, example: &CaptionExample) -> Result<Tensor> {
        let image_path = resolve_image_path(&self.vision_config.image_root, example)?;
        let image = self
            .preprocess
            .preprocess_path(&image_path, &self.device)?
            .unsqueeze(0)?;
        let patch_tokens = self.encoder.forward(&image)?.detach();
        let projected = self.projector.forward(&patch_tokens)?;
        let (text_tokens, caption_start_idx) = self.caption_tokens(&example.caption)?;
        let text = Tensor::from_vec(text_tokens.clone(), (1, text_tokens.len()), &self.device)?;
        let text_embeddings = self.llm.embed_tokens(&text)?.detach();
        let fused = interleave_image_tokens(&text_tokens, &text_embeddings, &projected, IMAGE_ID)?;
        let logits = self.llm.forward_embeddings_train(&fused)?;
        let image_tokens = projected.dims()[1];
        let (labels, mask) =
            labels_and_mask(&text_tokens, caption_start_idx, image_tokens, IMAGE_ID)?;
        let seq_len = labels.len();
        let labels = Tensor::from_vec(labels, (1, seq_len), &self.device)?;
        let mask = Tensor::from_vec(mask, (1, seq_len), &self.device)?;
        let language_loss = cross_entropy_loss(&logits, &labels, &mask)?;
        let alignment_loss =
            projector_alignment_loss(&projected, &text_embeddings, caption_start_idx)?
                .affine(0.1, 0.0)?;
        Ok((&language_loss + &alignment_loss)?)
    }

    fn caption_tokens(&self, caption: &str) -> Result<(Vec<u32>, usize)> {
        let mut prefix = self.tokenizer.encode(&format!("{IMAGE}{IMAGE_END}"))?;
        if !prefix.contains(&IMAGE_ID) || !prefix.contains(&IMAGE_END_ID) {
            return Err(AarambhError::Tokenizer(
                "tokenizer did not encode image boundary tokens".into(),
            ));
        }
        let caption_start = prefix.len();
        let mut caption_ids = self.tokenizer.encode(caption)?;
        let keep = self.vision_config.max_caption_tokens.saturating_sub(1);
        if caption_ids.len() > keep {
            caption_ids.truncate(keep);
        }
        caption_ids.push(ENDOFTEXT_ID);
        prefix.extend(caption_ids);
        Ok((prefix, caption_start))
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
                "projector backward produced no parameter gradients".into(),
            ));
        }
        for (name, grad) in updates {
            self.pending_grads.insert(name, grad);
        }
        Ok(())
    }

    fn optimizer_step(&mut self) -> Result<()> {
        let lr = self.schedule.lr_at_step(self.state.step);
        let grad_norm = clip_gradients(&mut self.pending_grads, self.train_config.clip_grad_norm)?;
        self.optimizer.step(&self.pending_grads, lr)?;
        self.pending_grads.clear();
        self.state.step += 1;
        self.after_step(lr, grad_norm)
    }

    fn after_step(&mut self, lr: f64, grad_norm: f64) -> Result<()> {
        let loss = self.last_loss.unwrap_or(0.0);
        if self.train_config.log_every_n_steps > 0
            && self
                .state
                .step
                .is_multiple_of(self.train_config.log_every_n_steps)
        {
            let samples_per_second = self.samples_per_second_since_last_log();
            println!(
                "vision_projector step={} loss={:.4} ppl={:.2} lr={:.6} grad_norm={:.4} samples/s={:.2}",
                self.state.step,
                loss,
                loss.exp(),
                lr,
                grad_norm,
                samples_per_second
            );
        }
        if self.train_config.save_every_n_steps > 0
            && self
                .state
                .step
                .is_multiple_of(self.train_config.save_every_n_steps)
        {
            self.checkpoint
                .save(&self.projector_varmap, &self.optimizer, &self.state)?;
        }
        Ok(())
    }

    fn samples_per_second_since_last_log(&mut self) -> f64 {
        let elapsed = self.last_log_at.elapsed().as_secs_f64();
        let samples = self.samples_since_log;
        self.samples_since_log = 0;
        self.last_log_at = Instant::now();
        if elapsed > 0.0 {
            samples as f64 / elapsed
        } else {
            0.0
        }
    }
}

fn projector_alignment_loss(
    projected: &Tensor,
    text_embeddings: &Tensor,
    caption_start_idx: usize,
) -> Result<Tensor> {
    let text_dims = text_embeddings.dims();
    if text_dims.len() != 3 || caption_start_idx >= text_dims[1] {
        return Err(AarambhError::Shape(format!(
            "caption_start_idx {caption_start_idx} is outside text embedding shape {text_dims:?}"
        )));
    }
    let caption_len = text_dims[1] - caption_start_idx;
    let image_mean = projected.mean(1)?;
    let caption = text_embeddings.narrow(1, caption_start_idx, caption_len)?;
    let caption_mean = caption.mean(1)?;
    Ok(image_mean.broadcast_sub(&caption_mean)?.sqr()?.mean_all()?)
}

fn resolve_image_path(root: &Path, example: &CaptionExample) -> Result<PathBuf> {
    let path = example
        .image_path
        .as_ref()
        .or(example.image.as_ref())
        .ok_or_else(|| AarambhError::Config("caption example is missing image path".into()))?;
    if path.is_absolute() {
        Ok(path.clone())
    } else {
        Ok(root.join(path))
    }
}

fn labels_and_mask(
    text_tokens: &[u32],
    caption_start_idx: usize,
    image_token_count: usize,
    image_placeholder_id: u32,
) -> Result<(Vec<u32>, Vec<u32>)> {
    let mut items = Vec::new();
    for (idx, token) in text_tokens.iter().enumerate() {
        if *token == image_placeholder_id {
            items.extend(std::iter::repeat_n((None, idx), image_token_count));
        } else {
            items.push((Some(*token), idx));
        }
    }
    let first_caption = items
        .iter()
        .find_map(|(token, original_idx)| (*original_idx >= caption_start_idx).then_some(*token))
        .flatten();
    let mut labels = Vec::with_capacity(items.len());
    let mut mask = Vec::with_capacity(items.len());
    for idx in 0..items.len() {
        match (items.get(idx).copied(), items.get(idx + 1).copied()) {
            (Some((None, _)), Some((Some(_), original_idx)))
                if original_idx < caption_start_idx && first_caption.is_some() =>
            {
                labels.push(first_caption.unwrap_or(0));
                mask.push(1);
            }
            (_, Some((Some(token), original_idx))) if original_idx >= caption_start_idx => {
                labels.push(token);
                mask.push(1);
            }
            (_, Some((Some(token), _))) => {
                labels.push(token);
                mask.push(0);
            }
            _ => {
                labels.push(0);
                mask.push(0);
            }
        }
    }
    if !mask.contains(&1) {
        return Err(AarambhError::Shape(
            "vision caption loss mask has no supervised tokens".into(),
        ));
    }
    Ok((labels, mask))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_mask_caption_only_after_image_prefix() {
        let text = vec![IMAGE_ID, IMAGE_END_ID, 10, 11, ENDOFTEXT_ID];
        let (labels, mask) = labels_and_mask(&text, 2, 3, IMAGE_ID).unwrap();
        assert_eq!(labels.len(), 7);
        assert_eq!(labels[2], 10);
        assert_eq!(mask, vec![0, 0, 1, 1, 1, 1, 0]);
    }
}
