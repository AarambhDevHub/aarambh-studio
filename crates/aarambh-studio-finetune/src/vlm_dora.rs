use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use aarambh_studio_audio::{
    AudioEncoderConfig, AudioPreprocessor, AudioProjector, AudioProjectorConfig, AudioQaExample,
    FrozenAudioEncoder, interleave_audio_tokens, load_audio_qa_jsonl,
};
use aarambh_studio_core::{AarambhError, Device, ModelConfig, Result, TokenizerLike, TrainConfig};
use aarambh_studio_tokenizer::{
    AUDIO, AUDIO_END, AUDIO_ID, BpeTokenizer, DOCUMENT, DOCUMENT_END, DOCUMENT_ID, FRAME_SEP,
    FRAME_SEP_ID, IMAGE, IMAGE_END, IMAGE_ID, PAGE_SEP, PAGE_SEP_ID, VIDEO, VIDEO_END, VIDEO_ID,
};
use aarambh_studio_train::optim::clip_gradients;
use aarambh_studio_train::{
    AdamW, AdamWConfig, AudioTrainingConfig, CosineScheduleWithWarmup, DocumentTrainingConfig,
    GradMap, TrainState, VideoTrainingConfig, VisionTrainingConfig, cross_entropy_loss,
};
use aarambh_studio_vision::{
    ClipVisionEncoder, DocQaExample, DocumentFeatureCache, DocumentFeatureCacheKey, DocumentSource,
    ImagePreprocessor, LayoutAwareProjector, LayoutEncodingKind, LayoutProjectorConfig,
    PageRasterizer, PageRasterizerConfig, ProjectorConfig, TemporalEncoder, TemporalEncodingConfig,
    TemporalEncodingKind, VideoFeatureCache, VideoFeatureCacheKey, VideoQaExample,
    VideoSamplingConfig, VisionEncoderConfig, VisionPreprocessConfig, VisionProjector, VqaExample,
    decode_sampled_video, interleave_document_tokens, interleave_image_tokens,
    interleave_video_tokens, load_document_qa_jsonl, load_video_qa, load_vqa_jsonl,
};
use candle_core::backprop::GradStore;
use candle_core::{DType, Tensor};
use candle_nn::{VarBuilder, VarMap};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::Serialize;

use crate::adapter::{AdapterMetadata, AdapterMethod, save_adapter};
use crate::dora::DoraAarambhModel;
use crate::lora::LoraConfig;
use crate::sft::ChatTemplate;

/// Configuration for one vision-language DoRA instruction-tuning run.
#[derive(Debug, Clone)]
pub struct VlmDoraRunConfig {
    /// Base model configuration.
    pub model_config: ModelConfig,
    /// Training hyperparameters.
    pub train_config: TrainConfig,
    /// Frozen base model path.
    pub base_model_path: PathBuf,
    /// Tokenizer JSON path. Must include Phase 19 vision tokens.
    pub tokenizer_path: PathBuf,
    /// VQA instruction JSONL data path.
    pub data_path: PathBuf,
    /// Output adapter directory.
    pub output_dir: PathBuf,
    /// DoRA adapter configuration.
    pub lora_config: LoraConfig,
    /// Logical training device.
    pub device: Device,
    /// Candle dtype used for model, encoder, and projector weights.
    pub dtype: DType,
    /// Whether to use QDoRA quantized base linear weights.
    pub qdora: bool,
    /// Whether to shuffle VQA examples each epoch.
    pub shuffle: bool,
    /// Phase 19 vision paths and dimensions.
    pub vision: VisionTrainingConfig,
    /// Phase 19 projector checkpoint to initialize from.
    pub projector_path: PathBuf,
    /// Whether projector parameters are updated alongside DoRA adapters.
    pub train_projector: bool,
}

/// Configuration for video-language DoRA instruction tuning.
#[derive(Debug, Clone)]
pub struct VideoVlmDoraRunConfig {
    /// Shared image VLM model, optimizer, and vision paths.
    pub vlm: VlmDoraRunConfig,
}

/// Configuration for document-language DoRA instruction tuning.
#[derive(Debug, Clone)]
pub struct DocumentVlmDoraRunConfig {
    /// Shared VLM model, optimizer, vision, and document paths.
    pub vlm: VlmDoraRunConfig,
}

/// Configuration for audio-language DoRA instruction tuning (Phase 42).
#[derive(Debug, Clone)]
pub struct AudioVlmDoraRunConfig {
    /// Shared VLM model, optimizer, vision, and audio paths.
    pub vlm: VlmDoraRunConfig,
}

/// Metrics emitted by a VLM DoRA optimizer step.
#[derive(Debug, Clone)]
pub struct VlmDoraMetrics {
    /// Current optimizer step.
    pub step: usize,
    /// Most recent micro-batch loss.
    pub loss: f64,
    /// Exponential of the loss.
    pub perplexity: f64,
    /// Learning rate.
    pub lr: f64,
    /// DoRA adapter gradient norm.
    pub dora_grad_norm: f64,
    /// Projector gradient norm when projector training is enabled.
    pub projector_grad_norm: Option<f64>,
    /// Temporal-position gradient norm for learned video positions.
    pub temporal_grad_norm: Option<f64>,
    /// Layout-position gradient norm for learned document positions.
    pub layout_grad_norm: Option<f64>,
    /// Examples processed per second since the last log.
    pub samples_per_second: f64,
}

/// Trainer for vision-language instruction tuning with a frozen encoder and DoRA LLM adapters.
pub struct VlmDoraTrainer {
    model: DoraAarambhModel,
    dora_varmap: VarMap,
    dora_optimizer: AdamW,
    projector: VisionProjector,
    projector_varmap: VarMap,
    projector_optimizer: Option<AdamW>,
    encoder: ClipVisionEncoder,
    preprocess: ImagePreprocessor,
    tokenizer: BpeTokenizer,
    train_config: TrainConfig,
    schedule: CosineScheduleWithWarmup,
    output_dir: PathBuf,
    metadata: AdapterMetadata,
    vlm_metadata: VlmArtifactsMetadata,
    examples: Vec<MultimodalExample>,
    video: Option<VideoTrainingRuntime>,
    document: Option<DocumentTrainingRuntime>,
    shuffle: bool,
    rng: StdRng,
    state: TrainState,
    dora_pending_grads: GradMap,
    projector_pending_grads: GradMap,
    device: candle_core::Device,
    last_loss: Option<f64>,
    samples_since_log: usize,
    last_log_at: Instant,
}

#[derive(Debug, Clone)]
enum MultimodalExample {
    Image(VqaExample),
    Video(VideoQaExample),
    Document(DocQaExample),
}

#[derive(Debug, Clone, Copy)]
enum MediaLayout {
    Image {
        patch_count: usize,
    },
    Video {
        frame_count: usize,
        patch_count: usize,
    },
    Document {
        page_count: usize,
        patch_count: usize,
    },
}

struct VideoTrainingRuntime {
    config: VideoTrainingConfig,
    temporal: TemporalEncoder,
    temporal_varmap: VarMap,
    temporal_optimizer: Option<AdamW>,
    temporal_pending_grads: GradMap,
    feature_cache: VideoFeatureCache,
    encoder_signature: String,
}

struct VideoRuntimeParts {
    config: VideoTrainingConfig,
    temporal: TemporalEncoder,
    temporal_varmap: VarMap,
    encoder_signature: String,
}

struct DocumentTrainingRuntime {
    config: DocumentTrainingConfig,
    layout: LayoutAwareProjector,
    layout_varmap: VarMap,
    layout_optimizer: Option<AdamW>,
    layout_pending_grads: GradMap,
    rasterizer: PageRasterizer,
    feature_cache: DocumentFeatureCache,
    encoder_signature: String,
}

struct DocumentRuntimeParts {
    config: DocumentTrainingConfig,
    layout: LayoutAwareProjector,
    layout_varmap: VarMap,
    rasterizer: PageRasterizer,
    encoder_signature: String,
}

impl VlmDoraTrainer {
    /// Create a VLM DoRA trainer from loaded model, vision, and data components.
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: DoraAarambhModel,
        dora_varmap: VarMap,
        projector: VisionProjector,
        projector_varmap: VarMap,
        encoder: ClipVisionEncoder,
        preprocess: ImagePreprocessor,
        tokenizer: BpeTokenizer,
        train_config: TrainConfig,
        output_dir: impl Into<PathBuf>,
        metadata: AdapterMetadata,
        vlm_metadata: VlmArtifactsMetadata,
        examples: Vec<MultimodalExample>,
        shuffle: bool,
        train_projector: bool,
        video_parts: Option<VideoRuntimeParts>,
        document_parts: Option<DocumentRuntimeParts>,
        device: candle_core::Device,
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
        if examples.is_empty() {
            return Err(AarambhError::Config(
                "VLM DoRA trainer needs at least one VQA example".into(),
            ));
        }
        let dora_optimizer = AdamW::from_varmap(&dora_varmap, AdamWConfig::from(&train_config))?;
        if dora_optimizer.parameters().is_empty() {
            return Err(AarambhError::Config(
                "VLM DoRA target_modules produced zero trainable tensors".into(),
            ));
        }
        let projector_optimizer = if train_projector {
            let optimizer =
                AdamW::from_varmap(&projector_varmap, AdamWConfig::from(&train_config))?;
            if optimizer.parameters().is_empty() {
                return Err(AarambhError::Config(
                    "projector training requested but projector has zero trainable tensors".into(),
                ));
            }
            Some(optimizer)
        } else {
            None
        };
        let video = video_parts
            .map(|parts| -> Result<VideoTrainingRuntime> {
                let temporal_optimizer =
                    if parts.config.temporal_encoding == TemporalEncodingKind::Learned {
                        let optimizer = AdamW::from_varmap(
                            &parts.temporal_varmap,
                            AdamWConfig::from(&train_config),
                        )?;
                        if optimizer.parameters().is_empty() {
                            return Err(AarambhError::Config(
                                "learned temporal encoding produced zero trainable tensors".into(),
                            ));
                        }
                        Some(optimizer)
                    } else {
                        None
                    };
                Ok(VideoTrainingRuntime {
                    feature_cache: VideoFeatureCache::new(parts.config.feature_cache_entries),
                    config: parts.config,
                    temporal: parts.temporal,
                    temporal_varmap: parts.temporal_varmap,
                    temporal_optimizer,
                    temporal_pending_grads: GradMap::new(),
                    encoder_signature: parts.encoder_signature,
                })
            })
            .transpose()?;
        let document = document_parts
            .map(|parts| -> Result<DocumentTrainingRuntime> {
                let layout_optimizer = if parts.config.layout_encoding
                    == LayoutEncodingKind::Learned
                {
                    let optimizer =
                        AdamW::from_varmap(&parts.layout_varmap, AdamWConfig::from(&train_config))?;
                    if optimizer.parameters().is_empty() {
                        return Err(AarambhError::Config(
                            "learned document layout encoding produced zero trainable tensors"
                                .into(),
                        ));
                    }
                    Some(optimizer)
                } else {
                    None
                };
                Ok(DocumentTrainingRuntime {
                    feature_cache: DocumentFeatureCache::new(parts.config.feature_cache_entries),
                    config: parts.config,
                    layout: parts.layout,
                    layout_varmap: parts.layout_varmap,
                    layout_optimizer,
                    layout_pending_grads: GradMap::new(),
                    rasterizer: parts.rasterizer,
                    encoder_signature: parts.encoder_signature,
                })
            })
            .transpose()?;

        let seed = train_config.seed;
        Ok(Self {
            model,
            dora_varmap,
            dora_optimizer,
            projector,
            projector_varmap,
            projector_optimizer,
            encoder,
            preprocess,
            tokenizer,
            schedule: CosineScheduleWithWarmup::from_train_config(&train_config),
            train_config,
            output_dir: output_dir.into(),
            metadata,
            vlm_metadata,
            examples,
            video,
            document,
            shuffle,
            rng: StdRng::seed_from_u64(seed),
            state: TrainState::default(),
            dora_pending_grads: GradMap::new(),
            projector_pending_grads: GradMap::new(),
            device,
            last_loss: None,
            samples_since_log: 0,
            last_log_at: Instant::now(),
        })
    }

    /// Return the training state.
    pub fn state(&self) -> &TrainState {
        &self.state
    }

    /// Run the full VLM DoRA training loop and save final artifacts.
    pub fn train(&mut self) -> Result<()> {
        let examples_per_step = self.train_config.batch_size * self.train_config.grad_accum_steps;
        let mut example_idx = 0usize;
        while self.state.epoch < self.train_config.max_epochs
            && self.state.step < self.train_config.max_steps
        {
            if example_idx == 0 && self.shuffle {
                self.examples.shuffle(&mut self.rng);
            }
            if example_idx >= self.examples.len() {
                example_idx = 0;
                self.state.epoch += 1;
                continue;
            }
            let example = self.examples[example_idx].clone();
            let loss = self.example_loss(&example)?;
            example_idx += 1;
            let loss_value = loss.to_scalar::<f32>()? as f64;
            if !loss_value.is_finite() {
                return Err(AarambhError::Config(format!(
                    "non-finite VLM DoRA loss: {loss_value}"
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
        if !self.dora_pending_grads.is_empty() && self.state.step < self.train_config.max_steps {
            self.optimizer_step()?;
        }
        self.save_final()
    }

    fn example_loss(&mut self, example: &MultimodalExample) -> Result<Tensor> {
        let (projected, text_tokens, target_start_idx, media) = match example {
            MultimodalExample::Image(example) => {
                let image_path =
                    resolve_media_path(&self.vlm_metadata.image_root, &example.image_path);
                let image = self
                    .preprocess
                    .preprocess_path(&image_path, &self.device)?
                    .unsqueeze(0)?;
                let patch_tokens = self.encoder.forward(&image)?.detach();
                let projected = self.projector.forward(&patch_tokens)?;
                let patch_count = projected.dims()[1];
                let (tokens, target) = self.multimodal_tokens(
                    &example.question,
                    &example.answer,
                    example.thinking.as_deref(),
                    &format!("{IMAGE}{IMAGE_END}"),
                    IMAGE_ID,
                    patch_count,
                )?;
                (
                    projected,
                    tokens,
                    target,
                    MediaLayout::Image { patch_count },
                )
            }
            MultimodalExample::Video(example) => {
                let (projected, frame_count, patch_count) = self.project_video(example)?;
                let prefix = video_marker_prefix(frame_count);
                let (tokens, target) = self.multimodal_tokens(
                    &example.question,
                    &example.answer,
                    example.thinking.as_deref(),
                    &prefix,
                    VIDEO_ID,
                    frame_count * patch_count,
                )?;
                (
                    projected,
                    tokens,
                    target,
                    MediaLayout::Video {
                        frame_count,
                        patch_count,
                    },
                )
            }
            MultimodalExample::Document(example) => {
                let (projected, page_count, patch_count) = self.project_document(example)?;
                let prefix = document_marker_prefix(page_count);
                let (tokens, target) = self.multimodal_tokens(
                    &example.question,
                    example.primary_answer(),
                    example.thinking.as_deref(),
                    &prefix,
                    DOCUMENT_ID,
                    page_count * patch_count,
                )?;
                (
                    projected,
                    tokens,
                    target,
                    MediaLayout::Document {
                        page_count,
                        patch_count,
                    },
                )
            }
        };
        let text = Tensor::from_vec(text_tokens.clone(), (1, text_tokens.len()), &self.device)?;
        let text_embeddings = self.model.embed_tokens(&text)?.detach();
        let fused = match media {
            MediaLayout::Image { .. } => {
                interleave_image_tokens(&text_tokens, &text_embeddings, &projected, IMAGE_ID)?
            }
            MediaLayout::Video { .. } => interleave_video_tokens(
                &text_tokens,
                &text_embeddings,
                &projected,
                VIDEO_ID,
                FRAME_SEP_ID,
            )?,
            MediaLayout::Document { .. } => interleave_document_tokens(
                &text_tokens,
                &text_embeddings,
                &projected,
                DOCUMENT_ID,
                PAGE_SEP_ID,
            )?,
        };
        let logits = self.model.forward_embeddings_train(&fused)?;
        let (labels, mask) = multimodal_labels_and_mask(&text_tokens, target_start_idx, media)?;
        let seq_len = labels.len();
        let labels = Tensor::from_vec(labels, (1, seq_len), &self.device)?;
        let mask = Tensor::from_vec(mask, (1, seq_len), &self.device)?;
        cross_entropy_loss(&logits, &labels, &mask)
    }

    fn multimodal_tokens(
        &self,
        question: &str,
        answer: &str,
        thinking: Option<&str>,
        media_prefix: &str,
        placeholder_id: u32,
        inserted_tokens: usize,
    ) -> Result<(Vec<u32>, usize)> {
        if inserted_tokens == 0 || inserted_tokens > self.model.config().max_seq_len {
            return Err(AarambhError::Shape(format!(
                "multimodal token count {inserted_tokens} is invalid for max_seq_len {}",
                self.model.config().max_seq_len
            )));
        }
        let template = ChatTemplate;
        let prefix = format!("{media_prefix}\n{}", template.prefix(question, None));
        let target = match thinking {
            Some(thinking) => template.thinking_target(thinking, answer),
            None => template.target(answer),
        };
        let prefix_ids = self.tokenizer.encode(&prefix)?;
        let mut target_ids = self.tokenizer.encode(&target)?;
        if !prefix_ids.contains(&placeholder_id) {
            return Err(AarambhError::Tokenizer(
                "VLM prefix did not encode the multimodal placeholder token".into(),
            ));
        }
        if target_ids.is_empty() {
            return Err(AarambhError::Config(
                "VLM target encoded to zero tokens".into(),
            ));
        }

        let max_text_tokens = self.model.config().max_seq_len + 1 - inserted_tokens;
        if prefix_ids.len() >= max_text_tokens {
            return Err(AarambhError::Shape(format!(
                "VLM prompt has {} text tokens plus {inserted_tokens} media tokens, exceeding max_seq_len {}",
                prefix_ids.len(),
                self.model.config().max_seq_len
            )));
        }
        let keep_target = max_text_tokens - prefix_ids.len();
        if target_ids.len() > keep_target {
            target_ids.truncate(keep_target);
        }
        let target_start_idx = prefix_ids.len();
        let mut tokens = prefix_ids;
        tokens.extend(target_ids);
        if tokens.len() < 2 {
            return Err(AarambhError::Config(
                "VLM sequence must contain at least two tokens".into(),
            ));
        }
        Ok((tokens, target_start_idx))
    }

    fn project_video(&mut self, example: &VideoQaExample) -> Result<(Tensor, usize, usize)> {
        let video = self.video.as_mut().ok_or_else(|| {
            AarambhError::Config("video example reached an image-only VLM trainer".into())
        })?;
        let path = resolve_media_path(&video.config.video_root, &example.video_path);
        let sampling = VideoSamplingConfig {
            frame_count: video.config.frame_count,
            max_frame_count: video.config.max_frame_count,
            strategy: video.config.sampling,
            scene_min_gap: video.config.scene_min_gap,
        };
        let cache_key =
            VideoFeatureCacheKey::new(&path, sampling.clone(), &video.encoder_signature)?;
        let patch_tokens = match video.feature_cache.get(&cache_key) {
            Some(features) => features,
            None => {
                let sampled = decode_sampled_video(&path, &sampling)?;
                let pixels = self
                    .preprocess
                    .preprocess_rgb_batch(&sampled.frames, &self.device)?;
                let mut chunks = Vec::new();
                for start in
                    (0..sampled.frames.len()).step_by(video.config.encoder_frame_batch_size)
                {
                    let len = video
                        .config
                        .encoder_frame_batch_size
                        .min(sampled.frames.len() - start);
                    chunks.push(self.encoder.forward(&pixels.narrow(0, start, len)?)?);
                }
                let references = chunks.iter().collect::<Vec<_>>();
                let features = Tensor::cat(&references, 0)?.detach();
                video.feature_cache.insert(cache_key, features.clone());
                features
            }
        };
        let temporal = video.temporal.forward(&patch_tokens)?;
        let projected = self.projector.forward(&temporal)?;
        Ok((projected, patch_tokens.dims()[0], patch_tokens.dims()[1]))
    }

    fn project_document(&mut self, example: &DocQaExample) -> Result<(Tensor, usize, usize)> {
        let document = self.document.as_mut().ok_or_else(|| {
            AarambhError::Config("document example reached a non-document VLM trainer".into())
        })?;
        let source = resolve_document_source(&document.config.document_root, &example.source());
        let rasterizer_config = PageRasterizerConfig {
            target_dpi: document.config.target_dpi,
            max_pages_per_document: document.config.max_pages_per_document,
            max_page_pixels: document.config.max_page_pixels,
        };
        let cache_key = DocumentFeatureCacheKey::new(
            &source,
            example.pages.as_deref(),
            rasterizer_config,
            &document.encoder_signature,
        )?;
        let patch_tokens = match document.feature_cache.get(&cache_key) {
            Some(features) => features,
            None => {
                let rendered = document
                    .rasterizer
                    .rasterize(&source, example.pages.as_deref())?;
                if rendered.truncated {
                    eprintln!(
                        "warning: document has {} pages; using the first {}",
                        rendered.source_page_count,
                        rendered.pages.len()
                    );
                }
                let pages = rendered
                    .pages
                    .into_iter()
                    .map(|page| page.image)
                    .collect::<Vec<_>>();
                let pixels = self
                    .preprocess
                    .preprocess_document_pages(&pages, &self.device)?;
                let mut chunks = Vec::new();
                for start in (0..pages.len()).step_by(document.config.encoder_page_batch_size) {
                    let len = document
                        .config
                        .encoder_page_batch_size
                        .min(pages.len() - start);
                    chunks.push(self.encoder.forward(&pixels.narrow(0, start, len)?)?);
                }
                let references = chunks.iter().collect::<Vec<_>>();
                let features = Tensor::cat(&references, 0)?.detach();
                document.feature_cache.insert(cache_key, features.clone());
                features
            }
        };
        let grid = (
            document.layout.config().patch_rows,
            document.layout.config().patch_cols,
        );
        let projected = document.layout.forward(&patch_tokens, grid)?;
        Ok((projected, patch_tokens.dims()[0], patch_tokens.dims()[1]))
    }

    fn accumulate_gradients(&mut self, grads: &GradStore) -> Result<()> {
        accumulate_for_optimizer(
            grads,
            &self.dora_optimizer,
            &mut self.dora_pending_grads,
            "VLM DoRA",
        )?;
        if let Some(projector_optimizer) = &self.projector_optimizer {
            accumulate_for_optimizer(
                grads,
                projector_optimizer,
                &mut self.projector_pending_grads,
                "VLM projector",
            )?;
        }
        if let Some(video) = &mut self.video
            && let Some(temporal_optimizer) = &video.temporal_optimizer
        {
            accumulate_for_optimizer(
                grads,
                temporal_optimizer,
                &mut video.temporal_pending_grads,
                "VLM temporal encoder",
            )?;
        }
        if let Some(document) = &mut self.document
            && let Some(layout_optimizer) = &document.layout_optimizer
        {
            accumulate_for_optimizer(
                grads,
                layout_optimizer,
                &mut document.layout_pending_grads,
                "VLM document layout encoder",
            )?;
        }
        Ok(())
    }

    fn optimizer_step(&mut self) -> Result<()> {
        let lr = self.schedule.lr_at_step(self.state.step);
        let dora_grad_norm = clip_gradients(
            &mut self.dora_pending_grads,
            self.train_config.clip_grad_norm,
        )?;
        self.dora_optimizer.step(&self.dora_pending_grads, lr)?;
        self.dora_pending_grads.clear();

        let projector_grad_norm = if let Some(projector_optimizer) = &mut self.projector_optimizer {
            let norm = clip_gradients(
                &mut self.projector_pending_grads,
                self.train_config.clip_grad_norm,
            )?;
            projector_optimizer.step(&self.projector_pending_grads, lr)?;
            self.projector_pending_grads.clear();
            Some(norm)
        } else {
            None
        };
        let temporal_grad_norm = if let Some(video) = &mut self.video
            && let Some(temporal_optimizer) = &mut video.temporal_optimizer
        {
            let norm = clip_gradients(
                &mut video.temporal_pending_grads,
                self.train_config.clip_grad_norm,
            )?;
            temporal_optimizer.step(&video.temporal_pending_grads, lr)?;
            video.temporal_pending_grads.clear();
            Some(norm)
        } else {
            None
        };
        let layout_grad_norm = if let Some(document) = &mut self.document
            && let Some(layout_optimizer) = &mut document.layout_optimizer
        {
            let norm = clip_gradients(
                &mut document.layout_pending_grads,
                self.train_config.clip_grad_norm,
            )?;
            layout_optimizer.step(&document.layout_pending_grads, lr)?;
            document.layout_pending_grads.clear();
            Some(norm)
        } else {
            None
        };

        self.state.step += 1;
        let metrics = VlmDoraMetrics {
            step: self.state.step,
            loss: self.last_loss.unwrap_or(0.0),
            perplexity: self.last_loss.unwrap_or(0.0).exp(),
            lr,
            dora_grad_norm,
            projector_grad_norm,
            temporal_grad_norm,
            layout_grad_norm,
            samples_per_second: self.samples_per_second_since_last_log(),
        };
        self.after_optimizer_step(&metrics)
    }

    fn after_optimizer_step(&self, metrics: &VlmDoraMetrics) -> Result<()> {
        if self.train_config.log_every_n_steps > 0
            && metrics
                .step
                .is_multiple_of(self.train_config.log_every_n_steps)
        {
            let projector = metrics
                .projector_grad_norm
                .map(|value| format!(" projector_grad_norm={value:.4}"))
                .unwrap_or_default();
            let temporal = metrics
                .temporal_grad_norm
                .map(|value| format!(" temporal_grad_norm={value:.4}"))
                .unwrap_or_default();
            let layout = metrics
                .layout_grad_norm
                .map(|value| format!(" layout_grad_norm={value:.4}"))
                .unwrap_or_default();
            println!(
                "vlm_dora step={} loss={:.4} ppl={:.2} lr={:.6} dora_grad_norm={:.4}{}{}{} samples/s={:.2}",
                metrics.step,
                metrics.loss,
                metrics.perplexity,
                metrics.lr,
                metrics.dora_grad_norm,
                projector,
                temporal,
                layout,
                metrics.samples_per_second
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

    fn save_final(&self) -> Result<()> {
        save_vlm_artifacts(
            &self.dora_varmap,
            &self.projector_varmap,
            MediaArtifactVarMaps {
                temporal: self.video.as_ref().map(|video| &video.temporal_varmap),
                layout: self
                    .document
                    .as_ref()
                    .map(|document| &document.layout_varmap),
            },
            &self.metadata,
            &self.vlm_metadata,
            &self.state,
            &self.output_dir,
        )
    }

    fn save_step(&self) -> Result<()> {
        let dir = self
            .output_dir
            .join("checkpoints")
            .join(format!("step_{:06}", self.state.step));
        save_vlm_artifacts(
            &self.dora_varmap,
            &self.projector_varmap,
            MediaArtifactVarMaps {
                temporal: self.video.as_ref().map(|video| &video.temporal_varmap),
                layout: self
                    .document
                    .as_ref()
                    .map(|document| &document.layout_varmap),
            },
            &self.metadata,
            &self.vlm_metadata,
            &self.state,
            dir,
        )
    }
}

/// Build and run a VLM DoRA trainer from a run configuration.
pub fn run_vlm_dora_from_config(config: VlmDoraRunConfig) -> Result<()> {
    config.lora_config.validate()?;
    let candle_device = config.device.to_candle()?;
    let tokenizer = BpeTokenizer::from_pretrained(&config.tokenizer_path)?;
    tokenizer.validate_vision_special_tokens()?;
    let mut model_config = config.model_config.clone();
    model_config.vocab_size = tokenizer.vocab_size();
    if model_config.moe.is_some() {
        return Err(AarambhError::Config(
            "VLM DoRA training for MoE models is not supported; train the MoE base model directly or use a dense config".into(),
        ));
    }

    let base = aarambh_studio_weights::load_any_model_with_dtype(
        &config.base_model_path,
        &model_config,
        &candle_device,
        config.dtype,
    )?;
    let base_tensors = base.named_tensors();
    drop(base);

    let (model, dora_varmap) = DoraAarambhModel::from_tensors(
        &model_config,
        &base_tensors,
        &config.lora_config,
        config.qdora,
        &candle_device,
    )?;
    eprintln!(
        "vlm adapter params: {} / {} ({:.3}%)",
        model.adapter_param_count(),
        model.base_param_count(),
        model.trainable_ratio() * 100.0
    );

    let encoder_config = VisionEncoderConfig::from_json(&config.vision.clip_config_path)?;
    let encoder = ClipVisionEncoder::load_pretrained(
        &config.vision.clip_weights_path,
        encoder_config.clone(),
        &candle_device,
        config.dtype,
    )?;
    let preprocess = ImagePreprocessor::new(VisionPreprocessConfig {
        image_size: encoder_config.image_size,
        ..VisionPreprocessConfig::default()
    })?;
    let projector_varmap = VarMap::new();
    let projector_vb = VarBuilder::from_varmap(&projector_varmap, config.dtype, &candle_device);
    let projector_config = ProjectorConfig {
        vit_d_model: encoder_config.vit_d_model,
        llm_d_model: model_config.hidden_dim,
        hidden_mult: config.vision.projector_hidden_mult,
    };
    let projector = VisionProjector::new(projector_config, projector_vb)?;
    let mut projector_varmap = projector_varmap;
    projector_varmap.load(&config.projector_path)?;

    let examples = load_vqa_jsonl(&config.data_path, config.vision.max_samples)?
        .into_iter()
        .map(MultimodalExample::Image)
        .collect();
    let metadata = AdapterMetadata::new_with_method(
        model_config,
        config.lora_config.clone(),
        Some(config.base_model_path.display().to_string()),
        config.qdora,
        AdapterMethod::Dora,
    );
    let vlm_metadata = VlmArtifactsMetadata {
        format_version: 1,
        projector_path: "projector.safetensors".into(),
        train_projector: config.train_projector,
        base_projector_path: config.projector_path.display().to_string(),
        clip_config_path: config.vision.clip_config_path.display().to_string(),
        clip_weights_path: config.vision.clip_weights_path.display().to_string(),
        image_root: config.vision.image_root.clone(),
        video: None,
        document: None,
    };
    let mut trainer = VlmDoraTrainer::new(
        model,
        dora_varmap,
        projector,
        projector_varmap,
        encoder,
        preprocess,
        tokenizer,
        config.train_config,
        config.output_dir,
        metadata,
        vlm_metadata,
        examples,
        config.shuffle,
        config.train_projector,
        None,
        None,
        candle_device,
    )?;
    trainer.train()
}

/// Build and run a video-language DoRA trainer using the shared VLM training loop.
pub fn run_video_vlm_dora_from_config(config: VideoVlmDoraRunConfig) -> Result<()> {
    let config = config.vlm;
    config.lora_config.validate()?;
    let video_config = config.vision.video.clone().ok_or_else(|| {
        AarambhError::Config("video VLM training requires a [vision.video] config block".into())
    })?;
    video_config.validate()?;
    let candle_device = config.device.to_candle()?;
    let tokenizer = BpeTokenizer::from_pretrained(&config.tokenizer_path)?;
    tokenizer.validate_video_special_tokens()?;
    let mut model_config = config.model_config.clone();
    model_config.vocab_size = tokenizer.vocab_size();
    if model_config.moe.is_some() {
        return Err(AarambhError::Config(
            "video VLM DoRA training currently requires a dense base model".into(),
        ));
    }
    let base = aarambh_studio_weights::load_any_model_with_dtype(
        &config.base_model_path,
        &model_config,
        &candle_device,
        config.dtype,
    )?;
    let base_tensors = base.named_tensors();
    drop(base);
    let (model, dora_varmap) = DoraAarambhModel::from_tensors(
        &model_config,
        &base_tensors,
        &config.lora_config,
        config.qdora,
        &candle_device,
    )?;
    eprintln!(
        "video VLM adapter params: {} / {} ({:.3}%)",
        model.adapter_param_count(),
        model.base_param_count(),
        model.trainable_ratio() * 100.0
    );

    let encoder_config = VisionEncoderConfig::from_json(&config.vision.clip_config_path)?;
    let encoder = ClipVisionEncoder::load_pretrained(
        &config.vision.clip_weights_path,
        encoder_config.clone(),
        &candle_device,
        config.dtype,
    )?;
    let preprocess = ImagePreprocessor::new(VisionPreprocessConfig {
        image_size: encoder_config.image_size,
        ..VisionPreprocessConfig::default()
    })?;
    let projector_varmap = VarMap::new();
    let projector_vb = VarBuilder::from_varmap(&projector_varmap, config.dtype, &candle_device);
    let projector = VisionProjector::new(
        ProjectorConfig {
            vit_d_model: encoder_config.vit_d_model,
            llm_d_model: model_config.hidden_dim,
            hidden_mult: config.vision.projector_hidden_mult,
        },
        projector_vb,
    )?;
    let mut projector_varmap = projector_varmap;
    projector_varmap.load(&config.projector_path)?;

    let temporal_varmap = VarMap::new();
    let temporal_vb = VarBuilder::from_varmap(&temporal_varmap, config.dtype, &candle_device);
    let temporal = TemporalEncoder::new(
        TemporalEncodingConfig {
            max_frames: video_config.max_frame_count,
            hidden_dim: encoder_config.vit_d_model,
            kind: video_config.temporal_encoding,
        },
        (video_config.temporal_encoding == TemporalEncodingKind::Learned).then_some(temporal_vb),
    )?;
    let mut temporal_varmap = temporal_varmap;
    if let Some(path) = &video_config.temporal_path {
        temporal_varmap.load(path)?;
    }

    let examples = load_video_qa(&config.data_path, config.vision.max_samples)?
        .into_iter()
        .map(MultimodalExample::Video)
        .collect();
    let metadata = AdapterMetadata::new_with_method(
        model_config,
        config.lora_config.clone(),
        Some(config.base_model_path.display().to_string()),
        config.qdora,
        AdapterMethod::Dora,
    );
    let temporal_path = (video_config.temporal_encoding == TemporalEncodingKind::Learned)
        .then(|| "temporal.safetensors".to_string());
    let vlm_metadata = VlmArtifactsMetadata {
        format_version: 2,
        projector_path: "projector.safetensors".into(),
        train_projector: config.train_projector,
        base_projector_path: config.projector_path.display().to_string(),
        clip_config_path: config.vision.clip_config_path.display().to_string(),
        clip_weights_path: config.vision.clip_weights_path.display().to_string(),
        image_root: config.vision.image_root.clone(),
        video: Some(VideoArtifactsMetadata {
            temporal_path,
            video_root: video_config.video_root.clone(),
            frame_count: video_config.frame_count,
            sampling: video_config.sampling,
            temporal_encoding: video_config.temporal_encoding,
        }),
        document: None,
    };
    let encoder_signature = format!(
        "{}:{}:{}:{}",
        config.vision.clip_weights_path.display(),
        encoder_config.image_size,
        encoder_config.patch_size,
        encoder_config.vit_d_model
    );
    let mut trainer = VlmDoraTrainer::new(
        model,
        dora_varmap,
        projector,
        projector_varmap,
        encoder,
        preprocess,
        tokenizer,
        config.train_config,
        config.output_dir,
        metadata,
        vlm_metadata,
        examples,
        config.shuffle,
        config.train_projector,
        Some(VideoRuntimeParts {
            config: video_config,
            temporal,
            temporal_varmap,
            encoder_signature,
        }),
        None,
        candle_device,
    )?;
    trainer.train()
}

/// Build and run document-language DoRA training using the shared VLM loop.
pub fn run_document_vlm_dora_from_config(config: DocumentVlmDoraRunConfig) -> Result<()> {
    let config = config.vlm;
    config.lora_config.validate()?;
    let document_config = config.vision.document.clone().ok_or_else(|| {
        AarambhError::Config(
            "document VLM training requires a [vision.document] config block".into(),
        )
    })?;
    document_config.validate()?;
    let candle_device = config.device.to_candle()?;
    let tokenizer = BpeTokenizer::from_pretrained(&config.tokenizer_path)?;
    tokenizer.validate_document_special_tokens()?;
    let mut model_config = config.model_config.clone();
    model_config.vocab_size = tokenizer.vocab_size();
    if model_config.moe.is_some() {
        return Err(AarambhError::Config(
            "document VLM DoRA training currently requires a dense base model".into(),
        ));
    }

    let base = aarambh_studio_weights::load_any_model_with_dtype(
        &config.base_model_path,
        &model_config,
        &candle_device,
        config.dtype,
    )?;
    let base_tensors = base.named_tensors();
    drop(base);
    let (model, dora_varmap) = DoraAarambhModel::from_tensors(
        &model_config,
        &base_tensors,
        &config.lora_config,
        config.qdora,
        &candle_device,
    )?;
    eprintln!(
        "document VLM adapter params: {} / {} ({:.3}%)",
        model.adapter_param_count(),
        model.base_param_count(),
        model.trainable_ratio() * 100.0
    );

    let encoder_config = VisionEncoderConfig::from_json(&config.vision.clip_config_path)?;
    let encoder = ClipVisionEncoder::load_pretrained(
        &config.vision.clip_weights_path,
        encoder_config.clone(),
        &candle_device,
        config.dtype,
    )?;
    let preprocess = ImagePreprocessor::new(VisionPreprocessConfig {
        image_size: encoder_config.image_size,
        ..VisionPreprocessConfig::default()
    })?;
    let projector_varmap = VarMap::new();
    let projector_vb = VarBuilder::from_varmap(&projector_varmap, config.dtype, &candle_device);
    let projector = VisionProjector::new(
        ProjectorConfig {
            vit_d_model: encoder_config.vit_d_model,
            llm_d_model: model_config.hidden_dim,
            hidden_mult: config.vision.projector_hidden_mult,
        },
        projector_vb,
    )?;
    let mut projector_varmap = projector_varmap;
    projector_varmap.load(&config.projector_path)?;

    let patch_side = encoder_config.image_size / encoder_config.patch_size;
    let layout_varmap = VarMap::new();
    let layout_vb = VarBuilder::from_varmap(&layout_varmap, config.dtype, &candle_device);
    let layout = LayoutAwareProjector::new(
        projector.clone(),
        LayoutProjectorConfig {
            patch_rows: patch_side,
            patch_cols: patch_side,
            hidden_dim: model_config.hidden_dim,
            encoding: document_config.layout_encoding,
        },
        (document_config.layout_encoding == LayoutEncodingKind::Learned).then_some(layout_vb),
    )?;
    let mut layout_varmap = layout_varmap;
    if let Some(path) = &document_config.layout_path {
        layout_varmap.load(path)?;
    }
    let rasterizer_config = PageRasterizerConfig {
        target_dpi: document_config.target_dpi,
        max_pages_per_document: document_config.max_pages_per_document,
        max_page_pixels: document_config.max_page_pixels,
    };
    let rasterizer = PageRasterizer::new(rasterizer_config)?;

    let examples = load_document_qa_jsonl(&config.data_path, config.vision.max_samples)?
        .into_iter()
        .map(MultimodalExample::Document)
        .collect();
    let metadata = AdapterMetadata::new_with_method(
        model_config,
        config.lora_config.clone(),
        Some(config.base_model_path.display().to_string()),
        config.qdora,
        AdapterMethod::Dora,
    );
    let layout_path = (document_config.layout_encoding == LayoutEncodingKind::Learned)
        .then(|| "layout.safetensors".to_string());
    let vlm_metadata = VlmArtifactsMetadata {
        format_version: 3,
        projector_path: "projector.safetensors".into(),
        train_projector: config.train_projector,
        base_projector_path: config.projector_path.display().to_string(),
        clip_config_path: config.vision.clip_config_path.display().to_string(),
        clip_weights_path: config.vision.clip_weights_path.display().to_string(),
        image_root: config.vision.image_root.clone(),
        video: None,
        document: Some(DocumentArtifactsMetadata {
            layout_path,
            document_root: document_config.document_root.clone(),
            target_dpi: document_config.target_dpi,
            max_pages_per_document: document_config.max_pages_per_document,
            layout_encoding: document_config.layout_encoding,
        }),
    };
    let encoder_signature = format!(
        "{}:{}:{}:{}:document-fit-pad",
        config.vision.clip_weights_path.display(),
        encoder_config.image_size,
        encoder_config.patch_size,
        encoder_config.vit_d_model
    );
    let mut trainer = VlmDoraTrainer::new(
        model,
        dora_varmap,
        projector,
        projector_varmap,
        encoder,
        preprocess,
        tokenizer,
        config.train_config,
        config.output_dir,
        metadata,
        vlm_metadata,
        examples,
        config.shuffle,
        config.train_projector,
        None,
        Some(DocumentRuntimeParts {
            config: document_config,
            layout,
            layout_varmap,
            rasterizer,
            encoder_signature,
        }),
        candle_device,
    )?;
    trainer.train()
}

/// Build and run an audio-language DoRA trainer (Phase 42).
///
/// Mirrors the two-stage frozen-encoder-plus-trainable-projector recipe v2 §25
/// established for vision, substituting audio for image. The frozen audio
/// spectrogram transformer encodes each clip into detached patch embeddings; a
/// trainable projector maps them into the decoder's hidden width; the result is
/// spliced into the token sequence at the `<audio>` placeholder. The DoRA-adapted
/// LLM and the projector train together on the audio-QA target.
pub fn run_audio_vlm_dora_from_config(config: AudioVlmDoraRunConfig) -> Result<()> {
    let config = config.vlm;
    config.lora_config.validate()?;
    let audio_config = config.vision.audio.clone().ok_or_else(|| {
        AarambhError::Config("audio VLM training requires a [vision.audio] config block".into())
    })?;
    audio_config.validate()?;
    let candle_device = config.device.to_candle()?;
    let tokenizer = BpeTokenizer::from_pretrained(&config.tokenizer_path)?;
    tokenizer.validate_audio_special_tokens()?;
    let mut model_config = config.model_config.clone();
    model_config.vocab_size = tokenizer.vocab_size();
    if model_config.moe.is_some() {
        return Err(AarambhError::Config(
            "audio VLM DoRA training currently requires a dense base model".into(),
        ));
    }

    let base = aarambh_studio_weights::load_any_model_with_dtype(
        &config.base_model_path,
        &model_config,
        &candle_device,
        config.dtype,
    )?;
    let base_tensors = base.named_tensors();
    drop(base);
    let (model, dora_varmap) = DoraAarambhModel::from_tensors(
        &model_config,
        &base_tensors,
        &config.lora_config,
        config.qdora,
        &candle_device,
    )?;
    eprintln!(
        "audio VLM adapter params: {} / {} ({:.3}%)",
        model.adapter_param_count(),
        model.base_param_count(),
        model.trainable_ratio() * 100.0
    );

    let encoder_config = AudioEncoderConfig::from_json(&audio_config.encoder_config_path)?;
    let encoder = FrozenAudioEncoder::load_pretrained(
        &audio_config.encoder_weights_path,
        encoder_config.clone(),
        &candle_device,
        config.dtype,
    )?;
    let preprocess = AudioPreprocessor::new(audio_config.mel.clone())?;
    let projector_varmap = VarMap::new();
    let projector_vb = VarBuilder::from_varmap(&projector_varmap, config.dtype, &candle_device);
    let projector = AudioProjector::new(
        AudioProjectorConfig {
            audio_d_model: encoder_config.audio_d_model,
            llm_d_model: model_config.hidden_dim,
            hidden_mult: config.vision.projector_hidden_mult,
        },
        projector_vb,
    )?;
    let mut projector_varmap = projector_varmap;
    projector_varmap.load(&config.projector_path)?;

    let examples = load_audio_qa_jsonl(&config.data_path, config.vision.max_samples)?;
    let metadata = AdapterMetadata::new_with_method(
        model_config.clone(),
        config.lora_config.clone(),
        Some(config.base_model_path.display().to_string()),
        config.qdora,
        AdapterMethod::Dora,
    );
    let mut dora_optimizer =
        AdamW::from_varmap(&dora_varmap, AdamWConfig::from(&config.train_config))?;
    if dora_optimizer.parameters().is_empty() {
        return Err(AarambhError::Config(
            "audio VLM DoRA target_modules produced zero trainable tensors".into(),
        ));
    }
    let mut projector_optimizer = if config.train_projector {
        Some(AdamW::from_varmap(
            &projector_varmap,
            AdamWConfig::from(&config.train_config),
        )?)
    } else {
        None
    };
    let schedule = CosineScheduleWithWarmup::new(
        config.train_config.lr,
        config.train_config.warmup_steps,
        config.train_config.max_steps,
        config.train_config.min_lr_ratio,
    );
    let mut rng = StdRng::seed_from_u64(config.train_config.seed);
    let mut examples = examples;
    if config.shuffle {
        examples.shuffle(&mut rng);
    }
    let output_dir = config.output_dir.clone();
    fs::create_dir_all(&output_dir)?;
    let mut dora_pending = GradMap::new();
    let mut projector_pending = GradMap::new();
    let mut step = 0usize;
    let mut example_idx = 0usize;
    let mut last_loss = 0.0f64;
    while step < config.train_config.max_steps {
        let example = &examples[example_idx % examples.len()];
        example_idx += 1;
        let loss = audio_example_loss(
            &model,
            &encoder,
            &preprocess,
            &projector,
            &tokenizer,
            &audio_config,
            example,
            &candle_device,
            model_config.max_seq_len,
        )?;
        last_loss = loss.to_scalar::<f32>()? as f64;
        let scaled = (loss / config.train_config.grad_accum_steps as f64)?;
        let grads = scaled.backward()?;
        accumulate_for_optimizer(&grads, &dora_optimizer, &mut dora_pending, "audio DoRA")?;
        if let Some(projector_optimizer) = projector_optimizer.as_ref() {
            accumulate_for_optimizer(
                &grads,
                projector_optimizer,
                &mut projector_pending,
                "audio projector",
            )?;
        }
        if example_idx.is_multiple_of(config.train_config.grad_accum_steps) {
            let lr = schedule.lr_at_step(step);
            clip_gradients(&mut dora_pending, config.train_config.clip_grad_norm)?;
            dora_optimizer.step(&dora_pending, lr)?;
            dora_pending.clear();
            if let Some(projector_optimizer) = projector_optimizer.as_mut() {
                clip_gradients(&mut projector_pending, config.train_config.clip_grad_norm)?;
                projector_optimizer.step(&projector_pending, lr)?;
                projector_pending.clear();
            }
            step += 1;
            if config.train_config.log_every_n_steps > 0
                && step.is_multiple_of(config.train_config.log_every_n_steps)
            {
                println!(
                    "audio_vlm_dora step={} loss={:.4} ppl={:.2} lr={:.6}",
                    step,
                    last_loss,
                    last_loss.exp(),
                    lr
                );
            }
        }
    }
    save_adapter(&dora_varmap, &metadata, &output_dir)?;
    projector_varmap.save(output_dir.join("projector.safetensors"))?;
    let vlm_metadata = AudioVlmArtifactsMetadata {
        format_version: 1,
        projector_path: "projector.safetensors".into(),
        train_projector: config.train_projector,
        base_projector_path: config.projector_path.display().to_string(),
        encoder_config_path: audio_config.encoder_config_path.display().to_string(),
        encoder_weights_path: audio_config.encoder_weights_path.display().to_string(),
        audio_root: audio_config.audio_root.display().to_string(),
        mel: audio_config.mel.clone(),
    };
    let metadata_json = serde_json::to_string_pretty(&vlm_metadata)?;
    fs::write(output_dir.join("audio_adapter_config.json"), metadata_json)?;
    eprintln!(
        "audio VLM DoRA training complete: {} steps, final loss {last_loss:.4}",
        config.train_config.max_steps
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn audio_example_loss(
    model: &DoraAarambhModel,
    encoder: &FrozenAudioEncoder,
    preprocess: &AudioPreprocessor,
    projector: &AudioProjector,
    tokenizer: &BpeTokenizer,
    audio_config: &AudioTrainingConfig,
    example: &AudioQaExample,
    device: &candle_core::Device,
    max_seq_len: usize,
) -> Result<Tensor> {
    let audio_path = resolve_media_path(&audio_config.audio_root, &example.audio_path);
    let spectrogram = preprocess
        .preprocess_path(&audio_path, device)?
        .unsqueeze(0)?;
    let patch_tokens = encoder.forward(&spectrogram)?.detach();
    let projected = projector.forward(&patch_tokens)?;
    let patch_count = projected.dims()[1];
    if patch_count == 0 || patch_count > max_seq_len {
        return Err(AarambhError::Shape(format!(
            "audio patch count {patch_count} is invalid for max_seq_len {max_seq_len}"
        )));
    }
    let template = ChatTemplate;
    let prefix = format!(
        "{AUDIO}{AUDIO_END}\n{}",
        template.prefix(&example.question, None)
    );
    let target = match example.thinking.as_deref() {
        Some(thinking) => template.thinking_target(thinking, &example.answer),
        None => template.target(&example.answer),
    };
    let prefix_ids = tokenizer.encode(&prefix)?;
    if !prefix_ids.contains(&AUDIO_ID) {
        return Err(AarambhError::Tokenizer(
            "audio prefix did not encode the <audio> placeholder token".into(),
        ));
    }
    let mut target_ids = tokenizer.encode(&target)?;
    if target_ids.is_empty() {
        return Err(AarambhError::Config(
            "audio target encoded to zero tokens".into(),
        ));
    }
    let max_text_tokens = max_seq_len + 1 - patch_count;
    if prefix_ids.len() >= max_text_tokens {
        return Err(AarambhError::Shape(format!(
            "audio prompt has {} text tokens plus {patch_count} media tokens, exceeding max_seq_len {max_seq_len}",
            prefix_ids.len()
        )));
    }
    let keep_target = max_text_tokens - prefix_ids.len();
    if target_ids.len() > keep_target {
        target_ids.truncate(keep_target);
    }
    let target_start_idx = prefix_ids.len();
    let text_tokens = {
        let mut combined = prefix_ids;
        combined.extend(target_ids);
        combined
    };
    if text_tokens.len() < 2 {
        return Err(AarambhError::Config(
            "audio sequence must contain at least two tokens".into(),
        ));
    }
    let text = Tensor::from_vec(text_tokens.clone(), (1, text_tokens.len()), device)?;
    let text_embeddings = model.embed_tokens(&text)?.detach();
    let fused = interleave_audio_tokens(&text_tokens, &text_embeddings, &projected, AUDIO_ID)?;
    let logits = model.forward_embeddings_train(&fused)?;
    let (labels, mask) = audio_labels_and_mask(&text_tokens, target_start_idx, patch_count)?;
    let seq_len = labels.len();
    let labels = Tensor::from_vec(labels, (1, seq_len), device)?;
    let mask = Tensor::from_vec(mask, (1, seq_len), device)?;
    cross_entropy_loss(&logits, &labels, &mask)
}

fn audio_labels_and_mask(
    text_tokens: &[u32],
    target_start_idx: usize,
    patch_count: usize,
) -> Result<(Vec<u32>, Vec<u32>)> {
    let mut items = Vec::new();
    let mut inserted = 0usize;
    for (idx, token) in text_tokens.iter().enumerate() {
        if *token == AUDIO_ID {
            items.extend(std::iter::repeat_n((None, idx), patch_count));
            inserted = 1;
        } else {
            items.push((Some(*token), idx));
        }
    }
    if inserted != 1 {
        return Err(AarambhError::Shape(format!(
            "expected one expanded audio item, found {inserted}"
        )));
    }
    let mut labels = Vec::with_capacity(items.len());
    let mut mask = Vec::with_capacity(items.len());
    for idx in 0..items.len() {
        match items.get(idx + 1).copied() {
            Some((Some(token), original_idx)) if original_idx >= target_start_idx => {
                labels.push(token);
                mask.push(1);
            }
            Some((Some(token), _)) => {
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
            "audio loss mask has no supervised answer tokens".into(),
        ));
    }
    Ok((labels, mask))
}

#[derive(Debug, Clone, Serialize)]
struct AudioVlmArtifactsMetadata {
    format_version: u32,
    projector_path: String,
    train_projector: bool,
    base_projector_path: String,
    encoder_config_path: String,
    encoder_weights_path: String,
    audio_root: String,
    mel: aarambh_studio_audio::MelSpectrogramConfig,
}

#[derive(Debug, Clone, Serialize)]
struct VlmArtifactsMetadata {
    format_version: u32,
    projector_path: String,
    train_projector: bool,
    base_projector_path: String,
    clip_config_path: String,
    clip_weights_path: String,
    image_root: PathBuf,
    video: Option<VideoArtifactsMetadata>,
    document: Option<DocumentArtifactsMetadata>,
}

#[derive(Debug, Clone, Serialize)]
struct VideoArtifactsMetadata {
    temporal_path: Option<String>,
    video_root: PathBuf,
    frame_count: usize,
    sampling: aarambh_studio_vision::FrameSamplingStrategy,
    temporal_encoding: TemporalEncodingKind,
}

#[derive(Debug, Clone, Serialize)]
struct DocumentArtifactsMetadata {
    layout_path: Option<String>,
    document_root: PathBuf,
    target_dpi: u32,
    max_pages_per_document: usize,
    layout_encoding: LayoutEncodingKind,
}

struct MediaArtifactVarMaps<'a> {
    temporal: Option<&'a VarMap>,
    layout: Option<&'a VarMap>,
}

fn accumulate_for_optimizer(
    grads: &GradStore,
    optimizer: &AdamW,
    pending: &mut GradMap,
    label: &str,
) -> Result<()> {
    let mut updates = Vec::new();
    for param in optimizer.parameters() {
        let Some(grad) = grads.get(param.tensor()) else {
            continue;
        };
        let grad = grad.detach();
        let next = match pending.get(param.name()) {
            Some(existing) => ((existing + &grad)?).detach(),
            None => grad,
        };
        updates.push((param.name().to_string(), next));
    }
    if updates.is_empty() {
        return Err(AarambhError::Config(format!(
            "{label} backward produced no trainable parameter gradients"
        )));
    }
    for (name, grad) in updates {
        pending.insert(name, grad);
    }
    Ok(())
}

fn save_vlm_artifacts(
    dora_varmap: &VarMap,
    projector_varmap: &VarMap,
    media_varmaps: MediaArtifactVarMaps<'_>,
    metadata: &AdapterMetadata,
    vlm_metadata: &VlmArtifactsMetadata,
    state: &TrainState,
    output_dir: impl AsRef<Path>,
) -> Result<()> {
    let output_dir = output_dir.as_ref();
    save_adapter(dora_varmap, metadata, output_dir)?;
    projector_varmap.save(output_dir.join("projector.safetensors"))?;
    if let Some(temporal_varmap) = media_varmaps.temporal
        && !temporal_varmap.all_vars().is_empty()
    {
        temporal_varmap.save(output_dir.join("temporal.safetensors"))?;
    }
    if let Some(layout_varmap) = media_varmaps.layout
        && !layout_varmap.all_vars().is_empty()
    {
        layout_varmap.save(output_dir.join("layout.safetensors"))?;
    }
    write_json(output_dir.join("vlm_config.json"), vlm_metadata)?;
    write_json(output_dir.join("train_state.json"), state)?;
    Ok(())
}

fn write_json(path: impl AsRef<Path>, value: &impl Serialize) -> Result<()> {
    let file = fs::File::create(path.as_ref())?;
    serde_json::to_writer_pretty(file, value).map_err(AarambhError::Json)?;
    Ok(())
}

fn resolve_media_path(root: &Path, media_path: &Path) -> PathBuf {
    if media_path.is_absolute() {
        media_path.to_path_buf()
    } else {
        root.join(media_path)
    }
}

fn resolve_document_source(root: &Path, source: &DocumentSource) -> DocumentSource {
    match source {
        DocumentSource::File(path) => DocumentSource::File(resolve_media_path(root, path)),
        DocumentSource::PageImages(paths) => DocumentSource::PageImages(
            paths
                .iter()
                .map(|path| resolve_media_path(root, path))
                .collect(),
        ),
    }
}

fn video_marker_prefix(frame_count: usize) -> String {
    let mut prefix = String::from(VIDEO);
    for _ in 1..frame_count {
        prefix.push_str(FRAME_SEP);
    }
    prefix.push_str(VIDEO_END);
    prefix
}

fn document_marker_prefix(page_count: usize) -> String {
    let mut prefix = String::from(DOCUMENT);
    for _ in 1..page_count {
        prefix.push_str(PAGE_SEP);
    }
    prefix.push_str(DOCUMENT_END);
    prefix
}

fn multimodal_labels_and_mask(
    text_tokens: &[u32],
    target_start_idx: usize,
    media: MediaLayout,
) -> Result<(Vec<u32>, Vec<u32>)> {
    let (placeholder_id, separator_id, media_count, patch_count) = match media {
        MediaLayout::Image { patch_count } => (IMAGE_ID, None, 1, patch_count),
        MediaLayout::Video {
            frame_count,
            patch_count,
        } => (VIDEO_ID, Some(FRAME_SEP_ID), frame_count, patch_count),
        MediaLayout::Document {
            page_count,
            patch_count,
        } => (DOCUMENT_ID, Some(PAGE_SEP_ID), page_count, patch_count),
    };
    let mut items = Vec::new();
    let mut inserted_media = 0usize;
    for (idx, token) in text_tokens.iter().enumerate() {
        if *token == placeholder_id {
            items.extend(std::iter::repeat_n((None, idx), patch_count));
            inserted_media = 1;
        } else {
            items.push((Some(*token), idx));
            if separator_id == Some(*token) {
                items.extend(std::iter::repeat_n((None, idx), patch_count));
                inserted_media += 1;
            }
        }
    }
    if inserted_media != media_count {
        return Err(AarambhError::Shape(format!(
            "expected {media_count} expanded media items, found {inserted_media}"
        )));
    }
    let mut labels = Vec::with_capacity(items.len());
    let mut mask = Vec::with_capacity(items.len());
    for idx in 0..items.len() {
        match items.get(idx + 1).copied() {
            Some((Some(token), original_idx)) if original_idx >= target_start_idx => {
                labels.push(token);
                mask.push(1);
            }
            Some((Some(token), _)) => {
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
            "VLM loss mask has no supervised answer tokens".into(),
        ));
    }
    Ok((labels, mask))
}

#[cfg(test)]
mod tests {
    use super::*;
    use aarambh_studio_tokenizer::{ENDOFTEXT_ID, IMAGE_END_ID};

    #[test]
    fn vlm_loss_mask_zeros_image_and_question_tokens() {
        let text = vec![IMAGE_ID, IMAGE_END_ID, 10, 11, 12, ENDOFTEXT_ID];
        let (labels, mask) =
            multimodal_labels_and_mask(&text, 4, MediaLayout::Image { patch_count: 3 }).unwrap();
        assert_eq!(labels.len(), 8);
        assert_eq!(mask, vec![0, 0, 0, 0, 0, 1, 1, 0]);
        assert_eq!(labels[5], 12);
        assert_eq!(labels[6], ENDOFTEXT_ID);
    }

    #[test]
    fn video_marker_prefix_has_one_separator_per_additional_frame() {
        assert_eq!(
            video_marker_prefix(3),
            format!("{VIDEO}{FRAME_SEP}{FRAME_SEP}{VIDEO_END}")
        );
    }

    #[test]
    fn video_loss_mask_includes_separator_but_not_frame_patches() {
        let text = vec![VIDEO_ID, FRAME_SEP_ID, 10, 20, 21, ENDOFTEXT_ID];
        let (_, mask) = multimodal_labels_and_mask(
            &text,
            4,
            MediaLayout::Video {
                frame_count: 2,
                patch_count: 2,
            },
        )
        .unwrap();
        assert_eq!(mask, vec![0, 0, 0, 0, 0, 0, 1, 1, 0]);
    }

    #[test]
    fn document_prefix_and_loss_mask_expand_pages() {
        assert_eq!(
            document_marker_prefix(3),
            format!("{DOCUMENT}{PAGE_SEP}{PAGE_SEP}{DOCUMENT_END}")
        );
        let text = vec![DOCUMENT_ID, PAGE_SEP_ID, 20, 21, ENDOFTEXT_ID];
        let (_, mask) = multimodal_labels_and_mask(
            &text,
            3,
            MediaLayout::Document {
                page_count: 2,
                patch_count: 2,
            },
        )
        .unwrap();
        assert_eq!(mask, vec![0, 0, 0, 0, 0, 1, 1, 0]);
    }
}
