use std::path::PathBuf;

use aarambh_studio_core::{AarambhError, ModelConfig, Result};
use aarambh_studio_finetune::{SftExample, Verifier};
use aarambh_studio_inference::{GenerationConfig, GenerationOutput, GenerationStep};
use aarambh_studio_tokenizer::{CURRENT_CHAT_TEMPLATE_VERSION, validate_chat_template_version};
use candle_core::{DType, Device as CandleDevice};

use crate::config::SelfLearnConfig;
use crate::critique::{CritiqueResult, critique_response};
use crate::forgetting_hook::{ForgettingHook, SelfLearnForgettingSummary};
use crate::metrics::{LearningMetrics, MetricsEvent};
use crate::online_grpo::{OnlineGrpo, OnlineGrpoBuildConfig, OnlineUpdate};
use crate::replay::{ReplayBuffer, ReplayEntry};
use crate::vision_cache::VisionCache;

#[derive(Debug, Clone)]
/// Build configuration for [`SelfLearnLoop`].
pub struct SelfLearnBuildConfig {
    /// Model architecture configuration.
    pub model_config: ModelConfig,
    /// Path to the base model checkpoint.
    pub base_model_path: PathBuf,
    /// Path to the frozen reference checkpoint.
    pub reference_model_path: PathBuf,
    /// Path to tokenizer JSON.
    pub tokenizer_path: PathBuf,
    /// Self-learning configuration.
    pub config: SelfLearnConfig,
    /// Candle device.
    pub device: CandleDevice,
    /// Model dtype.
    pub dtype: DType,
    /// Random seed.
    pub seed: u64,
}

#[derive(Debug)]
/// Generated draft awaiting safety approval and commit.
pub struct SelfLearnDraft {
    /// Original prompt.
    pub prompt: String,
    /// Generated output.
    pub output: GenerationOutput,
    /// Pending online update.
    pub update: OnlineUpdate,
    /// Critique result for the output.
    pub critique: CritiqueResult,
    /// Optional cached projected image-token reference for a vision draft.
    pub image_ref: Option<PathBuf>,
}

#[derive(Debug, Clone)]
/// Result returned after committing a self-learning draft.
pub struct SelfLearnResponse {
    /// Final response text.
    pub response: String,
    /// Critique score.
    pub critique_score: f32,
    /// Optional verifier score.
    pub verifier_score: Option<f32>,
    /// Whether critique rewrote the response.
    pub was_rewritten: bool,
    /// Whether response was stored in replay.
    pub stored_in_replay: bool,
    /// Whether GRPO was used for the update.
    pub used_grpo: bool,
    /// Optional cached projected image-token reference stored with replay.
    pub image_ref: Option<PathBuf>,
    /// Human-readable metrics summary.
    pub metrics_summary: String,
    /// Latest Phase 38 forgetting diagnostics when enabled.
    pub forgetting: Option<SelfLearnForgettingSummary>,
}

/// Coordinates online GRPO, critique, replay, and metric state.
pub struct SelfLearnLoop {
    online_grpo: OnlineGrpo,
    replay: ReplayBuffer,
    metrics: LearningMetrics,
    config: SelfLearnConfig,
    last_draft: Option<SelfLearnDraft>,
    forgetting: Option<ForgettingHook>,
    last_forgetting: Option<SelfLearnForgettingSummary>,
}

impl SelfLearnLoop {
    /// Build a self-learning loop from checkpoint paths.
    pub fn from_paths(build: SelfLearnBuildConfig) -> Result<Self> {
        build.config.validate()?;
        let replay =
            ReplayBuffer::load_jsonl(&build.config.replay.path, build.config.replay.clone())?;
        let metrics = LearningMetrics::load_jsonl(build.config.state_dir.join("metrics.jsonl"))?;
        let online_grpo = OnlineGrpo::from_paths(OnlineGrpoBuildConfig {
            model_config: build.model_config,
            base_model_path: build.base_model_path,
            reference_model_path: build.reference_model_path,
            tokenizer_path: build.tokenizer_path,
            state_dir: build.config.state_dir.clone(),
            config: build.config.grpo.clone(),
            mode: build.config.mode,
            device: build.device,
            dtype: build.dtype,
            seed: build.seed,
        })?;
        // Phase 52 (SELF_LEARNING_V4.md §55): a self-learning session refuses
        // to start if the checkpoint's declared chat-template version does not
        // match what this build's replay-buffer schema expects. A session that
        // ran for hours against a mismatched template would produce replay
        // entries built on a misinterpreted prompt structure, silently
        // corrupting the buffer — fail loud at session start instead.
        validate_chat_template_version(
            online_grpo.tokenizer().chat_template_version(),
            CURRENT_CHAT_TEMPLATE_VERSION,
            &[],
        )
        .map_err(AarambhError::Config)?;
        let mut forgetting = build
            .config
            .forgetting
            .clone()
            .filter(|config| config.enabled)
            .map(|config| ForgettingHook::new(config, online_grpo.tokenizer().clone()))
            .transpose()?;
        let last_forgetting = forgetting
            .as_mut()
            .map(|hook| hook.baseline(&online_grpo))
            .transpose()?;
        Ok(Self {
            online_grpo,
            replay,
            metrics,
            config: build.config,
            last_draft: None,
            forgetting,
            last_forgetting,
        })
    }

    /// Return self-learning configuration.
    pub fn config(&self) -> &SelfLearnConfig {
        &self.config
    }

    /// Return replay buffer state.
    pub fn replay(&self) -> &ReplayBuffer {
        &self.replay
    }

    /// Return learning metrics.
    pub fn metrics(&self) -> &LearningMetrics {
        &self.metrics
    }

    /// Generate and store a draft without committing the update.
    pub fn generate_draft(
        &mut self,
        prompt: &str,
        config: GenerationConfig,
        verifier: Option<&dyn Verifier>,
        ground_truth: Option<&str>,
    ) -> Result<&SelfLearnDraft> {
        let update = self
            .online_grpo
            .generate_update(prompt, config, verifier, ground_truth)?;
        let base_response = update.output.text.clone();
        let critique = if self.config.critique.enabled {
            critique_response(
                &mut self.online_grpo,
                prompt,
                &base_response,
                &self.config.critique,
            )?
        } else {
            CritiqueResult {
                response: base_response,
                score: 0.5,
                reason: "critique disabled".into(),
                was_rewritten: false,
            }
        };
        let mut output = update.output.clone();
        if critique.was_rewritten {
            output.text = critique.response.clone();
            output.answer_text = critique.response.clone();
            output.raw_text = critique.response.clone();
        }
        self.last_draft = Some(SelfLearnDraft {
            prompt: prompt.to_string(),
            output,
            update,
            critique,
            image_ref: None,
        });
        Ok(self.last_draft.as_ref().expect("draft inserted above"))
    }

    /// Generate and store a vision draft without committing the update.
    pub fn generate_vision_draft(
        &mut self,
        prompt: &str,
        image_tokens: &candle_core::Tensor,
        image_ref: PathBuf,
        config: GenerationConfig,
        verifier: Option<&dyn Verifier>,
        ground_truth: Option<&str>,
    ) -> Result<&SelfLearnDraft> {
        let update = self.online_grpo.generate_vision_update(
            prompt,
            image_tokens,
            config,
            verifier,
            ground_truth,
        )?;
        let base_response = update.output.text.clone();
        let critique = if let Some(score) = update.verifier_score {
            CritiqueResult {
                response: base_response,
                score,
                reason: "grounded vision verifier".into(),
                was_rewritten: false,
            }
        } else if self.config.critique.enabled {
            critique_response(
                &mut self.online_grpo,
                prompt,
                &base_response,
                &self.config.critique,
            )?
        } else {
            CritiqueResult {
                response: base_response,
                score: 0.5,
                reason: "critique disabled".into(),
                was_rewritten: false,
            }
        };
        let mut output = update.output.clone();
        if critique.was_rewritten {
            output.text = critique.response.clone();
            output.answer_text = critique.response.clone();
            output.raw_text = critique.response.clone();
        }
        self.last_draft = Some(SelfLearnDraft {
            prompt: prompt.to_string(),
            output,
            update,
            critique,
            image_ref: Some(image_ref),
        });
        Ok(self.last_draft.as_ref().expect("draft inserted above"))
    }

    /// Generate a draft and replay stored generation steps through a callback.
    pub fn generate_draft_with_callback<F>(
        &mut self,
        prompt: &str,
        config: GenerationConfig,
        verifier: Option<&dyn Verifier>,
        ground_truth: Option<&str>,
        mut on_step: F,
    ) -> Result<GenerationOutput>
    where
        F: FnMut(&GenerationStep) -> Result<()>,
    {
        let output = self
            .generate_draft(prompt, config, verifier, ground_truth)?
            .output
            .clone();
        for step in &output.steps {
            on_step(step)?;
        }
        Ok(output)
    }

    /// Generate a vision draft and replay stored generation steps through a callback.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_vision_draft_with_callback<F>(
        &mut self,
        prompt: &str,
        image_tokens: &candle_core::Tensor,
        image_ref: PathBuf,
        config: GenerationConfig,
        verifier: Option<&dyn Verifier>,
        ground_truth: Option<&str>,
        mut on_step: F,
    ) -> Result<GenerationOutput>
    where
        F: FnMut(&GenerationStep) -> Result<()>,
    {
        let output = self
            .generate_vision_draft(
                prompt,
                image_tokens,
                image_ref,
                config,
                verifier,
                ground_truth,
            )?
            .output
            .clone();
        for step in &output.steps {
            on_step(step)?;
        }
        Ok(output)
    }

    /// Commit the last draft after optional safety-filtered replacement text.
    pub fn commit_last_draft(
        &mut self,
        safe_response: Option<String>,
    ) -> Result<SelfLearnResponse> {
        let mut draft = self
            .last_draft
            .take()
            .ok_or_else(|| AarambhError::Config("no self-learning draft to commit".into()))?;
        if let Some(response) = safe_response {
            draft.output.text = response.clone();
            draft.output.answer_text = response.clone();
            draft.output.raw_text = response;
        }
        let score = draft.critique.score;
        let mut stored = false;
        if score >= self.config.replay.min_score {
            let entry = ReplayEntry::new_with_image_ref(
                &draft.prompt,
                &draft.output.text,
                score,
                draft.image_ref.clone(),
            );
            stored = self.replay.push(entry.clone());
            if stored {
                ReplayBuffer::append_jsonl(&self.config.replay.path, &entry)?;
            }
        }
        let verifier_score = draft.update.verifier_score;
        let used_grpo = draft.update.used_grpo;
        let mutation = self.online_grpo.commit_update(draft.update)?;
        if mutation.is_some() {
            self.run_forgetting("online")?;
        }
        self.metrics.record(score, &draft.prompt);
        if self.replay.should_replay(self.online_grpo.step_count()) {
            let _ = self.replay_finetune()?;
        }
        let metrics_event = MetricsEvent {
            step: self.metrics.total_steps(),
            topic: crate::replay::infer_topic(&draft.prompt),
            score,
        };
        LearningMetrics::append_event(self.config.state_dir.join("metrics.jsonl"), &metrics_event)?;
        Ok(SelfLearnResponse {
            response: draft.output.text,
            critique_score: score,
            verifier_score,
            was_rewritten: draft.critique.was_rewritten,
            stored_in_replay: stored,
            used_grpo,
            image_ref: draft.image_ref,
            metrics_summary: self.metrics.summary(),
            forgetting: self.last_forgetting.clone(),
        })
    }

    /// Drop the last draft without applying learning state.
    pub fn discard_last_draft(&mut self) {
        self.last_draft = None;
    }

    /// Apply any pending CPU-mode gradients.
    pub fn flush_pending_gradients(&mut self) -> Result<Option<f64>> {
        let norm = self.online_grpo.flush_pending_gradients()?;
        if norm.is_some() {
            self.run_forgetting("flush")?;
        }
        Ok(norm)
    }

    /// Run replay fine-tuning from sampled high-quality responses.
    pub fn replay_finetune(&mut self) -> Result<Option<f64>> {
        let batch = self.replay.sample_batch(self.config.replay.batch_size);
        if batch.is_empty() {
            return Ok(None);
        }
        let path = self.config.state_dir.join("replay_sft.jsonl");
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = std::fs::File::create(&path)?;
        let cache = VisionCache::new(&self.config.state_dir);
        let mut text_examples = Vec::new();
        let mut vision_examples = Vec::new();
        for entry in &batch {
            let example = SftExample {
                instruction: entry.prompt.clone(),
                input: None,
                response: entry.response.clone(),
            };
            serde_json::to_writer(&mut file, &example)?;
            use std::io::Write;
            writeln!(file)?;
            if let Some(image_ref) = &entry.image_ref {
                if let Some(tokens) =
                    cache.load_projected_tokens(image_ref, self.online_grpo.device())?
                {
                    vision_examples.push((example, tokens));
                }
            } else {
                text_examples.push(example);
            }
        }
        let mut norms = Vec::new();
        if let Some(norm) = self
            .online_grpo
            .replay_sft_batch(&text_examples, self.config.replay.batch_size)?
        {
            norms.push(norm);
        }
        if let Some(norm) = self
            .online_grpo
            .replay_vision_sft_batch(&vision_examples, self.config.replay.batch_size)?
        {
            norms.push(norm);
        }
        self.metrics.record_replay();
        if norms.is_empty() {
            Ok(None)
        } else {
            let norm = norms.iter().sum::<f64>() / norms.len() as f64;
            self.run_forgetting("replay")?;
            Ok(Some(norm))
        }
    }

    /// Return the latest forgetting summary when diagnostics are enabled.
    pub fn last_forgetting(&self) -> Option<&SelfLearnForgettingSummary> {
        self.last_forgetting.as_ref()
    }

    /// Return the configured forgetting store path.
    pub fn forgetting_store_path(&self) -> Option<&std::path::Path> {
        self.forgetting.as_ref().map(ForgettingHook::store_path)
    }

    fn run_forgetting(&mut self, update_kind: &str) -> Result<()> {
        let Some(mut hook) = self.forgetting.take() else {
            return Ok(());
        };
        let result = hook.after_update(&self.online_grpo, update_kind);
        self.forgetting = Some(hook);
        let summary = result?;
        let threshold = self
            .forgetting
            .as_ref()
            .map(ForgettingHook::threshold)
            .unwrap_or(0.02);
        println!(
            "[forgetting] current={} forgotten={} skipped={}",
            summary.current_id,
            summary.forgotten_count(threshold),
            summary.skipped.len()
        );
        self.last_forgetting = Some(summary);
        Ok(())
    }
}
