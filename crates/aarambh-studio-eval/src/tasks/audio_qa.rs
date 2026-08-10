//! Native audio question-answering evaluation task (Phase 42).
//!
//! Mirrors `video_qa::VideoQaTask`: loads audio instruction examples from JSONL,
//! encodes each clip with the frozen audio encoder, projects patch tokens into
//! the language model's hidden width, splices them into the prompt at the
//! `<audio>` placeholder, greedy-decodes, and scores the normalized answer.

use std::path::{Path, PathBuf};

use aarambh_studio_audio::{
    AudioEncoderConfig, AudioModel, AudioPreprocessor, AudioProjector, AudioProjectorConfig,
    FrozenAudioEncoder, interleave_audio_tokens, load_audio_qa_jsonl,
};
use aarambh_studio_core::{AarambhError, Result, TokenizerLike};
use aarambh_studio_tokenizer::{AUDIO, AUDIO_END, AUDIO_ID};
use aarambh_studio_train::TrainingRunConfig;
use candle_core::Tensor;

use crate::harness::{EvalConfig, EvalContext, EvalTask};
use crate::report::TaskScore;
use crate::tasks::first_existing;
use crate::tasks::vqa::greedy_generate_from_embeddings;

/// Audio question-answering evaluation task.
pub struct AudioQaTask;

impl EvalTask for AudioQaTask {
    fn name(&self) -> &'static str {
        "audio-qa"
    }

    fn run(&self, context: &EvalContext, config: &EvalConfig) -> Result<TaskScore> {
        let data_path = first_existing(&[
            config.data_dir.join("audio_qa").join("data.jsonl"),
            config.data_dir.join("audio_qa_smoke").join("data.jsonl"),
            config.data_dir.join("audio_qa.jsonl"),
        ])?;
        let examples = load_audio_qa_jsonl(&data_path, config.max_examples)?;
        let config_path = config
            .config_path
            .as_ref()
            .ok_or_else(|| AarambhError::Config("audio QA eval requires --config".into()))?;
        let run_config = TrainingRunConfig::from_toml(config_path)?;
        let audio_config = run_config
            .vision
            .as_ref()
            .and_then(|vision| vision.audio.as_ref())
            .ok_or_else(|| AarambhError::Config("audio QA eval requires [vision.audio]".into()))?;
        audio_config.validate()?;
        context.tokenizer().validate_audio_special_tokens()?;
        let runtime = load_audio_runtime(context, &run_config)?;
        let data_root = data_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| config.data_dir.clone());
        let mut passed = 0usize;
        for example in &examples {
            let path =
                resolve_audio_path(&audio_config.audio_root, &data_root, &example.audio_path);
            let prompt = audio_prompt(&format!("{}\n{}\n", example.question, example.answer));
            let prompt_ids = context.tokenizer().encode(&prompt)?;
            if prompt_ids.is_empty() {
                return Err(AarambhError::Config(
                    "audio QA prompt produced no tokens".into(),
                ));
            }
            let text =
                Tensor::from_vec(prompt_ids.clone(), (1, prompt_ids.len()), context.device())?;
            let text_embeddings = context.model().embed_tokens(&text)?;
            let spectrogram = runtime
                .preprocess
                .preprocess_path(&path, context.device())?
                .unsqueeze(0)?;
            let audio_tokens = runtime.model.forward(&spectrogram)?;
            let embeddings =
                interleave_audio_tokens(&prompt_ids, &text_embeddings, &audio_tokens, AUDIO_ID)?;
            let output =
                greedy_generate_from_embeddings(context, &embeddings, config.max_new_tokens)?;
            if normalize_answer(&output) == normalize_answer(&example.answer) {
                passed += 1;
            }
        }
        Ok(TaskScore::accuracy("audio-qa", passed, examples.len()))
    }
}

struct AudioRuntime {
    model: AudioModel,
    preprocess: AudioPreprocessor,
}

fn load_audio_runtime(
    context: &EvalContext,
    run_config: &TrainingRunConfig,
) -> Result<AudioRuntime> {
    let vision = run_config.vision.as_ref().ok_or_else(|| {
        AarambhError::Config("audio QA eval requires a [vision] config block".into())
    })?;
    let audio = vision.audio.as_ref().ok_or_else(|| {
        AarambhError::Config("audio QA eval requires a [vision.audio] block".into())
    })?;
    let encoder_config = AudioEncoderConfig::from_json(&audio.encoder_config_path)?;
    let encoder = FrozenAudioEncoder::load_pretrained(
        &audio.encoder_weights_path,
        encoder_config.clone(),
        context.device(),
        context.dtype(),
    )?;
    let projector_path = vision.projector_path.as_ref().ok_or_else(|| {
        AarambhError::Config("audio QA eval requires vision.projector_path".into())
    })?;
    let projector_config = AudioProjectorConfig {
        audio_d_model: encoder_config.audio_d_model,
        llm_d_model: run_config.model.hidden_dim,
        hidden_mult: vision.projector_hidden_mult,
    };
    let projector = AudioProjector::load_safetensors(
        projector_path,
        projector_config,
        context.device(),
        context.dtype(),
    )?;
    let preprocess = AudioPreprocessor::new(audio.mel.clone())?;
    Ok(AudioRuntime {
        model: AudioModel::new(encoder, projector),
        preprocess,
    })
}

fn audio_prompt(prompt: &str) -> String {
    if prompt.contains(AUDIO) {
        prompt.to_string()
    } else {
        format!("{AUDIO}{AUDIO_END}\n{prompt}")
    }
}

fn resolve_audio_path(config_root: &Path, data_root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }
    let configured = config_root.join(path);
    if configured.exists() {
        configured
    } else {
        data_root.join(path)
    }
}

fn normalize_answer(value: &str) -> String {
    value
        .trim()
        .to_ascii_lowercase()
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_prompt_inserts_placeholder_when_missing() {
        let prompt = audio_prompt("What sound is this?");
        assert!(prompt.contains(AUDIO));
        assert!(prompt.contains(AUDIO_END));
        assert!(prompt.contains("What sound is this?"));
    }

    #[test]
    fn audio_prompt_preserves_existing_placeholder() {
        let prompt = audio_prompt(&format!("{AUDIO} describe"));
        assert_eq!(prompt.matches(AUDIO).count(), 1);
    }

    #[test]
    fn answer_normalization_strips_punctuation_and_case() {
        assert_eq!(normalize_answer(" A.\n"), "a");
        assert_eq!(normalize_answer("Beep!"), "beep");
    }
}
