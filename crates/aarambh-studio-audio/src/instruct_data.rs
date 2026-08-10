//! Audio instruction data loading.
//!
//! Mirrors `aarambh_studio_vision::instruct_data`: a normalized
//! `AudioQaExample` plus a JSONL loader that accepts both simple
//! `{audio, question, answer}` records and LLaVA-style
//! `{audio, conversations:[...]}` records. Captions are accepted as a
//! degenerate QA pair whose question is empty.

use std::fs;
use std::path::{Path, PathBuf};

use aarambh_studio_core::{AarambhError, Result};
use serde::Deserialize;

/// Normalized audio instruction example.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioQaExample {
    /// Audio path, relative to the configured audio root when not absolute.
    pub audio_path: PathBuf,
    /// User question or instruction. Empty for caption-only examples.
    pub question: String,
    /// Assistant target answer or caption.
    pub answer: String,
    /// Optional hidden thinking target placed before the final answer.
    pub thinking: Option<String>,
}

impl AudioQaExample {
    /// Build a normalized audio QA example.
    pub fn new(
        audio_path: impl Into<PathBuf>,
        question: impl Into<String>,
        answer: impl Into<String>,
        thinking: Option<String>,
    ) -> Result<Self> {
        let question = strip_audio_marker(&question.into());
        let answer = answer.into().trim().to_string();
        if answer.is_empty() {
            return Err(AarambhError::Config(
                "audio QA example answer must not be empty".into(),
            ));
        }
        Ok(Self {
            audio_path: audio_path.into(),
            question,
            answer,
            thinking,
        })
    }

    /// Return whether this example is a caption (no question).
    pub fn is_caption(&self) -> bool {
        self.question.is_empty()
    }
}

/// Load audio instruction examples from JSONL.
pub fn load_audio_qa_jsonl(
    path: impl AsRef<Path>,
    max_samples: Option<usize>,
) -> Result<Vec<AudioQaExample>> {
    let path = path.as_ref();
    let content = fs::read_to_string(path).map_err(|err| {
        AarambhError::Io(std::io::Error::new(
            err.kind(),
            format!("failed to read {}: {err}", path.display()),
        ))
    })?;
    let mut examples = Vec::new();
    for (line_idx, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let raw: RawAudioRecord = serde_json::from_str(line).map_err(|err| {
            AarambhError::Config(format!(
                "failed to parse {} line {}: {err}",
                path.display(),
                line_idx + 1
            ))
        })?;
        examples.push(raw.into_example().map_err(|err| {
            AarambhError::Config(format!(
                "invalid audio example in {} line {}: {err}",
                path.display(),
                line_idx + 1
            ))
        })?);
        if max_samples.is_some_and(|max| examples.len() >= max) {
            break;
        }
    }
    if examples.is_empty() {
        return Err(AarambhError::Config(format!(
            "{} contains no audio examples",
            path.display()
        )));
    }
    Ok(examples)
}

#[derive(Debug, Deserialize)]
struct RawAudioRecord {
    #[serde(default, alias = "audio")]
    audio_path: Option<PathBuf>,
    #[serde(default)]
    question: Option<String>,
    #[serde(default)]
    answer: Option<String>,
    #[serde(default)]
    caption: Option<String>,
    #[serde(default)]
    thinking: Option<String>,
    #[serde(default)]
    conversations: Vec<ConversationTurn>,
}

impl RawAudioRecord {
    fn into_example(self) -> Result<AudioQaExample> {
        let audio_path = self.audio_path.ok_or_else(|| {
            AarambhError::Config("audio example is missing audio or audio_path".into())
        })?;
        if let Some(answer) = self.answer {
            let question = self.question.unwrap_or_default();
            return AudioQaExample::new(audio_path, question, answer, self.thinking);
        }
        if let Some(caption) = self.caption {
            return AudioQaExample::new(audio_path, "", caption, self.thinking);
        }
        let question = self
            .conversations
            .iter()
            .find(|turn| turn.is_human())
            .map(|turn| turn.value.clone())
            .ok_or_else(|| AarambhError::Config("audio record has no human turn".into()))?;
        let answer = self
            .conversations
            .iter()
            .skip_while(|turn| !turn.is_human())
            .find(|turn| turn.is_assistant())
            .map(|turn| turn.value.clone())
            .ok_or_else(|| {
                AarambhError::Config("audio record has no assistant turn after human turn".into())
            })?;
        AudioQaExample::new(audio_path, question, answer, self.thinking)
    }
}

#[derive(Debug, Deserialize)]
struct ConversationTurn {
    #[serde(default, alias = "role")]
    from: String,
    value: String,
}

impl ConversationTurn {
    fn is_human(&self) -> bool {
        matches!(
            self.from.trim().to_ascii_lowercase().as_str(),
            "human" | "user"
        )
    }

    fn is_assistant(&self) -> bool {
        matches!(
            self.from.trim().to_ascii_lowercase().as_str(),
            "gpt" | "assistant"
        )
    }
}

fn strip_audio_marker(value: &str) -> String {
    value
        .replace("<audio>", "")
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_caption_record() {
        let raw: RawAudioRecord =
            serde_json::from_str(r#"{"audio":"tone.wav","caption":"a pure tone"}"#).unwrap();
        let example = raw.into_example().unwrap();
        assert_eq!(example.audio_path, PathBuf::from("tone.wav"));
        assert!(example.is_caption());
        assert_eq!(example.answer, "a pure tone");
    }

    #[test]
    fn parses_simple_qa_record() {
        let raw: RawAudioRecord = serde_json::from_str(
            r#"{"audio":"beep.wav","question":"What sound?","answer":"a beep","thinking":"listen"}"#,
        )
        .unwrap();
        let example = raw.into_example().unwrap();
        assert_eq!(example.question, "What sound?");
        assert_eq!(example.answer, "a beep");
        assert_eq!(example.thinking.as_deref(), Some("listen"));
    }

    #[test]
    fn parses_conversation_record() {
        let raw: RawAudioRecord = serde_json::from_str(
            r#"{"audio":"chime.wav","conversations":[{"from":"human","value":"<audio>\nDescribe?"},{"from":"gpt","value":"A chime."}]}"#,
        )
        .unwrap();
        let example = raw.into_example().unwrap();
        assert_eq!(example.question, "Describe?");
        assert_eq!(example.answer, "A chime.");
    }

    #[test]
    fn rejects_empty_answer() {
        let raw: RawAudioRecord =
            serde_json::from_str(r#"{"audio":"x.wav","answer":"   "}"#).unwrap();
        assert!(raw.into_example().is_err());
    }
}
