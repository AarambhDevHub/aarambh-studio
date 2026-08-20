//! Phase 54 model-card CLI runner.
//!
//! Invoked from `cmd/eval.rs` when `--generate-model-card` is set. Reads the
//! metadata (TOML/JSON) + scorecard JSON + red-team report JSON, assembles a
//! [`ModelCard`], and writes `MODEL_CARD.md` (+ companion `.json`).
//!
//! No model is required — the model card is assembled from artefacts a real
//! eval-harness run (v2 §17) and a real red-team pass (v4 §67) already
//! produced. This mirrors the Phase 53 `--redteam` flag's no-model
//! short-circuit.

use std::path::{Path, PathBuf};

use aarambh_studio_eval::ModelCard;
use aarambh_studio_tokenizer::CURRENT_CHAT_TEMPLATE_VERSION;

/// Default metadata path if `--model-card-metadata` is unset.
///
/// Mirrors the project's `configs/*.toml` convention.
pub const DEFAULT_MODEL_CARD_METADATA_PATH: &str = "configs/model_card_metadata.toml";
/// Default scorecard path if `--model-card-scorecard` is unset.
pub const DEFAULT_SCORECARD_PATH: &str = "artifacts/eval_scorecard.json";
/// Default red-team report path (reuses Phase 53's default).
pub const DEFAULT_REDTEAM_REPORT_PATH: &str = "artifacts/redteam_report.json";
/// Default output Markdown path (reuses the Phase 54 roadmap mandate:
/// `aarambh-studio eval --generate-model-card --output MODEL_CARD.md`).
pub const DEFAULT_MODEL_CARD_OUTPUT_PATH: &str = "MODEL_CARD.md";

/// Run the Phase 54 model-card assembly: read the three inputs, assemble the
/// card, and write the Markdown + JSON companion.
///
/// Fails loudly (non-zero exit) if any input is missing or the red-team
/// report is not clean — a checkpoint cannot get a model card without a
/// clean red-team pass (ARCHITECTURE_V4.md §68 invariant).
#[allow(clippy::missing_errors_doc)]
pub fn run(
    metadata_path: &Path,
    scorecard_path: &Path,
    redteam_report_path: &Path,
    output: &Path,
    chat_template_version: u32,
) -> anyhow::Result<()> {
    let card = ModelCard::assemble_from_paths(
        metadata_path,
        scorecard_path,
        redteam_report_path,
        chat_template_version,
    )
    .map_err(|err| anyhow::anyhow!("{err}"))?;
    card.write(output)?;
    let json_companion = output.with_extension("json");
    println!(
        "Model card written to {} (and {} companion)",
        output.display(),
        json_companion.display()
    );
    Ok(())
}

/// Resolve the four CLI path flags to their defaults if unset.
///
/// Centralised here so `cmd/eval.rs::run()` stays a thin dispatcher and the
/// default-resolution logic is testable in isolation.
pub fn resolve_paths(
    metadata: Option<&Path>,
    scorecard: Option<&Path>,
    redteam: Option<&Path>,
    output: Option<&Path>,
) -> (PathBuf, PathBuf, PathBuf, PathBuf) {
    (
        metadata
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_CARD_METADATA_PATH)),
        scorecard
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_SCORECARD_PATH)),
        redteam
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_REDTEAM_REPORT_PATH)),
        output
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_CARD_OUTPUT_PATH)),
    )
}

/// Re-export the current chat-template version so `cmd/eval.rs` can default
/// the `--model-card-chat-template-version` flag without depending on the
/// tokenizer crate directly for a single constant.
pub fn default_chat_template_version() -> u32 {
    CURRENT_CHAT_TEMPLATE_VERSION
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The four default paths match the documented CLI defaults.
    #[test]
    fn default_paths_match_documented_defaults() {
        let (meta, score, rt, out) = resolve_paths(None, None, None, None);
        assert_eq!(meta, PathBuf::from("configs/model_card_metadata.toml"));
        assert_eq!(score, PathBuf::from("artifacts/eval_scorecard.json"));
        assert_eq!(rt, PathBuf::from("artifacts/redteam_report.json"));
        assert_eq!(out, PathBuf::from("MODEL_CARD.md"));
    }

    /// Explicit paths override the defaults.
    #[test]
    fn explicit_paths_override_defaults() {
        let (meta, score, rt, out) = resolve_paths(
            Some(Path::new("/tmp/m.toml")),
            Some(Path::new("/tmp/s.json")),
            Some(Path::new("/tmp/r.json")),
            Some(Path::new("/tmp/CARD.md")),
        );
        assert_eq!(meta, PathBuf::from("/tmp/m.toml"));
        assert_eq!(score, PathBuf::from("/tmp/s.json"));
        assert_eq!(rt, PathBuf::from("/tmp/r.json"));
        assert_eq!(out, PathBuf::from("/tmp/CARD.md"));
    }

    /// The default chat-template version is the current v4 version (4).
    #[test]
    fn default_chat_template_version_is_current() {
        assert_eq!(default_chat_template_version(), 4);
    }

    /// End-to-end: assembling against a missing red-team report fails loudly
    /// with a clear error message (the §68 invariant, exercised through the
    /// CLI runner).
    #[test]
    fn run_fails_loudly_when_redteam_report_is_missing() {
        let tmp = std::env::temp_dir();
        let nonce = std::process::id();
        // Use a `.json` metadata file so this test does not require the `toml`
        // crate as a direct CLI dependency (serde_json is already a dep).
        let meta = tmp.join(format!("phase54_cli_meta_{nonce}.json"));
        let score = tmp.join(format!("phase54_cli_score_{nonce}.json"));
        let missing_rt = tmp.join(format!("phase54_cli_missing_{nonce}.json"));
        let out = tmp.join(format!("phase54_cli_out_{nonce}.md"));
        let metadata = aarambh_studio_eval::ModelCardMetadata {
            intended_use: "test".into(),
            training_data: vec![],
            known_limitations: vec![],
            hardware_requirements: "test".into(),
        };
        std::fs::write(&meta, serde_json::to_string(&metadata).unwrap()).unwrap();
        let scorecard = aarambh_studio_eval::Scorecard::new(
            vec![aarambh_studio_eval::TaskScore::accuracy("mmlu", 1, 2)],
            4,
            8,
            None,
            None,
            None,
        );
        std::fs::write(&score, scorecard.to_json().unwrap()).unwrap();
        let err = run(&meta, &score, &missing_rt, &out, 4).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("red-team report is unreadable") || msg.contains("red-team"),
            "expected red-team error, got: {msg}"
        );
        let _ = std::fs::remove_file(&meta);
        let _ = std::fs::remove_file(&score);
    }
}
