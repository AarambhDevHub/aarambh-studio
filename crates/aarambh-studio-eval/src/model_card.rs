//! Phase 54 model-card assembly — one canonical, assembled document per
//! released checkpoint configuration.
//!
//! Generated from an eval-harness scorecard (v2 §17) plus the red-team
//! report (v4 §67) plus static metadata (dataset list, license, hardware
//! requirements). Capabilities and red-team sections are **pulled** from
//! real runs, never hand-entered — so a model card cannot silently drift
//! out of sync with a checkpoint's actual measured behavior.
//!
//! See [`ARCHITECTURE_V4.md` §68](../../ARCHITECTURE_V4.md) for the design
//! spec and [`docs/phase54_model_card.md`](../../docs/phase54_model_card.md)
//! for the runbook.

use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use aarambh_studio_core::{AarambhError, Result};
use aarambh_studio_safety::RedTeamReport;
use serde::{Deserialize, Serialize};

use crate::report::Scorecard;

/// Model-card schema version. Bumped only on breaking card-shape changes.
pub const MODEL_CARD_SCHEMA_VERSION: u32 = 1;

/// One training-data entry, license-tagged (ARCHITECTURE_V4.md §68).
///
/// Each entry records the dataset name, an optional public source URL, the
/// license (SPDX-style identifier), the approximate number of examples in
/// the split used, and the split name. The license tag is mandatory so a
/// downstream reader can audit provenance without re-deriving it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetEntry {
    /// Dataset name, e.g. `"wikitext-103"`.
    pub name: String,
    /// Optional public source URL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_url: Option<String>,
    /// License identifier (SPDX-style), e.g. `"CC-BY-3.0"`, `"MIT"`.
    pub license: String,
    /// Approximate number of examples in the split used.
    pub size_examples: usize,
    /// Split used, e.g. `"train"`, `"validation"`.
    pub split: String,
}

/// The static, hand-authored portion of a model card (ARCHITECTURE_V4.md §68).
///
/// Loaded once per release from a TOML or JSON file; everything else in a
/// [`ModelCard`] is pulled from real eval-harness and red-team runs. Keeping
/// the static portion in a dedicated struct makes the "authored once" fields
/// auditable separately from the "pulled from a real run" fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelCardMetadata {
    /// Intended-use statement.
    pub intended_use: String,
    /// Training-data provenance, license-tagged per entry.
    pub training_data: Vec<DatasetEntry>,
    /// Known limitations (static-authored; eval-derived ones may be appended
    /// at render time, but the authored list is the canonical baseline).
    pub known_limitations: Vec<String>,
    /// Hardware requirements summary.
    pub hardware_requirements: String,
}

impl ModelCardMetadata {
    /// Load metadata from a TOML file.
    ///
    /// TOML is the canonical format for hand-authored release metadata in this
    /// project (see `configs/*.toml`); the model-card metadata file mirrors
    /// that convention.
    pub fn from_toml_path(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)
            .map_err(|e| AarambhError::Config(format!("read metadata {}: {e}", path.display())))?;
        toml::from_str(&text).map_err(|e| {
            AarambhError::Config(format!("parse metadata TOML {}: {e}", path.display()))
        })
    }

    /// Load metadata from a JSON file.
    ///
    /// JSON is supported so a release pipeline can emit the metadata
    /// programmatically rather than authoring TOML by hand.
    pub fn from_json_path(path: &Path) -> Result<Self> {
        let file = fs::File::open(path)
            .map_err(|e| AarambhError::Config(format!("read metadata {}: {e}", path.display())))?;
        serde_json::from_reader(file).map_err(|e| {
            AarambhError::Config(format!("parse metadata JSON {}: {e}", path.display()))
        })
    }

    /// Load metadata from a path, auto-detecting TOML vs JSON by extension.
    ///
    /// `.toml` → TOML; `.json` → JSON; anything else → TOML (the canonical
    /// default). This lets the CLI accept `--model-card-metadata path` without
    /// a separate format flag.
    pub fn from_path(path: &Path) -> Result<Self> {
        match path.extension().and_then(|ext| ext.to_str()) {
            Some("json") => Self::from_json_path(path),
            _ => Self::from_toml_path(path),
        }
    }
}

/// Reasons model-card assembly fails loudly (ARCHITECTURE_V4.md §68).
///
/// The §68 invariant is: *"generation fails loudly if no red-team report is
/// present for the checkpoint being documented, rather than shipping a model
/// card with an empty or stale safety section."* Each variant names one
/// concrete loud-failure mode so the error message is actionable.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelCardError {
    /// No red-team report path was supplied (the caller passed `None`).
    MissingRedTeamReport,
    /// The red-team report file does not exist or is unreadable.
    RedTeamReportUnreadable(String),
    /// The red-team report exists but `is_clean()` is false — the checkpoint
    /// has a known, unaddressed red-team failure and must not be documented.
    RedTeamReportNotClean {
        /// Number of cases whose observed outcome did not match the expected.
        failed: usize,
        /// Total number of cases in the corpus.
        corpus_size: usize,
    },
    /// No scorecard was supplied (path or in-memory).
    MissingScorecard,
    /// The scorecard file does not exist or is unreadable.
    ScorecardUnreadable(String),
    /// The static metadata file does not exist or is unreadable.
    MetadataUnreadable(String),
}

impl std::fmt::Display for ModelCardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingRedTeamReport => write!(
                f,
                "model-card generation requires a red-team report (v4 §67); \
                 none was supplied. A checkpoint cannot get a model card \
                 without a clean red-team pass."
            ),
            Self::RedTeamReportUnreadable(path) => write!(
                f,
                "red-team report is unreadable at {path}; cannot assemble a \
                 model card with an empty or stale safety section."
            ),
            Self::RedTeamReportNotClean {
                failed,
                corpus_size,
            } => write!(
                f,
                "red-team report is not clean: {failed} of {corpus_size} cases \
                 did not match their labelled expected outcome. A checkpoint \
                 with a known, unaddressed red-team failure must not be \
                 documented in a model card."
            ),
            Self::MissingScorecard => write!(
                f,
                "model-card generation requires an eval-harness scorecard \
                 (v2 §17); none was supplied. Capabilities must be PULLED \
                 from an actual eval run, never hand-entered."
            ),
            Self::ScorecardUnreadable(path) => write!(
                f,
                "scorecard is unreadable at {path}; cannot assemble a model \
                 card without real eval-harness numbers."
            ),
            Self::MetadataUnreadable(path) => write!(
                f,
                "static metadata is unreadable at {path}; cannot assemble a \
                 model card without the hand-authored intended-use, \
                 training-data, and hardware-requirements fields."
            ),
        }
    }
}

impl std::error::Error for ModelCardError {}

/// The complete, assembled model card (ARCHITECTURE_V4.md §68).
///
/// Seven fields, in spec order. Three are **pulled** from real runs
/// (`capabilities`, `redteam_summary`, `chat_template_version`) and four are
/// **static** authored metadata (`intended_use`, `training_data`,
/// `known_limitations`, `hardware_requirements`). The pulled fields guarantee
/// the card cannot drift out of sync with a checkpoint's actual measured
/// behavior — a card assembled today from a stale scorecard would carry the
/// stale scorecard's timestamp, making the drift visible.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelCard {
    /// Schema version (`MODEL_CARD_SCHEMA_VERSION`).
    pub schema_version: u32,
    /// Generated-at Unix milliseconds.
    pub generated_at_unix_ms: u128,
    /// Static: intended use.
    pub intended_use: String,
    /// Static: training-data provenance, license-tagged.
    pub training_data: Vec<DatasetEntry>,
    /// PULLED from an actual eval-harness run (v2 §17), never hand-entered.
    pub capabilities: Scorecard,
    /// Static + eval-derived.
    pub known_limitations: Vec<String>,
    /// PULLED from v4 §67's actual red-team report, never hand-entered.
    pub redteam_summary: RedTeamReport,
    /// Static: hardware requirements.
    pub hardware_requirements: String,
    /// PULLED from v4 §66's chat-template version tag.
    pub chat_template_version: u32,
}

impl ModelCard {
    /// Assemble a model card from a scorecard, a red-team report, static
    /// metadata, and the chat-template version.
    ///
    /// Fails loudly (returns `Err`) if `redteam_report.is_clean()` is false —
    /// a checkpoint cannot get a model card without a clean red-team pass
    /// (ARCHITECTURE_V4.md §68 invariant; promised by `docs/phase53_redteam.md`
    /// lines 205–208).
    ///
    /// This is the in-memory entry point: both the scorecard and the red-team
    /// report are already deserialized. Use [`ModelCard::assemble_from_paths`]
    /// for the file-path entry point used by the CLI.
    pub fn assemble(
        metadata: ModelCardMetadata,
        scorecard: Scorecard,
        redteam_report: RedTeamReport,
        chat_template_version: u32,
    ) -> std::result::Result<Self, ModelCardError> {
        if !redteam_report.is_clean() {
            return Err(ModelCardError::RedTeamReportNotClean {
                failed: redteam_report.failed,
                corpus_size: redteam_report.corpus_size,
            });
        }
        Ok(Self {
            schema_version: MODEL_CARD_SCHEMA_VERSION,
            generated_at_unix_ms: now_unix_ms(),
            intended_use: metadata.intended_use,
            training_data: metadata.training_data,
            capabilities: scorecard,
            known_limitations: metadata.known_limitations,
            redteam_summary: redteam_report,
            hardware_requirements: metadata.hardware_requirements,
            chat_template_version,
        })
    }

    /// Assemble from file paths: metadata (TOML/JSON) + scorecard JSON +
    /// red-team report JSON.
    ///
    /// Fails loudly if any file is missing or the red-team report is not
    /// clean. This is the entry point the CLI `eval --generate-model-card`
    /// flag calls.
    #[allow(clippy::missing_errors_doc)]
    pub fn assemble_from_paths(
        metadata_path: &Path,
        scorecard_path: &Path,
        redteam_report_path: &Path,
        chat_template_version: u32,
    ) -> std::result::Result<Self, ModelCardError> {
        let metadata = ModelCardMetadata::from_path(metadata_path).map_err(|e| {
            ModelCardError::MetadataUnreadable(format!("{}: {e}", metadata_path.display()))
        })?;
        let scorecard = read_scorecard(scorecard_path).map_err(|e| {
            ModelCardError::ScorecardUnreadable(format!("{}: {e}", scorecard_path.display()))
        })?;
        let redteam_report = read_redteam_report(redteam_report_path).map_err(|e| {
            ModelCardError::RedTeamReportUnreadable(format!(
                "{}: {e}",
                redteam_report_path.display()
            ))
        })?;
        Self::assemble(metadata, scorecard, redteam_report, chat_template_version)
    }

    /// Serialize to pretty JSON (the machine-readable companion to the MD).
    ///
    /// The JSON carries the full `capabilities` scorecard and `redteam_summary`
    /// report inline, so a downstream reader can validate the card against the
    /// eval-harness and red-team outputs without re-running either.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(AarambhError::Json)
    }

    /// Render as the canonical `MODEL_CARD.md` Markdown — the seven §68
    /// sections, in spec order, with the capabilities and red-team sections
    /// pulled verbatim from the real `Scorecard`/`RedTeamReport` Markdown
    /// renderers (never re-rendered by hand).
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("# Model Card\n\n");
        out.push_str(&format!(
            "- schema version: `{}`\n- generated at (unix ms): `{}`\n- chat-template version: `{}`\n\n",
            self.schema_version, self.generated_at_unix_ms, self.chat_template_version,
        ));

        out.push_str("## Intended Use\n\n");
        out.push_str(&self.intended_use);
        out.push_str("\n\n");

        out.push_str("## Training Data & Licensing\n\n");
        out.push_str("| Dataset | Source | License | Examples | Split |\n");
        out.push_str("|---|---|---|---:|---|\n");
        for entry in &self.training_data {
            out.push_str(&format!(
                "| {} | {} | {} | {} | {} |\n",
                entry.name,
                entry.source_url.as_deref().unwrap_or("—"),
                entry.license,
                entry.size_examples,
                entry.split,
            ));
        }
        out.push('\n');

        out.push_str("## Capabilities\n\n");
        out.push_str(
            "Pulled directly from the eval-harness scorecard (v2 §17) — never \
             hand-entered.\n\n",
        );
        out.push_str(&self.capabilities.to_markdown());
        out.push_str("\n\n");

        out.push_str("## Known Limitations\n\n");
        if self.known_limitations.is_empty() {
            out.push_str("None recorded.\n");
        } else {
            for limit in &self.known_limitations {
                out.push_str(&format!("- {limit}\n"));
            }
        }
        out.push('\n');

        out.push_str("## Red-Team Summary\n\n");
        out.push_str(
            "Pulled directly from the Phase 53 red-team report (v4 §67) — never \
             hand-entered. Generation fails loudly if the report is not clean.\n\n",
        );
        out.push_str(&self.redteam_summary.to_markdown());
        out.push_str("\n\n");

        out.push_str("## Hardware Requirements\n\n");
        out.push_str(&self.hardware_requirements);
        out.push_str("\n\n");

        out.push_str("## Version & Chat-Template Compatibility\n\n");
        out.push_str(&format!(
            "Chat-template version: `{}` (v4 §66). The checkpoint's declared \
             template version must match (or be explicitly declared compatible \
             with) this version, or the server refuses to load it to avoid \
             silently misinterpreting prompt structure.\n",
            self.chat_template_version
        ));
        out.push_str("\n| Version | Template shape |\n");
        out.push_str("|---|---|\n");
        out.push_str("| `1` | v1.0.0 base `<imas>`/`</imas>` chat format |\n");
        out.push_str("| `2` | v2.0.0 + image tokens |\n");
        out.push_str("| `3` | v3.0.0 + video / document / tool tokens |\n");
        out.push_str("| `4` | v4.0.0 + system role formalized + audio tokens (current) |\n");

        out
    }

    /// Write both the Markdown card and its JSON companion.
    ///
    /// Given `output = "MODEL_CARD.md"`, writes `MODEL_CARD.md` and
    /// `MODEL_CARD.json` side-by-side so the machine-readable companion is
    /// always next to the human-readable card.
    pub fn write(&self, markdown_path: &Path) -> Result<()> {
        if let Some(parent) = markdown_path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        fs::write(markdown_path, self.to_markdown())?;
        let json_path = markdown_path.with_extension("json");
        fs::write(&json_path, self.to_json()?)?;
        Ok(())
    }
}

/// Read a [`Scorecard`] from a JSON file.
fn read_scorecard(path: &Path) -> Result<Scorecard> {
    let file = fs::File::open(path)?;
    serde_json::from_reader(file).map_err(AarambhError::Json)
}

/// Read a [`RedTeamReport`] from a JSON file.
fn read_redteam_report(path: &Path) -> Result<RedTeamReport> {
    let file = fs::File::open(path)?;
    serde_json::from_reader(file).map_err(AarambhError::Json)
}

/// Current Unix time in milliseconds.
fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::{Scorecard, TaskScore};
    use aarambh_studio_safety::{
        AdversarialCase, AdversarialInput, CaseOutcome, Corpus, ExpectedOutcome, ObservedOutcome,
        RedTeamReport, RedTeamSurface,
    };

    /// Build a minimal clean red-team report from the v4 corpus with every
    /// case matching its labelled expected outcome.
    fn clean_report() -> RedTeamReport {
        let corpus = Corpus::v4();
        let outcomes: Vec<CaseOutcome> = corpus
            .iter()
            .map(|c| {
                // Match the expected outcome exactly — a clean pass.
                let observed = match c.expected_outcome {
                    ExpectedOutcome::Refused => ObservedOutcome::Refused { reason: None },
                    ExpectedOutcome::Sanitized => ObservedOutcome::Sanitized { reason: None },
                    ExpectedOutcome::ExecutedSafely => ObservedOutcome::ExecutedSafely,
                };
                CaseOutcome::new(c, observed)
            })
            .collect();
        RedTeamReport::from_outcomes(outcomes)
    }

    /// Build a failing red-team report: every case observed as `Other`.
    fn failing_report() -> RedTeamReport {
        let corpus = Corpus::v4();
        let outcomes: Vec<CaseOutcome> = corpus
            .iter()
            .map(|c| {
                CaseOutcome::new(
                    c,
                    ObservedOutcome::Other {
                        label: "wrong".into(),
                    },
                )
            })
            .collect();
        RedTeamReport::from_outcomes(outcomes)
    }

    /// Build a minimal scorecard with one accuracy task.
    fn sample_scorecard() -> Scorecard {
        Scorecard::new(
            vec![TaskScore::accuracy("mmlu", 7, 10)],
            2048,
            128,
            Some("checkpoints/v4/model.safetensors".into()),
            Some("checkpoints/v4/tokenizer.json".into()),
            Some("configs/train.toml".into()),
        )
    }

    /// Build minimal static metadata.
    fn sample_metadata() -> ModelCardMetadata {
        ModelCardMetadata {
            intended_use: "Research checkpoint for instruction-following experiments.".into(),
            training_data: vec![DatasetEntry {
                name: "wikitext-103".into(),
                source_url: Some("https://huggingface.co/datasets/wikitext".into()),
                license: "CC-BY-3.0".into(),
                size_examples: 1801350,
                split: "train".into(),
            }],
            known_limitations: vec![
                "Not fine-tuned for safety; rely on the safety layer.".into(),
                "Context window is bounded.".into(),
            ],
            hardware_requirements: "CPU: 16 GB RAM (q4_k_m). GPU: 1× consumer GPU (bf16).".into(),
        }
    }

    // ---- The two ROADMAP-named acceptance tests ----

    /// ROADMAP_V4.md Phase 54 test #1 (verbatim name).
    ///
    /// The `capabilities` field of an assembled model card must be the exact
    /// scorecard produced by an eval-harness run — byte-for-byte equal, never
    /// re-derived or hand-entered. This is the "cannot silently drift out of
    /// sync with actual eval numbers" guarantee.
    #[test]
    fn model_card_eval_scores_match_the_actual_eval_harness_run_exactly() {
        let scorecard = sample_scorecard();
        let card = ModelCard::assemble(sample_metadata(), scorecard.clone(), clean_report(), 4)
            .expect("clean report assembles");
        assert_eq!(card.capabilities, scorecard);
        // The capabilities section in the rendered Markdown must contain the
        // exact scorecard Markdown — pulled verbatim, not re-rendered.
        let md = card.to_markdown();
        assert!(md.contains(&scorecard.to_markdown()));
        assert!(md.contains("mmlu"));
        assert!(md.contains("0.7000"));
    }

    /// ROADMAP_V4.md Phase 54 test #2 (verbatim name).
    ///
    /// Generation fails loudly if no red-team report is present. The
    /// `assemble_from_paths` entry point returns `MissingRedTeamReport` is
    /// not directly callable without a path, so this test exercises the
    /// in-memory `assemble` path's `RedTeamReportNotClean` failure (the
    /// "present but not clean" half) AND the CLI-level missing-file half
    /// (via `assemble_from_paths` against a non-existent path).
    #[test]
    fn model_card_generation_fails_loudly_if_no_redteam_report_is_present() {
        // Half 1: a present-but-not-clean report fails loudly.
        let report = failing_report();
        assert!(!report.is_clean());
        let err = ModelCard::assemble(sample_metadata(), sample_scorecard(), report, 4)
            .expect_err("not-clean report must fail");
        assert_eq!(
            err,
            ModelCardError::RedTeamReportNotClean {
                failed: 24,
                corpus_size: 24
            }
        );

        // Half 2: a missing red-team report file fails loudly.
        let tmp = std::env::temp_dir();
        let nonce = std::process::id();
        let metadata_path = tmp.join(format!("phase54_meta_{nonce}.toml"));
        let scorecard_path = tmp.join(format!("phase54_score_{nonce}.json"));
        let missing_redteam = tmp.join(format!("phase54_does_not_exist_{nonce}.json"));
        std::fs::write(&metadata_path, toml::to_string(&sample_metadata()).unwrap()).unwrap();
        std::fs::write(&scorecard_path, sample_scorecard().to_json().unwrap()).unwrap();
        let err =
            ModelCard::assemble_from_paths(&metadata_path, &scorecard_path, &missing_redteam, 4)
                .expect_err("missing red-team file must fail");
        assert!(matches!(err, ModelCardError::RedTeamReportUnreadable(_)));
        let _ = std::fs::remove_file(&metadata_path);
        let _ = std::fs::remove_file(&scorecard_path);
    }

    // ---- Supporting corollary tests ----

    /// A not-clean report fails loudly with the exact failure counts.
    #[test]
    fn model_card_generation_fails_loudly_if_redteam_report_is_not_clean() {
        let report = failing_report();
        let err = ModelCard::assemble(sample_metadata(), sample_scorecard(), report, 4)
            .expect_err("not-clean report must fail");
        match err {
            ModelCardError::RedTeamReportNotClean {
                failed,
                corpus_size,
            } => {
                assert_eq!(failed, 24);
                assert_eq!(corpus_size, 24);
            }
            other => panic!("expected RedTeamReportNotClean, got {other:?}"),
        }
    }

    /// A missing scorecard file fails loudly.
    #[test]
    fn model_card_generation_fails_loudly_if_scorecard_file_is_missing() {
        let tmp = std::env::temp_dir();
        let nonce = std::process::id();
        let metadata_path = tmp.join(format!("phase54_meta2_{nonce}.toml"));
        let missing_scorecard = tmp.join(format!("phase54_no_score_{nonce}.json"));
        let redteam_path = tmp.join(format!("phase54_rt_{nonce}.json"));
        std::fs::write(&metadata_path, toml::to_string(&sample_metadata()).unwrap()).unwrap();
        std::fs::write(&redteam_path, clean_report().to_json().unwrap()).unwrap();
        let err =
            ModelCard::assemble_from_paths(&metadata_path, &missing_scorecard, &redteam_path, 4)
                .expect_err("missing scorecard must fail");
        assert!(matches!(err, ModelCardError::ScorecardUnreadable(_)));
        let _ = std::fs::remove_file(&metadata_path);
        let _ = std::fs::remove_file(&redteam_path);
    }

    /// The rendered Markdown contains all seven §68 sections, in order.
    #[test]
    fn model_card_markdown_contains_all_seven_sections() {
        let card =
            ModelCard::assemble(sample_metadata(), sample_scorecard(), clean_report(), 4).unwrap();
        let md = card.to_markdown();
        let sections = [
            "## Intended Use",
            "## Training Data & Licensing",
            "## Capabilities",
            "## Known Limitations",
            "## Red-Team Summary",
            "## Hardware Requirements",
            "## Version & Chat-Template Compatibility",
        ];
        let mut last = 0;
        for section in sections {
            let idx = md
                .find(section)
                .unwrap_or_else(|| panic!("missing section {section} in:\n{md}"));
            assert!(
                idx >= last,
                "section {section} at {idx} is out of order (prev {last})"
            );
            last = idx;
        }
    }

    /// The JSON card round-trips through serde without loss.
    #[test]
    fn model_card_json_round_trips() {
        let card =
            ModelCard::assemble(sample_metadata(), sample_scorecard(), clean_report(), 4).unwrap();
        let json = card.to_json().unwrap();
        let back: ModelCard = serde_json::from_str(&json).unwrap();
        assert_eq!(card, back);
    }

    /// The capabilities section in the Markdown is the verbatim scorecard
    /// Markdown — not a hand-re-rendered copy.
    #[test]
    fn model_card_capabilities_section_matches_scorecard_markdown_verbatim() {
        let scorecard = sample_scorecard();
        let card =
            ModelCard::assemble(sample_metadata(), scorecard.clone(), clean_report(), 4).unwrap();
        let md = card.to_markdown();
        assert!(
            md.contains(&scorecard.to_markdown()),
            "capabilities section must be the verbatim scorecard Markdown"
        );
    }

    /// Metadata round-trips through TOML.
    #[test]
    fn model_card_metadata_toml_round_trips() {
        let metadata = sample_metadata();
        let toml_text = toml::to_string(&metadata).unwrap();
        let back: ModelCardMetadata = toml::from_str(&toml_text).unwrap();
        assert_eq!(metadata, back);
    }

    /// The schema version constant is 1.
    #[test]
    fn model_card_schema_version_is_one() {
        assert_eq!(MODEL_CARD_SCHEMA_VERSION, 1);
        let card =
            ModelCard::assemble(sample_metadata(), sample_scorecard(), clean_report(), 4).unwrap();
        assert_eq!(card.schema_version, 1);
    }

    /// `write()` produces both the `.md` and the `.json` companion file.
    #[test]
    fn model_card_write_produces_markdown_and_json() {
        let card =
            ModelCard::assemble(sample_metadata(), sample_scorecard(), clean_report(), 4).unwrap();
        let tmp = std::env::temp_dir();
        let nonce = std::process::id();
        let md_path = tmp.join(format!("phase54_write_{nonce}.md"));
        card.write(&md_path).unwrap();
        let json_path = md_path.with_extension("json");
        assert!(md_path.exists(), "md file written");
        assert!(json_path.exists(), "json companion written");
        let md_text = std::fs::read_to_string(&md_path).unwrap();
        assert!(md_text.contains("# Model Card"));
        let json_text = std::fs::read_to_string(&json_path).unwrap();
        let back: ModelCard = serde_json::from_str(&json_text).unwrap();
        assert_eq!(card, back);
        let _ = std::fs::remove_file(&md_path);
        let _ = std::fs::remove_file(&json_path);
    }

    /// The red-team summary section in the Markdown is the verbatim report
    /// Markdown — failures listed first, all-cases table present.
    #[test]
    fn model_card_redteam_section_matches_report_markdown_verbatim() {
        let report = clean_report();
        let card =
            ModelCard::assemble(sample_metadata(), sample_scorecard(), report.clone(), 4).unwrap();
        let md = card.to_markdown();
        assert!(
            md.contains(&report.to_markdown()),
            "red-team section must be the verbatim report Markdown"
        );
        // The v4 corpus has 24 cases; the all-cases table must list all 24.
        assert!(md.contains("## All cases"));
    }

    /// `ModelCardError` variants render actionable messages.
    #[test]
    fn model_card_error_messages_are_actionable() {
        let err = ModelCardError::MissingRedTeamReport.to_string();
        assert!(err.contains("red-team report"));
        assert!(err.contains("clean"));
        let err = ModelCardError::MissingScorecard.to_string();
        assert!(err.contains("scorecard"));
        assert!(err.contains("PULLED"));
        let err = ModelCardError::MetadataUnreadable("/tmp/meta.toml".to_string()).to_string();
        assert!(err.contains("static metadata"));
        assert!(err.contains("/tmp/meta.toml"));
    }

    /// A missing metadata file fails loudly with `MetadataUnreadable`.
    #[test]
    fn model_card_generation_fails_loudly_if_metadata_file_is_missing() {
        let tmp = std::env::temp_dir();
        let nonce = std::process::id();
        let missing_meta = tmp.join(format!("phase54_no_meta_{nonce}.toml"));
        let scorecard_path = tmp.join(format!("phase54_score_meta_{nonce}.json"));
        let redteam_path = tmp.join(format!("phase54_rt_meta_{nonce}.json"));
        std::fs::write(&scorecard_path, sample_scorecard().to_json().unwrap()).unwrap();
        std::fs::write(&redteam_path, clean_report().to_json().unwrap()).unwrap();
        let err = ModelCard::assemble_from_paths(&missing_meta, &scorecard_path, &redteam_path, 4)
            .expect_err("missing metadata must fail");
        assert!(matches!(err, ModelCardError::MetadataUnreadable(_)));
        let _ = std::fs::remove_file(&scorecard_path);
        let _ = std::fs::remove_file(&redteam_path);
    }

    /// A single hand-authored case assembles cleanly with a one-case report.
    #[test]
    fn model_card_assembles_with_single_case_clean_report() {
        let case = AdversarialCase {
            id: "t.single".into(),
            surface: RedTeamSurface::SystemTurnInjection,
            category: "test".into(),
            input: AdversarialInput::Prompt {
                prompt: "hello".into(),
            },
            expected_outcome: ExpectedOutcome::Refused,
            source: "hand-authored".into(),
        };
        let outcome = CaseOutcome::new(&case, ObservedOutcome::Refused { reason: None });
        let report = RedTeamReport::from_outcomes(vec![outcome]);
        assert!(report.is_clean());
        let card = ModelCard::assemble(sample_metadata(), sample_scorecard(), report, 4).unwrap();
        assert_eq!(card.redteam_summary.corpus_size, 1);
        assert_eq!(card.redteam_summary.passed, 1);
    }
}
