//! Red-team report types — pass/fail per adversarial case, surfaced plainly.
//!
//! The report is the single artefact the red-team pass produces. Every case
//! in the corpus appears in the report, including failures — a failing case is
//! never silently dropped, mirroring the "measure, don't assume" discipline
//! every capability claim in this project has held since v2 §17's eval harness.

use std::fs::File;
use std::io::Write;
use std::path::Path;

use aarambh_studio_core::Result;
use serde::{Deserialize, Serialize};

use crate::redteam::harness::{AdversarialCase, ExpectedOutcome, ObservedOutcome, RedTeamSurface};

/// Red-team report schema version. Bumped only on breaking report-shape changes.
pub const REDTEAM_REPORT_SCHEMA_VERSION: u32 = 1;

/// One case's outcome in the report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseOutcome {
    /// Stable snake_case case id, e.g. `"system_turn.injection.ignore_previous"`.
    pub id: String,
    /// Which v4.0 surface this case targeted.
    pub surface: RedTeamSurface,
    /// Human-readable category, e.g. `"system_turn_injection"`.
    pub category: String,
    /// What the system was expected to do.
    pub expected_outcome: ExpectedOutcome,
    /// What the system actually did when probed.
    pub observed: ObservedOutcome,
    /// True iff `observed` matches `expected_outcome`.
    pub passed: bool,
}

impl CaseOutcome {
    /// Build a case outcome by comparing the expected against the observed.
    ///
    /// A probe error is represented as `ObservedOutcome::Other { .. }`, which
    /// never matches a labelled expected outcome — the case is still surfaced
    /// in the report as a failure rather than silently dropped.
    pub fn new(case: &AdversarialCase, observed: ObservedOutcome) -> Self {
        let passed = observed.matches(case.expected_outcome);
        Self {
            id: case.id.clone(),
            surface: case.surface,
            category: case.category.clone(),
            expected_outcome: case.expected_outcome,
            observed,
            passed,
        }
    }
}

/// The complete red-team report.
///
/// Failures are surfaced plainly in both serialisations: JSON carries the full
/// `outcomes` vector; Markdown lists failures first, then the summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RedTeamReport {
    /// Report schema version (`REDTEAM_REPORT_SCHEMA_VERSION`).
    pub schema_version: u32,
    /// Report generation time in Unix milliseconds.
    pub generated_at_unix_ms: u128,
    /// Number of cases in the corpus.
    pub corpus_size: usize,
    /// Number of cases whose observed outcome matched the expected one.
    pub passed: usize,
    /// Number of cases whose observed outcome did NOT match the expected one.
    pub failed: usize,
    /// Every case outcome, including failures — never silently dropped.
    pub outcomes: Vec<CaseOutcome>,
}

impl RedTeamReport {
    /// Build a report from a corpus's worth of case outcomes.
    pub fn from_outcomes(outcomes: Vec<CaseOutcome>) -> Self {
        let passed = outcomes.iter().filter(|o| o.passed).count();
        let failed = outcomes.len() - passed;
        Self {
            schema_version: REDTEAM_REPORT_SCHEMA_VERSION,
            generated_at_unix_ms: now_unix_ms(),
            corpus_size: outcomes.len(),
            passed,
            failed,
            outcomes,
        }
    }

    /// True iff no case failed.
    pub fn is_clean(&self) -> bool {
        self.failed == 0
    }

    /// Render as pretty-printed JSON.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Render as Markdown. Failures are listed first, plainly.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("# Red-Team Report\n\n");
        out.push_str(&format!(
            "- corpus size: {}\n- passed: {}\n- failed: {}\n- schema version: {}\n\n",
            self.corpus_size, self.passed, self.failed, self.schema_version
        ));
        if self.is_clean() {
            out.push_str("All cases matched their labelled expected outcome.\n\n");
        } else {
            out.push_str("## Failures\n\n");
            out.push_str("| id | surface | category | expected | observed |\n");
            out.push_str("| --- | --- | --- | --- | --- |\n");
            for o in self.outcomes.iter().filter(|o| !o.passed) {
                out.push_str(&format!(
                    "| {} | {} | {} | {} | {} |\n",
                    o.id,
                    o.surface,
                    o.category,
                    o.expected_outcome,
                    o.observed.label(),
                ));
            }
            out.push('\n');
        }
        out.push_str("## All cases\n\n");
        out.push_str("| id | surface | category | expected | observed | passed |\n");
        out.push_str("| --- | --- | --- | --- | --- | --- |\n");
        for o in &self.outcomes {
            out.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} |\n",
                o.id,
                o.surface,
                o.category,
                o.expected_outcome,
                o.observed.label(),
                o.passed,
            ));
        }
        out
    }

    /// Write the JSON report to a path.
    pub fn write_json(&self, path: &Path) -> Result<()> {
        let mut file = File::create(path)?;
        serde_json::to_writer_pretty(&mut file, self)?;
        writeln!(file)?;
        Ok(())
    }

    /// Write the Markdown report to a path.
    pub fn write_markdown(&self, path: &Path) -> Result<()> {
        std::fs::write(path, self.to_markdown())?;
        Ok(())
    }
}

/// Current Unix time in milliseconds.
fn now_unix_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::redteam::harness::{
        AdversarialCase, AdversarialInput, Corpus, ExpectedOutcome, ObservedOutcome, RedTeamSurface,
    };

    fn case(id: &str, surface: RedTeamSurface, expected: ExpectedOutcome) -> AdversarialCase {
        AdversarialCase {
            id: id.to_string(),
            surface,
            category: "test".to_string(),
            input: AdversarialInput::Prompt {
                prompt: "hello".to_string(),
            },
            expected_outcome: expected,
            source: "hand-authored".to_string(),
        }
    }

    #[test]
    fn case_outcome_passes_when_observed_matches_expected() {
        let c = case(
            "t.1",
            RedTeamSurface::SystemTurnInjection,
            ExpectedOutcome::Refused,
        );
        let o = CaseOutcome::new(&c, ObservedOutcome::Refused { reason: None });
        assert!(o.passed);
    }

    #[test]
    fn case_outcome_fails_when_observed_differs_from_expected() {
        let c = case(
            "t.2",
            RedTeamSurface::AuthBypassAttempt,
            ExpectedOutcome::Refused,
        );
        let o = CaseOutcome::new(&c, ObservedOutcome::ExecutedSafely);
        assert!(!o.passed);
    }

    #[test]
    fn report_records_every_case_including_failures() {
        let corpus = Corpus::v4();
        // Always-wrong observed outcome: the opposite of every expectation.
        let outcomes: Vec<CaseOutcome> = corpus
            .iter()
            .map(|c| {
                CaseOutcome::new(
                    c,
                    ObservedOutcome::Other {
                        label: "wrong".to_string(),
                    },
                )
            })
            .collect();
        let report = RedTeamReport::from_outcomes(outcomes);
        assert_eq!(report.corpus_size, corpus.len());
        assert_eq!(report.passed, 0);
        assert_eq!(report.failed, corpus.len());
        assert_eq!(report.outcomes.len(), corpus.len());
    }

    #[test]
    fn report_markdown_surfaces_failures_first() {
        let c = case(
            "t.fail",
            RedTeamSurface::SystemTurnInjection,
            ExpectedOutcome::Refused,
        );
        let o = CaseOutcome::new(&c, ObservedOutcome::ExecutedSafely);
        let report = RedTeamReport::from_outcomes(vec![o]);
        let md = report.to_markdown();
        let fail_idx = md.find("## Failures").expect("failures section present");
        let all_idx = md.find("## All cases").expect("all-cases section present");
        assert!(
            fail_idx < all_idx,
            "failures must be listed before all-cases"
        );
        assert!(md.contains("t.fail"));
    }

    #[test]
    fn report_json_round_trips() {
        let c = case(
            "t.rt",
            RedTeamSurface::OrchestratorBoundBypass,
            ExpectedOutcome::Sanitized,
        );
        let o = CaseOutcome::new(&c, ObservedOutcome::Sanitized { reason: None });
        let report = RedTeamReport::from_outcomes(vec![o]);
        let json = report.to_json().expect("json");
        let back: RedTeamReport = serde_json::from_str(&json).expect("round-trip");
        assert_eq!(report, back);
    }

    #[test]
    fn report_schema_version_is_one() {
        assert_eq!(REDTEAM_REPORT_SCHEMA_VERSION, 1);
        let report = RedTeamReport::from_outcomes(vec![]);
        assert_eq!(report.schema_version, 1);
    }
}
