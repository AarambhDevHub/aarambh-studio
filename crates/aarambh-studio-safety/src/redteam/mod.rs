//! Red-team / adversarial safety evaluation (Phase 53, `ARCHITECTURE_V4.md` §67).
//!
//! A single systematic, end-to-end adversarial-testing pass run once near the
//! end of v4.0 against the complete attack surface: the safety layer
//! (`ARCHITECTURE.md` §13), the sandboxed tool-execution boundary (V4 §61),
//! the orchestrator hard bounds (V4 §62), the public inference server
//! (V4 §65), and the system-role/prompt-injection precedence rule (V4 §66).
//!
//! Every case carries a labelled [`ExpectedOutcome`]; a failing case
//! is surfaced plainly in the generated [`RedTeamReport`], never
//! silently dropped. Corpus content is hand-authored or drawn from free/public
//! sources only.

/// Adversarial case model, corpus, target trait, and harness runner.
pub mod harness;
/// Pass/fail report types — every case surfaced, failures first.
pub mod report;

pub use harness::{
    AdversarialCase, AdversarialInput, Corpus, ExpectedOutcome, ObservedOutcome, RedTeamHarness,
    RedTeamSurface, RedTeamTarget, SafetyLayerTarget,
};
pub use report::{CaseOutcome, REDTEAM_REPORT_SCHEMA_VERSION, RedTeamReport};
