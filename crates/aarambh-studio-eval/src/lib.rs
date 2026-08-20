//! Evaluation harness for language modeling, reasoning, preference, and vision tasks.
#![deny(missing_docs)]

/// Capability probes and persistent catastrophic-forgetting diagnostics.
pub mod forgetting;
/// Greedy generation helpers used by generative eval tasks.
pub mod generation;
/// Shared task runner and eval context types.
pub mod harness;
/// Phase 54 model-card assembly (generated, not hand-written).
pub mod model_card;
/// Perplexity-on-holdout evaluation.
pub mod ppl;
/// Scorecard serialization and comparison.
pub mod report;
/// Continuation log-probability scoring helpers.
pub mod scoring;
/// Built-in evaluation task implementations.
pub mod tasks;

pub use forgetting::{
    Capability, CapabilityProbe, CapabilityScore, DEFAULT_SIGNIFICANCE_THRESHOLD, ForgettingCurve,
    ForgettingDelta, ForgettingPoint, ForgettingRun, ForgettingStore, ManasForgettingRecord,
    ProbeManifest, ProbeSkip, RoutingDrift, RoutingSignature, run_capability_probes,
    tokenizer_fingerprint,
};
pub use generation::{
    BestOfNOptions, BestOfNResult, best_of_n_generate, greedy_generate, sample_generate,
};
pub use harness::{EvalConfig, EvalContext, EvalTask, run_all};
pub use model_card::{
    DatasetEntry, MODEL_CARD_SCHEMA_VERSION, ModelCard, ModelCardError, ModelCardMetadata,
};
pub use ppl::{PplResult, compute_ppl};
pub use report::{
    ForgettingReport, QatRobustnessReport, QatTaskRobustness, ScoreDelta, Scorecard,
    ScorecardComparison, TaskScore,
};
pub use scoring::{ContinuationScore, ContinuationScorer, ModelLogProbScorer};
