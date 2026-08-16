use std::time::{SystemTime, UNIX_EPOCH};

use aarambh_studio_core::Result;
use aarambh_studio_eval::{
    EvalConfig, EvalContext, ForgettingDelta, ForgettingStore, ProbeManifest, ProbeSkip,
    RoutingDrift, run_capability_probes, tokenizer_fingerprint,
};
use aarambh_studio_tokenizer::BpeTokenizer;
use serde::{Deserialize, Serialize};

use crate::config::SelfLearnForgettingConfig;
use crate::online_grpo::OnlineGrpo;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// Summary from one self-learning forgetting probe.
pub struct SelfLearnForgettingSummary {
    /// Baseline session identifier.
    pub baseline_id: String,
    /// Current update identifier.
    pub current_id: String,
    /// Per-capability score deltas.
    pub deltas: Vec<ForgettingDelta>,
    /// Optional MoE routing drift.
    pub routing_drift: Vec<RoutingDrift>,
    /// Capabilities skipped by this run.
    pub skipped: Vec<ProbeSkip>,
}

impl SelfLearnForgettingSummary {
    /// Count capabilities with a significant negative score delta.
    pub fn forgotten_count(&self, threshold: f64) -> usize {
        self.deltas
            .iter()
            .filter(|delta| delta.delta <= -threshold)
            .count()
    }
}

pub(crate) struct ForgettingHook {
    config: SelfLearnForgettingConfig,
    manifest: ProbeManifest,
    tokenizer: BpeTokenizer,
    tokenizer_sha256: String,
    session_id: String,
    baseline_id: String,
    capture_baseline: bool,
}

impl ForgettingHook {
    pub(crate) fn new(config: SelfLearnForgettingConfig, tokenizer: BpeTokenizer) -> Result<Self> {
        config.validate()?;
        let manifest = ProbeManifest::from_path(&config.manifest)?;
        let tokenizer_sha256 = tokenizer_fingerprint(&tokenizer)?;
        let session_id = unique_session_id();
        let capture_baseline = config.baseline_id.is_none();
        let baseline_id = config
            .baseline_id
            .clone()
            .unwrap_or_else(|| format!("{session_id}:start"));
        if !capture_baseline {
            let store = ForgettingStore::load_or_new(
                &config.store,
                &manifest,
                Some(tokenizer_sha256.clone()),
                config.significance_threshold,
            )?;
            if !store.contains_checkpoint_or_session(&baseline_id) {
                return Err(aarambh_studio_core::AarambhError::Config(format!(
                    "configured self-learning forgetting baseline '{baseline_id}' is not present in {}",
                    config.store.display()
                )));
            }
        }
        Ok(Self {
            config,
            manifest,
            tokenizer,
            tokenizer_sha256,
            session_id,
            baseline_id,
            capture_baseline,
        })
    }

    pub(crate) fn baseline(
        &mut self,
        online_grpo: &OnlineGrpo,
    ) -> Result<SelfLearnForgettingSummary> {
        if !self.capture_baseline {
            return Ok(SelfLearnForgettingSummary {
                baseline_id: self.baseline_id.clone(),
                current_id: self.baseline_id.clone(),
                deltas: Vec::new(),
                routing_drift: Vec::new(),
                skipped: Vec::new(),
            });
        }
        let id = self.baseline_id.clone();
        self.measure(online_grpo, id)
    }

    pub(crate) fn after_update(
        &mut self,
        online_grpo: &OnlineGrpo,
        update_kind: &str,
    ) -> Result<SelfLearnForgettingSummary> {
        let id = format!(
            "{}:{}_{:06}",
            self.session_id,
            update_kind,
            online_grpo.step_count()
        );
        self.measure(online_grpo, id)
    }

    pub(crate) fn store_path(&self) -> &std::path::Path {
        &self.config.store
    }

    pub(crate) fn threshold(&self) -> f64 {
        self.config.significance_threshold
    }

    fn measure(
        &mut self,
        online_grpo: &OnlineGrpo,
        current_id: String,
    ) -> Result<SelfLearnForgettingSummary> {
        let model = online_grpo.merged_eval_model()?;
        let context = EvalContext::new(
            model,
            self.tokenizer.clone(),
            online_grpo.device().clone(),
            online_grpo.dtype(),
        );
        let eval_config = EvalConfig {
            tasks: Vec::new(),
            data_dir: self.config.data_dir.clone(),
            max_examples: self.config.max_examples,
            max_new_tokens: self.config.max_new_tokens,
            agent_max_steps: self.config.agent_max_steps,
            allow_code_exec: self.config.allow_code_exec,
            thinking_mode: aarambh_studio_inference::ThinkingMode::None,
            best_of_n: None,
            best_of_n_selection: aarambh_studio_inference::SelectionStrategy::SelfConsistency,
            best_of_n_seed: 0,
            model_path: Some(format!("live-selflearn-step-{}", online_grpo.step_count())),
            tokenizer_path: None,
            config_path: self
                .config
                .config_path
                .as_ref()
                .map(|path| path.display().to_string()),
        };
        let run = run_capability_probes(
            &context,
            &eval_config,
            &self.manifest,
            &current_id,
            Some(self.tokenizer_sha256.clone()),
            self.config.require_all_probes,
        )?;
        let mut store = ForgettingStore::load_or_new(
            &self.config.store,
            &self.manifest,
            Some(self.tokenizer_sha256.clone()),
            self.config.significance_threshold,
        )?;
        store.record(&run)?;
        let deltas = if current_id == self.baseline_id {
            Vec::new()
        } else {
            store.deltas(&self.baseline_id, &current_id)?
        };
        store.save_atomic(&self.config.store)?;
        let routing_drift = store.routing_drift(&self.baseline_id, &current_id);
        if let Some(path) = &self.config.jsonl
            && current_id != self.baseline_id
        {
            store.export_jsonl(path, &self.baseline_id, &current_id)?;
        }
        Ok(SelfLearnForgettingSummary {
            baseline_id: self.baseline_id.clone(),
            current_id,
            deltas,
            routing_drift,
            skipped: run.skipped,
        })
    }
}

fn unique_session_id() -> String {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    format!("selflearn:{timestamp}:{}", std::process::id())
}
