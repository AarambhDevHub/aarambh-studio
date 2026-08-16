use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use aarambh_studio_core::{AarambhError, Result};
use aarambh_studio_eval::{
    EvalConfig, EvalContext, ForgettingStore, ProbeManifest, run_capability_probes,
    tokenizer_fingerprint,
};
use aarambh_studio_tokenizer::BpeTokenizer;
use aarambh_studio_train::{
    ForgettingTrainingConfig, TrainingObserver, TrainingObserverEvent, TrainingObserverFactory,
    TrainingObserverSnapshot, TrainingRunConfig,
};
use clap::Args;

#[derive(Debug, Args)]
pub struct TrainArgs {
    #[arg(long)]
    pub config: PathBuf,
}

pub fn run(args: TrainArgs) -> anyhow::Result<()> {
    let config = TrainingRunConfig::from_toml(&args.config)?;
    if config
        .forgetting
        .as_ref()
        .is_some_and(|forgetting| forgetting.enabled)
    {
        let mut factory = ForgettingObserverFactory {
            config_path: args.config.clone(),
        };
        aarambh_studio_train::run_training_from_config_with_observer(&args.config, &mut factory)?;
    } else {
        aarambh_studio_train::run_training_from_config(&args.config)?;
    }
    Ok(())
}

struct ForgettingObserverFactory {
    config_path: PathBuf,
}

impl TrainingObserverFactory for ForgettingObserverFactory {
    fn build(
        &mut self,
        config: &TrainingRunConfig,
        tokenizer: &BpeTokenizer,
    ) -> Result<Box<dyn TrainingObserver>> {
        let forgetting = config
            .forgetting
            .clone()
            .filter(|forgetting| forgetting.enabled)
            .ok_or_else(|| {
                AarambhError::Config(
                    "training observer requested without enabled [forgetting] config".into(),
                )
            })?;
        let manifest = ProbeManifest::from_path(&forgetting.manifest)?;
        let tokenizer_sha256 = tokenizer_fingerprint(tokenizer)?;
        let run_id = unique_run_id("train");
        let capture_baseline = forgetting.baseline_id.is_none();
        let baseline_id = forgetting
            .baseline_id
            .clone()
            .unwrap_or_else(|| format!("{run_id}:start"));
        if !capture_baseline {
            let store = ForgettingStore::load_or_new(
                &forgetting.store,
                &manifest,
                Some(tokenizer_sha256.clone()),
                forgetting.significance_threshold,
            )?;
            if !store.contains_checkpoint_or_session(&baseline_id) {
                return Err(AarambhError::Config(format!(
                    "configured forgetting baseline '{baseline_id}' is not present in {}",
                    forgetting.store.display()
                )));
            }
        }
        Ok(Box::new(ForgettingObserver {
            config_path: self.config_path.clone(),
            tokenizer: tokenizer.clone(),
            tokenizer_sha256,
            manifest,
            forgetting,
            run_id,
            baseline_id,
            capture_baseline,
            last_observed_step: None,
        }))
    }
}

struct ForgettingObserver {
    config_path: PathBuf,
    tokenizer: BpeTokenizer,
    tokenizer_sha256: String,
    manifest: ProbeManifest,
    forgetting: ForgettingTrainingConfig,
    run_id: String,
    baseline_id: String,
    capture_baseline: bool,
    last_observed_step: Option<usize>,
}

impl TrainingObserver for ForgettingObserver {
    fn should_observe(&self, event: TrainingObserverEvent, step: usize) -> bool {
        match event {
            TrainingObserverEvent::Start => self.capture_baseline,
            TrainingObserverEvent::OptimizerStep => {
                step.is_multiple_of(self.forgetting.every_n_steps)
            }
            TrainingObserverEvent::Finish => self.last_observed_step != Some(step),
        }
    }

    fn observe(&mut self, snapshot: TrainingObserverSnapshot<'_>) -> Result<()> {
        let current_id = match snapshot.event {
            TrainingObserverEvent::Start => self.baseline_id.clone(),
            TrainingObserverEvent::OptimizerStep | TrainingObserverEvent::Finish => {
                format!("{}:step_{:06}", self.run_id, snapshot.step)
            }
        };
        let context = EvalContext::new(
            snapshot.model.clone(),
            self.tokenizer.clone(),
            snapshot.device.clone(),
            snapshot.dtype,
        );
        let eval_config = EvalConfig {
            tasks: Vec::new(),
            data_dir: self.forgetting.data_dir.clone(),
            max_examples: self.forgetting.max_examples,
            max_new_tokens: self.forgetting.max_new_tokens,
            agent_max_steps: self.forgetting.agent_max_steps,
            allow_code_exec: self.forgetting.allow_code_exec,
            thinking_mode: aarambh_studio_inference::ThinkingMode::None,
            best_of_n: None,
            best_of_n_selection: aarambh_studio_inference::SelectionStrategy::SelfConsistency,
            best_of_n_seed: 0,
            model_path: Some(format!("live-training-step-{}", snapshot.step)),
            tokenizer_path: None,
            config_path: Some(self.config_path.display().to_string()),
        };
        let run = run_capability_probes(
            &context,
            &eval_config,
            &self.manifest,
            &current_id,
            Some(self.tokenizer_sha256.clone()),
            self.forgetting.require_all_probes,
        )?;
        let mut store = ForgettingStore::load_or_new(
            &self.forgetting.store,
            &self.manifest,
            Some(self.tokenizer_sha256.clone()),
            self.forgetting.significance_threshold,
        )?;
        store.record(&run)?;
        let deltas = if current_id == self.baseline_id {
            Vec::new()
        } else {
            store.deltas(&self.baseline_id, &current_id)?
        };
        store.save_atomic(&self.forgetting.store)?;
        let forgotten = deltas
            .iter()
            .filter(|delta| delta.delta <= -self.forgetting.significance_threshold)
            .count();
        println!(
            "forgetting step={} capabilities={} skipped={} forgotten={} baseline={}",
            snapshot.step,
            run.scores.len(),
            run.skipped.len(),
            forgotten,
            self.baseline_id
        );
        for delta in &deltas {
            println!(
                "forgetting capability={} before={:.4} after={:.4} delta={:+.4} significant={}",
                delta.capability_or_concept,
                delta.score_before,
                delta.score_after,
                delta.delta,
                delta.significant
            );
        }
        if let Some(path) = &self.forgetting.jsonl
            && current_id != self.baseline_id
        {
            store.export_jsonl(path, &self.baseline_id, &current_id)?;
        }
        self.last_observed_step = Some(snapshot.step);
        Ok(())
    }
}

fn unique_run_id(prefix: &str) -> String {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    format!("{prefix}:{timestamp}:{}", std::process::id())
}
