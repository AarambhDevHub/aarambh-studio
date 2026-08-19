use std::collections::HashMap;
use std::path::Path;

use aarambh_studio_core::{AarambhError, Configurable, ModelConfig, Result, TokenizerLike};
use aarambh_studio_model::AarambhModel;
use aarambh_studio_tokenizer::BpeTokenizer;
use candle_core::{DType, Device, Tensor};

use crate::dataset::{ReplayBatch, ScoredReferenceDataset};
use crate::rollout::StudentRollout;

/// Teacher signal requested by the configured distillation objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TeacherSignal {
    /// Packed token-distribution logits for forward KL.
    SoftLogits,
    /// One scalar quality reward per rollout.
    Reward,
}

/// Scalar teacher judgment for one student rollout.
#[derive(Debug, Clone, PartialEq)]
pub struct TeacherScore {
    /// Finite scalar quality signal.
    pub reward: f32,
    /// Optional teacher-approved corrected completion.
    pub corrected_completion: Option<String>,
}

/// Detached teacher feedback for one replay batch.
#[derive(Debug)]
pub struct TeacherBatchFeedback {
    /// One scalar score per rollout.
    pub scores: Vec<TeacherScore>,
    /// Packed completion-position teacher logits for soft-KL training.
    pub packed_logits: Option<Tensor>,
}

/// Batch-scoring interface shared by local and scored-reference teachers.
pub trait TeacherScorer {
    /// Human-readable backend name stored in manifests and logs.
    fn backend_name(&self) -> &'static str;

    /// Score a batch of student-generated rollouts.
    fn score_batch(
        &self,
        rollouts: &[StudentRollout],
        replay: &ReplayBatch,
        signal: TeacherSignal,
    ) -> Result<TeacherBatchFeedback>;
}

/// Frozen local Aarambh checkpoint used as a token-level teacher.
#[derive(Debug)]
pub struct LocalCheckpointTeacher {
    model: AarambhModel,
    device: Device,
}

impl LocalCheckpointTeacher {
    /// Construct a local teacher from an already loaded frozen model.
    pub fn new(model: AarambhModel, device: Device) -> Self {
        Self { model, device }
    }

    /// Load a frozen local teacher checkpoint with the requested dtype.
    pub fn from_paths(
        model_path: impl AsRef<Path>,
        config: &ModelConfig,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let model =
            aarambh_studio_weights::load_any_model_with_dtype(model_path, config, &device, dtype)?;
        Ok(Self::new(model, device))
    }

    /// Return the frozen teacher model.
    pub fn model(&self) -> &AarambhModel {
        &self.model
    }
}

impl TeacherScorer for LocalCheckpointTeacher {
    fn backend_name(&self) -> &'static str {
        "local_checkpoint"
    }

    fn score_batch(
        &self,
        rollouts: &[StudentRollout],
        replay: &ReplayBatch,
        signal: TeacherSignal,
    ) -> Result<TeacherBatchFeedback> {
        if rollouts.len() != replay.batch_size() {
            return Err(AarambhError::Shape(
                "teacher rollout count does not match replay batch".into(),
            ));
        }
        let sequence = replay.input_ids.dim(1)?;
        if sequence > self.model.config().max_seq_len {
            return Err(AarambhError::Shape(format!(
                "teacher replay sequence {sequence} exceeds max_seq_len {}",
                self.model.config().max_seq_len
            )));
        }
        let inputs = replay.input_ids.to_device(&self.device)?;
        let logits = self.model.forward_train(&inputs)?;
        let (_, _, vocab) = logits.dims3()?;
        let indices = replay.packed_row_indices.to_device(&self.device)?;
        let packed = logits
            .reshape((replay.batch_size() * sequence, vocab))?
            .index_select(&indices, 0)?;
        let labels = replay.packed_labels()?.to_device(&self.device)?;
        let selected = candle_nn::ops::log_softmax(&packed.to_dtype(DType::F32)?, 1)?
            .gather(&labels.reshape((replay.completion_tokens(), 1))?, 1)?
            .reshape(replay.completion_tokens())?;
        let mut scores = Vec::with_capacity(rollouts.len());
        let mut offset = 0usize;
        for &count in &replay.completion_counts {
            let reward = selected
                .narrow(0, offset, count)?
                .sum_all()?
                .affine(1.0 / count as f64, 0.0)?
                .to_scalar::<f32>()?;
            if !reward.is_finite() {
                return Err(AarambhError::Config(
                    "local teacher produced a non-finite reward".into(),
                ));
            }
            scores.push(TeacherScore {
                reward,
                corrected_completion: None,
            });
            offset += count;
        }
        let packed_logits = match signal {
            TeacherSignal::SoftLogits => {
                Some(packed.detach().to_device(replay.input_ids.device())?)
            }
            TeacherSignal::Reward => None,
        };
        Ok(TeacherBatchFeedback {
            scores,
            packed_logits,
        })
    }
}

/// Low-memory teacher that scores rollouts against pre-scored references.
#[derive(Debug, Clone)]
pub struct ScoredDatasetTeacher {
    dataset: ScoredReferenceDataset,
    tokenizer: BpeTokenizer,
    tokenized_references: HashMap<String, Vec<Vec<u32>>>,
}

impl ScoredDatasetTeacher {
    /// Build a scored-reference teacher and tokenize its references once.
    pub fn new(dataset: ScoredReferenceDataset, tokenizer: BpeTokenizer) -> Result<Self> {
        let mut tokenized_references = HashMap::with_capacity(dataset.records().len());
        for record in dataset.records() {
            let references = record
                .references
                .iter()
                .map(|reference| tokenizer.encode(&reference.completion))
                .collect::<Result<Vec<_>>>()?;
            tokenized_references.insert(record.id.clone(), references);
        }
        Ok(Self {
            dataset,
            tokenizer,
            tokenized_references,
        })
    }

    /// Return the underlying scored-reference dataset.
    pub fn dataset(&self) -> &ScoredReferenceDataset {
        &self.dataset
    }
}

impl TeacherScorer for ScoredDatasetTeacher {
    fn backend_name(&self) -> &'static str {
        "scored_dataset"
    }

    fn score_batch(
        &self,
        rollouts: &[StudentRollout],
        replay: &ReplayBatch,
        signal: TeacherSignal,
    ) -> Result<TeacherBatchFeedback> {
        if signal == TeacherSignal::SoftLogits {
            return Err(AarambhError::Unsupported(
                "scored-reference teachers support reward distillation only".into(),
            ));
        }
        if rollouts.len() != replay.batch_size() {
            return Err(AarambhError::Shape(
                "teacher rollout count does not match replay batch".into(),
            ));
        }
        let mut scores = Vec::with_capacity(rollouts.len());
        for rollout in rollouts {
            let record = self.dataset.get(&rollout.prompt_id).ok_or_else(|| {
                AarambhError::Config(format!(
                    "scored-reference teacher is missing prompt id '{}'",
                    rollout.prompt_id
                ))
            })?;
            if record.prompt != rollout.prompt {
                return Err(AarambhError::Config(format!(
                    "scored-reference prompt text mismatch for id '{}'",
                    rollout.prompt_id
                )));
            }
            let candidate = self.tokenizer.encode(&rollout.completion_text)?;
            let references = self
                .tokenized_references
                .get(&rollout.prompt_id)
                .ok_or_else(|| {
                    AarambhError::Config(format!(
                        "scored-reference token cache is missing prompt id '{}'",
                        rollout.prompt_id
                    ))
                })?;
            let mut best_reward = f32::NEG_INFINITY;
            let mut correction = None;
            for (reference, reference_tokens) in record.references.iter().zip(references) {
                let reward = reference.score * token_f1(&candidate, reference_tokens);
                if reward > best_reward {
                    best_reward = reward;
                    correction = Some(reference.completion.clone());
                }
            }
            scores.push(TeacherScore {
                reward: best_reward.max(0.0),
                corrected_completion: correction,
            });
        }
        Ok(TeacherBatchFeedback {
            scores,
            packed_logits: None,
        })
    }
}

fn token_f1(candidate: &[u32], reference: &[u32]) -> f32 {
    if candidate.is_empty() || reference.is_empty() {
        return 0.0;
    }
    if candidate == reference {
        return 1.0;
    }
    let mut candidate_counts = HashMap::<u32, usize>::new();
    let mut reference_counts = HashMap::<u32, usize>::new();
    for &token in candidate {
        *candidate_counts.entry(token).or_default() += 1;
    }
    for &token in reference {
        *reference_counts.entry(token).or_default() += 1;
    }
    let overlap = candidate_counts
        .iter()
        .map(|(token, count)| count.min(reference_counts.get(token).unwrap_or(&0)))
        .sum::<usize>();
    if overlap == 0 {
        return 0.0;
    }
    let precision = overlap as f32 / candidate.len() as f32;
    let recall = overlap as f32 / reference.len() as f32;
    2.0 * precision * recall / (precision + recall)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::ScoredReferenceRecord;
    use aarambh_studio_core::ModelConfig;
    use aarambh_studio_tokenizer::{
        ASSISTANT, ASSISTANT_ID, BOS, BOS_ID, ENDOFTEXT, ENDOFTEXT_ID, PAD, PAD_ID, THINK_END,
        THINK_END_ID, THINK_START, THINK_START_ID, USER, USER_ID, Vocab,
    };
    use candle_nn::VarBuilder;

    fn tokenizer() -> BpeTokenizer {
        let pairs: [(&str, u32); 12] = [
            (ENDOFTEXT, ENDOFTEXT_ID),
            (PAD, PAD_ID),
            (BOS, BOS_ID),
            (THINK_START, THINK_START_ID),
            (THINK_END, THINK_END_ID),
            (USER, USER_ID),
            (ASSISTANT, ASSISTANT_ID),
            ("H", 7),
            ("e", 8),
            ("l", 9),
            ("o", 10),
            (" ", 11),
        ];
        let token_to_id = pairs
            .iter()
            .map(|(token, id)| ((*token).to_string(), *id))
            .collect::<HashMap<_, _>>();
        let mut id_to_token = vec![String::new(); 12];
        for (token, id) in pairs {
            id_to_token[id as usize] = token.to_string();
        }
        BpeTokenizer {
            vocab: Vocab {
                token_to_id,
                id_to_token,
            },
            merges: Vec::new(),
            merge_rank: HashMap::new(),
            chat_template_version: None,
        }
    }

    fn rollout(id: &str, text: &str, index: usize) -> StudentRollout {
        StudentRollout {
            prompt_id: id.into(),
            prompt: "Hello".into(),
            prompt_token_ids: vec![BOS_ID, 7],
            completion_token_ids: vec![8, 9, 9, 10],
            completion_text: text.into(),
            loss_mask: vec![true; 4],
            rollout_index: index,
            finish_reason: crate::rollout::RolloutFinish::MaxTokens,
        }
    }

    #[test]
    fn teacher_trait_accepts_local_and_dataset_backends() {
        let device = Device::Cpu;
        let model_config = ModelConfig {
            vocab_size: 12,
            hidden_dim: 64,
            ffn_dim: 128,
            n_layers: 1,
            n_heads: 1,
            n_kv_heads: 1,
            max_seq_len: 16,
            rope_theta: 10_000.0,
            rope_scaling: None,
            moe: None,
            attention_schedule: None,
            dsa_config: None,
            mtp: None,
            qat: None,
            norm_eps: 1e-5,
            tie_embeddings: true,
            chat_template_version: None,
        };
        let model =
            AarambhModel::new(&model_config, VarBuilder::zeros(DType::F32, &device)).unwrap();
        let local = LocalCheckpointTeacher::new(model, device.clone());
        let dataset = ScoredReferenceDataset::from_records(vec![ScoredReferenceRecord {
            id: "p".into(),
            prompt: "Hello".into(),
            references: vec![crate::dataset::ReferenceAnswer {
                completion: "Hello".into(),
                score: 1.0,
            }],
        }])
        .unwrap();
        let scored = ScoredDatasetTeacher::new(dataset, tokenizer()).unwrap();
        let scorers: [&dyn TeacherScorer; 2] = [&local, &scored];
        assert_eq!(scorers[0].backend_name(), "local_checkpoint");
        assert_eq!(scorers[1].backend_name(), "scored_dataset");
    }

    #[test]
    fn local_teacher_returns_detached_packed_logits_and_scores() {
        let device = Device::Cpu;
        let config = ModelConfig {
            vocab_size: 12,
            hidden_dim: 64,
            ffn_dim: 128,
            n_layers: 1,
            n_heads: 1,
            n_kv_heads: 1,
            max_seq_len: 16,
            rope_theta: 10_000.0,
            rope_scaling: None,
            moe: None,
            attention_schedule: None,
            dsa_config: None,
            mtp: None,
            qat: None,
            norm_eps: 1e-5,
            tie_embeddings: true,
            chat_template_version: None,
        };
        let model = AarambhModel::new(&config, VarBuilder::zeros(DType::F32, &device)).unwrap();
        let teacher = LocalCheckpointTeacher::new(model, device.clone());
        let rollouts = vec![rollout("p", "ello", 0)];
        let replay = ReplayBatch::from_rollouts(&rollouts, PAD_ID, &device).unwrap();
        let feedback = teacher
            .score_batch(&rollouts, &replay, TeacherSignal::SoftLogits)
            .unwrap();
        assert_eq!(feedback.scores.len(), 1);
        assert!(feedback.scores[0].reward.is_finite());
        assert_eq!(
            feedback.packed_logits.unwrap().dims(),
            [replay.completion_tokens(), 12]
        );
    }

    #[test]
    fn scored_dataset_teacher_rewards_exact_reference() {
        let dataset = ScoredReferenceDataset::from_records(vec![ScoredReferenceRecord {
            id: "p".into(),
            prompt: "Hello".into(),
            references: vec![crate::dataset::ReferenceAnswer {
                completion: "ello".into(),
                score: 0.8,
            }],
        }])
        .unwrap();
        let teacher = ScoredDatasetTeacher::new(dataset, tokenizer()).unwrap();
        let rollouts = vec![rollout("p", "ello", 0)];
        let replay = ReplayBatch::from_rollouts(&rollouts, PAD_ID, &Device::Cpu).unwrap();
        let feedback = teacher
            .score_batch(&rollouts, &replay, TeacherSignal::Reward)
            .unwrap();
        assert!((feedback.scores[0].reward - 0.8).abs() < 1e-6);
        assert!(feedback.packed_logits.is_none());
    }

    #[test]
    fn token_f1_handles_exact_partial_and_disjoint_sequences() {
        assert_eq!(token_f1(&[1, 2], &[1, 2]), 1.0);
        assert!((token_f1(&[1, 2], &[2, 3]) - 0.5).abs() < 1e-6);
        assert_eq!(token_f1(&[1], &[2]), 0.0);
    }
}
