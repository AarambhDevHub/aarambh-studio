//! Fine-tuning support.
//!
//! Phase 9 implements LoRA, QLoRA, SFT loss masking, and adapter merge support.
//! Phase 10 adds deterministic-verifier GRPO for adapter-only RL fine-tuning.
//! Phase 18 adds DoRA and QDoRA supervised fine-tuning.
//! Phase 20 adds vision-language DoRA instruction tuning.
//! Phase 24 adds DoRA/QDoRA Direct Preference Optimization.
//! Phase 35 extends the shared VLM trainer to native video instruction tuning.
//! Phase 36 extends it to PDF and scanned-document instruction tuning.
//! Phase 46 adds RLAIF (Reinforcement Learning from AI Feedback), an
//! offline data-generation front end that produces (chosen, rejected)
//! preference pairs in the exact DPO schema from AI-judged self-sampled
//! candidates, with position-swap bias correction.
#![deny(missing_docs)]

/// Adapter metadata and serialization helpers.
pub mod adapter;
/// DoRA adapter layers and DoRA-wrapped model implementation.
pub mod dora;
/// Direct Preference Optimization data loading, loss, and adapter training.
pub mod dpo;
/// Group Relative Policy Optimization data loading, rollout, and training.
pub mod grpo;
/// LoRA adapter layers and configuration.
pub mod lora;
/// LoRA-wrapped Aarambh model implementation.
pub mod model;
/// Phase 46 — RLAIF (Reinforcement Learning from AI Feedback).
pub mod rlaif;
/// Supervised fine-tuning datasets, templates, and batches.
pub mod sft;
/// Function-calling supervised datasets and protocol formatting.
pub mod tool_sft;
/// SFT trainer and adapter merge entrypoints.
pub mod trainer;
/// Rule-based verifiers used by GRPO and self-learning.
pub mod verifier;
/// Vision-language DoRA instruction tuning.
pub mod vlm_dora;

pub use adapter::{AdapterMetadata, AdapterMethod, load_adapter_metadata, save_adapter};
pub use dora::{DoraAarambhModel, DoraConfig, DoraLinear};
pub use dpo::{
    DpoBatch, DpoConfig, DpoDataLoader, DpoDataset, DpoExample, DpoMetrics, DpoRunConfig,
    DpoSaveMetadata, DpoTrainer, dpo_loss, run_dpo_from_config, sequence_log_probs,
};
pub use grpo::{
    GrpoConfig, GrpoDataset, GrpoExample, GrpoMetrics, GrpoRunConfig, GrpoThinkingMode,
    GrpoTrainer, Rollout, RolloutFinish, compute_advantages, grpo_loss, run_grpo_from_config,
    sample_group,
};
pub use lora::{BaseLinear, LoraConfig, LoraLinear};
pub use model::LoraAarambhModel;
pub use rlaif::{
    AgreementLevel, BiasCorrectedPair, CandidatePair, CandidateSampler, JudgeChoice,
    JudgeGenerator, JudgeVerdict, RLAIF_PROVENANCE, RlaifConfig, RlaifPair, RlaifRunConfig,
    RlaifSummary, build_judge_prompt, default_judge_template, form_pairs, generate_rlaif_dataset,
    judge_pair, judge_pair_both_orderings, parse_judge_verdict, read_prompts_jsonl,
    resolve_preference, run_rlaif_with_engines, write_preference_jsonl,
};
pub use sft::{
    ChatTemplate, SftBatch, SftDataLoader, SftDataset, SftExample, ThinkingSftExample,
    format_thinking_sft,
};
pub use tool_sft::{
    MultiStepToolSftExample, ToolSftCall, ToolSftDataset, ToolSftDefinition, ToolSftExample,
    ToolSftResult, ToolSftResultContent, ToolSftResultStatus, ToolSftTurn,
};
pub use trainer::{
    AdapterSftModel, DoraTrainer, SftRunConfig, SftTrainer, merge_adapter_from_paths,
    merge_dora_from_paths, merge_lora_from_paths, run_dora_from_config, run_sft_from_config,
    run_tool_sft_from_config,
};
pub use verifier::{
    CodeVerifier, CompositeVerifier, FormatVerifier, MathVerifier, Verifier, VerifierKind,
    extract_final_number,
};
pub use vlm_dora::{
    AudioVlmDoraRunConfig, DocumentVlmDoraRunConfig, VideoVlmDoraRunConfig, VlmDoraMetrics,
    VlmDoraRunConfig, VlmDoraTrainer, run_audio_vlm_dora_from_config,
    run_document_vlm_dora_from_config, run_video_vlm_dora_from_config, run_vlm_dora_from_config,
};
