use std::fs;
use std::path::{Path, PathBuf};

use aarambh_studio_core::TokenizerLike;
use aarambh_studio_tokenizer::{BpeTokenizer, FRAME_SEP_ID, IMAGE_END_ID, IMAGE_ID};
use aarambh_studio_train::TrainingRunConfig;
use aarambh_studio_weights::{
    GgufFormat, HfArch, VocabularyExpansion, convert_hf_with_arch, expand_safetensors_vocabulary,
    load_any_model, save_gguf, save_model,
};
use clap::Args;

#[derive(Debug, Args)]
/// Convert checkpoints between HF SafeTensors and GGUF, or expand vocabularies.
pub struct ConvertArgs {
    /// Training/model TOML configuration (provides architecture + device).
    #[arg(long, default_value = "configs/tiny_shakespeare.toml")]
    pub config: PathBuf,
    /// Source checkpoint path to convert.
    #[arg(long)]
    pub input: PathBuf,
    /// Output converted checkpoint path.
    #[arg(long)]
    pub output: PathBuf,
    /// HF architecture family: llama3 (used for HF -> SafeTensors migration).
    #[arg(long, default_value = "llama3")]
    pub arch: String,
    /// Emit a GGUF checkpoint instead of native SafeTensors.
    #[arg(long)]
    pub gguf: bool,
    /// GGUF quantisation format: q4_k_m, q5_k_m, or q8_0.
    #[arg(long, default_value = "q4_k_m")]
    pub format: String,
    /// Expand the SafeTensors model with the Phase 35 video vocabulary.
    #[arg(long, requires = "tokenizer", requires = "output_tokenizer")]
    pub upgrade_video_vocab: bool,
    /// Expand the SafeTensors model with the Phase 36 document vocabulary.
    #[arg(
        long,
        requires = "tokenizer",
        requires = "output_tokenizer",
        conflicts_with = "upgrade_video_vocab"
    )]
    pub upgrade_document_vocab: bool,
    /// Expand the SafeTensors model with the Phase 42 audio vocabulary.
    #[arg(
        long,
        requires = "tokenizer",
        requires = "output_tokenizer",
        conflicts_with_all = ["upgrade_video_vocab", "upgrade_document_vocab"]
    )]
    pub upgrade_audio_vocab: bool,
    /// Source tokenizer JSON path (required for vocabulary upgrades).
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,
    /// Output tokenizer JSON path for the upgraded vocabulary.
    #[arg(long)]
    pub output_tokenizer: Option<PathBuf>,
}

pub fn run(args: ConvertArgs) -> anyhow::Result<()> {
    let run_config = TrainingRunConfig::from_toml(&args.config)?;
    let device = run_config.device()?.to_candle()?;
    write_parent_dir(&args.output)?;

    if args.upgrade_video_vocab {
        if args.gguf {
            return Err(anyhow::anyhow!(
                "--upgrade-video-vocab requires SafeTensors input; migrate first, then quantise the migrated checkpoint"
            ));
        }
        let tokenizer_path = args.tokenizer.as_ref().expect("required by clap");
        let output_tokenizer = args.output_tokenizer.as_ref().expect("required by clap");
        let tokenizer = BpeTokenizer::from_pretrained(tokenizer_path)?;
        if tokenizer.validate_video_special_tokens().is_ok() {
            return Err(anyhow::anyhow!(
                "tokenizer {} already contains the Phase 35 video vocabulary",
                tokenizer_path.display()
            ));
        }
        tokenizer.validate_vision_special_tokens()?;
        let old_vocab_size = tokenizer.vocab_size();
        let upgraded = tokenizer.upgraded_for_video()?;
        write_parent_dir(output_tokenizer)?;
        let report = expand_safetensors_vocabulary(
            &args.input,
            &args.output,
            old_vocab_size,
            &VocabularyExpansion {
                insertion_id: 9,
                source_ids: vec![
                    IMAGE_ID as usize,
                    IMAGE_END_ID as usize,
                    IMAGE_END_ID as usize,
                ],
            },
        )?;
        upgraded.save_pretrained(output_tokenizer)?;
        eprintln!(
            "upgraded video vocabulary {} -> {} rows across {} tensors; model={} tokenizer={}",
            report.old_vocab_size,
            report.new_vocab_size,
            report.expanded_tensors,
            args.output.display(),
            output_tokenizer.display()
        );
        return Ok(());
    }

    if args.upgrade_document_vocab {
        if args.gguf {
            return Err(anyhow::anyhow!(
                "--upgrade-document-vocab requires SafeTensors input; migrate first, then quantise the migrated checkpoint"
            ));
        }
        let tokenizer_path = args.tokenizer.as_ref().expect("required by clap");
        let output_tokenizer = args.output_tokenizer.as_ref().expect("required by clap");
        let tokenizer = BpeTokenizer::from_pretrained(tokenizer_path)?;
        if tokenizer.validate_document_special_tokens().is_ok() {
            return Err(anyhow::anyhow!(
                "tokenizer {} already contains the Phase 36 document vocabulary",
                tokenizer_path.display()
            ));
        }
        tokenizer.validate_video_special_tokens()?;
        let old_vocab_size = tokenizer.vocab_size();
        let upgraded = tokenizer.upgraded_for_document()?;
        write_parent_dir(output_tokenizer)?;
        let report = expand_safetensors_vocabulary(
            &args.input,
            &args.output,
            old_vocab_size,
            &VocabularyExpansion {
                insertion_id: 12,
                source_ids: vec![
                    IMAGE_ID as usize,
                    IMAGE_END_ID as usize,
                    FRAME_SEP_ID as usize,
                ],
            },
        )?;
        upgraded.save_pretrained(output_tokenizer)?;
        eprintln!(
            "upgraded document vocabulary {} -> {} rows across {} tensors; model={} tokenizer={}",
            report.old_vocab_size,
            report.new_vocab_size,
            report.expanded_tensors,
            args.output.display(),
            output_tokenizer.display()
        );
        return Ok(());
    }

    if args.upgrade_audio_vocab {
        if args.gguf {
            return Err(anyhow::anyhow!(
                "--upgrade-audio-vocab requires SafeTensors input; migrate first, then quantise the migrated checkpoint"
            ));
        }
        let tokenizer_path = args.tokenizer.as_ref().expect("required by clap");
        let output_tokenizer = args.output_tokenizer.as_ref().expect("required by clap");
        let tokenizer = BpeTokenizer::from_pretrained(tokenizer_path)?;
        if tokenizer.validate_audio_special_tokens().is_ok() {
            return Err(anyhow::anyhow!(
                "tokenizer {} already contains the Phase 42 audio vocabulary",
                tokenizer_path.display()
            ));
        }
        tokenizer.validate_document_special_tokens()?;
        let old_vocab_size = tokenizer.vocab_size();
        let upgraded = tokenizer.upgraded_for_audio()?;
        write_parent_dir(output_tokenizer)?;
        let report = expand_safetensors_vocabulary(
            &args.input,
            &args.output,
            old_vocab_size,
            &VocabularyExpansion {
                insertion_id: aarambh_studio_tokenizer::AUDIO_ID as usize,
                source_ids: vec![IMAGE_ID as usize, IMAGE_END_ID as usize],
            },
        )?;
        upgraded.save_pretrained(output_tokenizer)?;
        eprintln!(
            "upgraded audio vocabulary {} -> {} rows across {} tensors; model={} tokenizer={}",
            report.old_vocab_size,
            report.new_vocab_size,
            report.expanded_tensors,
            args.output.display(),
            output_tokenizer.display()
        );
        return Ok(());
    }

    if args.gguf {
        let format = GgufFormat::from_name(&args.format)?;
        let model = load_any_model(&args.input, &run_config.model, &device)?;
        save_gguf(&model, format, &args.output)?;
        eprintln!(
            "converted {} to {:?} GGUF at {}",
            args.input.display(),
            format,
            args.output.display()
        );
        return Ok(());
    }

    let arch = HfArch::from_name(&args.arch)?;
    let model = convert_hf_with_arch(&args.input, &run_config.model, arch, &device)?;
    save_model(&model, &args.output)?;
    eprintln!(
        "converted HF checkpoint {} ({arch:?}) to {}",
        args.input.display(),
        args.output.display()
    );
    Ok(())
}

fn write_parent_dir(path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}
