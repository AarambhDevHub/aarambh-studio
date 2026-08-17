use std::fs;
use std::path::{Path, PathBuf};

use aarambh_studio_core::TokenizerLike;
use aarambh_studio_data::dataset::{PlaintextDataset, TextDataset};
use aarambh_studio_model::AarambhModel;
use aarambh_studio_quant::{CalibrationStats, GgufFormat, QuantMethod};
use aarambh_studio_tokenizer::BpeTokenizer;
use aarambh_studio_train::TrainingRunConfig;
use aarambh_studio_weights::{load_any_model, save_gguf};
use candle_core::{Device, Tensor};
use clap::Args;

#[derive(Debug, Args)]
/// Quantise a trained checkpoint into a smaller GGUF format.
pub struct QuantiseArgs {
    /// Training/model TOML configuration (provides architecture + device).
    #[arg(long, default_value = "configs/tiny_shakespeare.toml")]
    pub config: PathBuf,
    /// Source model checkpoint path to quantise.
    #[arg(long)]
    pub model: PathBuf,
    /// Optional tokenizer JSON path; falls back to the configured tokenizer.
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,
    /// Quantisation method: int8, awq, gptq, q4_k_m, q5_k_m, or q8_0.
    #[arg(long, default_value = "int8")]
    pub method: String,
    /// Target quantisation bit width (4 for awq/gptq, 5 for q5_k_m, 8 for int8/q8_0).
    #[arg(long, default_value_t = 8)]
    pub bits: u8,
    /// Calibration plaintext dataset path (required for awq and gptq).
    #[arg(long)]
    pub calibration_data: Option<PathBuf>,
    /// Maximum calibration samples to draw from the dataset.
    #[arg(long, default_value_t = 128)]
    pub samples: usize,
    /// Output quantised GGUF checkpoint path.
    #[arg(long)]
    pub output: PathBuf,
}

pub fn run(args: QuantiseArgs) -> anyhow::Result<()> {
    let run_config = TrainingRunConfig::from_toml(&args.config)?;
    let device = run_config.device()?.to_candle()?;
    let method = QuantMethod::from_name(&args.method)?;
    let format = format_for(method, args.bits)?;

    let mut model_config = run_config.model.clone();
    let tokenizer_path = args
        .tokenizer
        .clone()
        .or_else(|| run_config.tokenizer_path.clone())
        .or_else(|| run_config.tokenizer_save_path.clone())
        .unwrap_or_else(|| run_config.train.checkpoint_dir.join("tokenizer.json"));
    if tokenizer_path.exists() {
        let tokenizer = BpeTokenizer::from_pretrained(&tokenizer_path)?;
        tokenizer.validate_special_tokens()?;
        model_config.vocab_size = tokenizer.vocab_size();

        if matches!(method, QuantMethod::AwqInt4 | QuantMethod::GptqInt4) {
            let calibration_path = args.calibration_data.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "--calibration-data is required for {} quantisation",
                    args.method
                )
            })?;
            let dataset = PlaintextDataset::from_file(calibration_path)?;
            let model = load_any_model(&args.model, &model_config, &device)?;
            let stats = run_calibration(
                &model,
                &tokenizer,
                &dataset,
                args.samples,
                model_config.max_seq_len,
                &device,
                matches!(method, QuantMethod::GptqInt4),
            )?;
            eprintln!(
                "calibration: {} layers from up to {} samples",
                stats.layer_names().len(),
                args.samples
            );
            write_parent_dir(&args.output)?;
            save_gguf(&model, format, &args.output)?;
            eprintln!(
                "quantised {:?} checkpoint written to {}",
                format,
                args.output.display()
            );
            return Ok(());
        }
    } else if matches!(method, QuantMethod::AwqInt4 | QuantMethod::GptqInt4) {
        return Err(anyhow::anyhow!(
            "tokenizer {} is required for calibration",
            tokenizer_path.display()
        ));
    }

    let model = load_any_model(&args.model, &model_config, &device)?;
    write_parent_dir(&args.output)?;
    save_gguf(&model, format, &args.output)?;
    eprintln!(
        "quantised {:?} checkpoint written to {}",
        format,
        args.output.display()
    );
    Ok(())
}

fn format_for(method: QuantMethod, bits: u8) -> anyhow::Result<GgufFormat> {
    match (method, bits) {
        (QuantMethod::Int8Absmax | QuantMethod::Q80, 8) => Ok(GgufFormat::Q80),
        (QuantMethod::AwqInt4 | QuantMethod::GptqInt4 | QuantMethod::Q4KM, 4) => {
            Ok(GgufFormat::Q4KM)
        }
        (QuantMethod::Q5KM, 5) => Ok(GgufFormat::Q5KM),
        _ => Err(anyhow::anyhow!(
            "invalid method/bits combination: method={method:?}, bits={bits}; use int8/8 or awq|gptq/4"
        )),
    }
}

fn write_parent_dir(path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn run_calibration(
    model: &AarambhModel,
    tokenizer: &dyn TokenizerLike,
    dataset: &dyn TextDataset,
    n_samples: usize,
    max_seq_len: usize,
    device: &Device,
    with_hessian: bool,
) -> anyhow::Result<CalibrationStats> {
    if n_samples == 0 {
        return Err(anyhow::anyhow!("calibration sample count must be non-zero"));
    }
    let mut stats = CalibrationStats::default();
    let mut seen = 0usize;
    for index in 0..dataset.len() {
        if seen >= n_samples {
            break;
        }
        let mut ids = tokenizer.encode(dataset.get(index))?;
        if ids.len() < 2 {
            continue;
        }
        ids.truncate(max_seq_len.max(1));
        let input = Tensor::from_vec(ids.clone(), (1, ids.len()), device)?;
        for (name, activations) in model.linear_inputs(&input)? {
            stats.observe(&name, &activations, with_hessian)?;
        }
        seen += 1;
    }
    if seen == 0 {
        return Err(anyhow::anyhow!(
            "calibration dataset produced no usable samples"
        ));
    }
    Ok(stats)
}
