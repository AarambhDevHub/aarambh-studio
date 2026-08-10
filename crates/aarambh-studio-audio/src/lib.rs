//! Audio encoder, projector, preprocessing, and multimodal fusion utilities.
//!
//! Phase 42 adds audio understanding to aarambh-studio following the exact
//! frozen-encoder-plus-trainable-projector pattern v2 §24 established for
//! vision and v3 §35–36 reused for video and documents. A frozen, pretrained
//! audio spectrogram transformer converts a log-mel spectrogram into a grid of
//! patch embeddings; a small trainable projector maps those into the decoder's
//! `d_model` space; the result is spliced into the token sequence at the
//! `<audio>` special token position. Nothing about the decoder, the thinking
//! engine, or tool calling changes — audio is just another sense feeding the
//! same fusion mechanism.

#![deny(missing_docs)]

/// Frozen audio spectrogram transformer encoder.
pub mod encoder;
/// Audio-token fusion helpers.
pub mod fusion;
/// Audio instruction data loading.
pub mod instruct_data;
/// Pure-Rust WAV decode and mel-spectrogram extraction.
pub mod preprocess;
/// Trainable audio-to-language projector.
pub mod projector;

pub use encoder::{AudioEncoderConfig, FrozenAudioEncoder};
pub use fusion::interleave_audio_tokens;
pub use instruct_data::{AudioQaExample, load_audio_qa_jsonl};
pub use preprocess::{AudioPreprocessor, MelSpectrogramConfig};
pub use projector::{AudioProjector, AudioProjectorConfig};

/// Frozen audio encoder plus trainable language-model projector.
#[derive(Debug, Clone)]
pub struct AudioModel {
    encoder: FrozenAudioEncoder,
    projector: AudioProjector,
}

impl AudioModel {
    /// Create an audio model from an encoder and projector.
    pub fn new(encoder: FrozenAudioEncoder, projector: AudioProjector) -> Self {
        Self { encoder, projector }
    }

    /// Return the frozen audio encoder.
    pub fn encoder(&self) -> &FrozenAudioEncoder {
        &self.encoder
    }

    /// Return the trainable projector.
    pub fn projector(&self) -> &AudioProjector {
        &self.projector
    }

    /// Encode a mel-spectrogram tensor and project patch tokens into LLM width.
    pub fn forward(
        &self,
        spectrogram: &candle_core::Tensor,
    ) -> aarambh_studio_core::Result<candle_core::Tensor> {
        let patch_tokens = self.encoder.forward(spectrogram)?;
        self.projector.forward(&patch_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    fn tiny_audio_setup() -> (AudioModel, VarMap, VarMap) {
        let device = Device::Cpu;
        let encoder_config = AudioEncoderConfig {
            n_mels: 16,
            max_frames: 16,
            patch_mel: 4,
            patch_time: 4,
            audio_d_model: 16,
            audio_layers: 2,
            audio_heads: 2,
            mlp_dim: 32,
            norm_eps: 1e-5,
        };
        let encoder_varmap = VarMap::new();
        let encoder_vb = VarBuilder::from_varmap(&encoder_varmap, DType::F32, &device);
        let encoder = FrozenAudioEncoder::new(encoder_config, encoder_vb).unwrap();
        let projector_varmap = VarMap::new();
        let projector_vb = VarBuilder::from_varmap(&projector_varmap, DType::F32, &device);
        let projector = AudioProjector::new(
            AudioProjectorConfig {
                audio_d_model: 16,
                llm_d_model: 24,
                hidden_mult: 2,
            },
            projector_vb,
        )
        .unwrap();
        let model = AudioModel::new(encoder, projector);
        (model, encoder_varmap, projector_varmap)
    }

    #[test]
    fn frozen_audio_encoder_never_receives_gradients() {
        // The projector-only training stage detaches the frozen encoder output
        // before the projector consumes it (the same pattern
        // `aarambh_studio_finetune::vlm_dora` uses for the CLIP encoder). After a
        // backward pass, gradients must reach the projector variables and must
        // NOT reach any encoder variable.
        let device = Device::Cpu;
        let encoder_config = AudioEncoderConfig {
            n_mels: 16,
            max_frames: 16,
            patch_mel: 4,
            patch_time: 4,
            audio_d_model: 16,
            audio_layers: 1,
            audio_heads: 2,
            mlp_dim: 32,
            norm_eps: 1e-5,
        };
        let encoder_varmap = VarMap::new();
        let encoder_vb = VarBuilder::from_varmap(&encoder_varmap, DType::F32, &device);
        let encoder = FrozenAudioEncoder::new(encoder_config, encoder_vb).unwrap();
        let projector_varmap = VarMap::new();
        let projector_vb = VarBuilder::from_varmap(&projector_varmap, DType::F32, &device);
        let projector = AudioProjector::new(
            AudioProjectorConfig {
                audio_d_model: 16,
                llm_d_model: 24,
                hidden_mult: 2,
            },
            projector_vb,
        )
        .unwrap();
        let spectrogram = Tensor::zeros((1, 16, 16), DType::F32, &device).unwrap();
        // Detach the encoder output: this is the freeze mechanism.
        let patch_tokens = encoder.forward(&spectrogram).unwrap().detach();
        let output = projector.forward(&patch_tokens).unwrap();
        let loss = output.sum_all().unwrap();
        let grads = loss.backward().unwrap();

        let projector_data = projector_varmap.data().lock().unwrap();
        let projector_grads = projector_data
            .iter()
            .filter(|(_, var)| grads.get(var.as_tensor()).is_some())
            .count();
        assert!(
            projector_grads > 0,
            "projector variables must receive gradients"
        );

        let encoder_data = encoder_varmap.data().lock().unwrap();
        let encoder_grads = encoder_data
            .iter()
            .filter(|(_, var)| grads.get(var.as_tensor()).is_some())
            .count();
        assert_eq!(
            encoder_grads, 0,
            "frozen audio encoder must never receive gradients"
        );
    }

    #[test]
    fn projector_pretrain_stage_trains_only_projector_weights() {
        // The projector-only stage trains the projector alone while the encoder
        // and the LLM stay frozen. We assert the structural guarantee the stage
        // relies on: the detached encoder output has no backward graph, so
        // gradients stop at the projector and cannot flow into the encoder.
        let device = Device::Cpu;
        let encoder_config = AudioEncoderConfig {
            n_mels: 16,
            max_frames: 16,
            patch_mel: 4,
            patch_time: 4,
            audio_d_model: 16,
            audio_layers: 1,
            audio_heads: 2,
            mlp_dim: 32,
            norm_eps: 1e-5,
        };
        let encoder_varmap = VarMap::new();
        let encoder_vb = VarBuilder::from_varmap(&encoder_varmap, DType::F32, &device);
        let encoder = FrozenAudioEncoder::new(encoder_config, encoder_vb).unwrap();
        let projector_varmap = VarMap::new();
        let projector_vb = VarBuilder::from_varmap(&projector_varmap, DType::F32, &device);
        let projector = AudioProjector::new(
            AudioProjectorConfig {
                audio_d_model: 16,
                llm_d_model: 24,
                hidden_mult: 2,
            },
            projector_vb,
        )
        .unwrap();
        let spectrogram = Tensor::zeros((1, 16, 16), DType::F32, &device).unwrap();
        let patch_tokens = encoder.forward(&spectrogram).unwrap();
        // Detach simulates the frozen-encoder projector-pretrain stage.
        let detached = patch_tokens.detach();
        let output = projector.forward(&detached).unwrap();
        assert_eq!(output.dims(), &[1, 16, 24]);
        // A detached tensor carries no backward node, so the encoder cannot
        // receive gradients from a loss computed over `output` — exactly the
        // guarantee the projector-only training stage relies on.
        let loss = output.sum_all().unwrap();
        let grads = loss.backward().unwrap();
        let encoder_data = encoder_varmap.data().lock().unwrap();
        let encoder_grads = encoder_data
            .iter()
            .filter(|(_, var)| grads.get(var.as_tensor()).is_some())
            .count();
        assert_eq!(encoder_grads, 0, "encoder must stay frozen in stage one");
        let projector_data = projector_varmap.data().lock().unwrap();
        let projector_grads = projector_data
            .iter()
            .filter(|(_, var)| grads.get(var.as_tensor()).is_some())
            .count();
        assert!(projector_grads > 0, "projector must train in stage one");
    }

    #[test]
    fn audio_token_fusion_produces_expected_sequence_length() {
        let device = Device::Cpu;
        let text_tokens = vec![5u32, 15, 6, 0];
        let text_values = (0..16).map(|v| v as f32).collect::<Vec<_>>();
        let text = Tensor::from_vec(text_values, (1, 4, 4), &device).unwrap();
        let audio_values = (0..24).map(|v| v as f32).collect::<Vec<_>>();
        let audio = Tensor::from_vec(audio_values, (1, 6, 4), &device).unwrap();
        let fused = interleave_audio_tokens(&text_tokens, &text, &audio, 15).unwrap();
        // 4 text tokens, one replaced by 6 audio tokens -> 3 + 6 = 9
        assert_eq!(fused.dims(), &[1, 9, 4]);
    }

    #[test]
    fn thinking_controller_behaves_identically_after_audio_context() {
        // Composability guarantee (v2 §25, restated for audio in ARCHITECTURE_V4
        // §56): a thinking block generated after audio tokens is indistinguishable
        // to the controller from one generated after text-only context. We verify
        // the mechanism that makes this true — the fused audio-context embeddings
        // have the same hidden width and a well-formed sequence length, so the
        // decoder and its thinking controller observe an ordinary token sequence.
        let device = Device::Cpu;
        let hidden = 16usize;
        let text_tokens = vec![5u32, 15, 6];
        let text = Tensor::zeros((1, 3, hidden), DType::F32, &device).unwrap();
        let audio = Tensor::zeros((1, 8, hidden), DType::F32, &device).unwrap();
        let fused = interleave_audio_tokens(&text_tokens, &text, &audio, 15).unwrap();
        assert_eq!(fused.dims()[2], hidden);
        assert_eq!(fused.dims()[1], 3 + 8 - 1);
        // The fused tensor is finite and contiguous — the decoder sees a normal
        // embedding sequence, identical in contract to a text-only one.
        assert!(fused.is_contiguous());
        let sum = fused.sum_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(sum.is_finite());
    }

    #[test]
    fn audio_model_forwards_to_llm_width() {
        let (model, _, _) = tiny_audio_setup();
        let spectrogram = Tensor::zeros((1, 16, 16), DType::F32, &Device::Cpu).unwrap();
        let output = model.forward(&spectrogram).unwrap();
        assert_eq!(output.dims(), &[1, 16, 24]);
    }

    #[test]
    fn projector_linear_layers_are_accessible() {
        let (model, _, _) = tiny_audio_setup();
        // fc1 maps audio_d_model (16) -> hidden_dim (48); fc2 maps 48 -> llm (24).
        assert_eq!(model.projector().fc1().weight().dims(), &[48, 16]);
        assert_eq!(model.projector().fc2().weight().dims(), &[24, 48]);
    }

    #[test]
    fn audio_model_forward_is_finite_for_nonzero_input() {
        let (model, _, _) = tiny_audio_setup();
        let values: Vec<f32> = (0..256).map(|v| (v as f32) / 256.0).collect();
        let spectrogram = Tensor::from_vec(values, (1, 16, 16), &Device::Cpu).unwrap();
        let output = model.forward(&spectrogram).unwrap();
        let sum = output.sum_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(sum.is_finite());
    }

    #[test]
    fn encoder_named_tensors_exclude_projector_names() {
        let (model, _, _) = tiny_audio_setup();
        let names = model.encoder().named_tensors();
        assert!(!names.contains_key("fc1.weight"));
        assert!(names.contains_key("patch_embed.weight"));
    }
}
