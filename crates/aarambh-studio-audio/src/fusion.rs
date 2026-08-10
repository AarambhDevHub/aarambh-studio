//! Audio-token fusion helpers.
//!
//! Generalises v2 §24's `interleave_image_tokens` into the shared
//! modal-token-splicing pattern: the single `<audio>` placeholder token in the
//! text sequence is replaced by the projected audio patch embeddings, producing
//! one contiguous sequence the decoder consumes exactly as it consumes text.

use aarambh_studio_core::{AarambhError, Result};
use candle_core::Tensor;

/// Replace one audio placeholder token embedding with projected audio tokens.
///
/// `text_tokens` is the token-id sequence of the prompt (which must contain
/// exactly one `audio_placeholder_id`). `text_embeddings` is the embedding
/// lookup of that sequence with shape `[1, seq, hidden_dim]`.
/// `audio_embeddings` is the projected audio patch tokens with shape
/// `[1, audio_tokens, hidden_dim]`. The result concatenates the text-token
/// embeddings with the audio embeddings spliced in at the placeholder position,
/// yielding shape `[1, seq - 1 + audio_tokens, hidden_dim]`.
pub fn interleave_audio_tokens(
    text_tokens: &[u32],
    text_embeddings: &Tensor,
    audio_embeddings: &Tensor,
    audio_placeholder_id: u32,
) -> Result<Tensor> {
    let text_dims = text_embeddings.dims();
    let audio_dims = audio_embeddings.dims();
    if text_dims.len() != 3 || text_dims[0] != 1 {
        return Err(AarambhError::Shape(format!(
            "text_embeddings must have shape [1, seq, hidden_dim], got {text_dims:?}"
        )));
    }
    if audio_dims.len() != 3 || audio_dims[0] != 1 {
        return Err(AarambhError::Shape(format!(
            "audio_embeddings must have shape [1, audio_tokens, hidden_dim], got {audio_dims:?}"
        )));
    }
    if text_dims[1] != text_tokens.len() {
        return Err(AarambhError::Shape(format!(
            "text token count {} does not match text embedding seq {}",
            text_tokens.len(),
            text_dims[1]
        )));
    }
    if text_dims[2] != audio_dims[2] {
        return Err(AarambhError::Shape(format!(
            "text hidden dim {} does not match audio hidden dim {}",
            text_dims[2], audio_dims[2]
        )));
    }

    let placeholder_count = text_tokens
        .iter()
        .filter(|token| **token == audio_placeholder_id)
        .count();
    if placeholder_count != 1 {
        return Err(AarambhError::Config(format!(
            "expected exactly one audio placeholder token id {audio_placeholder_id}, found {placeholder_count}"
        )));
    }

    let mut parts = Vec::with_capacity(text_tokens.len() + 1);
    for (idx, token_id) in text_tokens.iter().enumerate() {
        if *token_id == audio_placeholder_id {
            parts.push(audio_embeddings.clone());
        } else {
            parts.push(text_embeddings.narrow(1, idx, 1)?);
        }
    }
    Ok(Tensor::cat(&parts, 1)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn audio_token_interleave_produces_expected_sequence_length() {
        let device = Device::Cpu;
        let text_tokens = vec![10u32, 7, 11, 12];
        let text_values = (0..12).map(|value| value as f32).collect::<Vec<_>>();
        let audio_values = vec![100f32, 101., 102., 103., 104., 105., 106., 107., 108.];
        let text = Tensor::from_vec(text_values, (1, 4, 3), &device).unwrap();
        let audio = Tensor::from_vec(audio_values, (1, 3, 3), &device).unwrap();
        let fused = interleave_audio_tokens(&text_tokens, &text, &audio, 7).unwrap();
        // 4 text tokens, one replaced by 3 audio tokens -> 3 + 3 = 6
        assert_eq!(fused.dims(), &[1, 6, 3]);
        let rows = fused.squeeze(0).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(rows[0], vec![0., 1., 2.]);
        assert_eq!(rows[1], vec![100., 101., 102.]);
        assert_eq!(rows[2], vec![103., 104., 105.]);
        assert_eq!(rows[3], vec![106., 107., 108.]);
        assert_eq!(rows[4], vec![6., 7., 8.]);
        assert_eq!(rows[5], vec![9., 10., 11.]);
    }

    #[test]
    fn audio_interleave_rejects_missing_placeholder() {
        let device = Device::Cpu;
        let text_tokens = vec![10u32, 11, 12];
        let text = Tensor::zeros((1, 3, 4), candle_core::DType::F32, &device).unwrap();
        let audio = Tensor::zeros((1, 2, 4), candle_core::DType::F32, &device).unwrap();
        let result = interleave_audio_tokens(&text_tokens, &text, &audio, 7);
        assert!(result.is_err());
    }
}
