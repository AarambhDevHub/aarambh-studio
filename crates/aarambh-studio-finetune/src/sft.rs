use std::fs;
use std::path::Path;

use aarambh_studio_core::{AarambhError, Device, Result, TokenizerLike};
use aarambh_studio_tokenizer::{
    ASSISTANT, ENDOFTEXT, PAD_ID, SYSTEM, THINK_END, THINK_START, USER,
};
use candle_core::Tensor;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Supervised fine-tuning instruction example.
pub struct SftExample {
    /// User instruction.
    pub instruction: String,
    #[serde(default)]
    /// Optional extra input context.
    pub input: Option<String>,
    /// Assistant response.
    pub response: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Supervised fine-tuning example with explicit thinking text.
pub struct ThinkingSftExample {
    /// User instruction.
    pub instruction: String,
    /// Thinking text placed inside thinking markers.
    pub thinking: String,
    /// Assistant response.
    pub response: String,
}

#[derive(Debug, Clone, Default)]
/// Chat template used by SFT and GRPO datasets.
pub struct ChatTemplate;

impl ChatTemplate {
    /// Format a standard SFT example.
    pub fn format(&self, example: &SftExample) -> String {
        format_sft(example)
    }

    /// Format an SFT example containing thinking text.
    pub fn format_with_thinking(&self, example: &ThinkingSftExample) -> String {
        format_thinking_sft(example)
    }

    /// Format the prompt prefix before assistant target tokens.
    pub fn prefix(&self, instruction: &str, input: Option<&str>) -> String {
        let instruction = join_instruction_input(instruction, input);
        format!("{USER}\n{instruction}\n{ASSISTANT}\n")
    }

    /// Format the prompt prefix with a leading operator-set system turn (Phase 52).
    ///
    /// Produces ` IMS\n{system}\n IMS\n{instruction}[\n{input}]\n IMS\n`. The
    /// system turn is optional and single-use: a caller that has no system
    /// instructions uses [`prefix`](Self::prefix) instead, which reproduces the
    /// v1.0.0 ` IMS... IMS` format exactly.
    ///
    /// The existing loss-mask rule ([`build_loss_mask`]) masks every position
    /// before the ` IMS` token, so a leading system turn is excluded from the
    /// loss by construction — no training-code change is needed, only this
    /// documented prefix.
    pub fn prefix_with_system(
        &self,
        system: &str,
        instruction: &str,
        input: Option<&str>,
    ) -> String {
        let instruction = join_instruction_input(instruction, input);
        format!("{SYSTEM}\n{system}\n{USER}\n{instruction}\n{ASSISTANT}\n")
    }

    /// Format the assistant target text.
    pub fn target(&self, response: &str) -> String {
        format!("{response}{ENDOFTEXT}")
    }

    /// Format a thinking target followed by final response text.
    pub fn thinking_target(&self, thinking: &str, response: &str) -> String {
        format!("{THINK_START}\n{thinking}\n{THINK_END}\n{response}{ENDOFTEXT}")
    }
}

/// Format a standard SFT example.
pub fn format_sft(example: &SftExample) -> String {
    let template = ChatTemplate;
    format!(
        "{}{}",
        template.prefix(&example.instruction, example.input.as_deref()),
        template.target(&example.response)
    )
}

/// Format an SFT example with thinking markers.
pub fn format_thinking_sft(example: &ThinkingSftExample) -> String {
    let template = ChatTemplate;
    format!(
        "{}{}",
        template.prefix(&example.instruction, None),
        template.thinking_target(&example.thinking, &example.response)
    )
}

#[derive(Debug, Clone)]
struct SftSequence {
    input_ids: Vec<u32>,
    labels: Vec<u32>,
    loss_mask: Vec<u32>,
}

#[derive(Debug, Clone)]
/// Tokenized supervised fine-tuning dataset.
pub struct SftDataset {
    sequences: Vec<SftSequence>,
    max_seq_len: usize,
}

impl SftDataset {
    /// Load tokenized SFT sequences from JSONL.
    pub fn from_jsonl(
        path: impl AsRef<Path>,
        tokenizer: &dyn TokenizerLike,
        max_seq_len: usize,
    ) -> Result<Self> {
        if max_seq_len == 0 {
            return Err(AarambhError::Config("max_seq_len must be non-zero".into()));
        }
        let content = fs::read_to_string(path.as_ref())?;
        let template = ChatTemplate;
        let mut sequences = Vec::new();
        for (line_idx, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let raw: RawSftRecord = serde_json::from_str(line).map_err(|err| {
                AarambhError::Config(format!("invalid SFT JSONL at line {}: {err}", line_idx + 1))
            })?;
            let prefix = template.prefix(&raw.instruction, raw.input.as_deref());
            let target = match raw.thinking {
                Some(thinking) => template.thinking_target(&thinking, &raw.response),
                None => template.target(&raw.response),
            };
            let sequence = encode_sft_sequence(tokenizer, &prefix, &target, max_seq_len)?;
            sequences.push(sequence);
        }
        if sequences.is_empty() {
            return Err(AarambhError::Config(format!(
                "SFT dataset {} produced no usable examples",
                path.as_ref().display()
            )));
        }
        Ok(Self {
            sequences,
            max_seq_len,
        })
    }

    /// Build a tokenized SFT dataset from examples.
    pub fn from_examples(
        examples: &[SftExample],
        tokenizer: &dyn TokenizerLike,
        max_seq_len: usize,
    ) -> Result<Self> {
        let template = ChatTemplate;
        let mut sequences = Vec::with_capacity(examples.len());
        for example in examples {
            sequences.push(encode_sft_sequence(
                tokenizer,
                &template.prefix(&example.instruction, example.input.as_deref()),
                &template.target(&example.response),
                max_seq_len,
            )?);
        }
        if sequences.is_empty() {
            return Err(AarambhError::Config(
                "SFT dataset must contain at least one example".into(),
            ));
        }
        Ok(Self {
            sequences,
            max_seq_len,
        })
    }

    pub(crate) fn from_masked_id_sequences(
        sequences: &[(Vec<u32>, Vec<u32>)],
        max_seq_len: usize,
    ) -> Result<Self> {
        if max_seq_len == 0 {
            return Err(AarambhError::Config("max_seq_len must be non-zero".into()));
        }
        let mut encoded = Vec::with_capacity(sequences.len());
        for (ids, token_mask) in sequences {
            if ids.len() != token_mask.len() {
                return Err(AarambhError::Config(format!(
                    "masked SFT ids length {} does not match token mask length {}",
                    ids.len(),
                    token_mask.len()
                )));
            }
            if ids.len() > max_seq_len + 1 {
                return Err(AarambhError::Config(format!(
                    "multi-step tool SFT sequence has {} tokens and would truncate at max_seq_len {max_seq_len}",
                    ids.len()
                )));
            }
            if ids.len() < 2 {
                return Err(AarambhError::Config(
                    "masked SFT sequence must contain at least two tokens".into(),
                ));
            }
            if token_mask[1..].iter().all(|value| *value == 0) {
                return Err(AarambhError::Config(
                    "masked SFT sequence has no supervised assistant tokens".into(),
                ));
            }
            encoded.push(SftSequence {
                input_ids: ids[..ids.len() - 1].to_vec(),
                labels: ids[1..].to_vec(),
                loss_mask: token_mask[1..].to_vec(),
            });
        }
        if encoded.is_empty() {
            return Err(AarambhError::Config(
                "masked SFT dataset must contain at least one example".into(),
            ));
        }
        Ok(Self {
            sequences: encoded,
            max_seq_len,
        })
    }

    /// Return the number of tokenized examples.
    pub fn len(&self) -> usize {
        self.sequences.len()
    }

    /// Return true when the dataset has no examples.
    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }

    /// Return the configured maximum sequence length.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}

#[derive(Debug)]
/// Tensor batch for SFT training.
pub struct SftBatch {
    /// Input token ids.
    pub input_ids: Tensor,
    /// Next-token labels.
    pub labels: Tensor,
    /// Mask selecting assistant target positions.
    pub loss_mask: Tensor,
}

/// Mini-batch loader for SFT datasets.
pub struct SftDataLoader {
    sequences: Vec<SftSequence>,
    batch_size: usize,
    max_seq_len: usize,
    shuffle: bool,
    rng: StdRng,
    pos: usize,
    device: Device,
}

impl SftDataLoader {
    /// Create an SFT data loader.
    pub fn new(
        dataset: &SftDataset,
        batch_size: usize,
        shuffle: bool,
        seed: u64,
        device: Device,
    ) -> Result<Self> {
        if batch_size == 0 {
            return Err(AarambhError::Config("batch_size must be non-zero".into()));
        }
        Ok(Self {
            sequences: dataset.sequences.clone(),
            batch_size,
            max_seq_len: dataset.max_seq_len,
            shuffle,
            rng: StdRng::seed_from_u64(seed),
            pos: 0,
            device,
        })
    }

    /// Reset iteration and reshuffle when enabled.
    pub fn reset(&mut self) {
        self.pos = 0;
        if self.shuffle {
            self.sequences.shuffle(&mut self.rng);
        }
    }

    /// Return the number of full batches.
    pub fn len(&self) -> usize {
        self.sequences.len() / self.batch_size
    }

    /// Return true when there are no full batches.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Iterator for SftDataLoader {
    type Item = Result<SftBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.sequences.len() {
            return None;
        }
        let end = (self.pos + self.batch_size).min(self.sequences.len());
        if end - self.pos < self.batch_size {
            self.pos = end;
            return None;
        }
        let batch = &self.sequences[self.pos..end];
        self.pos = end;
        Some(batch_to_tensors(
            batch,
            self.batch_size,
            self.max_seq_len,
            &self.device,
        ))
    }
}

#[derive(Debug, Deserialize)]
struct RawSftRecord {
    instruction: String,
    #[serde(default)]
    input: Option<String>,
    #[serde(default)]
    thinking: Option<String>,
    response: String,
}

fn encode_sft_sequence(
    tokenizer: &dyn TokenizerLike,
    prefix: &str,
    target: &str,
    max_seq_len: usize,
) -> Result<SftSequence> {
    let prefix_ids = tokenizer.encode(prefix)?;
    let target_ids = tokenizer.encode(target)?;
    if target_ids.is_empty() {
        return Err(AarambhError::Config(
            "SFT target encoded to zero tokens".into(),
        ));
    }
    let prefix_len = prefix_ids.len();
    let mut ids = prefix_ids;
    ids.extend(target_ids);
    if ids.len() > max_seq_len + 1 {
        ids.truncate(max_seq_len + 1);
    }
    if ids.len() < 2 {
        return Err(AarambhError::Config(
            "SFT sequence must contain at least two tokens".into(),
        ));
    }

    let input_ids = ids[..ids.len() - 1].to_vec();
    let labels = ids[1..].to_vec();
    let loss_mask = build_loss_mask(prefix_len, ids.len());
    if loss_mask.iter().all(|value| *value == 0) {
        return Err(AarambhError::Config(
            "SFT sequence has no assistant tokens after truncation".into(),
        ));
    }
    Ok(SftSequence {
        input_ids,
        labels,
        loss_mask,
    })
}

/// Build a loss mask that ignores user/prefix tokens.
pub fn build_loss_mask(prefix_len: usize, token_count: usize) -> Vec<u32> {
    if token_count < 2 {
        return Vec::new();
    }
    (0..token_count - 1)
        .map(|idx| u32::from(idx + 1 >= prefix_len))
        .collect()
}

fn batch_to_tensors(
    batch: &[SftSequence],
    batch_size: usize,
    max_seq_len: usize,
    device: &Device,
) -> Result<SftBatch> {
    let candle_device = device.to_candle()?;
    let mut input_ids = Vec::with_capacity(batch_size * max_seq_len);
    let mut labels = Vec::with_capacity(batch_size * max_seq_len);
    let mut loss_mask = Vec::with_capacity(batch_size * max_seq_len);

    for sequence in batch {
        push_padded(&mut input_ids, &sequence.input_ids, max_seq_len, PAD_ID);
        push_padded(&mut labels, &sequence.labels, max_seq_len, PAD_ID);
        push_padded(&mut loss_mask, &sequence.loss_mask, max_seq_len, 0);
    }

    Ok(SftBatch {
        input_ids: Tensor::from_vec(input_ids, (batch_size, max_seq_len), &candle_device)?,
        labels: Tensor::from_vec(labels, (batch_size, max_seq_len), &candle_device)?,
        loss_mask: Tensor::from_vec(loss_mask, (batch_size, max_seq_len), &candle_device)?,
    })
}

fn push_padded(dst: &mut Vec<u32>, values: &[u32], max_len: usize, pad: u32) {
    let take = values.len().min(max_len);
    dst.extend_from_slice(&values[..take]);
    dst.extend(std::iter::repeat_n(pad, max_len - take));
}

fn join_instruction_input(instruction: &str, input: Option<&str>) -> String {
    match input {
        Some(input) if !input.trim().is_empty() => format!("{instruction}\n{input}"),
        _ => instruction.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aarambh_studio_tokenizer::{
        ASSISTANT_ID, BpeTokenizer, ENDOFTEXT_ID, SYSTEM_ID, THINK_END_ID, THINK_START_ID, USER_ID,
        Vocab,
    };
    use std::collections::HashMap;

    #[test]
    fn thinking_sft_format_matches_phase_9_contract() {
        let example = ThinkingSftExample {
            instruction: "What is 23 x 47?".into(),
            thinking: "23 x 40 = 920; 23 x 7 = 161; total = 1081".into(),
            response: "The answer is 1081.".into(),
        };

        let formatted = format_thinking_sft(&example);
        assert_eq!(
            formatted,
            "<|user|>\nWhat is 23 x 47?\n<|assistant|>\n<think>\n23 x 40 = 920; 23 x 7 = 161; total = 1081\n</think>\nThe answer is 1081.<|endoftext|>"
        );
    }

    #[test]
    fn thinking_sft_format_uses_reserved_special_token_ids() {
        let tokenizer = test_tokenizer();
        let formatted = format_thinking_sft(&ThinkingSftExample {
            instruction: "Hi".into(),
            thinking: "Plan".into(),
            response: "Hello".into(),
        });
        let ids = tokenizer.encode(&formatted).unwrap();

        assert!(ids.contains(&USER_ID));
        assert!(ids.contains(&ASSISTANT_ID));
        assert!(ids.contains(&THINK_START_ID));
        assert!(ids.contains(&THINK_END_ID));
        assert!(ids.contains(&ENDOFTEXT_ID));
    }

    #[test]
    fn loss_mask_starts_at_first_target_token() {
        let mask = build_loss_mask(4, 8);
        assert_eq!(mask, vec![0, 0, 0, 1, 1, 1, 1]);
    }

    #[test]
    fn sft_batch_pads_and_masks_prompt_tokens() {
        let tokenizer = test_tokenizer();
        let dataset = SftDataset::from_examples(
            &[SftExample {
                instruction: "Hi".into(),
                input: None,
                response: "Hello".into(),
            }],
            &tokenizer,
            16,
        )
        .unwrap();
        let mut loader = SftDataLoader::new(&dataset, 1, false, 42, Device::Cpu).unwrap();
        let batch = loader.next().unwrap().unwrap();
        let mask = batch.loss_mask.to_vec2::<u32>().unwrap();
        assert!(mask[0].contains(&1));
        assert_eq!(mask[0].last().copied(), Some(0));
    }

    #[test]
    fn sft_loss_mask_correctly_covers_a_leading_system_turn() {
        // Phase 52: a leading operator-set system turn must be excluded from the
        // SFT loss. `build_loss_mask` masks every position before the assistant
        // (` IMS`) position, so a system turn placed before the user turn is
        // covered by construction — this test pins that invariant.
        let tokenizer = test_tokenizer();
        let template = ChatTemplate;
        let prefix = template.prefix_with_system("Hi", "Plan", None);
        let target = template.target("Hello");

        let prefix_ids = tokenizer.encode(&prefix).unwrap();
        let target_ids = tokenizer.encode(&target).unwrap();
        assert!(
            prefix_ids.contains(&SYSTEM_ID),
            "prefix must contain the system marker token, got {prefix_ids:?}"
        );

        let prefix_len = prefix_ids.len();
        let mut ids = prefix_ids.clone();
        ids.extend(target_ids);
        let mask = build_loss_mask(prefix_len, ids.len());

        // The system marker sits inside the prefix, so its position is masked.
        let system_pos = ids
            .iter()
            .position(|&id| id == SYSTEM_ID)
            .expect("system marker present in the sequence");
        assert!(
            system_pos < prefix_len,
            "system marker must live in the masked prefix region"
        );
        // mask[i] corresponds to predicting token i+1. When the system marker is
        // not the very first token, the position predicting it (system_pos - 1)
        // must carry no loss. When it is the first token there is no predicting
        // position to check — it is trivially outside the supervised region.
        if system_pos > 0 {
            assert_eq!(
                mask[system_pos - 1],
                0,
                "no loss on the system-turn position"
            );
        }

        // Every position before the assistant target is masked (0); every
        // position from the assistant target onward is supervised (1).
        for (idx, &value) in mask.iter().enumerate() {
            if idx + 1 < prefix_len {
                assert_eq!(value, 0, "prefix position {idx} must be masked");
            } else {
                assert_eq!(value, 1, "target position {idx} must be supervised");
            }
        }
    }

    fn test_tokenizer() -> BpeTokenizer {
        let mut token_to_id = HashMap::from([
            (USER.to_string(), USER_ID),
            (ASSISTANT.to_string(), ASSISTANT_ID),
            (SYSTEM.to_string(), SYSTEM_ID),
            (THINK_START.to_string(), THINK_START_ID),
            (THINK_END.to_string(), THINK_END_ID),
            (ENDOFTEXT.to_string(), ENDOFTEXT_ID),
        ]);
        let mut id_to_token = vec![String::new(); 32];
        for (token, id) in &token_to_id {
            id_to_token[*id as usize] = token.clone();
        }
        for (token, id) in [
            ("H", 7),
            ("i", 8),
            ("P", 9),
            ("l", 10),
            ("a", 11),
            ("n", 12),
            ("e", 13),
            ("o", 14),
        ] {
            token_to_id.insert(token.to_string(), id);
            id_to_token[id as usize] = token.to_string();
        }
        BpeTokenizer {
            vocab: Vocab {
                token_to_id,
                id_to_token,
            },
            merges: vec![],
            merge_rank: HashMap::new(),
            chat_template_version: None,
        }
    }
}
