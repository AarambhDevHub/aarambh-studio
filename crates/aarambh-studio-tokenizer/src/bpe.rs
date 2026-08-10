use std::collections::HashMap;
use std::path::Path;

use aarambh_studio_core::{AarambhError, Result, TokenizerLike};

use crate::special;
use crate::vocab::Vocab;

#[derive(Debug, Clone)]
/// Pure-Rust BPE tokenizer loaded from or saved to tokenizer JSON.
pub struct BpeTokenizer {
    /// Token-to-id and id-to-token vocabulary.
    pub vocab: Vocab,
    /// Ordered BPE merge pairs.
    pub merges: Vec<(String, String)>,
    /// Rank lookup for merge pairs.
    pub merge_rank: HashMap<(String, String), usize>,
}

impl BpeTokenizer {
    /// Verify that another tokenizer has the same vocabulary and merge rules.
    pub fn validate_compatible(&self, other: &Self) -> Result<()> {
        if self.vocab.token_to_id != other.vocab.token_to_id {
            return Err(AarambhError::Tokenizer(
                "draft and target tokenizer vocabularies do not match".into(),
            ));
        }
        if self.vocab.id_to_token != other.vocab.id_to_token {
            return Err(AarambhError::Tokenizer(
                "draft and target tokenizer token ids do not match".into(),
            ));
        }
        if self.merges != other.merges {
            return Err(AarambhError::Tokenizer(
                "draft and target tokenizer merge rules do not match".into(),
            ));
        }
        self.validate_special_tokens()?;
        other.validate_special_tokens()?;
        Ok(())
    }

    /// Train a BPE tokenizer from a whitespace corpus file.
    pub fn train(corpus_path: impl AsRef<Path>, vocab_size: usize) -> Result<Self> {
        use tokenizers::models::bpe::{BPE, BpeTrainer};
        use tokenizers::tokenizer::{Tokenizer as HfTokenizer, Trainer};

        if vocab_size <= special::SPECIAL_TOKEN_COUNT {
            return Err(AarambhError::Tokenizer(format!(
                "vocab_size must be greater than {} to reserve special tokens",
                special::SPECIAL_TOKEN_COUNT
            )));
        }

        let content = std::fs::read_to_string(corpus_path.as_ref()).map_err(AarambhError::Io)?;

        let words: Vec<String> = content.split_whitespace().map(String::from).collect();

        let learned_vocab_size = vocab_size - special::SPECIAL_TOKEN_COUNT;
        let mut trainer = BpeTrainer::new(2, learned_vocab_size);
        trainer
            .feed(words.iter(), |s| Ok(vec![s.to_owned()]))
            .map_err(|e| AarambhError::Tokenizer(e.to_string()))?;

        let mut model = BPE::default();
        trainer
            .train(&mut model)
            .map_err(|e| AarambhError::Tokenizer(e.to_string()))?;

        let hf = HfTokenizer::new(model);
        let tmp = std::env::temp_dir().join(format!("aarambh_bpe_{}.json", std::process::id()));
        hf.save(&tmp, false)
            .map_err(|e| AarambhError::Tokenizer(format!("Save failed: {e}")))?;

        let result = Self::from_pretrained(&tmp).and_then(|tokenizer| {
            tokenizer.with_reserved_special_tokens(special::SPECIAL_TOKEN_COUNT as u32)
        });
        let _ = std::fs::remove_file(&tmp);
        result
    }

    /// Load a HuggingFace-compatible `tokenizer.json` file.
    pub fn from_pretrained(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())?;
        let json: serde_json::Value = serde_json::from_str(&content).map_err(AarambhError::Json)?;

        let vocab_obj = json["model"]["vocab"].as_object().ok_or_else(|| {
            AarambhError::Tokenizer("missing model.vocab in tokenizer.json".into())
        })?;

        let token_to_id: HashMap<String, u32> = vocab_obj
            .iter()
            .map(|(k, v)| (k.clone(), v.as_u64().unwrap_or(0) as u32))
            .collect();

        let merges: Vec<(String, String)> = json["model"]["merges"]
            .as_array()
            .ok_or_else(|| {
                AarambhError::Tokenizer("missing model.merges in tokenizer.json".into())
            })?
            .iter()
            .filter_map(parse_merge)
            .collect();

        let max_id = token_to_id.values().copied().max().unwrap_or(0);
        let mut id_to_token: Vec<String> = vec![String::new(); (max_id + 1) as usize];
        for (token, id) in &token_to_id {
            id_to_token[*id as usize] = token.clone();
        }

        let merge_rank: HashMap<(String, String), usize> = merges
            .iter()
            .enumerate()
            .map(|(i, (a, b))| ((a.clone(), b.clone()), i))
            .collect();

        let vocab = Vocab {
            token_to_id,
            id_to_token,
        };

        Ok(Self {
            vocab,
            merges,
            merge_rank,
        })
    }

    /// Save only the vocabulary JSON.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        self.vocab.save_json(path)
    }

    /// Save a HuggingFace-compatible tokenizer JSON.
    pub fn save_pretrained(&self, path: impl AsRef<Path>) -> Result<()> {
        let vocab = self
            .vocab
            .token_to_id
            .iter()
            .map(|(token, id)| (token.clone(), serde_json::Value::from(*id)))
            .collect::<serde_json::Map<_, _>>();
        let merges = self
            .merges
            .iter()
            .map(|(left, right)| {
                serde_json::Value::Array(vec![
                    serde_json::Value::from(left.clone()),
                    serde_json::Value::from(right.clone()),
                ])
            })
            .collect::<Vec<_>>();
        let json = serde_json::json!({
            "model": {
                "type": "BPE",
                "vocab": vocab,
                "merges": merges,
            }
        });
        let content = serde_json::to_string_pretty(&json).map_err(AarambhError::Json)?;
        std::fs::write(path.as_ref(), content).map_err(AarambhError::Io)?;
        Ok(())
    }

    /// Verify that reserved Aarambh special tokens use their required ids.
    pub fn validate_special_tokens(&self) -> Result<()> {
        for (token, id) in special::TEXT_SPECIAL_TOKENS {
            self.validate_special_token(token, id)?;
        }
        Ok(())
    }

    /// Verify that multimodal v2 special tokens use their required ids.
    pub fn validate_vision_special_tokens(&self) -> Result<()> {
        for (token, id) in special::VISION_SPECIAL_TOKENS {
            self.validate_special_token(token, id)?;
        }
        Ok(())
    }

    /// Verify that Phase 35 video tokens and all earlier special tokens use their required ids.
    pub fn validate_video_special_tokens(&self) -> Result<()> {
        for (token, id) in special::VIDEO_SPECIAL_TOKENS {
            self.validate_special_token(token, id)?;
        }
        Ok(())
    }

    /// Verify that Phase 36 document tokens and all earlier special tokens use required ids.
    pub fn validate_document_special_tokens(&self) -> Result<()> {
        for (token, id) in special::SPECIAL_TOKENS {
            self.validate_special_token(token, id)?;
        }
        Ok(())
    }

    /// Verify that Phase 42 audio tokens and all earlier special tokens use required ids.
    pub fn validate_audio_special_tokens(&self) -> Result<()> {
        for (token, id) in special::AUDIO_SPECIAL_TOKENS {
            self.validate_special_token(token, id)?;
        }
        Ok(())
    }

    /// Return an image-capable tokenizer upgraded to the Phase 35 video vocabulary layout.
    ///
    /// Existing learned token ids beginning at 9 are shifted by three. Callers must apply
    /// the same row migration to the model embedding and untied language-model head.
    pub fn upgraded_for_video(&self) -> Result<Self> {
        self.validate_vision_special_tokens()?;
        if self.validate_video_special_tokens().is_ok() {
            return Ok(self.clone());
        }
        for (token, _) in special::VIDEO_SPECIAL_TOKENS
            .iter()
            .skip(special::VISION_SPECIAL_TOKENS.len())
        {
            if let Some(id) = self.vocab.get_id(token) {
                return Err(AarambhError::Tokenizer(format!(
                    "cannot reserve video token {token:?}: it already exists at id {id}"
                )));
            }
        }

        let insertion = special::VIDEO_ID;
        let added =
            (special::VIDEO_SPECIAL_TOKENS.len() - special::VISION_SPECIAL_TOKENS.len()) as u32;
        let mut token_to_id = HashMap::with_capacity(self.vocab.token_to_id.len() + added as usize);
        for (token, id) in &self.vocab.token_to_id {
            let migrated = if *id >= insertion { *id + added } else { *id };
            token_to_id.insert(token.clone(), migrated);
        }
        for (token, id) in special::VIDEO_SPECIAL_TOKENS
            .iter()
            .skip(special::VISION_SPECIAL_TOKENS.len())
        {
            token_to_id.insert((*token).to_string(), *id);
        }

        let max_id = token_to_id.values().copied().max().unwrap_or(0);
        let mut id_to_token = vec![String::new(); (max_id + 1) as usize];
        for (token, id) in &token_to_id {
            id_to_token[*id as usize] = token.clone();
        }
        let upgraded = Self {
            vocab: Vocab {
                token_to_id,
                id_to_token,
            },
            merges: self.merges.clone(),
            merge_rank: self.merge_rank.clone(),
        };
        upgraded.validate_video_special_tokens()?;
        Ok(upgraded)
    }

    /// Return a video-capable tokenizer upgraded to the Phase 36 document layout.
    ///
    /// Existing learned token ids beginning at 12 are shifted by three. Callers must
    /// apply the same migration to model embeddings and an untied language-model head.
    pub fn upgraded_for_document(&self) -> Result<Self> {
        self.validate_video_special_tokens()?;
        if self.validate_document_special_tokens().is_ok() {
            return Ok(self.clone());
        }
        for (token, _) in special::SPECIAL_TOKENS
            .iter()
            .skip(special::VIDEO_SPECIAL_TOKENS.len())
        {
            if let Some(id) = self.vocab.get_id(token) {
                return Err(AarambhError::Tokenizer(format!(
                    "cannot reserve document token {token:?}: it already exists at id {id}"
                )));
            }
        }

        let insertion = special::DOCUMENT_ID;
        let added = (special::SPECIAL_TOKENS.len() - special::VIDEO_SPECIAL_TOKENS.len()) as u32;
        let mut token_to_id = HashMap::with_capacity(self.vocab.token_to_id.len() + added as usize);
        for (token, id) in &self.vocab.token_to_id {
            let migrated = if *id >= insertion { *id + added } else { *id };
            token_to_id.insert(token.clone(), migrated);
        }
        for (token, id) in special::SPECIAL_TOKENS
            .iter()
            .skip(special::VIDEO_SPECIAL_TOKENS.len())
        {
            token_to_id.insert((*token).to_string(), *id);
        }

        let max_id = token_to_id.values().copied().max().unwrap_or(0);
        let mut id_to_token = vec![String::new(); (max_id + 1) as usize];
        for (token, id) in &token_to_id {
            id_to_token[*id as usize] = token.clone();
        }
        let upgraded = Self {
            vocab: Vocab {
                token_to_id,
                id_to_token,
            },
            merges: self.merges.clone(),
            merge_rank: self.merge_rank.clone(),
        };
        upgraded.validate_document_special_tokens()?;
        Ok(upgraded)
    }

    /// Return a document-capable tokenizer upgraded to the Phase 42 audio layout.
    ///
    /// Existing learned token ids beginning at 15 are shifted by two. Callers must
    /// apply the same migration to model embeddings and an untied language-model head.
    pub fn upgraded_for_audio(&self) -> Result<Self> {
        self.validate_document_special_tokens()?;
        if self.validate_audio_special_tokens().is_ok() {
            return Ok(self.clone());
        }
        for (token, _) in special::AUDIO_SPECIAL_TOKENS
            .iter()
            .skip(special::SPECIAL_TOKENS.len())
        {
            if let Some(id) = self.vocab.get_id(token) {
                return Err(AarambhError::Tokenizer(format!(
                    "cannot reserve audio token {token:?}: it already exists at id {id}"
                )));
            }
        }

        let insertion = special::AUDIO_ID;
        let added = (special::AUDIO_SPECIAL_TOKENS.len() - special::SPECIAL_TOKENS.len()) as u32;
        let mut token_to_id = HashMap::with_capacity(self.vocab.token_to_id.len() + added as usize);
        for (token, id) in &self.vocab.token_to_id {
            let migrated = if *id >= insertion { *id + added } else { *id };
            token_to_id.insert(token.clone(), migrated);
        }
        for (token, id) in special::AUDIO_SPECIAL_TOKENS
            .iter()
            .skip(special::SPECIAL_TOKENS.len())
        {
            token_to_id.insert((*token).to_string(), *id);
        }

        let max_id = token_to_id.values().copied().max().unwrap_or(0);
        let mut id_to_token = vec![String::new(); (max_id + 1) as usize];
        for (token, id) in &token_to_id {
            id_to_token[*id as usize] = token.clone();
        }
        let upgraded = Self {
            vocab: Vocab {
                token_to_id,
                id_to_token,
            },
            merges: self.merges.clone(),
            merge_rank: self.merge_rank.clone(),
        };
        upgraded.validate_audio_special_tokens()?;
        Ok(upgraded)
    }

    fn validate_special_token(&self, token: &str, id: u32) -> Result<()> {
        if self.vocab.get_id(token) != Some(id) {
            return Err(AarambhError::Tokenizer(format!(
                "special token {token:?} must have id {id}"
            )));
        }
        if self.vocab.get_token(id) != Some(token) {
            return Err(AarambhError::Tokenizer(format!(
                "special id {id} must decode to {token:?}"
            )));
        }
        Ok(())
    }

    fn with_reserved_special_tokens(self, offset: u32) -> Result<Self> {
        let mut token_to_id =
            HashMap::with_capacity(self.vocab.token_to_id.len() + offset as usize);
        let mut max_id = 0u32;

        for (token, id) in special::SPECIAL_TOKENS {
            token_to_id.insert(token.to_string(), id);
            max_id = max_id.max(id);
        }

        for (token, id) in self.vocab.token_to_id {
            if special::SPECIAL_TOKENS
                .iter()
                .any(|(special_token, _)| *special_token == token)
            {
                continue;
            }
            let shifted = id + offset;
            max_id = max_id.max(shifted);
            token_to_id.insert(token, shifted);
        }

        let mut id_to_token = vec![String::new(); (max_id + 1) as usize];
        for (token, id) in &token_to_id {
            id_to_token[*id as usize] = token.clone();
        }

        let tokenizer = Self {
            vocab: Vocab {
                token_to_id,
                id_to_token,
            },
            merges: self.merges,
            merge_rank: self.merge_rank,
        };
        tokenizer.validate_document_special_tokens()?;
        Ok(tokenizer)
    }

    fn encode_regular_text(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for word in text.split_inclusive(|c: char| c.is_whitespace()) {
            ids.extend(self.encode_word(word));
        }
        ids
    }

    fn encode_word(&self, word: &str) -> Vec<u32> {
        let mut symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();

        loop {
            if symbols.len() <= 1 {
                break;
            }

            let mut best_idx = None;
            let mut best_rank = usize::MAX;

            for i in 0..symbols.len() - 1 {
                let pair = (symbols[i].clone(), symbols[i + 1].clone());
                if let Some(rank) = self.merge_rank.get(&pair)
                    && *rank < best_rank
                {
                    best_rank = *rank;
                    best_idx = Some(i);
                }
            }

            match best_idx {
                Some(idx) => {
                    let merged = format!("{}{}", symbols[idx], symbols[idx + 1]);
                    symbols[idx] = merged;
                    symbols.remove(idx + 1);
                }
                None => break,
            }
        }

        symbols
            .into_iter()
            .filter_map(|s| self.vocab.token_to_id.get(&s).copied())
            .collect()
    }
}

fn parse_merge(value: &serde_json::Value) -> Option<(String, String)> {
    if let Some(s) = value.as_str() {
        let parts = s.split_once(' ')?;
        return Some((parts.0.to_string(), parts.1.to_string()));
    }

    let array = value.as_array()?;
    if array.len() != 2 {
        return None;
    }
    Some((
        array[0].as_str()?.to_string(),
        array[1].as_str()?.to_string(),
    ))
}

impl TokenizerLike for BpeTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        let mut rest = text;

        while !rest.is_empty() {
            let next_special = special::AUDIO_SPECIAL_TOKENS
                .iter()
                .filter(|(token, id)| self.vocab.get_id(token) == Some(*id))
                .filter_map(|(token, id)| rest.find(token).map(|pos| (pos, *token, *id)))
                .min_by_key(|(pos, _, _)| *pos);

            match next_special {
                Some((0, token, id)) => {
                    ids.push(id);
                    rest = &rest[token.len()..];
                }
                Some((pos, _, _)) => {
                    ids.extend(self.encode_regular_text(&rest[..pos]));
                    rest = &rest[pos..];
                }
                None => {
                    ids.extend(self.encode_regular_text(rest));
                    break;
                }
            }
        }
        Ok(ids)
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut text = String::new();
        for &id in ids {
            match self.vocab.get_token(id) {
                Some(t) => text.push_str(t),
                None => {
                    return Err(AarambhError::Tokenizer(format!("unknown token id: {id}")));
                }
            }
        }
        Ok(text)
    }

    fn vocab_size(&self) -> usize {
        self.vocab.id_to_token.len()
    }

    fn eos_token_id(&self) -> u32 {
        special::ENDOFTEXT_ID
    }

    fn bos_token_id(&self) -> Option<u32> {
        Some(special::BOS_ID)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::special::AUDIO_SPECIAL_TOKENS;
    use aarambh_studio_core::TokenizerLike;

    fn audio_tokenizer() -> BpeTokenizer {
        // Build a minimal audio-capable tokenizer: the 17 reserved special tokens
        // (ids 0..=16) plus two learned tokens (ids 17, 18).
        let mut token_to_id = std::collections::HashMap::new();
        for (token, id) in AUDIO_SPECIAL_TOKENS {
            token_to_id.insert((*token).to_string(), id);
        }
        token_to_id.insert("hello".to_string(), 17);
        token_to_id.insert("world".to_string(), 18);
        let max_id = *token_to_id.values().max().unwrap();
        let mut id_to_token = vec![String::new(); (max_id + 1) as usize];
        for (token, id) in &token_to_id {
            id_to_token[*id as usize] = token.clone();
        }
        BpeTokenizer {
            vocab: Vocab {
                token_to_id,
                id_to_token,
            },
            merges: Vec::new(),
            merge_rank: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn audio_tokenizer_encodes_audio_placeholder_as_single_token() {
        // Regression test for Phase 42: the encoder must recognize <audio> as a
        // single special token (id 15), not BPE-split it into characters. Before
        // the encode path used AUDIO_SPECIAL_TOKENS, <audio> was invisible to the
        // special-token splitter and the placeholder was never emitted.
        let tokenizer = audio_tokenizer();
        tokenizer.validate_audio_special_tokens().unwrap();
        let ids = tokenizer.encode("prefix <audio> suffix").unwrap();
        assert!(
            ids.contains(&special::AUDIO_ID),
            "encode must emit AUDIO id 15, got {ids:?}"
        );
        // The placeholder must appear exactly once.
        assert_eq!(
            ids.iter().filter(|&&id| id == special::AUDIO_ID).count(),
            1,
            "exactly one audio placeholder, got {ids:?}"
        );
    }

    #[test]
    fn audio_tokenizer_encodes_audio_end_boundary() {
        let tokenizer = audio_tokenizer();
        let ids = tokenizer.encode("<audio><audio_end>").unwrap();
        assert_eq!(ids, vec![special::AUDIO_ID, special::AUDIO_END_ID]);
    }

    #[test]
    fn text_only_tokenizer_does_not_match_audio_placeholder() {
        // A text-only tokenizer (no audio tokens in vocab) must not treat the
        // literal string "<audio>" as a special token — it BPE-splits it.
        let mut token_to_id = std::collections::HashMap::new();
        for (token, id) in special::TEXT_SPECIAL_TOKENS {
            token_to_id.insert((*token).to_string(), id);
        }
        token_to_id.insert("a".to_string(), 7);
        let max_id = *token_to_id.values().max().unwrap();
        let mut id_to_token = vec![String::new(); (max_id + 1) as usize];
        for (token, id) in &token_to_id {
            id_to_token[*id as usize] = token.clone();
        }
        let tokenizer = BpeTokenizer {
            vocab: Vocab {
                token_to_id,
                id_to_token,
            },
            merges: Vec::new(),
            merge_rank: std::collections::HashMap::new(),
        };
        let ids = tokenizer.encode("<audio>").unwrap();
        // No id 15 in a text-only tokenizer; the literal is BPE-split.
        assert!(!ids.contains(&special::AUDIO_ID));
    }
}
