use std::collections::HashMap;

use aarambh_studio_core::TokenizerLike;
use aarambh_studio_tokenizer::{
    BpeTokenizer, Vocab,
    special::{
        ASSISTANT, ASSISTANT_ID, BOS, BOS_ID, DOCUMENT, DOCUMENT_END, DOCUMENT_END_ID, DOCUMENT_ID,
        ENDOFTEXT, ENDOFTEXT_ID, FRAME_SEP, FRAME_SEP_ID, IMAGE, IMAGE_END, IMAGE_END_ID, IMAGE_ID,
        PAD, PAD_ID, PAGE_SEP, PAGE_SEP_ID, SPECIAL_TOKEN_COUNT, THINK_END, THINK_END_ID,
        THINK_START, THINK_START_ID, USER, USER_ID, VIDEO, VIDEO_END, VIDEO_END_ID, VIDEO_ID,
        VIDEO_SPECIAL_TOKENS, VISION_SPECIAL_TOKENS,
    },
};

#[test]
fn special_token_ids_are_correct() {
    assert_eq!(ENDOFTEXT_ID, 0);
    assert_eq!(PAD_ID, 1);
    assert_eq!(BOS_ID, 2);
    assert_eq!(THINK_START_ID, 3);
    assert_eq!(THINK_END_ID, 4);
    assert_eq!(USER_ID, 5);
    assert_eq!(ASSISTANT_ID, 6);
    assert_eq!(IMAGE_ID, 7);
    assert_eq!(IMAGE_END_ID, 8);
    assert_eq!(VIDEO_ID, 9);
    assert_eq!(VIDEO_END_ID, 10);
    assert_eq!(FRAME_SEP_ID, 11);
    assert_eq!(DOCUMENT_ID, 12);
    assert_eq!(DOCUMENT_END_ID, 13);
    assert_eq!(PAGE_SEP_ID, 14);
    assert_eq!(ENDOFTEXT, "<|endoftext|>");
    assert_eq!(PAD, "<|pad|>");
    assert_eq!(BOS, "<|bos|>");
    assert_eq!(THINK_START, "<think>");
    assert_eq!(THINK_END, "</think>");
    assert_eq!(USER, "<|user|>");
    assert_eq!(ASSISTANT, "<|assistant|>");
    assert_eq!(IMAGE, "<image>");
    assert_eq!(IMAGE_END, "<image_end>");
    assert_eq!(VIDEO, "<video>");
    assert_eq!(VIDEO_END, "<video_end>");
    assert_eq!(FRAME_SEP, "<frame_sep>");
    assert_eq!(DOCUMENT, "<document>");
    assert_eq!(DOCUMENT_END, "<document_end>");
    assert_eq!(PAGE_SEP, "<page_sep>");
}

#[test]
fn vocab_get_id_and_get_token() {
    let token_to_id = HashMap::from([
        ("hello".into(), 0u32),
        ("world".into(), 1u32),
        (" ".into(), 2u32),
    ]);
    let id_to_token = vec!["hello".into(), "world".into(), " ".into()];
    let vocab = Vocab {
        token_to_id,
        id_to_token,
    };

    assert_eq!(vocab.get_id("hello"), Some(0));
    assert_eq!(vocab.get_id("world"), Some(1));
    assert_eq!(vocab.get_id("unknown"), None);
    assert_eq!(vocab.get_token(0), Some("hello"));
    assert_eq!(vocab.get_token(1), Some("world"));
    assert_eq!(vocab.get_token(99), None);
}

#[test]
fn vocab_roundtrip_via_json() {
    let token_to_id = HashMap::from([("a".into(), 0u32), ("b".into(), 1u32)]);
    let id_to_token = vec!["a".into(), "b".into()];
    let original = Vocab {
        token_to_id,
        id_to_token,
    };

    let dir = std::env::temp_dir();
    let path = dir.join("test_vocab.json");
    original.save_json(&path).unwrap();
    let loaded = Vocab::from_json(&path).unwrap();
    let _ = std::fs::remove_file(&path);

    assert_eq!(loaded.get_id("a"), Some(0));
    assert_eq!(loaded.get_id("b"), Some(1));
    assert_eq!(loaded.get_token(0), Some("a"));
}

#[test]
fn bpe_tokenizer_encode_decode_roundtrip() {
    let token_to_id = HashMap::from([
        ("h".into(), 0u32),
        ("e".into(), 1u32),
        ("l".into(), 2u32),
        ("o".into(), 3u32),
        (" ".into(), 4u32),
        ("w".into(), 5u32),
        ("r".into(), 6u32),
        ("d".into(), 7u32),
        ("he".into(), 8u32),
        ("ll".into(), 9u32),
        ("hell".into(), 10u32),
        ("hello".into(), 11u32),
        ("wo".into(), 12u32),
        ("rl".into(), 13u32),
        ("worl".into(), 14u32),
        ("world".into(), 15u32),
    ]);
    let id_to_token = vec![
        "h".into(),
        "e".into(),
        "l".into(),
        "o".into(),
        " ".into(),
        "w".into(),
        "r".into(),
        "d".into(),
        "he".into(),
        "ll".into(),
        "hell".into(),
        "hello".into(),
        "wo".into(),
        "rl".into(),
        "worl".into(),
        "world".into(),
    ];
    let merges: Vec<(String, String)> = vec![
        ("h".into(), "e".into()),
        ("l".into(), "l".into()),
        ("he".into(), "ll".into()),
        ("hell".into(), "o".into()),
        ("w".into(), "o".into()),
        ("r".into(), "l".into()),
        ("wo".into(), "rl".into()),
        ("worl".into(), "d".into()),
    ];

    let vocab = Vocab {
        token_to_id,
        id_to_token,
    };

    let merge_rank: HashMap<(String, String), usize> = merges
        .iter()
        .enumerate()
        .map(|(i, (a, b))| ((a.clone(), b.clone()), i))
        .collect();

    let tokenizer = BpeTokenizer {
        vocab,
        merges,
        merge_rank,
        chat_template_version: None,
    };

    let text = "hello world";
    let ids = tokenizer.encode(text).unwrap();
    let decoded = tokenizer.decode(&ids).unwrap();
    assert_eq!(decoded, text);
}

#[test]
fn bpe_tokenizer_implements_tokenizer_like() {
    let token_to_id = HashMap::from([("a".into(), 0u32), ("b".into(), 1u32)]);
    let id_to_token = vec!["a".into(), "b".into()];
    let vocab = Vocab {
        token_to_id,
        id_to_token,
    };
    let merge_rank = HashMap::new();

    let tokenizer = BpeTokenizer {
        vocab,
        merges: vec![],
        merge_rank,
        chat_template_version: None,
    };

    assert_eq!(tokenizer.vocab_size(), 2);
    assert_eq!(tokenizer.eos_token_id(), 0);
    assert_eq!(tokenizer.bos_token_id(), Some(2));
}

#[test]
fn bpe_save_pretrained_roundtrip_preserves_merges() {
    let token_to_id = HashMap::from([("a".into(), 0u32), ("b".into(), 1u32), ("ab".into(), 2u32)]);
    let id_to_token = vec!["a".into(), "b".into(), "ab".into()];
    let merges = vec![("a".into(), "b".into())];
    let merge_rank = HashMap::from([(("a".into(), "b".into()), 0usize)]);
    let tokenizer = BpeTokenizer {
        vocab: Vocab {
            token_to_id,
            id_to_token,
        },
        merges,
        merge_rank,
        chat_template_version: None,
    };
    let path = std::env::temp_dir().join(format!(
        "aarambh_tokenizer_roundtrip_{}.json",
        std::process::id()
    ));

    tokenizer.save_pretrained(&path).unwrap();
    let loaded = BpeTokenizer::from_pretrained(&path).unwrap();
    let _ = std::fs::remove_file(&path);

    assert_eq!(loaded.encode("ab").unwrap(), vec![2]);
    assert_eq!(loaded.decode(&[2]).unwrap(), "ab");
}

#[test]
fn trained_bpe_reserves_special_token_ids() {
    let path = std::env::temp_dir().join(format!(
        "aarambh_tokenizer_train_specials_{}.txt",
        std::process::id()
    ));
    std::fs::write(
        &path,
        "the thou thee therefore there their thing think thinking",
    )
    .unwrap();

    let tokenizer = BpeTokenizer::train(&path, 32).unwrap();
    let _ = std::fs::remove_file(&path);

    tokenizer.validate_special_tokens().unwrap();
    assert!(tokenizer.vocab_size() >= SPECIAL_TOKEN_COUNT);
    assert!(tokenizer.vocab_size() <= 32);
    for (token, id) in &tokenizer.vocab.token_to_id {
        if !token.starts_with('<') {
            assert!((*id as usize) >= SPECIAL_TOKEN_COUNT);
        }
    }
    assert_eq!(
        tokenizer
            .encode("<|user|>hello<|assistant|>")
            .unwrap()
            .first()
            .copied(),
        Some(USER_ID)
    );
    assert_eq!(tokenizer.decode(&[ENDOFTEXT_ID]).unwrap(), ENDOFTEXT);
}

#[test]
fn validate_special_tokens_rejects_plain_character_ids() {
    let token_to_id = HashMap::from([("!".into(), 0u32), ("$".into(), 1u32)]);
    let id_to_token = vec!["!".into(), "$".into()];
    let tokenizer = BpeTokenizer {
        vocab: Vocab {
            token_to_id,
            id_to_token,
        },
        merges: vec![],
        merge_rank: HashMap::new(),
        chat_template_version: None,
    };

    let err = tokenizer.validate_special_tokens().unwrap_err();
    assert!(err.to_string().contains("special token"));
}

#[test]
fn image_tokenizer_upgrades_video_tokens_without_changing_learned_token_text() {
    let mut token_to_id = HashMap::new();
    let mut id_to_token = Vec::new();
    for (token, id) in VISION_SPECIAL_TOKENS {
        token_to_id.insert(token.to_string(), id);
        id_to_token.push(token.to_string());
    }
    token_to_id.insert("hello".into(), 9);
    id_to_token.push("hello".into());
    let tokenizer = BpeTokenizer {
        vocab: Vocab {
            token_to_id,
            id_to_token,
        },
        merges: Vec::new(),
        merge_rank: HashMap::new(),
        chat_template_version: None,
    };

    tokenizer.validate_vision_special_tokens().unwrap();
    let upgraded = tokenizer.upgraded_for_video().unwrap();
    upgraded.validate_video_special_tokens().unwrap();
    assert_eq!(upgraded.vocab.get_id("hello"), Some(12));
    assert_eq!(upgraded.decode(&[12]).unwrap(), "hello");
    assert_eq!(upgraded.encode("<video><video_end>").unwrap(), vec![9, 10]);
}

#[test]
fn legacy_image_tokenizer_does_not_emit_unmapped_video_ids() {
    let mut token_to_id = HashMap::new();
    let mut id_to_token = Vec::new();
    for (token, id) in VISION_SPECIAL_TOKENS {
        token_to_id.insert(token.to_string(), id);
        id_to_token.push(token.to_string());
    }
    let tokenizer = BpeTokenizer {
        vocab: Vocab {
            token_to_id,
            id_to_token,
        },
        merges: Vec::new(),
        merge_rank: HashMap::new(),
        chat_template_version: None,
    };

    assert!(!tokenizer.encode("<video>").unwrap().contains(&VIDEO_ID));
}

#[test]
fn video_tokenizer_upgrades_document_tokens_without_changing_learned_text() {
    let mut token_to_id = HashMap::new();
    let mut id_to_token = Vec::new();
    for (token, id) in VIDEO_SPECIAL_TOKENS {
        token_to_id.insert(token.to_string(), id);
        id_to_token.push(token.to_string());
    }
    token_to_id.insert("invoice".into(), 12);
    id_to_token.push("invoice".into());
    let tokenizer = BpeTokenizer {
        vocab: Vocab {
            token_to_id,
            id_to_token,
        },
        merges: Vec::new(),
        merge_rank: HashMap::new(),
        chat_template_version: None,
    };

    let upgraded = tokenizer.upgraded_for_document().unwrap();
    upgraded.validate_document_special_tokens().unwrap();
    assert_eq!(upgraded.vocab.get_id("invoice"), Some(15));
    assert_eq!(upgraded.decode(&[15]).unwrap(), "invoice");
    assert_eq!(
        upgraded
            .encode("<document><page_sep><document_end>")
            .unwrap(),
        vec![12, 14, 13]
    );
}
