//! BPE tokenizer wrapper, vocabulary utilities, and Aarambh special tokens.
#![deny(missing_docs)]

/// Byte-pair-encoding tokenizer implementation.
pub mod bpe;
/// Reserved special token definitions.
pub mod special;
/// Deterministic virtual-token encoding for structured tool calls.
pub mod tool_protocol;
/// Vocabulary lookup table.
pub mod vocab;

pub use bpe::BpeTokenizer;
pub use special::{
    ASSISTANT, ASSISTANT_ID, AUDIO, AUDIO_END, AUDIO_END_ID, AUDIO_ID, AUDIO_SPECIAL_TOKENS, BOS,
    BOS_ID, CURRENT_CHAT_TEMPLATE_VERSION, DOCUMENT, DOCUMENT_END, DOCUMENT_END_ID, DOCUMENT_ID,
    ENDOFTEXT, ENDOFTEXT_ID, FRAME_SEP, FRAME_SEP_ID, IMAGE, IMAGE_END, IMAGE_END_ID, IMAGE_ID,
    PAD, PAD_ID, PAGE_SEP, PAGE_SEP_ID, SPECIAL_TOKEN_COUNT, SPECIAL_TOKENS, SYSTEM, SYSTEM_ID,
    SYSTEM_SPECIAL_TOKENS, TEXT_SPECIAL_TOKENS, THINK_END, THINK_END_ID, THINK_START,
    THINK_START_ID, USER, USER_ID, VIDEO, VIDEO_END, VIDEO_END_ID, VIDEO_ID, VIDEO_SPECIAL_TOKENS,
    VISION_SPECIAL_TOKENS, validate_chat_template_version,
};
pub use tool_protocol::{
    VIRTUAL_ASCII_BASE, VIRTUAL_ASCII_END, VIRTUAL_ASCII_FIRST, VIRTUAL_ASCII_LAST,
    encode_virtual_json, tool_json_token_text,
};
pub use vocab::Vocab;
