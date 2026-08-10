/// End-of-text token string.
pub const ENDOFTEXT: &str = "<|endoftext|>";
/// Padding token string.
pub const PAD: &str = "<|pad|>";
/// Beginning-of-sequence token string.
pub const BOS: &str = "<|bos|>";
/// Thinking-section start token string.
pub const THINK_START: &str = "<think>";
/// Thinking-section end token string.
pub const THINK_END: &str = "</think>";
/// User role token string.
pub const USER: &str = "<|user|>";
/// Assistant role token string.
pub const ASSISTANT: &str = "<|assistant|>";
/// Image placeholder token string.
pub const IMAGE: &str = "<image>";
/// Image prefix boundary token string.
pub const IMAGE_END: &str = "<image_end>";
/// Video placeholder token string.
pub const VIDEO: &str = "<video>";
/// Video prefix boundary token string.
pub const VIDEO_END: &str = "<video_end>";
/// Separator token inserted between sampled video frames.
pub const FRAME_SEP: &str = "<frame_sep>";
/// Document placeholder token string.
pub const DOCUMENT: &str = "<document>";
/// Document prefix boundary token string.
pub const DOCUMENT_END: &str = "<document_end>";
/// Separator token inserted between rendered document pages.
pub const PAGE_SEP: &str = "<page_sep>";
/// Audio placeholder token string.
pub const AUDIO: &str = "<audio>";
/// Audio prefix boundary token string.
pub const AUDIO_END: &str = "<audio_end>";

/// End-of-text token id.
pub const ENDOFTEXT_ID: u32 = 0;
/// Padding token id.
pub const PAD_ID: u32 = 1;
/// Beginning-of-sequence token id.
pub const BOS_ID: u32 = 2;
/// Thinking-section start token id.
pub const THINK_START_ID: u32 = 3;
/// Thinking-section end token id.
pub const THINK_END_ID: u32 = 4;
/// User role token id.
pub const USER_ID: u32 = 5;
/// Assistant role token id.
pub const ASSISTANT_ID: u32 = 6;
/// Image placeholder token id.
pub const IMAGE_ID: u32 = 7;
/// Image prefix boundary token id.
pub const IMAGE_END_ID: u32 = 8;
/// Video placeholder token id.
pub const VIDEO_ID: u32 = 9;
/// Video prefix boundary token id.
pub const VIDEO_END_ID: u32 = 10;
/// Sampled-frame separator token id.
pub const FRAME_SEP_ID: u32 = 11;
/// Document placeholder token id.
pub const DOCUMENT_ID: u32 = 12;
/// Document prefix boundary token id.
pub const DOCUMENT_END_ID: u32 = 13;
/// Rendered-page separator token id.
pub const PAGE_SEP_ID: u32 = 14;
/// Audio placeholder token id.
pub const AUDIO_ID: u32 = 15;
/// Audio prefix boundary token id.
pub const AUDIO_END_ID: u32 = 16;

/// Reserved special token table in required id order.
pub const SPECIAL_TOKENS: [(&str, u32); 15] = [
    (ENDOFTEXT, ENDOFTEXT_ID),
    (PAD, PAD_ID),
    (BOS, BOS_ID),
    (THINK_START, THINK_START_ID),
    (THINK_END, THINK_END_ID),
    (USER, USER_ID),
    (ASSISTANT, ASSISTANT_ID),
    (IMAGE, IMAGE_ID),
    (IMAGE_END, IMAGE_END_ID),
    (VIDEO, VIDEO_ID),
    (VIDEO_END, VIDEO_END_ID),
    (FRAME_SEP, FRAME_SEP_ID),
    (DOCUMENT, DOCUMENT_ID),
    (DOCUMENT_END, DOCUMENT_END_ID),
    (PAGE_SEP, PAGE_SEP_ID),
];

/// Audio-capable reserved token table accepted by Phase 42 checkpoints.
pub const AUDIO_SPECIAL_TOKENS: [(&str, u32); 17] = [
    (ENDOFTEXT, ENDOFTEXT_ID),
    (PAD, PAD_ID),
    (BOS, BOS_ID),
    (THINK_START, THINK_START_ID),
    (THINK_END, THINK_END_ID),
    (USER, USER_ID),
    (ASSISTANT, ASSISTANT_ID),
    (IMAGE, IMAGE_ID),
    (IMAGE_END, IMAGE_END_ID),
    (VIDEO, VIDEO_ID),
    (VIDEO_END, VIDEO_END_ID),
    (FRAME_SEP, FRAME_SEP_ID),
    (DOCUMENT, DOCUMENT_ID),
    (DOCUMENT_END, DOCUMENT_END_ID),
    (PAGE_SEP, PAGE_SEP_ID),
    (AUDIO, AUDIO_ID),
    (AUDIO_END, AUDIO_END_ID),
];

/// Video-capable reserved token table accepted by Phase 35 checkpoints.
pub const VIDEO_SPECIAL_TOKENS: [(&str, u32); 12] = [
    (ENDOFTEXT, ENDOFTEXT_ID),
    (PAD, PAD_ID),
    (BOS, BOS_ID),
    (THINK_START, THINK_START_ID),
    (THINK_END, THINK_END_ID),
    (USER, USER_ID),
    (ASSISTANT, ASSISTANT_ID),
    (IMAGE, IMAGE_ID),
    (IMAGE_END, IMAGE_END_ID),
    (VIDEO, VIDEO_ID),
    (VIDEO_END, VIDEO_END_ID),
    (FRAME_SEP, FRAME_SEP_ID),
];

/// Image-capable reserved token table accepted by v2 checkpoints.
pub const VISION_SPECIAL_TOKENS: [(&str, u32); 9] = [
    (ENDOFTEXT, ENDOFTEXT_ID),
    (PAD, PAD_ID),
    (BOS, BOS_ID),
    (THINK_START, THINK_START_ID),
    (THINK_END, THINK_END_ID),
    (USER, USER_ID),
    (ASSISTANT, ASSISTANT_ID),
    (IMAGE, IMAGE_ID),
    (IMAGE_END, IMAGE_END_ID),
];

/// Text-only reserved special token table accepted by legacy checkpoints.
pub const TEXT_SPECIAL_TOKENS: [(&str, u32); 7] = [
    (ENDOFTEXT, ENDOFTEXT_ID),
    (PAD, PAD_ID),
    (BOS, BOS_ID),
    (THINK_START, THINK_START_ID),
    (THINK_END, THINK_END_ID),
    (USER, USER_ID),
    (ASSISTANT, ASSISTANT_ID),
];

/// Number of reserved special tokens.
pub const SPECIAL_TOKEN_COUNT: usize = SPECIAL_TOKENS.len();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_table_is_strict_superset_of_document_table() {
        // Every (token, id) pair in the document table must appear, in order, at
        // the front of the audio table, with the two new audio tokens appended.
        assert_eq!(AUDIO_SPECIAL_TOKENS.len(), SPECIAL_TOKENS.len() + 2);
        for (idx, entry) in SPECIAL_TOKENS.iter().enumerate() {
            assert_eq!(AUDIO_SPECIAL_TOKENS[idx], *entry);
        }
        assert_eq!(AUDIO_SPECIAL_TOKENS[15], (AUDIO, AUDIO_ID));
        assert_eq!(AUDIO_SPECIAL_TOKENS[16], (AUDIO_END, AUDIO_END_ID));
    }

    #[test]
    fn audio_token_ids_are_contiguous_after_document_tokens() {
        // Phase 42 audio tokens occupy ids 15 and 16, immediately following the
        // Phase 36 document table (ids 0..=14).
        assert_eq!(AUDIO_ID, 15);
        assert_eq!(AUDIO_END_ID, 16);
        assert_eq!(AUDIO_ID, PAGE_SEP_ID + 1);
        assert_eq!(AUDIO_END_ID, AUDIO_ID + 1);
    }

    #[test]
    fn audio_table_ids_are_contiguous_from_zero() {
        for (idx, &(_, id)) in AUDIO_SPECIAL_TOKENS.iter().enumerate() {
            assert_eq!(
                id, idx as u32,
                "audio table id at position {idx} must be {idx}"
            );
        }
    }
}
