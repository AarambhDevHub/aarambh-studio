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
/// System role marker token string (Phase 52).
///
/// `<|system|>` is the first-class, optional system-turn marker formalized in
/// Phase 52 (`ARCHITECTURE_V4.md` §66). A session may include at most one
/// `<|system|>` turn, placed before any `<|user|>` turn, carrying operator-
/// or application-set instructions. Omitting it entirely reproduces every
/// prior version's `<|user|>...<|assistant|>` format exactly — the system
/// role is purely additive.
pub const SYSTEM: &str = "<|system|>";

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
/// System role marker token id (Phase 52).
///
/// Reserved at id 17, immediately after the Phase 42 audio tokens, following
/// the project's append-never-reassign discipline. `ARCHITECTURE_V4.md` §66
/// notes the historical docs referred to "id 7"; id 7 has been `IMAGE` since
/// v2.0.0 and is never reassigned, so the system marker takes the next free id.
pub const SYSTEM_ID: u32 = 17;

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

/// System-capable reserved token table accepted by v4.0 (Phase 52) checkpoints.
///
/// This is the canonical v4.0 special-token table: the Phase 42 audio table
/// plus the `<|system|>` system-role marker at id 17. It is a strict superset of
/// [`AUDIO_SPECIAL_TOKENS`] — every audio-table pair appears, in order, at the
/// front, with the single new system marker appended.
pub const SYSTEM_SPECIAL_TOKENS: [(&str, u32); 18] = [
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
    (SYSTEM, SYSTEM_ID),
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

/// The chat-template shape version this build of the tokenizer expects.
///
/// Bumped exactly once per template-shape change in the project's history
/// (`ARCHITECTURE_V4.md` §66):
///
/// | version | template shape                                              |
/// |---------|-------------------------------------------------------------|
/// | `1`     | v1.0.0 base `<|user|>`/`<|assistant|>` chat format        |
/// | `2`     | v2.0.0 + image tokens                                       |
/// | `3`     | v3.0.0 + video / document / tool tokens                     |
/// | `4`     | v4.0.0 + system role formalized (`<|system|>`) + audio tokens |
pub const CURRENT_CHAT_TEMPLATE_VERSION: u32 = 4;

/// Validate a checkpoint's declared chat-template version against an expected one.
///
/// A served checkpoint's `chat_template_version` must match (or be explicitly
/// declared compatible with) the server's expected version. A concrete mismatch
/// is a clear error, never a silent misinterpretation of prompt structure.
///
/// - `declared = Some(v)` where `v == expected` or `compatible.contains(&v)` → `Ok`.
/// - `declared = Some(v)` otherwise → `Err` (clear mismatch message).
/// - `declared = None` → `Ok` (undeclared / pre-Phase-52 legacy checkpoint; the
///   project holds many such checkpoints, and absence is not a mismatch).
pub fn validate_chat_template_version(
    declared: Option<u32>,
    expected: u32,
    compatible: &[u32],
) -> std::result::Result<(), String> {
    match declared {
        Some(version) if version == expected || compatible.contains(&version) => Ok(()),
        Some(version) => Err(format!(
            "chat_template_version mismatch: checkpoint declares {version}, \
             but this build expects {expected} (compatible: {compatible:?}). \
             Refusing to load to avoid silently misinterpreting prompt structure."
        )),
        None => Ok(()),
    }
}

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

    #[test]
    fn system_table_is_strict_superset_of_audio_table() {
        // Every (token, id) pair in the audio table must appear, in order, at
        // the front of the system table, with the single system marker appended.
        assert_eq!(SYSTEM_SPECIAL_TOKENS.len(), AUDIO_SPECIAL_TOKENS.len() + 1);
        for (idx, entry) in AUDIO_SPECIAL_TOKENS.iter().enumerate() {
            assert_eq!(SYSTEM_SPECIAL_TOKENS[idx], *entry);
        }
        assert_eq!(SYSTEM_SPECIAL_TOKENS[17], (SYSTEM, SYSTEM_ID));
    }

    #[test]
    fn system_token_id_is_contiguous_after_audio_tokens() {
        // Phase 52 system marker occupies id 17, immediately following the
        // Phase 42 audio table (ids 0..=16).
        assert_eq!(SYSTEM_ID, 17);
        assert_eq!(SYSTEM_ID, AUDIO_END_ID + 1);
    }

    #[test]
    fn system_table_ids_are_contiguous_from_zero() {
        for (idx, &(_, id)) in SYSTEM_SPECIAL_TOKENS.iter().enumerate() {
            assert_eq!(
                id, idx as u32,
                "system table id at position {idx} must be {idx}"
            );
        }
    }

    #[test]
    fn chat_template_version_matches_current_v4_shape() {
        assert_eq!(CURRENT_CHAT_TEMPLATE_VERSION, 4);
    }

    #[test]
    fn chat_template_version_mismatch_is_rejected() {
        // A checkpoint declaring version 2 (image-only) is not compatible with
        // a v4 build that expects version 4 and declares no compatible fallbacks.
        let err = validate_chat_template_version(Some(2), 4, &[]).unwrap_err();
        assert!(
            err.contains("mismatch"),
            "error must mention mismatch, got: {err}"
        );
        assert!(
            err.contains("2") && err.contains("4"),
            "error must name both versions, got: {err}"
        );
    }

    #[test]
    fn chat_template_version_match_is_accepted() {
        assert!(validate_chat_template_version(Some(4), 4, &[]).is_ok());
    }

    #[test]
    fn chat_template_version_compatible_fallback_is_accepted() {
        // A v3 checkpoint served by a v4 build that explicitly declares v3
        // shape-compatible is accepted.
        assert!(validate_chat_template_version(Some(3), 4, &[3]).is_ok());
    }

    #[test]
    fn chat_template_version_undeclared_is_accepted_as_legacy() {
        // Pre-Phase-52 checkpoints do not declare a version; absence is not a
        // mismatch and must not block loading.
        assert!(validate_chat_template_version(None, 4, &[]).is_ok());
    }
}
