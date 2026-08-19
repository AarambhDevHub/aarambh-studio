//! Multi-turn context-truncation policy (Phase 52, `ARCHITECTURE_V4.md` §66).
//!
//! Long agentic chains (v3 §46, v4 §61–62) and RAG-augmented sessions (v4 §63)
//! can exceed the model's context window. Prior to Phase 52 every long-context
//! feature handled this ad hoc, if at all. This module defines one canonical
//! [`ContextTruncationPolicy`] referenced consistently by all of them, rather
//! than each feature inventing its own truncation behaviour.

use aarambh_studio_core::{AarambhError, Result};

/// Canonical multi-turn context-truncation policy.
///
/// One policy, applied by every long-context feature in the project. The
/// system turn, when present as the leading turn, is **never evicted** —
/// operator-set instructions must survive truncation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ContextTruncationPolicy {
    /// Drop the oldest non-system turns first until the transcript fits the
    /// token budget. The leading system turn, if any, is never evicted.
    #[default]
    SlidingWindow,
    /// Replace evicted turns with a single generated summary turn, reusing the
    /// project's existing self-critique-style summarization. The summary token
    /// ids are supplied by the caller (which has the tokenizer and can run the
    /// model); this policy only decides *what* to evict and *where* the summary
    /// goes.
    Summarize,
    /// Refuse to proceed rather than silently drop context. The mandatory
    /// default for anything safety- or execution-sensitive — e.g. sandboxed
    /// tool-execution sessions (v4 §61) and orchestration (v4 §62) — where
    /// silently losing a turn would change the meaning of the session.
    Reject,
}

/// The role of a single conversational turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnRole {
    /// Operator- or application-set system turn (` IMS`).
    System,
    /// User-authored turn (` IMS`).
    User,
    /// Model-produced turn (` IMS`).
    Assistant,
}

/// One encoded conversational turn consumed by the truncation policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextTurn {
    /// Role of this turn.
    pub role: TurnRole,
    /// Encoded token ids for this turn's content (role markers included by the
    /// caller if they belong to the turn).
    pub token_ids: Vec<u32>,
}

impl ContextTurn {
    /// Number of tokens in this turn.
    pub fn len(&self) -> usize {
        self.token_ids.len()
    }

    /// Whether this turn carries no tokens.
    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }
}

/// Return the total token count across all turns.
pub fn total_tokens(turns: &[ContextTurn]) -> usize {
    turns.iter().map(|turn| turn.len()).sum()
}

/// Apply a truncation policy to a multi-turn transcript.
///
/// `max_tokens` is the model's available context budget for these turns (the
/// caller is responsible for reserving space for the generated continuation).
/// `summary_ids` is used only by [`ContextTruncationPolicy::Summarize`]; it is
/// the encoded text of the summary turn that replaces evicted turns.
///
/// - If the transcript already fits, it is returned unchanged.
/// - [`ContextTruncationPolicy::Reject`] returns an error when the transcript
///   exceeds the budget — never a silent drop.
/// - [`ContextTruncationPolicy::SlidingWindow`] drops oldest non-system turns
///   until the transcript fits; a leading system turn is always retained.
/// - [`ContextTruncationPolicy::Summarize`] behaves like `SlidingWindow` but
///   inserts one summary turn (role [`TurnRole::System`]) in place of the
///   evicted block, positioned just after the leading system turn (or at the
///   front when there is no leading system turn).
pub fn apply(
    policy: ContextTruncationPolicy,
    turns: &[ContextTurn],
    max_tokens: usize,
    summary_ids: &[u32],
) -> Result<Vec<ContextTurn>> {
    if total_tokens(turns) <= max_tokens {
        return Ok(turns.to_vec());
    }

    match policy {
        ContextTruncationPolicy::Reject => Err(AarambhError::Config(format!(
            "context-truncation policy is `Reject`: transcript of {} tokens exceeds the \
             {}-token budget; refusing to proceed rather than silently drop context",
            total_tokens(turns),
            max_tokens
        ))),
        ContextTruncationPolicy::SlidingWindow => Ok(sliding_window(turns, max_tokens)),
        ContextTruncationPolicy::Summarize => Ok(summarize(turns, max_tokens, summary_ids)),
    }
}

/// Drop the oldest non-system turns until the transcript fits `max_tokens`.
///
/// The leading turn, when it is a [`TurnRole::System`] turn, is never evicted —
/// operator-set instructions survive truncation. Turns are dropped from the
/// front of the non-system region (oldest first) until the budget is met, then
/// the most recent turns are retained in order.
fn sliding_window(turns: &[ContextTurn], max_tokens: usize) -> Vec<ContextTurn> {
    let (head, body) = split_leading_system(turns);
    let head_tokens: usize = head.iter().map(|turn| turn.len()).sum();
    let budget = max_tokens.saturating_sub(head_tokens);

    // Walk the body from the end, keeping the most recent turns until adding the
    // next-older turn would exceed the budget. This drops the oldest turns first.
    let mut kept: Vec<ContextTurn> = Vec::new();
    let mut used = 0usize;
    for turn in body.iter().rev() {
        if used + turn.len() > budget && !kept.is_empty() {
            break;
        }
        used += turn.len();
        kept.push(turn.clone());
    }
    kept.reverse();

    let mut out = head.to_vec();
    out.extend(kept);
    out
}

/// Evict the oldest non-system turns and insert a single summary turn in their
/// place, re-running [`sliding_window`] over the remaining body so the result
/// fits `max_tokens`.
fn summarize(turns: &[ContextTurn], max_tokens: usize, summary_ids: &[u32]) -> Vec<ContextTurn> {
    let summary_turn = ContextTurn {
        role: TurnRole::System,
        token_ids: summary_ids.to_vec(),
    };

    let (head, body) = split_leading_system(turns);
    let head_tokens: usize = head.iter().map(|turn| turn.len()).sum();
    let summary_tokens = summary_turn.len();
    let budget = max_tokens
        .saturating_sub(head_tokens)
        .saturating_sub(summary_tokens);

    // Keep the most recent turns that fit the remaining budget; everything
    // older than the kept region is represented by the summary turn.
    let mut kept: Vec<ContextTurn> = Vec::new();
    let mut used = 0usize;
    for turn in body.iter().rev() {
        if used + turn.len() > budget && !kept.is_empty() {
            break;
        }
        used += turn.len();
        kept.push(turn.clone());
    }
    kept.reverse();

    let mut out = head.to_vec();
    // The summary replaces the evicted (older) turns, so it sits just before the
    // retained recent turns — after the leading system turn (if any).
    if !summary_turn.is_empty() {
        out.push(summary_turn);
    }
    out.extend(kept);
    out
}

/// Split `turns` into the leading system turn (if the first turn is a system
/// turn) and the remaining body.
fn split_leading_system(turns: &[ContextTurn]) -> (&[ContextTurn], &[ContextTurn]) {
    if matches!(turns.first(), Some(turn) if turn.role == TurnRole::System) {
        (&turns[..1], &turns[1..])
    } else {
        (&[], turns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(role: TurnRole, ids: &[u32]) -> ContextTurn {
        ContextTurn {
            role,
            token_ids: ids.to_vec(),
        }
    }

    #[test]
    fn context_policy_reject_refuses_rather_than_silently_drops_context() {
        // A transcript that exceeds the budget under `Reject` must error, never
        // silently return a truncated transcript.
        let turns = vec![
            turn(TurnRole::User, &[1, 2, 3, 4]),
            turn(TurnRole::Assistant, &[5, 6, 7, 8]),
        ];
        let err = apply(ContextTruncationPolicy::Reject, &turns, 4, &[]).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Reject"),
            "error must name the Reject policy, got: {msg}"
        );
        assert!(
            msg.contains("8") && msg.contains("4"),
            "error must name the budget and actual size, got: {msg}"
        );
    }

    #[test]
    fn context_policy_reject_accepts_transcript_under_budget() {
        let turns = vec![turn(TurnRole::User, &[1, 2])];
        let out = apply(ContextTruncationPolicy::Reject, &turns, 4, &[]).unwrap();
        assert_eq!(out, turns);
    }

    #[test]
    fn context_policy_sliding_window_never_evicts_the_system_turn() {
        // A leading system turn plus three exchanges that together exceed a
        // tight budget. SlidingWindow must keep the system turn and drop the
        // oldest non-system turns until the transcript fits.
        let turns = vec![
            turn(TurnRole::System, &[100]),   // system, never evicted
            turn(TurnRole::User, &[1, 2, 3]), // oldest non-system
            turn(TurnRole::Assistant, &[4, 5, 6]),
            turn(TurnRole::User, &[7, 8, 9]),
            turn(TurnRole::Assistant, &[10, 11, 12]), // most recent
        ];
        // Budget: 1 (system) + 6 (two most recent turns) = 7.
        let out = apply(ContextTruncationPolicy::SlidingWindow, &turns, 7, &[]).unwrap();
        assert_eq!(out.len(), 3, "keeps system + two most recent turns");
        assert_eq!(out[0].role, TurnRole::System, "system turn retained");
        assert_eq!(out[0].token_ids, vec![100]);
        assert_eq!(
            out[1].token_ids,
            vec![7, 8, 9],
            "oldest non-system turn dropped"
        );
        assert_eq!(out[2].token_ids, vec![10, 11, 12]);
    }

    #[test]
    fn context_policy_sliding_window_without_system_turn_drops_oldest() {
        let turns = vec![
            turn(TurnRole::User, &[1, 2]),
            turn(TurnRole::Assistant, &[3, 4]),
            turn(TurnRole::User, &[5, 6]),
        ];
        // Budget fits only the two most recent turns.
        let out = apply(ContextTruncationPolicy::SlidingWindow, &turns, 4, &[]).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].token_ids, vec![3, 4]);
        assert_eq!(out[1].token_ids, vec![5, 6]);
    }

    #[test]
    fn context_policy_summarize_inserts_summary_after_system_turn() {
        let turns = vec![
            turn(TurnRole::System, &[100]),
            turn(TurnRole::User, &[1, 2, 3]), // evicted -> summarized
            turn(TurnRole::Assistant, &[4, 5, 6]), // evicted -> summarized
            turn(TurnRole::User, &[7, 8]),    // retained
            turn(TurnRole::Assistant, &[9, 10]), // retained
        ];
        let summary = vec![200, 201];
        // Budget: 1 (system) + 2 (summary) + 4 (two retained) = 7.
        let out = apply(ContextTruncationPolicy::Summarize, &turns, 7, &summary).unwrap();
        assert_eq!(out[0].role, TurnRole::System);
        assert_eq!(out[0].token_ids, vec![100]);
        assert_eq!(out[1].role, TurnRole::System, "summary turn is system-role");
        assert_eq!(out[1].token_ids, vec![200, 201]);
        assert_eq!(out[2].token_ids, vec![7, 8]);
        assert_eq!(out[3].token_ids, vec![9, 10]);
    }

    #[test]
    fn context_policy_summarize_with_no_system_turn_places_summary_first() {
        let turns = vec![
            turn(TurnRole::User, &[1, 2, 3]),
            turn(TurnRole::Assistant, &[4, 5]),
        ];
        let summary = vec![200];
        // Budget: 1 (summary) + 2 (most recent) = 3.
        let out = apply(ContextTruncationPolicy::Summarize, &turns, 3, &summary).unwrap();
        assert_eq!(out[0].token_ids, vec![200]);
        assert_eq!(out[1].token_ids, vec![4, 5]);
    }

    #[test]
    fn context_policy_under_budget_returns_unchanged() {
        let turns = vec![turn(TurnRole::System, &[1]), turn(TurnRole::User, &[2, 3])];
        let out = apply(ContextTruncationPolicy::SlidingWindow, &turns, 100, &[]).unwrap();
        assert_eq!(out, turns);
    }
}
