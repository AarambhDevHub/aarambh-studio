//! Self-consistency majority-vote selection for test-time compute scaling.
//!
//! This module implements the answer-extraction and majority-vote helpers
//! used by [`crate::best_of_n::SelectionStrategy::SelfConsistency`]. It is
//! deliberately dependency-free: the canonical `extract_final_number`
//! helper lives in `aarambh-studio-finetune::extract_final_number`, but the
//! inference crate is architecturally lower-level than the finetune crate
//! (the eval crate depends on both as siblings), so this module re-declares
//! a byte-identical copy with an attribution doc-comment rather than
//! pulling the finetune crate into the inference dependency graph.

use std::collections::HashMap;
use std::hash::Hash;

use crate::best_of_n::SelectionRationale;

/// Extract the final numeric answer from a generated completion.
///
/// This is a byte-identical re-declaration of
/// `aarambh_studio_finetune::extract_final_number` (verifier.rs:178), kept
/// locally so the inference crate does not depend on the finetune crate.
/// The algorithm matches the finetune helper exactly: prefer the text after
/// the last `####` marker (GSM8K convention), otherwise scan the whole text;
/// return the last parsable finite floating-point number, ignoring commas.
pub fn extract_final_number(text: &str) -> Option<f64> {
    let source = text
        .rsplit_once("####")
        .map(|(_, answer)| answer)
        .unwrap_or(text);
    let mut last = None;
    let mut current = String::new();
    let mut has_digit = false;

    for ch in source.chars() {
        let sign_at_start = matches!(ch, '-' | '+') && current.is_empty();
        let numeric_char = ch.is_ascii_digit() || ch == '.' || ch == ',';
        if sign_at_start || numeric_char {
            if ch.is_ascii_digit() {
                has_digit = true;
            }
            current.push(ch);
            continue;
        }
        if has_digit && let Some(value) = parse_number(&current) {
            last = Some(value);
        }
        current.clear();
        has_digit = false;
    }

    if has_digit && let Some(value) = parse_number(&current) {
        last = Some(value);
    }
    last
}

fn parse_number(value: &str) -> Option<f64> {
    let normalized = value
        .trim_matches(|ch: char| !ch.is_ascii_digit() && !matches!(ch, '-' | '+' | '.'))
        .replace(',', "");
    if normalized.is_empty() || matches!(normalized.as_str(), "+" | "-" | ".") {
        return None;
    }
    normalized.parse::<f64>().ok().filter(|v| v.is_finite())
}

/// Extract a canonical final-answer string from a completion.
///
/// For numeric answers, returns the number formatted via Rust's default
/// `f64`-to-string (so `4.0` and `4` both map to `"4"`). For non-numeric
/// completions, returns the last non-empty trimmed line of the text — a
/// reasonable fallback for code-completion or short-answer tasks where the
/// final answer is the last line of the generation.
pub fn extract_final_answer(text: &str) -> Option<String> {
    if let Some(number) = extract_final_number(text) {
        return Some(format_number_answer(number));
    }
    text.lines()
        .rev()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(ToString::to_string)
}

fn format_number_answer(number: f64) -> String {
    if number.fract() == 0.0 {
        format!("{number:.0}")
    } else {
        format!("{number}")
    }
}

/// Return the most common element of `values` and its count, breaking ties
/// by first occurrence.
///
/// Returns `None` only when `values` is empty. Identical elements are
/// determined by `Eq` and `Hash`; the count is the number of occurrences of
/// the winning element. When multiple elements share the maximum count, the
/// one that appears first in `values` wins.
pub fn majority_vote<T>(values: &[T]) -> Option<(T, usize)>
where
    T: Hash + Eq + Clone,
{
    if values.is_empty() {
        return None;
    }
    let mut counts: HashMap<&T, usize> = HashMap::new();
    for value in values {
        *counts.entry(value).or_insert(0) += 1;
    }
    let max_count = counts.values().copied().max().expect("non-empty input");
    for value in values.iter() {
        if counts.get(value).copied().unwrap_or(0) == max_count {
            return Some((value.clone(), max_count));
        }
    }
    unreachable!("max_count is attained by at least one value")
}

/// Run self-consistency selection over a slice of completion strings.
///
/// Extracts a final answer from each completion, majority-votes the
/// extracted answers, and returns the index of the first completion whose
/// extracted answer matches the winning answer, plus the
/// [`SelectionRationale::SelfConsistency`] describing the vote.
///
/// When no completion yields an extractable answer, falls back to a raw
/// majority vote on the completion strings themselves (the
/// [`SelectionRationale::Majority`] rationale), so self-consistency never
/// silently produces an empty selection.
pub fn self_consistency_select(completions: &[String]) -> (usize, SelectionRationale) {
    let answers: Vec<Option<String>> = completions
        .iter()
        .map(|completion| extract_final_answer(completion))
        .collect();
    if answers.iter().all(Option::is_none) {
        let (winner, count) =
            majority_vote(completions).expect("non-empty completions guarantee a winner");
        let index = completions
            .iter()
            .position(|completion| completion == &winner)
            .expect("winner came from completions");
        return (
            index,
            SelectionRationale::Majority {
                count,
                total: completions.len(),
            },
        );
    }
    let present: Vec<&String> = answers.iter().filter_map(Option::as_ref).collect();
    let (winner, count) =
        majority_vote(&present).expect("at least one extracted answer when not all are None");
    let index = answers
        .iter()
        .position(|maybe| maybe.as_ref() == Some(winner))
        .expect("winner came from answers");
    (
        index,
        SelectionRationale::SelfConsistency {
            answer: winner.clone(),
            count,
            total: completions.len(),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_final_number_matches_gsm8k_marker() {
        assert_eq!(extract_final_number("work\n#### 42"), Some(42.0));
        assert_eq!(extract_final_number("answer: -4.5"), Some(-4.5));
        assert_eq!(extract_final_number("no number here"), None);
    }

    #[test]
    fn extract_final_number_handles_commas_and_decimals() {
        assert_eq!(extract_final_number("#### 1,081"), Some(1081.0));
        assert_eq!(extract_final_number("value is 2.5 approx"), Some(2.5));
    }

    #[test]
    fn extract_final_answer_prefers_number_then_last_line() {
        assert_eq!(extract_final_answer("#### 4"), Some("4".into()));
        assert_eq!(
            extract_final_answer("def add(a,b):\n    return a+b"),
            Some("return a+b".into())
        );
    }

    #[test]
    fn self_consistency_majority_vote_selects_the_most_common_final_answer() {
        let completions = vec![
            "Let me think.\n2+2=4\n#### 4".to_string(),
            "So 2+2 equals 4.\n#### 4".to_string(),
            "Hmm, 7.\n#### 7".to_string(),
            "The answer is 4.\n#### 4".to_string(),
        ];
        let (index, rationale) = self_consistency_select(&completions);
        assert_eq!(index, 0);
        match rationale {
            SelectionRationale::SelfConsistency {
                answer,
                count,
                total,
            } => {
                assert_eq!(answer, "4");
                assert_eq!(count, 3);
                assert_eq!(total, 4);
            }
            other => panic!("expected SelfConsistency, got {other:?}"),
        }
    }

    #[test]
    fn self_consistency_falls_back_to_majority_when_no_answer_extractable() {
        let completions = vec!["".to_string(), "   ".to_string(), "".to_string()];
        let (index, rationale) = self_consistency_select(&completions);
        assert_eq!(index, 0);
        match rationale {
            SelectionRationale::Majority { count, total } => {
                assert_eq!(count, 2);
                assert_eq!(total, 3);
            }
            other => panic!("expected Majority, got {other:?}"),
        }
    }

    #[test]
    fn majority_vote_breaks_ties_by_first_occurrence() {
        let values = vec!["a", "b", "a", "b"];
        let (winner, count) = majority_vote(&values).unwrap();
        assert_eq!(winner, "a");
        assert_eq!(count, 2);
    }
}
