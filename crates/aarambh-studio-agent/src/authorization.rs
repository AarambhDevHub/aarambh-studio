//! Operator-controlled per-tool authorization.
//!
//! Phase 47 (`ARCHITECTURE_V4.md` §61) makes tool execution an operator
//! decision, not a model decision: a model can *declare* any tool in its
//! schema and *request* execution of anything it declares, but only the
//! tools an operator explicitly enabled at server or CLI startup are ever
//! actually executed. This module is the data structure that carries that
//! operator decision.
//!
//! An [`AuthorizationScope`] is a closed set of tool names. It is built once
//! at startup from the operator's allowlist and never widened by the model.
//! [`AuthorizationScope::intersect`] supports Phase 48's multi-agent
//! orchestration, where a sub-agent's authorized scope can only be a
//! *subset* of its orchestrator's — orchestration can never escalate tool
//! access beyond what the operator enabled at the top level.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::AgentError;

/// Validate a tool name against the same character rule
/// `aarambh-studio-inference` enforces for tool declarations, so an
/// operator cannot enable a name the model could never legally declare.
fn validate_tool_name(name: &str) -> Result<(), AgentError> {
    let mut chars = name.chars();
    let first = chars
        .next()
        .ok_or_else(|| AgentError::Config("tool name must not be empty".into()))?;
    if name.len() > 64
        || !(first.is_ascii_alphabetic() || first == '_')
        || !chars.all(|char| char.is_ascii_alphanumeric() || matches!(char, '_' | '.' | '-'))
    {
        return Err(AgentError::Config(format!(
            "invalid tool name {name:?}; expected [A-Za-z_][A-Za-z0-9_.-]{{0,63}}"
        )));
    }
    Ok(())
}

/// The closed set of tool names an operator explicitly enabled at startup.
///
/// Authorization is intentionally separate from the closed-world allowlist
/// of registered [`crate::ToolExecutor`] implementations: a tool can be
/// authorized (the operator said "yes, this name may execute") without a
/// matching executor being registered, in which case execution is still a
/// hard refusal via [`crate::ExecError::UnknownTool`]. The two checks are
/// applied in sequence, mirroring `ARCHITECTURE_V4.md` §61's pipeline.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AuthorizationScope {
    enabled: BTreeSet<String>,
}

impl AuthorizationScope {
    /// Build an empty scope (no tool is authorized).
    pub fn empty() -> Self {
        Self::default()
    }

    /// Build a scope from an iterator of tool names, validating each.
    pub fn new<I, S>(names: I) -> Result<Self, AgentError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut scope = Self::empty();
        for name in names {
            scope.enable(&name.into())?;
        }
        Ok(scope)
    }

    /// Enable one tool name, validating its format. Idempotent.
    pub fn enable(&mut self, name: &str) -> Result<(), AgentError> {
        validate_tool_name(name)?;
        self.enabled.insert(name.to_string());
        Ok(())
    }

    /// Returns true when the operator enabled this tool name.
    pub fn is_authorized(&self, name: &str) -> bool {
        self.enabled.contains(name)
    }

    /// The sorted, deduplicated set of authorized tool names.
    pub fn allowed(&self) -> &BTreeSet<String> {
        &self.enabled
    }

    /// Number of tools the operator authorized.
    pub fn len(&self) -> usize {
        self.enabled.len()
    }

    /// Whether no tool is authorized (execution is fully disabled).
    pub fn is_empty(&self) -> bool {
        self.enabled.is_empty()
    }

    /// Restrict this scope to the intersection with `other`.
    ///
    /// Used by Phase 48 multi-agent orchestration so a sub-agent's scope
    /// can only narrow, never widen, its orchestrator's scope. The result
    /// is always a subset of both inputs.
    pub fn intersect(&self, other: &AuthorizationScope) -> AuthorizationScope {
        let enabled = self
            .enabled
            .intersection(&other.enabled)
            .cloned()
            .collect::<BTreeSet<_>>();
        AuthorizationScope { enabled }
    }
}

#[cfg(test)]
mod tests {
    use super::AuthorizationScope;

    #[test]
    fn empty_scope_authorizes_nothing() {
        let scope = AuthorizationScope::empty();
        assert!(scope.is_empty());
        assert!(!scope.is_authorized("read_file_in_workdir"));
    }

    #[test]
    fn enable_authorizes_named_tool() {
        let mut scope = AuthorizationScope::empty();
        scope.enable("read_file_in_workdir").unwrap();
        assert!(scope.is_authorized("read_file_in_workdir"));
        assert!(!scope.is_authorized("lookup"));
        assert_eq!(scope.len(), 1);
    }

    #[test]
    fn invalid_tool_names_are_rejected() {
        let mut scope = AuthorizationScope::empty();
        assert!(scope.enable("").is_err());
        assert!(scope.enable("1starts_with_digit").is_err());
        assert!(scope.enable("has space").is_err());
        assert!(scope.enable(&"a".repeat(65)).is_err());
        // Valid forms.
        assert!(scope.enable("lookup").is_ok());
        assert!(scope.enable("read_file_in_workdir").is_ok());
        assert!(scope.enable("http.get_v2").is_ok());
    }

    #[test]
    fn enable_is_idempotent() {
        let mut scope = AuthorizationScope::empty();
        scope.enable("lookup").unwrap();
        scope.enable("lookup").unwrap();
        assert_eq!(scope.len(), 1);
    }

    #[test]
    fn intersect_is_subset_of_both() {
        let parent =
            AuthorizationScope::new(["read_file_in_workdir", "lookup", "shipping_quote"]).unwrap();
        let child = AuthorizationScope::new(["lookup", "dangerous_shell"]).unwrap();
        let sub = parent.intersect(&child);
        assert!(sub.is_authorized("lookup"));
        assert!(!sub.is_authorized("read_file_in_workdir"));
        assert!(!sub.is_authorized("dangerous_shell"));
        // A sub-scope can never reach a tool the parent lacked.
        assert!(sub.allowed().iter().all(|name| parent.is_authorized(name)));
    }

    #[test]
    fn intersect_with_empty_disables_everything() {
        let scope = AuthorizationScope::new(["read_file_in_workdir", "lookup"]).unwrap();
        let sub = scope.intersect(&AuthorizationScope::empty());
        assert!(sub.is_empty());
    }
}
