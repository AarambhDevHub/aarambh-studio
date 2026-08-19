//! Per-key API identity and rate limiting for the public inference server.
//!
//! # Layout
//!
//! [`ApiKeyStore`] holds the validated set of issued keys (loaded from a JSON
//! key file via [`AuthConfig`]). [`AuthGate`] resolves an incoming request's
//! `Authorization: Bearer <secret>` header to a [`TenantId`] — or rejects it
//! before admission. [`RateLimiter`] enforces per-key requests/minute and
//! tokens/minute limits, independently per tenant, at the same admission
//! point.
//!
//! # Honesty boundary
//!
//! When no key store is configured (the documented default), the gate runs in
//! loopback-open mode: every request is mapped to a synthetic `local` tenant
//! with no rate limiting. This preserves the v2 §31 single-user default
//! byte-for-byte. Multi-tenant auth is strictly opt-in.

#![deny(missing_docs)]

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use aarambh_studio_core::{AarambhError, Result};
use axum::http::HeaderMap;
use serde::{Deserialize, Serialize};

/// Window over which per-key rate limits are measured.
pub const RATE_WINDOW: Duration = Duration::from_secs(60);

/// Opaque tenant identifier resolved from a validated API key, or the
/// synthetic `local` tenant when auth is disabled.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TenantId(String);

impl TenantId {
    /// Construct a tenant id from a verified string.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// The synthetic tenant used when auth is disabled.
    pub fn local() -> Self {
        Self::new("local")
    }

    /// Return the tenant's identifier string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for TenantId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Per-key rate limits enforced at admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RateLimit {
    /// Maximum requests admitted per rolling 60-second window.
    pub requests_per_minute: u32,
    /// Maximum prompt+completion tokens admitted per rolling 60-second window.
    pub tokens_per_minute: u32,
}

impl RateLimit {
    /// Unlimited rate limit (used by the synthetic local tenant).
    pub const UNLIMITED: Self = Self {
        requests_per_minute: u32::MAX,
        tokens_per_minute: u32::MAX,
    };
}

/// One issued API key with its owning tenant and per-key rate limits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey {
    /// The shared secret presented in the `Authorization: Bearer` header.
    pub secret: String,
    /// The tenant this key authenticates as.
    pub tenant: TenantId,
    /// Per-key rate limits.
    #[serde(default = "default_rate_limit")]
    pub limits: RateLimit,
}

fn default_rate_limit() -> RateLimit {
    RateLimit {
        requests_per_minute: 60,
        tokens_per_minute: 10_000,
    }
}

/// On-disk key-file entry (admits either the bare [`ApiKey`] shape or the
/// OpenAI-style `{secret, tenant, limits}` object).
#[derive(Debug, Clone, Deserialize)]
pub struct KeyEntry {
    /// The shared secret presented in the `Authorization: Bearer` header.
    pub secret: String,
    /// The tenant this key authenticates as.
    pub tenant: TenantId,
    /// Per-key rate limits (defaults to 60 RPM / 10 000 TPM when omitted).
    #[serde(default = "default_rate_limit")]
    pub limits: RateLimit,
}

impl From<KeyEntry> for ApiKey {
    fn from(entry: KeyEntry) -> Self {
        Self {
            secret: entry.secret,
            tenant: entry.tenant,
            limits: entry.limits,
        }
    }
}

/// On-disk multi-tenant key file: `{ "keys": [ {secret, tenant, limits}, ... ] }`.
#[derive(Debug, Clone, Deserialize)]
pub struct AuthConfig {
    /// The issued keys.
    pub keys: Vec<KeyEntry>,
}

impl AuthConfig {
    /// Read and parse a key file from disk.
    pub fn from_path(path: &std::path::Path) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|error| {
            AarambhError::Config(format!(
                "failed to read API-key file {}: {error}",
                path.display()
            ))
        })?;
        let config: Self = serde_json::from_slice(&bytes).map_err(|error| {
            AarambhError::Config(format!(
                "failed to parse API-key file {}: {error}",
                path.display()
            ))
        })?;
        Ok(config)
    }
}

/// In-memory validated key set.
pub struct ApiKeyStore {
    keys: Vec<ApiKey>,
}

impl ApiKeyStore {
    /// Build a key store from a parsed [`AuthConfig`].
    pub fn from_config(config: &AuthConfig) -> Result<Self> {
        if config.keys.is_empty() {
            return Err(AarambhError::Config(
                "API-key file must declare at least one key".into(),
            ));
        }
        let mut seen = std::collections::HashSet::new();
        for entry in &config.keys {
            if entry.secret.trim().is_empty() {
                return Err(AarambhError::Config(
                    "API-key file declares an empty secret".into(),
                ));
            }
            if entry.tenant.as_str().trim().is_empty() {
                return Err(AarambhError::Config(
                    "API-key file declares an empty tenant".into(),
                ));
            }
            if !seen.insert(entry.secret.clone()) {
                return Err(AarambhError::Config(
                    "API-key file declares a duplicate secret".into(),
                ));
            }
        }
        Ok(Self {
            keys: config.keys.iter().cloned().map(ApiKey::from).collect(),
        })
    }

    /// Validate a presented bearer secret in constant time.
    pub fn validate(&self, presented: &str) -> Option<&ApiKey> {
        let presented = presented.as_bytes();
        let mut matched: Option<&ApiKey> = None;
        for key in &self.keys {
            if constant_time_eq(presented, key.secret.as_bytes()) && matched.is_none() {
                matched = Some(key);
            }
        }
        matched
    }

    /// Iterate over the issued keys (for tooling / tests).
    pub fn keys(&self) -> &[ApiKey] {
        &self.keys
    }
}

/// Authorisation outcome for one request.
#[derive(Debug)]
pub enum AuthOutcome<'a> {
    /// A valid key was presented; the request is authenticated as `key.tenant`.
    Authenticated {
        /// The resolved key (carries the tenant id and rate limits).
        key: &'a ApiKey,
    },
    /// Auth is disabled and loopback-open mode is active; the request maps to
    /// the synthetic `local` tenant with no rate limiting.
    UnauthenticatedLocal,
    /// Auth is enabled and the request failed authorisation.
    Rejected(AuthRejection),
}

/// Reason a request was rejected by the auth gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthRejection {
    /// No `Authorization: Bearer` header was present.
    MissingKey,
    /// A bearer header was present but no key matched.
    InvalidKey,
}

/// Authorisation gate: resolves an incoming request to a tenant or a rejection.
pub struct AuthGate {
    store: Option<ApiKeyStore>,
}

impl AuthGate {
    /// Build a gate. When `store` is `None`, the gate runs in loopback-open
    /// mode (every request maps to the synthetic `local` tenant).
    pub fn new(store: Option<ApiKeyStore>) -> Self {
        Self { store }
    }

    /// Resolve the bearer header.
    pub fn authorize(&self, headers: &HeaderMap) -> AuthOutcome<'_> {
        let Some(store) = self.store.as_ref() else {
            return AuthOutcome::UnauthenticatedLocal;
        };
        let Some(presented) = bearer_token(headers) else {
            return AuthOutcome::Rejected(AuthRejection::MissingKey);
        };
        match store.validate(presented) {
            Some(key) => AuthOutcome::Authenticated { key },
            None => AuthOutcome::Rejected(AuthRejection::InvalidKey),
        }
    }

    /// Whether this gate is running in loopback-open mode.
    pub fn is_open(&self) -> bool {
        self.store.is_none()
    }
}

fn bearer_token(headers: &HeaderMap) -> Option<&str> {
    headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .map(str::trim)
}

/// Constant-time byte-slice equality comparison.
pub fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right)
        .fold(0u8, |difference, (left, right)| difference | (left ^ right))
        == 0
}

/// Per-tenant sliding-window rate limiter (requests/minute + tokens/minute).
pub struct RateLimiter {
    inner: Mutex<HashMap<TenantId, Vec<(Instant, usize)>>>,
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

impl RateLimiter {
    /// Create an empty rate limiter.
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
        }
    }

    /// Check whether a request from `tenant` estimated at `tokens` prompt +
    /// completion tokens may be admitted under `limit`.
    pub fn check(&self, tenant: &TenantId, limit: RateLimit, tokens: usize) -> bool {
        let mut windows = self.inner.lock().expect("rate limiter mutex poisoned");
        let now = Instant::now();
        let entries = windows.entry(tenant.clone()).or_default();
        entries.retain(|(stamp, _)| now.duration_since(*stamp) < RATE_WINDOW);
        let current_rpm = entries.len() as u32;
        let current_tpm: usize = entries.iter().map(|(_, t)| *t).sum();
        let rpm_ok = current_rpm.saturating_add(1) <= limit.requests_per_minute;
        let tpm_ok = current_tpm.saturating_add(tokens) <= limit.tokens_per_minute as usize;
        if rpm_ok && tpm_ok {
            entries.push((now, tokens));
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    fn headers_with_bearer(token: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        if !token.is_empty() {
            headers.insert(
                "authorization",
                HeaderValue::from_str(&format!("Bearer {token}")).unwrap(),
            );
        }
        headers
    }

    #[test]
    fn constant_time_compare_rejects_prefix_and_unequal_lengths() {
        assert!(constant_time_eq(b"secret", b"secret"));
        assert!(!constant_time_eq(b"secret", b"secre"));
        assert!(!constant_time_eq(b"secret", b"secrex"));
        assert!(!constant_time_eq(b"secret", b"secret-extra"));
        assert!(constant_time_eq(b"", b""));
    }

    #[test]
    fn api_key_store_validates_presented_secret_and_rejects_unknown() {
        let config = AuthConfig {
            keys: vec![KeyEntry {
                secret: "alpha-secret".into(),
                tenant: TenantId::new("tenant-a"),
                limits: default_rate_limit(),
            }],
        };
        let store = ApiKeyStore::from_config(&config).unwrap();
        assert_eq!(
            store.validate("alpha-secret").map(|k| k.tenant.as_str()),
            Some("tenant-a")
        );
        assert!(store.validate("wrong").is_none());
        assert!(store.validate("").is_none());
    }

    #[test]
    fn auth_gate_open_mode_maps_every_request_to_local() {
        let gate = AuthGate::new(None);
        let outcome = gate.authorize(&headers_with_bearer("ignored"));
        assert!(matches!(outcome, AuthOutcome::UnauthenticatedLocal));
        assert!(gate.is_open());
    }

    #[test]
    fn auth_gate_rejects_missing_key_before_admission() {
        let config = AuthConfig {
            keys: vec![KeyEntry {
                secret: "alpha-secret".into(),
                tenant: TenantId::new("tenant-a"),
                limits: default_rate_limit(),
            }],
        };
        let store = ApiKeyStore::from_config(&config).unwrap();
        let gate = AuthGate::new(Some(store));
        let mut empty = HeaderMap::new();
        empty.insert("content-type", HeaderValue::from_static("application/json"));
        match gate.authorize(&empty) {
            AuthOutcome::Rejected(AuthRejection::MissingKey) => {}
            other => panic!("expected MissingKey, got {other:?}"),
        }
    }

    #[test]
    fn auth_gate_rejects_invalid_key() {
        let config = AuthConfig {
            keys: vec![KeyEntry {
                secret: "alpha-secret".into(),
                tenant: TenantId::new("tenant-a"),
                limits: default_rate_limit(),
            }],
        };
        let store = ApiKeyStore::from_config(&config).unwrap();
        let gate = AuthGate::new(Some(store));
        match gate.authorize(&headers_with_bearer("nope")) {
            AuthOutcome::Rejected(AuthRejection::InvalidKey) => {}
            other => panic!("expected InvalidKey, got {other:?}"),
        }
    }

    #[test]
    fn rate_limiter_enforces_rpm_independently_per_tenant() {
        let limiter = RateLimiter::new();
        let limit = RateLimit {
            requests_per_minute: 2,
            tokens_per_minute: u32::MAX,
        };
        let a = TenantId::new("a");
        let b = TenantId::new("b");
        assert!(limiter.check(&a, limit, 1));
        assert!(limiter.check(&a, limit, 1));
        assert!(
            !limiter.check(&a, limit, 1),
            "tenant a third request must be rejected"
        );
        assert!(limiter.check(&b, limit, 1));
        assert!(limiter.check(&b, limit, 1));
    }

    #[test]
    fn rate_limiter_enforces_tpm_ceiling() {
        let limiter = RateLimiter::new();
        let limit = RateLimit {
            requests_per_minute: u32::MAX,
            tokens_per_minute: 100,
        };
        let tenant = TenantId::new("a");
        assert!(limiter.check(&tenant, limit, 60));
        assert!(limiter.check(&tenant, limit, 40));
        assert!(!limiter.check(&tenant, limit, 1), "token budget exhausted");
    }

    #[test]
    fn key_file_rejects_duplicate_and_empty_secrets() {
        let dup = AuthConfig {
            keys: vec![
                KeyEntry {
                    secret: "s".into(),
                    tenant: TenantId::new("a"),
                    limits: default_rate_limit(),
                },
                KeyEntry {
                    secret: "s".into(),
                    tenant: TenantId::new("b"),
                    limits: default_rate_limit(),
                },
            ],
        };
        assert!(ApiKeyStore::from_config(&dup).is_err());
        let empty = AuthConfig {
            keys: vec![KeyEntry {
                secret: "".into(),
                tenant: TenantId::new("a"),
                limits: default_rate_limit(),
            }],
        };
        assert!(ApiKeyStore::from_config(&empty).is_err());
        assert!(ApiKeyStore::from_config(&AuthConfig { keys: vec![] }).is_err());
    }
}
