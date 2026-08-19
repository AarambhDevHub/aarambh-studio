//! Acceptance tests for Phase 51 multi-tenant auth (auth.rs).
//!
//! #1 — `request_with_missing_or_invalid_api_key_is_rejected_before_admission`
//! #2 — `per_tenant_rate_limit_is_enforced_independently_per_key`

mod common;

use std::sync::Arc;

use aarambh_studio_serve::{
    ApiKeyStore, AuthConfig, AuthGate, AuthOutcome, AuthRejection, KeyEntry, RateLimit,
    RateLimiter, ServeConfig, TenantId, build_router,
};
use aarambh_studio_serve::{BatcherConfig, ServerMetrics};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use tower::ServiceExt;

use common::test_engine;

fn two_tenant_config() -> ServeConfig {
    ServeConfig {
        auth: Some(AuthConfig {
            keys: vec![
                KeyEntry {
                    secret: "alpha-secret".into(),
                    tenant: TenantId::new("tenant-a"),
                    limits: RateLimit {
                        requests_per_minute: 2,
                        tokens_per_minute: 100,
                    },
                },
                KeyEntry {
                    secret: "beta-secret".into(),
                    tenant: TenantId::new("tenant-b"),
                    limits: RateLimit {
                        requests_per_minute: 2,
                        tokens_per_minute: 100,
                    },
                },
            ],
        }),
        safety_policy: None,
        max_request_tokens: 8,
        ..ServeConfig::default()
    }
}

#[test]
fn request_with_missing_or_invalid_api_key_is_rejected_before_admission() {
    // Verbatim roadmap acceptance test #1: a request without a valid bearer
    // key must be rejected with HTTP 401 before being admitted to the
    // continuous batcher. We assert this at the HTTP layer (the response
    // carries the rejection) and at the auth-gate layer (the gate's outcome
    // discriminates between missing and invalid).
    let metrics = Arc::new(ServerMetrics::default());
    let batcher = aarambh_studio_serve::BatcherHandle::start(
        test_engine(1),
        BatcherConfig {
            max_batch_size: 2,
            queue_capacity: 8,
            batch_wait: std::time::Duration::from_millis(1),
            prefill_chunk_size: 8,
        },
        None,
        metrics.clone(),
    )
    .unwrap();
    let router = build_router(two_tenant_config(), batcher.clone(), metrics.clone()).unwrap();

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    // Missing bearer: 401.
    let response = rt
        .block_on(router.clone().oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"aarambh-studio-local","messages":[{"role":"user","content":"Hello"}],"max_tokens":1}"#,
                ))
                .unwrap(),
        ))
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    let body = rt
        .block_on(response.into_body().collect())
        .unwrap()
        .to_bytes();
    let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(value["error"]["code"], "invalid_api_key");
    // No request reached the batcher: the request counter is still zero.
    assert_eq!(batcher_metrics(&metrics).requests_total, 0);

    // Invalid bearer: 401 as well, but the gate reports InvalidKey.
    let response = rt
        .block_on(router.oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("authorization", "Bearer not-a-real-key")
                .body(Body::from(
                    r#"{"model":"aarambh-studio-local","messages":[{"role":"user","content":"Hello"}],"max_tokens":1}"#,
                ))
                .unwrap(),
        ))
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert!(batcher_metrics(&metrics).requests_total == 0);

    batcher.shutdown();
}

#[test]
fn per_tenant_rate_limit_is_enforced_independently_per_key() {
    // Verbatim roadmap acceptance test #2: rate limits must be enforced
    // independently per tenant — tenant A hitting its ceiling must not block
    // tenant B. We exercise the RateLimiter directly so the test is
    // deterministic (no wall-clock dependency on the inference worker).
    let limiter = RateLimiter::new();
    let limit = RateLimit {
        requests_per_minute: 2,
        tokens_per_minute: u32::MAX,
    };
    let a = TenantId::new("tenant-a");
    let b = TenantId::new("tenant-b");

    // Both tenants start fresh: each can use its full quota independently.
    assert!(limiter.check(&a, limit, 1), "tenant A 1st request");
    assert!(limiter.check(&a, limit, 1), "tenant A 2nd request");
    assert!(limiter.check(&b, limit, 1), "tenant B 1st request");
    assert!(limiter.check(&b, limit, 1), "tenant B 2nd request");

    // Tenant A's third request is rejected; tenant B is unaffected.
    assert!(
        !limiter.check(&a, limit, 1),
        "tenant A third request must be rejected"
    );
    assert!(
        !limiter.check(&b, limit, 1),
        "tenant B third request must also be rejected (limit is per-tenant, not shared)"
    );
}

#[test]
fn auth_gate_resolves_known_secret_to_authenticated_outcome() {
    let store = ApiKeyStore::from_config(&two_tenant_config().auth.unwrap()).unwrap();
    let gate = AuthGate::new(Some(store));
    let mut headers = axum::http::HeaderMap::new();
    headers.insert(
        "authorization",
        axum::http::HeaderValue::from_static("Bearer alpha-secret"),
    );
    match gate.authorize(&headers) {
        AuthOutcome::Authenticated { key } => {
            assert_eq!(key.tenant.as_str(), "tenant-a");
            assert_eq!(key.limits.requests_per_minute, 2);
        }
        other => panic!("expected Authenticated, got {other:?}"),
    }
    assert!(!gate.is_open());
}

#[test]
fn auth_gate_distinguishes_missing_from_invalid_key() {
    let store = ApiKeyStore::from_config(&two_tenant_config().auth.unwrap()).unwrap();
    let gate = AuthGate::new(Some(store));
    let empty = axum::http::HeaderMap::new();
    match gate.authorize(&empty) {
        AuthOutcome::Rejected(AuthRejection::MissingKey) => {}
        other => panic!("expected MissingKey, got {other:?}"),
    }
    let mut headers = axum::http::HeaderMap::new();
    headers.insert(
        "authorization",
        axum::http::HeaderValue::from_static("Bearer wrong-secret"),
    );
    match gate.authorize(&headers) {
        AuthOutcome::Rejected(AuthRejection::InvalidKey) => {}
        other => panic!("expected InvalidKey, got {other:?}"),
    }
}

#[test]
fn auth_config_rejects_duplicate_and_empty_secrets() {
    let dup = AuthConfig {
        keys: vec![
            KeyEntry {
                secret: "shared".into(),
                tenant: TenantId::new("a"),
                limits: RateLimit::UNLIMITED,
            },
            KeyEntry {
                secret: "shared".into(),
                tenant: TenantId::new("b"),
                limits: RateLimit::UNLIMITED,
            },
        ],
    };
    assert!(ApiKeyStore::from_config(&dup).is_err());
    let empty = AuthConfig {
        keys: vec![KeyEntry {
            secret: "  ".into(),
            tenant: TenantId::new("a"),
            limits: RateLimit::UNLIMITED,
        }],
    };
    assert!(ApiKeyStore::from_config(&empty).is_err());
}

fn batcher_metrics(metrics: &ServerMetrics) -> aarambh_studio_serve::MetricsSnapshot {
    metrics.snapshot()
}
