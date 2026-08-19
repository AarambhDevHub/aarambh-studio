//! Acceptance tests for Phase 51 tenant isolation (tenant_isolation.rs).
//!
//! #5 — `one_tenants_request_burst_does_not_starve_another_tenants_admitted_queue`

mod common;

use std::sync::Arc;
use std::time::Duration;

use aarambh_studio_serve::{
    AuthConfig, BatcherConfig, KeyEntry, RateLimit, ServeConfig, ServerMetrics, TenantBusy,
    TenantId, TenantIsolationConfig, TenantLimiter, build_router,
};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use tower::ServiceExt;

use common::test_engine;

fn bounded_config(capacity: usize) -> ServeConfig {
    ServeConfig {
        auth: Some(AuthConfig {
            keys: vec![
                KeyEntry {
                    secret: "alpha-secret".into(),
                    tenant: TenantId::new("tenant-a"),
                    limits: RateLimit::UNLIMITED,
                },
                KeyEntry {
                    secret: "beta-secret".into(),
                    tenant: TenantId::new("tenant-b"),
                    limits: RateLimit::UNLIMITED,
                },
            ],
        }),
        tenant_isolation: TenantIsolationConfig {
            max_concurrent_per_tenant: capacity,
        },
        safety_policy: None,
        max_request_tokens: 8,
        ..ServeConfig::default()
    }
}

#[test]
fn one_tenants_request_burst_does_not_starve_another_tenants_admitted_queue() {
    // Verbatim roadmap acceptance test #5: one tenant's burst of requests
    // cannot starve another tenant's already-admitted queue. We exercise the
    // TenantLimiter directly so the assertion is deterministic — no
    // inference-worker timing dependency.
    let limiter = TenantLimiter::new(TenantIsolationConfig {
        max_concurrent_per_tenant: 1,
    });
    let a = TenantId::new("tenant-a");
    let b = TenantId::new("tenant-b");

    // Tenant A bursts: acquires its single concurrency slot.
    let a_permit = limiter.try_admit(&a).expect("tenant A first admit");

    // Tenant A's next in-flight request is rejected (ceiling reached) —
    // but this rejection must NOT affect tenant B.
    let err = limiter.try_admit(&a).unwrap_err();
    assert_eq!(err, TenantBusy, "tenant A second request must be throttled");

    // Tenant B is independently admitted.
    let _b_permit = limiter
        .try_admit(&b)
        .expect("tenant B admitted despite A's burst");

    // When A's permit is released, A can be admitted again — the ceiling
    // is per-in-flight, not per-history.
    drop(a_permit);
    let _a_permit = limiter
        .try_admit(&a)
        .expect("tenant A re-admitted after release");
}

#[test]
fn tenant_limiter_unlimited_config_admits_concurrently() {
    // The loopback-open default must not cap any tenant. Two simultaneous
    // admits for the same tenant both succeed.
    let limiter = TenantLimiter::new(TenantIsolationConfig::UNLIMITED);
    let tenant = TenantId::new("local");
    let _p1 = limiter.try_admit(&tenant).expect("first admit");
    let _p2 = limiter.try_admit(&tenant).expect("second admit");
}

#[tokio::test]
async fn tenant_busy_request_returns_429_at_http_layer() {
    // End-to-end HTTP integration: when a tenant is at its concurrency
    // ceiling, the next request from that tenant is rejected with HTTP 429
    // while a request from a different tenant succeeds with HTTP 200. A1 is
    // a streaming request so its admission permit is moved into the spawned
    // SSE consumer task and held deterministically (no sleep).
    let metrics = Arc::new(ServerMetrics::default());
    let batcher = aarambh_studio_serve::BatcherHandle::start(
        test_engine(1),
        BatcherConfig {
            max_batch_size: 4,
            queue_capacity: 16,
            batch_wait: Duration::from_millis(1),
            prefill_chunk_size: 8,
        },
        None,
        metrics.clone(),
    )
    .unwrap();
    let router = build_router(bounded_config(1), batcher.clone(), metrics).unwrap();

    // A1: tenant A streaming request. The handler admits A (acquires the
    // permit), spawns the SSE consumer task (which now owns the permit),
    // and returns the SSE Response. We hold the Response so the spawned
    // task continues to hold the permit.
    let a1_response = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("authorization", "Bearer alpha-secret")
                .body(Body::from(
                    r#"{"model":"aarambh-studio-local","messages":[{"role":"user","content":"Hello"}],"max_tokens":2,"stream":true}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(a1_response.status(), StatusCode::OK);

    // A2: tenant A's second concurrent request — must be rejected with 429
    // because tenant A is at its ceiling of 1.
    let a2_response = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("authorization", "Bearer alpha-secret")
                .body(Body::from(
                    r#"{"model":"aarambh-studio-local","messages":[{"role":"user","content":"Hello"}],"max_tokens":2}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        a2_response.status(),
        StatusCode::TOO_MANY_REQUESTS,
        "tenant A second concurrent request must be throttled"
    );
    let body = a2_response.into_body().collect().await.unwrap().to_bytes();
    let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(value["error"]["code"], "tenant_busy");

    // B: tenant B's request — must succeed with 200 because the ceiling is
    // per-tenant, not global. This is the "does not starve another tenant"
    // half of the acceptance property, exercised at the HTTP layer.
    let b_response = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("authorization", "Bearer beta-secret")
                .body(Body::from(
                    r#"{"model":"aarambh-studio-local","messages":[{"role":"user","content":"Hello"}],"max_tokens":2}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        b_response.status(),
        StatusCode::OK,
        "tenant B must not be starved by tenant A's burst"
    );

    // Drain A1's response body so the spawned SSE consumer task can exit
    // and release the permit cleanly.
    let _ = a1_response.into_body().collect().await.unwrap().to_bytes();
    batcher.shutdown();
}
