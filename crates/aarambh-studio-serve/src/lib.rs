//! Local and public OpenAI-compatible HTTP/SSE inference server.
#![deny(missing_docs)]

/// OpenAI-compatible HTTP data types.
pub mod api;
/// Per-key API identity and rate limiting (Phase 51).
pub mod auth;
/// Continuous inference scheduling and request sessions.
pub mod batching;
/// Atomic server telemetry.
pub mod metrics;
/// Prompt-prefix → cached KV state store (Phase 51).
pub mod prefix_cache;
/// Axum routing and server lifecycle.
pub mod server;
/// Per-tenant concurrent-in-flight ceiling (Phase 51).
pub mod tenant_isolation;

pub use auth::{
    ApiKey, ApiKeyStore, AuthConfig, AuthGate, AuthOutcome, AuthRejection, KeyEntry, RATE_WINDOW,
    RateLimit, RateLimiter, TenantId,
};
pub use batching::{BatcherConfig, BatcherHandle, GenerationEvent, GenerationRequest};
pub use metrics::{MetricsSnapshot, ServerMetrics};
pub use prefix_cache::{
    KvFootprint, PrefixCache, PrefixCacheConfig, PrefixCacheStats, PrefixLookup,
};
pub use server::{ServeConfig, build_router, run_server};
pub use tenant_isolation::{TenantBusy, TenantIsolationConfig, TenantLimiter, TenantPermit};
