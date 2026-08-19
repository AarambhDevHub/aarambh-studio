use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use serde::Serialize;

#[derive(Debug, Default)]
/// Lock-free counters shared by HTTP handlers and the inference worker.
pub struct ServerMetrics {
    queued: AtomicUsize,
    active: AtomicUsize,
    requests_total: AtomicU64,
    requests_completed: AtomicU64,
    requests_cancelled: AtomicU64,
    requests_rejected: AtomicU64,
    safety_blocked: AtomicU64,
    generated_tokens: AtomicU64,
    decode_batches: AtomicU64,
    batch_items: AtomicU64,
    inference_errors: AtomicU64,
    auth_rejections: AtomicU64,
    rate_limited: AtomicU64,
    tenant_throttled: AtomicU64,
    prefix_cache_hits: AtomicU64,
    prefix_cache_misses: AtomicU64,
    prefix_cache_evictions: AtomicU64,
    prefix_cache_prefill_tokens_saved: AtomicU64,
}

impl ServerMetrics {
    pub(crate) fn request_queued(&self) {
        self.queued.fetch_add(1, Ordering::Relaxed);
        self.requests_total.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn request_queue_rollback(&self) {
        self.queued.fetch_sub(1, Ordering::Relaxed);
        self.requests_total.fetch_sub(1, Ordering::Relaxed);
    }

    pub(crate) fn request_admitted(&self) {
        self.queued.fetch_sub(1, Ordering::Relaxed);
        self.active.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn request_completed(&self) {
        self.active.fetch_sub(1, Ordering::Relaxed);
        self.requests_completed.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn request_cancelled(&self) {
        self.active.fetch_sub(1, Ordering::Relaxed);
        self.requests_cancelled.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn request_rejected(&self) {
        self.requests_rejected.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn safety_blocked(&self) {
        self.safety_blocked.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn generated_token(&self) {
        self.generated_tokens.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn decode_batch(&self, size: usize) {
        self.decode_batches.fetch_add(1, Ordering::Relaxed);
        self.batch_items.fetch_add(size as u64, Ordering::Relaxed);
    }

    pub(crate) fn inference_error(&self) {
        self.inference_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an authentication rejection (missing or invalid bearer key).
    pub fn auth_rejection(&self) {
        self.auth_rejections.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a request that was rejected by the per-key rate limiter.
    pub fn rate_limited(&self) {
        self.rate_limited.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a request rejected because the tenant was at its concurrency ceiling.
    pub fn tenant_throttled(&self) {
        self.tenant_throttled.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a prefix-cache hit and the number of prefill tokens it saved.
    pub fn prefix_cache_hit(&self, tokens_saved: u64) {
        self.prefix_cache_hits.fetch_add(1, Ordering::Relaxed);
        self.prefix_cache_prefill_tokens_saved
            .fetch_add(tokens_saved, Ordering::Relaxed);
    }

    /// Record a prefix-cache miss.
    pub fn prefix_cache_miss(&self) {
        self.prefix_cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a prefix-cache LRU eviction.
    pub fn prefix_cache_eviction(&self) {
        self.prefix_cache_evictions.fetch_add(1, Ordering::Relaxed);
    }

    /// Capture all counters as a serializable value.
    pub fn snapshot(&self) -> MetricsSnapshot {
        let decode_batches = self.decode_batches.load(Ordering::Relaxed);
        let batch_items = self.batch_items.load(Ordering::Relaxed);
        MetricsSnapshot {
            queued: self.queued.load(Ordering::Relaxed),
            active: self.active.load(Ordering::Relaxed),
            requests_total: self.requests_total.load(Ordering::Relaxed),
            requests_completed: self.requests_completed.load(Ordering::Relaxed),
            requests_cancelled: self.requests_cancelled.load(Ordering::Relaxed),
            requests_rejected: self.requests_rejected.load(Ordering::Relaxed),
            safety_blocked: self.safety_blocked.load(Ordering::Relaxed),
            generated_tokens: self.generated_tokens.load(Ordering::Relaxed),
            decode_batches,
            average_batch_size: if decode_batches == 0 {
                0.0
            } else {
                batch_items as f64 / decode_batches as f64
            },
            inference_errors: self.inference_errors.load(Ordering::Relaxed),
            auth_rejections: self.auth_rejections.load(Ordering::Relaxed),
            rate_limited: self.rate_limited.load(Ordering::Relaxed),
            tenant_throttled: self.tenant_throttled.load(Ordering::Relaxed),
            prefix_cache_hits: self.prefix_cache_hits.load(Ordering::Relaxed),
            prefix_cache_misses: self.prefix_cache_misses.load(Ordering::Relaxed),
            prefix_cache_evictions: self.prefix_cache_evictions.load(Ordering::Relaxed),
            prefix_cache_prefill_tokens_saved: self
                .prefix_cache_prefill_tokens_saved
                .load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
/// Point-in-time inference server metrics.
pub struct MetricsSnapshot {
    /// Requests waiting for admission.
    pub queued: usize,
    /// Requests currently generating.
    pub active: usize,
    /// Requests accepted since startup.
    pub requests_total: u64,
    /// Requests completed successfully.
    pub requests_completed: u64,
    /// Requests cancelled after client disconnect.
    pub requests_cancelled: u64,
    /// Requests rejected because the queue was full.
    pub requests_rejected: u64,
    /// Requests stopped by output safety.
    pub safety_blocked: u64,
    /// Generated token count.
    pub generated_tokens: u64,
    /// Shared decode pass count.
    pub decode_batches: u64,
    /// Mean number of requests per shared decode pass.
    pub average_batch_size: f64,
    /// Inference failures.
    pub inference_errors: u64,
    /// Authentication rejections (missing or invalid bearer key).
    pub auth_rejections: u64,
    /// Requests rejected by the per-key rate limiter.
    pub rate_limited: u64,
    /// Requests rejected by the per-tenant concurrency limiter.
    pub tenant_throttled: u64,
    /// Prefix-cache hits since startup.
    pub prefix_cache_hits: u64,
    /// Prefix-cache misses since startup.
    pub prefix_cache_misses: u64,
    /// Prefix-cache LRU evictions since startup.
    pub prefix_cache_evictions: u64,
    /// Prompt tokens whose prefill was skipped by a prefix-cache hit.
    pub prefix_cache_prefill_tokens_saved: u64,
}
