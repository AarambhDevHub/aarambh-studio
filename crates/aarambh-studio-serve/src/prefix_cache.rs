//! Prompt-prefix → cached KV state store with LRU eviction.
//!
//! # Layout
//!
//! [`PrefixCache`] maps a prompt's token-id sequence to the prefilled
//! `KvCache` produced by
//! `InferenceEngine::prepare_session_with_prefix_cache`. On lookup it finds
//! the longest cached entry whose token-id sequence is a prefix of the
//! incoming prompt's token ids, and returns the matching KV (cloned) plus
//! the matched length. On store it inserts or refreshes the entry,
//! evicting least-recently-used entries to stay under the configured memory
//! ceiling.
//!
//! # Honesty boundary
//!
//! The memory ceiling is an *estimate* computed from the model configuration
//! ([`KvFootprint`]): `n_layers × 2 × n_kv_heads × head_dim × dtype_bytes` per
//! cached sequence token. It is honest about being an estimate — the actual
//! per-layer KV storage may differ slightly for MLA/Gated DeltaNet layers —
//! but it is deterministic and is the value the LRU evictor uses. A cache hit
//! is reported via [`PrefixCacheStats`] as a counted saving
//! (`prefill_tokens_saved`), never an assumed one.

#![deny(missing_docs)]

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use aarambh_studio_core::ModelConfig;
use aarambh_studio_inference::KvCache;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Honest KV-memory estimate per cached sequence token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvFootprint {
    bytes_per_token: usize,
}

impl KvFootprint {
    /// Build a footprint estimate from a model configuration and dtype size.
    pub fn from_model_config(config: &ModelConfig, dtype_bytes: usize) -> Self {
        let head_dim = config.hidden_dim.checked_div(config.n_heads).unwrap_or(0);
        let bytes_per_token = config.n_layers * 2 * config.n_kv_heads * head_dim * dtype_bytes;
        Self { bytes_per_token }
    }

    /// Build a footprint directly from a known per-token byte cost (for tests).
    pub fn from_bytes_per_token(bytes_per_token: usize) -> Self {
        Self { bytes_per_token }
    }

    /// Estimated bytes occupied by a cached KV sequence of `seq_len` tokens.
    pub fn bytes_for(&self, seq_len: usize) -> usize {
        self.bytes_per_token.saturating_mul(seq_len)
    }

    /// Per-token byte cost.
    pub fn bytes_per_token(&self) -> usize {
        self.bytes_per_token
    }
}

/// Prefix-cache configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefixCacheConfig {
    /// Maximum estimated KV bytes across all cached prefixes (`0` disables).
    pub max_bytes: usize,
    /// Maximum number of cached entries (hard cap; LRU-evict beyond this).
    pub max_entries: usize,
}

impl Default for PrefixCacheConfig {
    fn default() -> Self {
        Self::DISABLED
    }
}

impl PrefixCacheConfig {
    /// Disabled configuration (the documented default — no prefix caching).
    pub const DISABLED: Self = Self {
        max_bytes: 0,
        max_entries: 0,
    };

    /// Whether prefix caching is enabled.
    pub fn is_enabled(&self) -> bool {
        self.max_bytes > 0 && self.max_entries > 0
    }
}

/// Result of looking up a prompt's prefix in the cache.
#[derive(Debug)]
pub enum PrefixLookup {
    /// A cached prefix was found. `cache` is a clone of the cached KV (the
    /// caller truncates it to `matched_len`); `matched_len` is capped at
    /// `prompt_ids.len() - 1` so at least one token remains to prefill.
    Hit {
        /// Cloned KV for the matched prefix.
        cache: KvCache,
        /// Number of prompt token ids covered by the cached prefix.
        matched_len: usize,
    },
    /// No usable cached prefix was found.
    Miss,
}

struct CacheEntry {
    token_ids: Vec<u32>,
    cache: KvCache,
    seq_len: usize,
    last_used: Instant,
}

struct PrefixCacheInner {
    entries: HashMap<u64, CacheEntry>,
    lru: Vec<u64>,
    bytes: usize,
}

struct PrefixCacheStatsAtomic {
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    prefill_tokens_saved: AtomicU64,
    entries: AtomicUsize,
    bytes: AtomicUsize,
}

/// Point-in-time prefix-cache statistics.
#[derive(Debug, Clone, Default, Serialize)]
pub struct PrefixCacheStats {
    /// Cache hits since startup.
    pub hits: u64,
    /// Cache misses since startup.
    pub misses: u64,
    /// LRU evictions since startup.
    pub evictions: u64,
    /// Prompt tokens whose prefill was skipped thanks to a hit.
    pub prefill_tokens_saved: u64,
    /// Number of entries currently cached.
    pub entries: usize,
    /// Estimated bytes currently cached.
    pub bytes: usize,
}

/// Prompt-prefix → cached KV store with LRU eviction.
pub struct PrefixCache {
    inner: Mutex<PrefixCacheInner>,
    footprint: KvFootprint,
    config: PrefixCacheConfig,
    stats: PrefixCacheStatsAtomic,
}

impl PrefixCache {
    /// Create a new prefix cache.
    pub fn new(config: PrefixCacheConfig, footprint: KvFootprint) -> Self {
        Self {
            inner: Mutex::new(PrefixCacheInner {
                entries: HashMap::new(),
                lru: Vec::new(),
                bytes: 0,
            }),
            footprint,
            config,
            stats: PrefixCacheStatsAtomic {
                hits: AtomicU64::new(0),
                misses: AtomicU64::new(0),
                evictions: AtomicU64::new(0),
                prefill_tokens_saved: AtomicU64::new(0),
                entries: AtomicUsize::new(0),
                bytes: AtomicUsize::new(0),
            },
        }
    }

    /// Find the longest cached entry whose token ids are a prefix of
    /// `prompt_ids`.
    pub fn lookup(&self, prompt_ids: &[u32]) -> PrefixLookup {
        if !self.config.is_enabled() || prompt_ids.len() <= 1 {
            self.stats.misses.fetch_add(1, Ordering::Relaxed);
            return PrefixLookup::Miss;
        }
        let mut inner = self.inner.lock().expect("prefix cache mutex poisoned");
        let cap = prompt_ids.len().saturating_sub(1);
        let mut best: Option<(u64, usize)> = None;
        for (hash, entry) in inner.entries.iter() {
            let matched = common_prefix_len(&entry.token_ids, prompt_ids).min(cap);
            if matched > 0 && best.map(|(_, b)| matched > b).unwrap_or(true) {
                best = Some((*hash, matched));
            }
        }
        match best {
            Some((hash, matched_len)) => {
                let entry = inner.entries.get(&hash).expect("entry present");
                let cache = entry.cache.snapshot();
                touch_lru(&mut inner.lru, hash);
                inner
                    .entries
                    .get_mut(&hash)
                    .expect("entry present")
                    .last_used = Instant::now();
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                self.stats
                    .prefill_tokens_saved
                    .fetch_add(matched_len as u64, Ordering::Relaxed);
                self.sync_usage(&inner);
                PrefixLookup::Hit { cache, matched_len }
            }
            None => {
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                self.sync_usage(&inner);
                PrefixLookup::Miss
            }
        }
    }

    /// Store (or refresh) the full-prefix KV for `prompt_ids`.
    pub fn store(&self, prompt_ids: &[u32], cache: &KvCache) {
        if !self.config.is_enabled() || prompt_ids.is_empty() {
            return;
        }
        let hash = hash_token_ids(prompt_ids);
        let seq_len = prompt_ids.len();
        let entry_bytes = self.footprint.bytes_for(seq_len);
        let mut inner = self.inner.lock().expect("prefix cache mutex poisoned");
        if let Some(existing) = inner.entries.get_mut(&hash) {
            if existing.token_ids == prompt_ids {
                existing.last_used = Instant::now();
                touch_lru(&mut inner.lru, hash);
                self.sync_usage(&inner);
                return;
            }
            let prev_bytes = self.footprint.bytes_for(existing.seq_len);
            inner.bytes = inner.bytes.saturating_sub(prev_bytes);
        }
        while (inner.bytes.saturating_add(entry_bytes) > self.config.max_bytes
            && !inner.entries.is_empty())
            || (inner.entries.len() >= self.config.max_entries && !inner.entries.is_empty())
        {
            let Some(victim) = inner.lru.first().copied() else {
                break;
            };
            if let Some(removed) = inner.entries.remove(&victim) {
                inner.bytes = inner
                    .bytes
                    .saturating_sub(self.footprint.bytes_for(removed.seq_len));
                inner.lru.remove(0);
                self.stats.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }
        inner.entries.insert(
            hash,
            CacheEntry {
                token_ids: prompt_ids.to_vec(),
                cache: cache.snapshot(),
                seq_len,
                last_used: Instant::now(),
            },
        );
        inner.bytes = inner.bytes.saturating_add(entry_bytes);
        touch_lru(&mut inner.lru, hash);
        self.sync_usage(&inner);
    }

    /// Snapshot the cache statistics.
    pub fn stats(&self) -> PrefixCacheStats {
        PrefixCacheStats {
            hits: self.stats.hits.load(Ordering::Relaxed),
            misses: self.stats.misses.load(Ordering::Relaxed),
            evictions: self.stats.evictions.load(Ordering::Relaxed),
            prefill_tokens_saved: self.stats.prefill_tokens_saved.load(Ordering::Relaxed),
            entries: self.stats.entries.load(Ordering::Relaxed),
            bytes: self.stats.bytes.load(Ordering::Relaxed),
        }
    }

    fn sync_usage(&self, inner: &PrefixCacheInner) {
        self.stats
            .entries
            .store(inner.entries.len(), Ordering::Relaxed);
        self.stats.bytes.store(inner.bytes, Ordering::Relaxed);
    }
}

fn touch_lru(lru: &mut Vec<u64>, hash: u64) {
    lru.retain(|entry| *entry != hash);
    lru.push(hash);
}

fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

fn hash_token_ids(token_ids: &[u32]) -> u64 {
    let mut hasher = Sha256::new();
    for id in token_ids {
        hasher.update(id.to_le_bytes());
    }
    let digest = hasher.finalize();
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    u64::from_le_bytes(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn footprint(per_token: usize) -> KvFootprint {
        KvFootprint::from_bytes_per_token(per_token)
    }

    #[test]
    fn disabled_config_is_a_noop() {
        let cache = PrefixCache::new(PrefixCacheConfig::DISABLED, footprint(8));
        assert!(matches!(cache.lookup(&[1, 2, 3, 4]), PrefixLookup::Miss));
        cache.store(&[1, 2, 3, 4], &stub_cache());
        assert_eq!(cache.stats().entries, 0);
        assert_eq!(cache.stats().hits, 0);
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn lookup_returns_miss_for_empty_cache() {
        let cache = PrefixCache::new(
            PrefixCacheConfig {
                max_bytes: 1024,
                max_entries: 4,
            },
            footprint(8),
        );
        assert!(matches!(cache.lookup(&[1, 2, 3, 4]), PrefixLookup::Miss));
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn lookup_returns_hit_for_identical_prompt_after_store() {
        let cache = PrefixCache::new(
            PrefixCacheConfig {
                max_bytes: 1024,
                max_entries: 4,
            },
            footprint(8),
        );
        cache.store(&[1, 2, 3, 4, 5], &stub_cache());
        match cache.lookup(&[1, 2, 3, 4, 5]) {
            PrefixLookup::Hit { matched_len, .. } => {
                assert_eq!(matched_len, 4);
            }
            other => panic!("expected hit, got {other:?}"),
        }
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.prefill_tokens_saved, 4);
    }

    #[test]
    fn lookup_returns_partial_hit_for_shared_prefix() {
        let cache = PrefixCache::new(
            PrefixCacheConfig {
                max_bytes: 1024,
                max_entries: 4,
            },
            footprint(8),
        );
        cache.store(&[1, 2, 3, 4, 5], &stub_cache());
        match cache.lookup(&[1, 2, 3, 9, 10, 11]) {
            PrefixLookup::Hit { matched_len, .. } => assert_eq!(matched_len, 3),
            other => panic!("expected partial hit, got {other:?}"),
        }
    }

    #[test]
    fn lookup_picks_the_longest_matching_prefix() {
        let cache = PrefixCache::new(
            PrefixCacheConfig {
                max_bytes: 1024,
                max_entries: 8,
            },
            footprint(8),
        );
        cache.store(&[1, 2, 3], &stub_cache());
        cache.store(&[1, 2, 3, 4, 5], &stub_cache());
        match cache.lookup(&[1, 2, 3, 4, 5, 6]) {
            PrefixLookup::Hit { matched_len, .. } => assert_eq!(matched_len, 5),
            other => panic!("expected longest-prefix hit, got {other:?}"),
        }
    }

    #[test]
    fn store_refreshes_existing_entry_without_growing_entries() {
        let cache = PrefixCache::new(
            PrefixCacheConfig {
                max_bytes: 1024,
                max_entries: 4,
            },
            footprint(8),
        );
        cache.store(&[1, 2, 3, 4], &stub_cache());
        cache.store(&[1, 2, 3, 4], &stub_cache());
        assert_eq!(cache.stats().entries, 1);
        assert_eq!(cache.stats().evictions, 0);
    }

    #[test]
    fn memory_ceiling_evicts_lru() {
        let cache = PrefixCache::new(
            PrefixCacheConfig {
                max_bytes: 32,
                max_entries: 8,
            },
            footprint(8),
        );
        cache.store(&[1, 2, 3, 4], &stub_cache());
        assert_eq!(cache.stats().entries, 1);
        assert_eq!(cache.stats().bytes, 32);
        cache.store(&[5, 6, 7, 8], &stub_cache());
        assert!(cache.stats().evictions >= 1);
        assert_eq!(cache.stats().bytes, 32, "byte ceiling is respected");
        match cache.lookup(&[1, 2, 3, 4, 5]) {
            PrefixLookup::Miss => {}
            other => panic!("expected miss after eviction, got {other:?}"),
        }
        match cache.lookup(&[5, 6, 7, 8, 9]) {
            PrefixLookup::Hit { matched_len, .. } => assert_eq!(matched_len, 4),
            other => panic!("expected hit for surviving entry, got {other:?}"),
        }
    }

    #[test]
    fn entry_count_ceiling_evicts_lru() {
        let cache = PrefixCache::new(
            PrefixCacheConfig {
                max_bytes: 1024,
                max_entries: 2,
            },
            footprint(8),
        );
        cache.store(&[1, 2, 3, 4], &stub_cache());
        cache.store(&[5, 6, 7, 8], &stub_cache());
        cache.store(&[9, 10, 11, 12], &stub_cache());
        assert_eq!(cache.stats().entries, 2);
        assert!(cache.stats().evictions >= 1);
        match cache.lookup(&[1, 2, 3, 4, 5]) {
            PrefixLookup::Miss => {}
            other => panic!("expected miss for evicted entry, got {other:?}"),
        }
    }

    fn stub_cache() -> KvCache {
        KvCache::new(0)
    }
}
