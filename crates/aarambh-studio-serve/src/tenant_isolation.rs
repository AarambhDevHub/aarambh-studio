//! Per-tenant concurrent-in-flight request ceiling.

#![deny(missing_docs)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::auth::TenantId;

/// Per-tenant isolation configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TenantIsolationConfig {
    /// Maximum simultaneous in-flight (queued + active) requests per tenant.
    pub max_concurrent_per_tenant: usize,
}

impl Default for TenantIsolationConfig {
    fn default() -> Self {
        Self {
            max_concurrent_per_tenant: usize::MAX,
        }
    }
}

impl TenantIsolationConfig {
    /// A configuration that does not cap any tenant (the loopback-open default).
    pub const UNLIMITED: Self = Self {
        max_concurrent_per_tenant: usize::MAX,
    };

    /// Whether this configuration enforces a real ceiling.
    pub fn is_bounded(&self) -> bool {
        self.max_concurrent_per_tenant != usize::MAX
    }
}

/// Failure to admit a tenant's request because the tenant is at its ceiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TenantBusy;

impl std::fmt::Display for TenantBusy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("tenant concurrent request ceiling reached")
    }
}

impl std::error::Error for TenantBusy {}

/// RAII permit: dropped → releases the tenant's semaphore slot.
#[derive(Debug)]
pub struct TenantPermit {
    _permit: OwnedSemaphorePermit,
}

/// Per-tenant concurrent-in-flight limiter.
pub struct TenantLimiter {
    per_tenant: Mutex<HashMap<TenantId, Arc<Semaphore>>>,
    max_concurrent: usize,
}

impl TenantLimiter {
    /// Build a limiter from its configuration.
    pub fn new(config: TenantIsolationConfig) -> Self {
        Self {
            per_tenant: Mutex::new(HashMap::new()),
            max_concurrent: config.max_concurrent_per_tenant,
        }
    }

    /// Try to admit a tenant's request.
    pub fn try_admit(&self, tenant: &TenantId) -> Result<TenantPermit, TenantBusy> {
        if self.max_concurrent == usize::MAX {
            let sem = self.semaphore_for(tenant);
            return sem
                .clone()
                .try_acquire_owned()
                .map(|permit| TenantPermit { _permit: permit })
                .map_err(|_| TenantBusy);
        }
        let sem = self.semaphore_for(tenant);
        sem.clone()
            .try_acquire_owned()
            .map(|permit| TenantPermit { _permit: permit })
            .map_err(|_| TenantBusy)
    }

    fn semaphore_for(&self, tenant: &TenantId) -> Arc<Semaphore> {
        let mut map = self
            .per_tenant
            .lock()
            .expect("tenant limiter mutex poisoned");
        if let Some(existing) = map.get(tenant) {
            return existing.clone();
        }
        let capacity = if self.max_concurrent == usize::MAX {
            1_000_000
        } else {
            self.max_concurrent
        };
        let sem = Arc::new(Semaphore::new(capacity));
        map.insert(tenant.clone(), sem.clone());
        sem
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unlimited_config_admits_everything() {
        let limiter = TenantLimiter::new(TenantIsolationConfig::UNLIMITED);
        let tenant = TenantId::new("a");
        let p1 = limiter.try_admit(&tenant).expect("first admit");
        let p2 = limiter.try_admit(&tenant).expect("second admit");
        let _ = (p1, p2);
    }

    #[test]
    fn permits_are_independent_per_tenant() {
        let limiter = TenantLimiter::new(TenantIsolationConfig {
            max_concurrent_per_tenant: 1,
        });
        let a = TenantId::new("a");
        let b = TenantId::new("b");
        let _a_permit = limiter.try_admit(&a).expect("a admitted");
        let _b_permit = limiter.try_admit(&b).expect("b admitted");
    }

    #[tokio::test]
    async fn permit_releases_on_drop_so_next_admission_succeeds() {
        let limiter = TenantLimiter::new(TenantIsolationConfig {
            max_concurrent_per_tenant: 1,
        });
        let tenant = TenantId::new("a");
        {
            let _permit = limiter.try_admit(&tenant).expect("first admit");
            assert!(
                limiter.try_admit(&tenant).is_err(),
                "second must be busy while held"
            );
        }
        let _permit = limiter.try_admit(&tenant).expect("admit after drop");
    }
}
