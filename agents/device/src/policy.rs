//! Owner policy enforced locally (spec FR 07 to FR 09). The coordinator can tighten these limits
//! but never loosen them: the owner's local settings are the ceiling, even when offline.

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct OwnerPolicy {
    pub memory_cap_bytes: u64,
    pub allow_battery: bool,
    pub battery_stop_pct: f64,
    pub battery_resume_pct: f64,
    pub paused: bool,
    #[serde(default = "default_warm_hold")]
    pub warm_hold_minutes: u32,
}

fn default_warm_hold() -> u32 {
    30
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Power {
    Ac,
    Battery(Option<f64>),
    Unknown,
}

impl Power {
    pub fn label(&self) -> &'static str {
        match self {
            Power::Ac => "ac",
            Power::Battery(_) => "battery",
            Power::Unknown => "unknown",
        }
    }
    pub fn pct(&self) -> Option<f64> {
        match self {
            Power::Battery(p) => *p,
            _ => None,
        }
    }
}

/// Combine the owner's local policy with what the coordinator sent. Every field moves only toward
/// less contribution: smaller memory cap, battery use only if both allow it, paused if either says so.
pub fn merge(local: &OwnerPolicy, remote: &OwnerPolicy) -> OwnerPolicy {
    OwnerPolicy {
        memory_cap_bytes: local.memory_cap_bytes.min(remote.memory_cap_bytes),
        allow_battery: local.allow_battery && remote.allow_battery,
        battery_stop_pct: local.battery_stop_pct.max(remote.battery_stop_pct),
        battery_resume_pct: local.battery_resume_pct.max(remote.battery_resume_pct),
        paused: local.paused || remote.paused,
        warm_hold_minutes: local.warm_hold_minutes.min(remote.warm_hold_minutes),
    }
}

/// Decide whether contribution is paused. `battery_hold` carries hysteresis between calls so a
/// laptop near the threshold does not repeatedly load and unload (FR 08).
pub fn effective_paused(p: &OwnerPolicy, power: Power, battery_hold: &mut bool) -> bool {
    if p.paused {
        return true;
    }
    match power {
        Power::Ac => {
            *battery_hold = false;
            false
        }
        Power::Unknown => {
            // Unknown power defaults to paused battery contribution until the owner opts in.
            *battery_hold = !p.allow_battery;
            !p.allow_battery
        }
        Power::Battery(pct) => {
            if !p.allow_battery {
                *battery_hold = true;
                return true;
            }
            let Some(pct) = pct else {
                *battery_hold = true;
                return true;
            };
            if *battery_hold {
                *battery_hold = pct < p.battery_resume_pct;
            } else {
                *battery_hold = pct <= p.battery_stop_pct;
            }
            *battery_hold
        }
    }
}

/// Budget the agent reports: never above the owner's cap or what the OS says is available.
pub fn allocatable(cap: u64, available: u64) -> u64 {
    cap.min(available)
}

/// Execution authorization lease (spec 9.4): work stops when it is not renewed in time.
pub struct Lease {
    renewed: Option<Instant>,
    length: Duration,
}

impl Lease {
    pub fn new(length: Duration) -> Self {
        Lease { renewed: None, length }
    }
    pub fn renew(&mut self, now: Instant, length: Duration) {
        self.renewed = Some(now);
        self.length = length;
    }
    pub fn valid(&self, now: Instant) -> bool {
        matches!(self.renewed, Some(t) if now.duration_since(t) <= self.length)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> OwnerPolicy {
        OwnerPolicy {
            memory_cap_bytes: 8_000_000_000,
            allow_battery: true,
            battery_stop_pct: 30.0,
            battery_resume_pct: 40.0,
            paused: false,
            warm_hold_minutes: 10,
        }
    }

    #[test]
    fn coordinator_cannot_raise_limits() {
        let local = OwnerPolicy { allow_battery: false, ..policy() };
        let remote = OwnerPolicy { memory_cap_bytes: 64_000_000_000, allow_battery: true, ..policy() };
        let m = merge(&local, &remote);
        assert_eq!(m.memory_cap_bytes, 8_000_000_000);
        assert!(!m.allow_battery);
    }

    #[test]
    fn coordinator_can_pause_or_tighten() {
        let remote = OwnerPolicy { paused: true, memory_cap_bytes: 4_000_000_000, ..policy() };
        let m = merge(&policy(), &remote);
        assert!(m.paused);
        assert_eq!(m.memory_cap_bytes, 4_000_000_000);
    }

    #[test]
    fn battery_defaults_to_paused_unless_allowed() {
        let mut hold = false;
        let p = OwnerPolicy { allow_battery: false, ..policy() };
        assert!(effective_paused(&p, Power::Battery(Some(90.0)), &mut hold));
        assert!(!effective_paused(&p, Power::Ac, &mut hold));
        assert!(effective_paused(&p, Power::Unknown, &mut hold));
    }

    #[test]
    fn battery_hysteresis_between_stop_and_resume() {
        let p = policy();
        let mut hold = false;
        assert!(!effective_paused(&p, Power::Battery(Some(35.0)), &mut hold));
        assert!(effective_paused(&p, Power::Battery(Some(30.0)), &mut hold)); // hits stop threshold
        assert!(effective_paused(&p, Power::Battery(Some(35.0)), &mut hold)); // still below resume
        assert!(!effective_paused(&p, Power::Battery(Some(40.0)), &mut hold)); // resumes at 40
        assert!(effective_paused(&p, Power::Battery(None), &mut hold)); // unknown level pauses
    }

    #[test]
    fn owner_pause_always_wins() {
        let mut hold = false;
        let p = OwnerPolicy { paused: true, ..policy() };
        assert!(effective_paused(&p, Power::Ac, &mut hold));
    }

    #[test]
    fn allocatable_never_exceeds_cap_or_available() {
        assert_eq!(allocatable(64, 100), 64);
        assert_eq!(allocatable(64, 10), 10);
    }

    #[test]
    fn lease_expires_without_renewal() {
        let start = Instant::now();
        let mut lease = Lease::new(Duration::from_secs(30));
        assert!(!lease.valid(start));
        lease.renew(start, Duration::from_secs(30));
        assert!(lease.valid(start + Duration::from_secs(29)));
        assert!(!lease.valid(start + Duration::from_secs(31)));
    }
}
