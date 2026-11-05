//! Capability probes (spec FR 06): OS, architecture, available memory, power, and a short
//! memory-bandwidth measurement, since decode speed is bounded mainly by memory bandwidth.
//! Simulation profiles stand in for the reference devices when testing on one machine.

use crate::policy::Power;
use std::process::Command;
use std::time::Instant;
use sysinfo::System;

#[derive(Clone, Debug)]
pub struct Probe {
    pub os: String,
    pub arch: String,
    pub available_bytes: u64,
    pub power: Power,
    pub mem_bandwidth_gbps: f64,
    pub simulated_rtt_ms: f64,
}

pub fn real() -> Probe {
    let mut sys = System::new();
    sys.refresh_memory();
    Probe {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        available_bytes: sys.available_memory(),
        power: detect_power(),
        mem_bandwidth_gbps: measure_bandwidth(),
        simulated_rtt_ms: 0.0,
    }
}

/// Reference-group stand-ins from spec 3 and 3.2. Numbers are planning values, not measurements.
pub fn simulated(profile: &str) -> Option<Probe> {
    let p = |os: &str, arch: &str, gb: f64, power, bw, rtt| Probe {
        os: os.into(),
        arch: arch.into(),
        available_bytes: (gb * 1e9) as u64,
        power,
        mem_bandwidth_gbps: bw,
        simulated_rtt_ms: rtt,
    };
    match profile {
        "mac-studio" => Some(p("macos", "aarch64", 110.0, Power::Ac, 800.0, 1.0)),
        "hp-elitebook" => Some(p("windows", "x86_64", 10.2, Power::Ac, 51.0, 5.0)),
        "hp-elitebook-battery" => Some(p("windows", "x86_64", 10.2, Power::Battery(Some(80.0)), 51.0, 5.0)),
        "ubuntu-vm" => Some(p("linux", "x86_64", 5.0, Power::Ac, 15.0, 60.0)),
        _ => None,
    }
}

fn detect_power() -> Power {
    if cfg!(target_os = "macos") {
        if let Ok(out) = Command::new("pmset").args(["-g", "batt"]).output() {
            let text = String::from_utf8_lossy(&out.stdout);
            if text.contains("AC Power") {
                return Power::Ac;
            }
            if text.contains("Battery Power") {
                return Power::Battery(parse_pct(&text));
            }
        }
        return Power::Unknown;
    }
    if cfg!(target_os = "linux") {
        let mut on_ac = None;
        let mut pct = None;
        if let Ok(entries) = std::fs::read_dir("/sys/class/power_supply") {
            for e in entries.flatten() {
                let path = e.path();
                let kind = std::fs::read_to_string(path.join("type")).unwrap_or_default();
                if kind.trim() == "Mains" {
                    on_ac = Some(std::fs::read_to_string(path.join("online")).unwrap_or_default().trim() == "1");
                } else if kind.trim() == "Battery" {
                    pct = std::fs::read_to_string(path.join("capacity")).ok().and_then(|s| s.trim().parse().ok());
                }
            }
        }
        return match (on_ac, pct) {
            (Some(true), _) => Power::Ac,
            (Some(false), p) => Power::Battery(p),
            (None, None) => Power::Ac, // desktops and VMs without power supplies report nothing
            (None, p) => Power::Battery(p),
        };
    }
    Power::Unknown // Windows probe is a Phase 1 item
}

fn parse_pct(text: &str) -> Option<f64> {
    let idx = text.find('%')?;
    let digits: String = text[..idx].chars().rev().take_while(|c| c.is_ascii_digit()).collect();
    digits.chars().rev().collect::<String>().parse().ok()
}

/// Copy a 64 MiB buffer several times and report read+write bytes per second, in GB/s.
pub fn measure_bandwidth() -> f64 {
    const BYTES: usize = 64 << 20;
    const ITERS: usize = 6;
    let src = vec![1u8; BYTES];
    let mut dst = vec![0u8; BYTES];
    dst.copy_from_slice(&src); // warm up and fault in pages
    let start = Instant::now();
    for i in 0..ITERS {
        dst.copy_from_slice(&src);
        std::hint::black_box(&dst[i]);
    }
    let secs = start.elapsed().as_secs_f64().max(1e-9);
    (2 * BYTES * ITERS) as f64 / secs / 1e9
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_battery_percentage() {
        let sample = "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t76%; discharging;";
        assert_eq!(parse_pct(sample), Some(76.0));
    }

    #[test]
    fn simulated_profiles_cover_reference_group() {
        for name in ["mac-studio", "hp-elitebook", "ubuntu-vm"] {
            assert!(simulated(name).is_some(), "{name}");
        }
        assert!(simulated("unknown").is_none());
    }

    #[test]
    fn bandwidth_is_positive() {
        assert!(measure_bandwidth() > 0.0);
    }
}
