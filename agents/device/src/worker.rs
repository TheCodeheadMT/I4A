//! Agent-managed inference worker (llama.cpp rpc-server) under the owner's memory cap (FR 07 to FR 09).
//!
//! llama.cpp's RPC worker advertises all host memory and has no cap option (Phase 1 finding), so the
//! agent owns the process: it starts the worker only while contribution is allowed and the lease is
//! valid, stops it on pause or lease expiry, and kills it if resident memory exceeds the cap. The
//! watchdog polls every 250 ms, so memory can exceed the cap for at most one poll interval; hard
//! prevention needs OS limits (cgroups on Linux, Job Objects on Windows), which are a Phase 2 item.

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use sysinfo::{Pid, ProcessesToUpdate, System};

#[derive(Clone, Debug)]
pub struct WorkerConfig {
    pub bin: PathBuf,
    pub host: String,
    pub port: u16,
    pub threads: u32,
}

impl WorkerConfig {
    pub fn endpoint(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

#[derive(Debug, PartialEq)]
pub enum Decision {
    Start,
    Stop(&'static str),
    Kill(&'static str),
    Keep,
}

/// Pure decision rule so the policy can be tested without processes.
pub fn decide(should_run: bool, running: bool, rss: Option<u64>, cap: u64) -> Decision {
    if running {
        if rss.is_some_and(|r| r > cap) {
            return Decision::Kill("memory cap exceeded");
        }
        if !should_run {
            return Decision::Stop("contribution paused or lease not valid");
        }
        return Decision::Keep;
    }
    if should_run {
        Decision::Start
    } else {
        Decision::Keep
    }
}

pub struct Worker {
    cfg: WorkerConfig,
    child: Option<Child>,
    sys: System,
    pub rss: u64,
    pub peak_rss: u64,
    pub cap_kills: u32,
    pub last_stop: Option<String>,
    /// After a cap kill, stay stopped until the owner raises the cap above what the worker needed.
    blocked_below: Option<u64>,
}

impl Worker {
    pub fn new(cfg: WorkerConfig) -> Self {
        Worker { cfg, child: None, sys: System::new(), rss: 0, peak_rss: 0, cap_kills: 0, last_stop: None,
                 blocked_below: None }
    }

    pub fn running(&mut self) -> bool {
        match self.child.as_mut().map(|c| c.try_wait()) {
            Some(Ok(None)) => true,
            Some(_) => {
                self.child = None;
                self.last_stop.get_or_insert_with(|| "worker exited".into());
                false
            }
            None => false,
        }
    }

    fn measure(&mut self) -> Option<u64> {
        let pid = Pid::from_u32(self.child.as_ref()?.id());
        self.sys.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
        let rss = self.sys.process(pid)?.memory();
        self.rss = rss;
        self.peak_rss = self.peak_rss.max(rss);
        Some(rss)
    }

    /// Apply the decision rule once. Returns a log line when something changed.
    pub fn tick(&mut self, should_run: bool, cap: u64) -> Option<String> {
        let running = self.running();
        let rss = if running { self.measure() } else { None };
        let allowed = should_run && self.blocked_below.is_none_or(|need| cap > need);
        match decide(allowed, running, rss, cap) {
            Decision::Start => match Command::new(&self.cfg.bin)
                .args(["-H", &self.cfg.host, "-p", &self.cfg.port.to_string(), "-d", "CPU", "-t",
                       &self.cfg.threads.to_string()])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
            {
                Ok(child) => {
                    self.child = Some(child);
                    self.blocked_below = None;
                    self.last_stop = None;
                    Some(format!("worker started on {}", self.cfg.endpoint()))
                }
                Err(e) => Some(format!("worker failed to start: {e}")),
            },
            Decision::Stop(reason) => {
                self.halt(reason);
                Some(format!("worker stopped: {reason}"))
            }
            Decision::Kill(reason) => {
                let used = self.rss;
                self.halt(reason);
                self.cap_kills += 1;
                self.blocked_below = Some(used);
                Some(format!("worker killed: {reason} ({:.0} MB used, cap {:.0} MB)", used as f64 / 1e6,
                             cap as f64 / 1e6))
            }
            Decision::Keep => None,
        }
    }

    pub fn halt(&mut self, reason: &str) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.rss = 0;
        self.last_stop = Some(reason.to_string());
    }

    pub fn status(&mut self) -> serde_json::Value {
        let running = self.running();
        serde_json::json!({
            "state": if running { "running" } else { "stopped" },
            "endpoint": self.cfg.endpoint(),
            "rss_bytes": if running { self.rss } else { 0 },
            "peak_rss_bytes": self.peak_rss,
            "cap_kills": self.cap_kills,
            "last_stop_reason": self.last_stop,
        })
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.halt("agent exiting");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_only_when_allowed() {
        assert_eq!(decide(true, false, None, 100), Decision::Start);
        assert_eq!(decide(false, false, None, 100), Decision::Keep);
    }

    #[test]
    fn stops_when_paused_or_lease_invalid() {
        assert!(matches!(decide(false, true, Some(10), 100), Decision::Stop(_)));
    }

    #[test]
    fn kills_above_cap_even_when_allowed() {
        assert!(matches!(decide(true, true, Some(101), 100), Decision::Kill(_)));
        assert_eq!(decide(true, true, Some(100), 100), Decision::Keep);
    }

    #[test]
    fn cap_kill_takes_priority_over_stop() {
        assert!(matches!(decide(false, true, Some(500), 100), Decision::Kill(_)));
    }
}
