//! difa-agent: enrolls a device with a one-time code, then heartbeats with probed capabilities and
//! the locally enforced contribution state. Prototype transport is HTTPS/JSON with a device token;
//! the spec's gRPC over mutual TLS replaces it in Phase 2.

mod policy;
mod probe;

use clap::{Parser, Subcommand};
use policy::{allocatable, effective_paused, merge, Lease, OwnerPolicy};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const DEFAULT_BACKEND_DIGEST: &str = "fake-runtime-0.1";
const GB: f64 = 1e9;

#[derive(Parser)]
#[command(name = "difa-agent", version, about = "DIFA device agent")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Enroll this device using a one-time code from the DIFA app.
    Enroll {
        #[arg(long)]
        coordinator: String,
        #[arg(long)]
        code: String,
        #[arg(long)]
        name: String,
        /// Most memory this device may contribute, in GB. Required: owners choose explicitly (FR 07).
        #[arg(long)]
        memory_cap_gb: f64,
        #[arg(long, default_value_t = false)]
        allow_battery: bool,
        #[arg(long, default_value = "difa-agent-state.json")]
        state: PathBuf,
        /// Use a reference-device stand-in instead of probing this machine.
        #[arg(long)]
        simulate: Option<String>,
        /// Inference backend build this device runs (must match the coordinator's deployment).
        #[arg(long, default_value = DEFAULT_BACKEND_DIGEST)]
        backend_digest: String,
    },
    /// Send heartbeats and enforce owner policy.
    Run {
        #[arg(long, default_value = "difa-agent-state.json")]
        state: PathBuf,
        /// Stop after this many heartbeats (for tests).
        #[arg(long)]
        beats: Option<u64>,
    },
    /// Print what this device would report, without contacting a coordinator.
    Probe {
        #[arg(long)]
        simulate: Option<String>,
    },
}

#[derive(Serialize, Deserialize)]
struct State {
    coordinator: String,
    device_id: String,
    device_token: String,
    name: String,
    simulate: Option<String>,
    owner_policy: OwnerPolicy,
    #[serde(default = "default_digest")]
    backend_digest: String,
}

fn default_digest() -> String {
    DEFAULT_BACKEND_DIGEST.to_string()
}

#[derive(Deserialize)]
struct HeartbeatReply {
    policy: OwnerPolicy,
    lease_s: u64,
    assignment: Option<serde_json::Value>,
}

fn take_probe(simulate: &Option<String>) -> Result<probe::Probe, String> {
    match simulate {
        Some(name) => probe::simulated(name).ok_or_else(|| format!("unknown simulation profile {name}")),
        None => Ok(probe::real()),
    }
}

fn capabilities(p: &probe::Probe, cap: u64, rtt_ms: f64, digest: &str) -> serde_json::Value {
    json!({
        "os": p.os, "arch": p.arch, "backend_digest": digest, "formats": ["gguf"],
        "allocatable_bytes": allocatable(cap, p.available_bytes),
        "mem_bandwidth_gbps": (p.mem_bandwidth_gbps * 10.0).round() / 10.0,
        "rtt_ms": rtt_ms, "power": p.power.label(), "battery_pct": p.power.pct(),
    })
}

fn save_state(path: &Path, state: &State) -> std::io::Result<()> {
    let body = serde_json::to_vec_pretty(state)?;
    std::fs::write(path, body)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

fn placeholder_public_key() -> String {
    // Stand-in until device key pairs and mTLS land; it identifies the device and is not a secret.
    use std::io::Read;
    let mut bytes = [0u8; 16];
    let _ = std::fs::File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut bytes));
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn client() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder().timeout(Duration::from_secs(10)).build().expect("http client")
}

#[allow(clippy::too_many_arguments)]
fn enroll(coordinator: String, code: String, name: String, memory_cap_gb: f64, allow_battery: bool, state: PathBuf,
          simulate: Option<String>, backend_digest: String) -> Result<(), String> {
    let p = take_probe(&simulate)?;
    let owner_policy = OwnerPolicy {
        memory_cap_bytes: (memory_cap_gb * GB) as u64,
        allow_battery,
        battery_stop_pct: 30.0,
        battery_resume_pct: 40.0,
        paused: false,
        warm_hold_minutes: 30,
    };
    let body = json!({
        "code": code, "name": name, "public_key": placeholder_public_key(),
        "capabilities": capabilities(&p, owner_policy.memory_cap_bytes, p.simulated_rtt_ms, &backend_digest),
        "policy": owner_policy,
    });
    let url = format!("{}/api/v1/devices/enroll", coordinator.trim_end_matches('/'));
    let resp = client().post(url).json(&body).send().map_err(|e| format!("cannot reach coordinator: {e}"))?;
    let status = resp.status();
    let reply: serde_json::Value = resp.json().map_err(|e| e.to_string())?;
    if !status.is_success() {
        return Err(format!("enrollment failed ({status}): {}", reply["error"]["message"]));
    }
    let st = State {
        coordinator,
        device_id: reply["device_id"].as_str().unwrap_or_default().to_string(),
        device_token: reply["device_token"].as_str().unwrap_or_default().to_string(),
        name,
        simulate,
        owner_policy,
        backend_digest,
    };
    save_state(&state, &st).map_err(|e| e.to_string())?;
    println!("enrolled device {} as '{}'; state saved to {}", st.device_id, st.name, state.display());
    Ok(())
}

fn run(state_path: PathBuf, beats: Option<u64>) -> Result<(), String> {
    let raw = std::fs::read(&state_path).map_err(|e| format!("cannot read {}: {e}", state_path.display()))?;
    let st: State = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
    let http = client();
    let url = format!("{}/api/v1/devices/{}/heartbeat", st.coordinator.trim_end_matches('/'), st.device_id);
    let mut effective = st.owner_policy.clone();
    let mut battery_hold = false;
    let mut lease = Lease::new(Duration::from_secs(30));
    let mut last_rtt = 0.0;
    let mut assignment: Option<serde_json::Value> = None;
    let mut count = 0u64;
    loop {
        let p = take_probe(&st.simulate)?;
        let paused = effective_paused(&effective, p.power, &mut battery_hold);
        let rtt = if st.simulate.is_some() { p.simulated_rtt_ms } else { last_rtt };
        let body = json!({"capabilities": capabilities(&p, effective.memory_cap_bytes, rtt, &st.backend_digest),
                          "paused": paused});
        let started = Instant::now();
        match http.post(&url).bearer_auth(&st.device_token).json(&body).send() {
            Ok(resp) if resp.status().is_success() => {
                last_rtt = (started.elapsed().as_secs_f64() * 1000.0 * 10.0).round() / 10.0;
                let reply: HeartbeatReply = resp.json().map_err(|e| e.to_string())?;
                effective = merge(&st.owner_policy, &reply.policy);
                lease.renew(Instant::now(), Duration::from_secs(reply.lease_s));
                if reply.assignment != assignment {
                    println!("assignment: {}", reply.assignment.as_ref().map_or("none".into(), |a| a.to_string()));
                    assignment = reply.assignment;
                }
                println!("heartbeat ok: paused={paused} power={} cap={:.1}GB rtt={last_rtt}ms",
                         p.power.label(), effective.memory_cap_bytes as f64 / GB);
            }
            Ok(resp) if resp.status().as_u16() == 401 || resp.status().as_u16() == 403 => {
                println!("device authorization revoked: stopping all local work");
                return Err("revoked".into());
            }
            Ok(resp) => println!("heartbeat rejected: {}", resp.status()),
            Err(e) => println!("coordinator unreachable: {e}"),
        }
        if assignment.is_some() && !lease.valid(Instant::now()) {
            println!("lease expired: releasing assigned work");
            assignment = None;
        }
        count += 1;
        if beats.is_some_and(|b| count >= b) {
            return Ok(());
        }
        std::thread::sleep(Duration::from_secs(5));
    }
}

fn main() {
    let result = match Cli::parse().cmd {
        Cmd::Enroll { coordinator, code, name, memory_cap_gb, allow_battery, state, simulate, backend_digest } => {
            enroll(coordinator, code, name, memory_cap_gb, allow_battery, state, simulate, backend_digest)
        }
        Cmd::Run { state, beats } => run(state, beats),
        Cmd::Probe { simulate } => take_probe(&simulate).map(|p| {
            let caps = capabilities(&p, u64::MAX, p.simulated_rtt_ms, DEFAULT_BACKEND_DIGEST);
            println!("{}", serde_json::to_string_pretty(&caps).unwrap());
        }),
    };
    if let Err(e) = result {
        eprintln!("difa-agent: {e}");
        std::process::exit(if e == "revoked" { 2 } else { 1 });
    }
}
