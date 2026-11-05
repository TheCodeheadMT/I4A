# DIFA prototype

Phase 0 prototype of **Distributed Inference For All**, built against
`DIFA_Product_Specification_v0_3.docx` and `DIFA_Design_Whitepaper_v0_3.docx` in the parent folder.

> **The inference runtime is fake.** It streams deterministic text and reports which devices it
> "ran" on. Everything here exercises the product contracts (groups, privacy, queueing,
> placement, owner controls, API); none of it is evidence that distributed inference works.
> That evidence is Phase 1.

## What is here

| Path | What it is |
|---|---|
| `services/coordinator/difa/domain/` | Pure rules: roles and visibility (`access.py`), request lifecycle, fair queue, memory ledger, filter-and-score placement with latency budget |
| `services/coordinator/difa/api/` | FastAPI gateway: native `/api/v1` and the OpenAI-compatible `/v1` subset |
| `services/coordinator/difa/scheduler.py` | Per-group admission, reservation, execution, cancellation, device-loss handling |
| `services/coordinator/difa/runtime/` | Adapter contract (`base.py`) and the fake runtime |
| `services/coordinator/difa/catalog.py` | Synthetic model fixtures, including `pilot-test-c` sized for the v0.3 pilot decision |
| `agents/device/` | Rust `difa-agent`: one-time-code enrollment, capability and bandwidth probes, owner policy enforced locally, heartbeats and leases |
| `contracts/http/openapi.json` | Exported HTTP contract |
| `tests/` | Unit and integration tests (pytest) |
| `scripts/` | Dev server, three-device smoke test, contract export |

## Run it

Python 3.10+ and Rust stable. The virtual environment and Cargo build output live outside this
iCloud folder so they are not synced.

```bash
python3 -m venv ~/.venvs/difa
~/.venvs/difa/bin/python -m pip install -e ".[dev]"
```

```bash
~/.venvs/difa/bin/python -m pytest -q
```

```bash
(cd agents/device && CARGO_TARGET_DIR=~/.cache/difa-target cargo test)
```

Start the coordinator on loopback (database in `~/.local/share/difa/`), then, in another terminal,
run the smoke test, which enrolls three simulated reference devices with real agent processes:

```bash
~/.venvs/difa/bin/python scripts/dev_server.py --fresh
```

```bash
(cd agents/device && CARGO_TARGET_DIR=~/.cache/difa-target cargo build) && ~/.venvs/difa/bin/python scripts/smoke.py
```

Interactive API docs are at http://127.0.0.1:8765/docs while the server runs.

Regenerate the contract after API changes:

```bash
~/.venvs/difa/bin/python scripts/export_openapi.py
```

## Phase 1: llama.cpp feasibility (in progress)

`services/coordinator/difa/runtime/llamacpp.py` is a real backend adapter behind the same contract as
the fake runtime. The coordinator host runs `llama-server` on its own Metal device and reaches other
stages through llama.cpp RPC workers; the placement pipeline's layer counts become `--tensor-split`
shares, and the adapter refuses a deployment unless llama.cpp's load log shows weights on every stage.
`benchmarks/phase1_llamacpp.py` measures single-device baselines and cooperation topologies with
100 warm requests each and a pre-declared greedy-match acceptance rule; results land in
`benchmarks/results/`.

Research-only constraints, as the spec requires: llama.cpp RPC is upstream-documented as insecure, so the
adapter refuses non-loopback RPC endpoints unless explicitly allowed, binds `llama-server` to 127.0.0.1
with a random API key, and all prompts are synthetic. So far every RPC worker is a CPU process on the
same Mac; the real HP and VM, and real network paths, are still untested.

One-time setup (pinned build and model, outside iCloud):

```bash
brew install cmake && git clone --depth 1 --branch v0.9.0 https://github.com/ggml-org/llama.cpp.git ~/.cache/difa-phase1/llama.cpp
```

```bash
cd ~/.cache/difa-phase1/llama.cpp && cmake -B build -DGGML_METAL=ON -DGGML_RPC=ON -DLLAMA_CURL=OFF && cmake --build build -j --target llama-server llama-bench ggml-rpc-server
```

```bash
mkdir -p ~/.cache/difa-phase1/models && curl -L -o ~/.cache/difa-phase1/models/qwen2.5-0.5b-instruct-q8_0.gguf https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf
```

```bash
~/.venvs/difa/bin/python benchmarks/phase1_llamacpp.py
```

Full write-up with numbers: [benchmarks/results/phase1-findings.md](benchmarks/results/phase1-findings.md).
Findings so far that change the design:

- Metal on the Mac Studio reports a working-set limit of 103,079 MB, inside the spec 3.2 planning range.
- An RPC worker advertises all host memory and has no memory-cap option, so owner limits (FR 07) must be
  enforced by the agent around the worker, not by llama.cpp.
- A small tensor-split share can round a device down to zero layers while it still appears as a
  participant; the harness and adapter now fail any placement where a stage holds no weights.
- llama.cpp puts the output head on the last device. The planner now orders stages so the strongest
  device runs last and counts the head against its budget.
- llama.cpp labels every RPC buffer `RPC0[host:port]`; only the endpoint identifies the device.

## Spec coverage

Implemented and tested: FR 01 to FR 05 (groups, invitations, removal), FR 06 to FR 09 and FR 26
(enrollment, owner limits, battery hysteresis, warm-hold setting), FR 13 to FR 15 (per-device ledger,
pinned requests), FR 16 to FR 21 (streaming chat, private/group branches, fork rules, single writer),
FR 22 to FR 25 (API, credentials and scopes, fair bounded queue, explicit rejections), FR 31
(filter-and-score placement with explanations), spec 7.1 lifecycle, 7.2 latency budget estimate,
spec 10 device-loss handling.

Prototype deviations to close before Phase 2 sign-off:

- Agent transport is HTTPS/JSON with a device token, not gRPC over mutual TLS. Device public keys are placeholders.
- Stored content is not yet encrypted with the group data key; no-history mode, retention purge jobs,
  and backups are not implemented.
- The client-disconnect grace window, deadline-aware rejection (FR 32), validated profiles (FR 33),
  conversation cache, and multiple placements (FR 34) are not implemented.
- Warm-hold unloading and protect-my-computer mode (FR 26, FR 27) are stored as policy but not acted on.
- Token counts are approximate; a real tokenizer comes with the Phase 1 backend.
- Windows power probing returns "unknown", which pauses battery contribution by default.
- No browser chat UI or Tauri desktop app yet.

## OpenAI-compatible subset

`GET /v1/models` and `POST /v1/chat/completions` with an application credential (`Authorization: Bearer`).
Supported: text `messages`, `max_tokens` / `max_completion_tokens`, `temperature`, `seed`, `stream`.
Rejected with `UNSUPPORTED_FIELD`: tools, functions, response formats, logprobs, `n`, audio, and non-text
content. Extensions: `difa_visibility` (`private` default, or `group` with a share-scoped key and
`difa_branch_id`) and `difa_client_request_id` for idempotency. Streaming errors after headers are sent
arrive as a terminal `data: {"error": ...}` event with no `[DONE]` marker.
