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
