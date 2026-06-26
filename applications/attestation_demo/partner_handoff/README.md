# Partner handoff — BoundaryAttest integration (PoC v2 + EBE-9)

**Audience:** BoundaryAttest adapter authors and integration operators.

## Base URL

| Environment | URL |
|-------------|-----|
| Local dev | `http://127.0.0.1:8097` |
| Docker | `http://<host>:8097` (see [`docs/BUILD_AND_DEPLOY.md`](../docs/BUILD_AND_DEPLOY.md)) |

## Authentication

When `INTERGRAX_HARNESS_API_KEY` is set on the host, send:

```http
X-Api-Key: <key>
```

or `Authorization: Bearer <key>`. When the env var is unset (local dev default), requests pass without credentials.

## Primary integration flow (default: EBE-9 host signing)

1. `POST /v1/attestation_demo/poc/run` with body from [`poc_run_request.v1.json`](poc_run_request.v1.json)
2. Read `boundary_events[]` from the JSON response — each element includes `signed`, `host_attestation` (when signing enabled), and full `execution_boundary_event.v1` fields
3. **Verify Intergrax host signature** per event (see [`EBE-9_HOST_SIGNING.md`](EBE-9_HOST_SIGNING.md) and golden vector [`ebe9_golden_vector.v1.json`](ebe9_golden_vector.v1.json))
4. **Create one receipt per boundary event** (not one composite receipt per run)
5. Map each verified event → BoundaryAttest `createSignedReceipt` with a separate `receiptRole: "client_observed"` wrapper (partner key — not the host key)
6. Persist via partner `LocalFileReceiptSink`; run `verify` / `chain`
7. Optional journal compare: `GET /debug/tasks/{run_id}/trace` on the same host (run/task-level correlation only — see below)

**Unsigned v2 reference shapes** (when `host_signing_enabled=false`): [`poc_run_response.v2.json`](poc_run_response.v2.json), failure example [`poc_run_response.failed.v2.json`](poc_run_response.failed.v2.json).

## Trace vs boundary correlation

| Source | Correlates | Does not expose |
|--------|------------|-----------------|
| `boundary_events[]` from `POST /poc/run` | Per-event `event_id`, `event_sequence`, `boundary_type`, `tool_id`, `step_id`, hashes | Full HOS spine |
| `GET /debug/tasks/{run_id}/trace` | Same `run_id`, agent, capability, graph node, critic, task state | EBE `event_id`, `step_id`, `tool_id` |

Use **`boundary_events[]` as the authoritative per-event source** for receipt keys and tool/harness claims. Use trace for optional run-level journal comparison.

**Partner validated (EBE-8 unsigned v2):** Intergrax `106aee776fcc6053e8265b9c3656638d107d351d` on `agent_experiment_runtime` (live Docker, 2026-06).

**Partner validated (EBE-9 host signing):** Intergrax live @ `96b7f9974869e484406cbade3160b61c71b2980c`; handoff docs @ `13102cfaff1a7a9d212c16cd16587477cc533dc0` on `agent_experiment_runtime`. BoundaryAttest verifier @ `61be9918bc8f91fc8f160e0392d2914f38f3d4cb` (golden vector, 39/39 tests, unsigned v2 regression).

**Default manifest:** `host_signing_enabled=true`, `public_key_id=attestation-demo-host-1`. Set `host_signing_enabled=false` to reproduce unsigned v2.

## PoC v2 event shape

Each successful demo run returns **two** events in `boundary_events[]`, ordered by `event_sequence`:

| `event_sequence` | `boundary_type` | Claim |
|------------------|-----------------|-------|
| 1 | `tool_execution` | `records.put` executed at tool invoker |
| 2 | `harness_step` | HarnessKernel step completed (policy + outcome) |

Group related receipts with `run_id`, `step_id`, and `lineage.ref`. Use `event_id` as the stable per-event identifier.

> **Legacy:** [`poc_run_response.v1.json`](poc_run_response.v1.json) documents PoC v1 (single tool event only).

## Failure path (dual claims)

When `records.put` fails, Intergrax still returns **two** boundary events:

| `event_sequence` | `boundary_type` | Typical `action_status` | Meaning |
|------------------|-----------------|-------------------------|---------|
| 1 | `tool_execution` | `failed` | Tool invoker boundary — side effect did not succeed |
| 2 | `harness_step` | `completed` | Harness kernel finished policy/merge for the step |

This is **intentional**: tool and harness are separate claims. Sign **one receipt per event**; verify each independently. Do not collapse into one composite run receipt.

See [`poc_run_response.failed.v2.json`](poc_run_response.failed.v2.json) for the expected shape.

## Field mapping (boundary event → BoundaryAttest)

| Intergrax `boundary_events[]` | BoundaryAttest |
|-------------------------------|--------------|
| `event_id` | stable receipt / evidence key |
| `event_sequence` | ordering within run |
| `boundary_type` | distinguishes tool vs harness claim |
| `agent_id` | `agentId` |
| `tool_id` | `tool` (tool events only) |
| `action_status` | `actionStatus` |
| `input` / `output` | `input` / `output` (hash via partner `stableJson`) |
| `policy_verdicts` / `step_outcome` | harness-step metadata (optional mapping) |
| `lineage.ref` | `lineage.ref` |
| `lineage.type` | `lineage.type` |
| `host_attestation` | verify with pinned `public_key_ed25519` before partner wrapper |
| — | `receiptRole: "client_observed"` on partner wrapper (separate from host claim) |

## Trust model

**Default (EBE-9 enabled):**

- Intergrax emits **host-signed** boundary facts (`signed: true`) with `host_attestation` envelope per event.
- Response `trust_model.recommended_receipt_role` is **`host_attested`** — honest runtime/host claim only.
- BoundaryAttest verifies the host signature, then may add its own **`client_observed`** wrapper with a separate partner key.
- Two signatures, two claims — do **not** conflate them.
- Do **not** use `server_attested` for Intergrax unless co-located deployment is explicitly documented.

**Unsigned fallback (`host_signing_enabled=false`):**

- Intergrax emits `signed: false`, `host_attestation: null`; `trust_model.recommended_receipt_role` is `client_observed`.
- Partner signs locally — proves what the adapter recorded, not platform attestation.

## Deferred

- Webhook delivery of boundary events
- Run-level composite signed receipt (derived summary may come later)

Full design: [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) · Application ADR: [`docs/adr/ADR-ATTESTATION_DEMO-001.md`](../docs/adr/ADR-ATTESTATION_DEMO-001.md)
Operator verify: [`docs/DOCKER_VERIFY_RUNBOOK.md`](../docs/DOCKER_VERIFY_RUNBOOK.md)
