# Partner handoff — AgentReceipt integration (PoC v2)

**Audience:** AgentReceipt adapter authors and integration operators.

## Base URL

| Environment | URL |
|-------------|-----|
| Local dev | `http://127.0.0.1:8097` |
| Docker | `http://<host>:8097` (see [`BUILD_AND_DEPLOY.md`](../BUILD_AND_DEPLOY.md)) |

## Authentication

When `INTERGRAX_HARNESS_API_KEY` is set on the host, send:

```http
X-Api-Key: <key>
```

or `Authorization: Bearer <key>`. When the env var is unset (local dev default), requests pass without credentials.

## Primary integration flow

1. `POST /v1/attestation_demo/poc/run` with body from [`poc_run_request.v1.json`](poc_run_request.v1.json)
2. Read `boundary_events[]` from the JSON response (shapes: [`poc_run_response.v2.json`](poc_run_response.v2.json), failure example: [`poc_run_response.failed.v2.json`](poc_run_response.failed.v2.json))
3. **Create one receipt per boundary event** (not one composite receipt per run)
4. Map each event → AgentReceipt `createSignedReceipt` with `receiptRole: "client_observed"`
5. Persist via partner `LocalFileReceiptSink`; run `verify` / `chain`
6. Optional journal compare: `GET /debug/tasks/{run_id}/trace` on the same host (run/task-level correlation only — see below)

## Trace vs boundary correlation

| Source | Correlates | Does not expose |
|--------|------------|-----------------|
| `boundary_events[]` from `POST /poc/run` | Per-event `event_id`, `event_sequence`, `boundary_type`, `tool_id`, `step_id`, hashes | Full HOS spine |
| `GET /debug/tasks/{run_id}/trace` | Same `run_id`, agent, capability, graph node, critic, task state | EBE `event_id`, `step_id`, `tool_id` |

Use **`boundary_events[]` as the authoritative per-event source** for receipt keys and tool/harness claims. Use trace for optional run-level journal comparison.

**Partner validated:** commit `106aee776fcc6053e8265b9c3656638d107d351d` on branch `agent_experiment_runtime` (live Docker, BoundaryAttest adapter, 2026-06).

**EBE-9 (host signing):** see [`EBE-9_HOST_SIGNING.md`](EBE-9_HOST_SIGNING.md) and golden vector [`ebe9_golden_vector.v1.json`](ebe9_golden_vector.v1.json). Default `attestation_demo` manifest enables signing; set `host_signing_enabled=false` for unsigned v2.

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

## Field mapping (boundary event → AgentReceipt)

| Intergrax `boundary_events[]` | AgentReceipt |
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
| — | `receiptRole: "client_observed"` (recommended) |

## Trust model

- Intergrax emits **unsigned** facts (`signed: false`). Response includes `trust_model` metadata.
- Partner signs locally — does **not** prove Intergrax server attestation.
- Do **not** use `server_attested` unless co-located deployment is explicitly documented.

## Deferred (not in PoC v2)

- Webhook delivery of boundary events
- Intergrax host-side signing
- Run-level composite signed receipt (derived summary may come later)

Full design: [`ARCHITECTURE.md`](../ARCHITECTURE.md) · Application ADR: [`adr/ADR-ATTESTATION_DEMO-001.md`](../adr/ADR-ATTESTATION_DEMO-001.md)  
Operator verify: [`DOCKER_VERIFY_RUNBOOK.md`](../DOCKER_VERIFY_RUNBOOK.md)
