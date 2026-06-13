# Partner handoff — AgentReceipt integration (PoC v1)

**Audience:** Cullen Meyers / AgentReceipt adapter authors.

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
2. Read `boundary_events[]` from the JSON response (shape: [`poc_run_response.v1.json`](poc_run_response.v1.json))
3. Map each event → AgentReceipt `createSignedReceipt` with `receiptRole: "client_observed"`
4. Persist via partner `LocalFileReceiptSink`; run `verify` / `chain`
5. Optional journal compare: `GET /debug/tasks/{run_id}/trace` on the same host

## Field mapping (boundary event → AgentReceipt)

| Intergrax `boundary_events[]` | AgentReceipt |
|-------------------------------|--------------|
| `agent_id` | `agentId` |
| `tool_id` | `tool` |
| `action_status` | `actionStatus` |
| `input` / `output` | `input` / `output` (hash via partner `stableJson`) |
| `lineage.ref` | `lineage.ref` |
| `lineage.type` | `lineage.type` |
| — | `receiptRole: "client_observed"` (recommended) |

## Trust model

- Intergrax emits **unsigned** facts (`signed: false`). Response includes `trust_model` metadata.
- Partner signs locally — does **not** prove Intergrax server attestation.
- Do **not** use `server_attested` unless co-located deployment is explicitly documented.

## Deferred (not in PoC v1)

- Webhook delivery of boundary events
- Intergrax host-side signing
- HarnessKernel step-level export

Full design: [`ARCHITECTURE.md`](../ARCHITECTURE.md) · Application ADR: [`adr/ADR-ATTESTATION_DEMO-001.md`](../adr/ADR-ATTESTATION_DEMO-001.md)  
Operator verify: [`DOCKER_VERIFY_RUNBOOK.md`](../DOCKER_VERIFY_RUNBOOK.md)
