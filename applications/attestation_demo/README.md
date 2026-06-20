# Attestation Demo — Partner PoC Quickstart

**Audience:** BoundaryAttest partner integration and internal harness reviewers.

This Tier-3 host demonstrates **Execution Boundary Export (EBE)**: Intergrax emits `execution_boundary_event.v1` records at tool and harness boundaries. **By default (EBE-9)** each event includes an Ed25519 **host attestation** envelope; BoundaryAttest verifies the host claim and may add a separate `client_observed` wrapper.

## Documentation

| Document | Purpose |
|----------|---------|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Host design, EBE contract, trust model |
| [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) | Task queue and verification |
| [`BUILD_AND_DEPLOY.md`](BUILD_AND_DEPLOY.md) | Local run, Docker, deploy runbook |
| [`DOCKER_VERIFY_RUNBOOK.md`](DOCKER_VERIFY_RUNBOOK.md) | **Step-by-step** — build image, run, verify PoC assumptions |
| [`adr/README.md`](adr/README.md) | Application architecture decisions |
| [`partner_handoff/README.md`](partner_handoff/README.md) | **Partner integration** — auth, mapping, EBE-9 golden vector |
| [`partner_handoff/EBE-9_HOST_SIGNING.md`](partner_handoff/EBE-9_HOST_SIGNING.md) | Host signing verifier spec |

## Run locally

```bash
# From repository root (requires .env with lab LLM settings or stub adapters)
uv run python -m attestation_demo.host.main
```

Default base URL: `http://127.0.0.1:8097`

Environment overrides:

| Variable | Default |
|----------|---------|
| `ATTESTATION_DEMO_BACKEND_HOST` | `127.0.0.1` |
| `ATTESTATION_DEMO_BACKEND_PORT` | `8097` |
| `ATTESTATION_DEMO_ROUTE_PREFIX` | `/v1/attestation_demo` |

## Primary PoC endpoint

`POST /v1/attestation_demo/poc/run`

### Example request

```bash
curl -s -X POST "http://127.0.0.1:8097/v1/attestation_demo/poc/run" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Partner PoC sample",
    "capability": "attestation.demo",
    "partition_key": "attestation_demo",
    "row_key": "poc-001",
    "record_data": { "title": "PoC report", "version": 1 }
  }'
```

### Example response (truncated — EBE-9 default)

Successful runs return **two** signed events (`tool_execution` seq 1, `harness_step` seq 2):

```json
{
  "task_id": "run-…",
  "run_id": "run-…",
  "state": "completed",
  "agent_id": "boundary_demo_agent",
  "boundary_events": [
    {
      "schema_id": "execution_boundary_event.v1",
      "signed": true,
      "boundary_type": "tool_execution",
      "event_sequence": 1,
      "tool_id": "records.put",
      "host_attestation": {
        "schema_id": "host_attestation_envelope.v1",
        "context": "boundaryattest.host-attestation.v1",
        "signed_payload_hash": "sha256:…",
        "public_key_id": "attestation-demo-host-1",
        "signature": "…"
      }
    }
  ],
  "trust_model": {
    "platform_signed": "true",
    "recommended_receipt_role": "host_attested"
  }
}
```

Full shapes: [`partner_handoff/ebe9_golden_vector.v1.json`](partner_handoff/ebe9_golden_vector.v1.json) · unsigned v2 reference [`partner_handoff/poc_run_response.v2.json`](partner_handoff/poc_run_response.v2.json).

## Debug endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/v1/attestation_demo/poc/runs/{run_id}/boundary-events` | Buffered events for a run |
| `GET` | `/v1/attestation_demo/agents` | Agent roster |
| `GET` | `/debug/tasks/{run_id}/trace` | Internal journal (compare with receipts) |

## Partner adapter mapping (external)

1. Verify `host_attestation` per event (EBE-9).
2. Map each `boundary_events[]` item to BoundaryAttest `createSignedReceipt` with partner `client_observed` wrapper.

| Boundary event | Partner field |
|----------------|---------------|
| `event_id` | stable receipt / evidence key |
| `event_sequence` | ordering within run |
| `boundary_type` | tool vs harness claim |
| `agent_id` | `agentId` |
| `tool_id` | `tool` |
| `action_status` | `actionStatus` |
| `input` / `output` | `input` / `output` |
| `lineage.ref` | `lineage.ref` |
| `host_attestation` | host verify (pinned pubkey) |
| — | `receiptRole: "client_observed"` on partner wrapper |

Intergrax does **not** ship the adapter or sign partner receipts.

## Trust model

**Default (EBE-9):** Intergrax host-signs each boundary event (`host_attested`). BoundaryAttest keeps a **separate** `client_observed` wrapper — two signatures, two claims.

**Unsigned mode:** set `host_signing_enabled=false` in manifest → `signed: false`, `client_observed` recommended.

Do not label receipts as `server_attested` by Intergrax unless co-located deployment is explicitly documented.

See `ARCHITECTURE.md` and `partner_handoff/EBE-9_HOST_SIGNING.md` for full design.
