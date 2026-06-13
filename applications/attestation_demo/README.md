# Attestation Demo — Partner PoC Quickstart

**Audience:** AgentReceipt integration (Cullen Meyers) and internal harness reviewers.

This Tier-3 host demonstrates **Execution Boundary Export (EBE)**: Intergrax emits **unsigned** `execution_boundary_event.v1` records at the `RuntimeToolInvoker` boundary. The partner adapter signs receipts externally (`client_observed` recommended).

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

### Example response (truncated)

```json
{
  "task_id": "run-…",
  "run_id": "run-…",
  "state": "completed",
  "agent_id": "boundary_demo_agent",
  "boundary_events": [
    {
      "schema_id": "execution_boundary_event.v1",
      "signed": false,
      "tool_id": "records.put",
      "agent_id": "boundary_demo_agent",
      "action_status": "executed",
      "input": {
        "partition_key": "attestation_demo",
        "row_key": "poc-001",
        "data": { "title": "PoC report", "version": 1 }
      },
      "output": {
        "stored": true,
        "partition_key": "attestation_demo",
        "row_key": "poc-001"
      },
      "lineage": { "ref": "run-…:store_demo_record", "type": "execution_record" }
    }
  ],
  "trust_model": {
    "platform_signed": "false",
    "recommended_receipt_role": "client_observed"
  }
}
```

## Debug endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/v1/attestation_demo/poc/runs/{run_id}/boundary-events` | Buffered events for a run |
| `GET` | `/v1/attestation_demo/agents` | Agent roster |
| `GET` | `/debug/tasks/{run_id}/trace` | Internal journal (compare with receipts) |

## Partner adapter mapping (external)

Map each `boundary_events[]` item to AgentReceipt `createSignedReceipt`:

| Boundary event | AgentReceipt field |
|----------------|-------------------|
| `agent_id` | `agentId` |
| `tool_id` | `tool` |
| `action_status` | `actionStatus` |
| `input` / `output` | `input` / `output` (hashed via partner `stableJson`) |
| `lineage.ref` | `lineage.ref` |
| — | `receiptRole: "client_observed"` (recommended) |

Intergrax does **not** ship the adapter or sign receipts.

## Trust model (PoC v1)

- Intergrax emits **unsigned** facts (`signed: false`).
- Partner signs locally — proves the adapter recorded a payload, not platform attestation.
- Do not label receipts as `server_attested` by Intergrax unless co-located deployment is explicitly documented.

See `ARCHITECTURE.md` for full design.
