# OBSERVABILITY vendor proofs

This satellite records completed live vendor proof evidence without expanding the main OBSERVABILITY plan hub.

## OBS-VENDOR-7 - Done

**Date:** 2026-06-30  
**Backend:** Elasticsearch/OpenSearch-compatible local Docker Compose backend  
**Proof artifact:** `applications/local_workspace_application/docs/ELASTICSEARCH_OBSERVABILITY_PROOF_2026_06_30.md`

Live proof result:

```text
run_id=run_d28d5f36f5ca4240b8693ae46eaa5946
records=24
duplicate_check=0
safety_check=passed
proof_result=PASS
```

Validated path:

```text
LKW run
→ ObservabilityExportEnvelope
→ ObservabilityExportPolicy
→ ElasticsearchObservabilityIntegration
→ Elasticsearch/OpenSearch-compatible backend
→ readback by run_id
→ duplicate check
→ safety-key check
```

Observed required events:

```text
tool_requested   agent_id=local_search  tool_id=rag.retrieve
tool_completed   agent_id=local_search  tool_id=rag.retrieve
task_completed
```

Acceptance result:

- backend readback by real `run_id`: **passed**
- `tool_requested` / `tool_completed` appeared once for the proof run: **passed**
- duplicate check: **duplicate groups = 0**
- safety-key check: **0 forbidden keys**
- no raw prompt/content/chunks/tool args/secrets/full paths observed by the readback guardrail
- no runtime/LKW vendor SDK branching introduced

Operational hardening remains separate under **OBS-VENDOR-6**.
