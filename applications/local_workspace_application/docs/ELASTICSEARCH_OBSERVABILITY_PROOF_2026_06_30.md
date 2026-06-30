# Elasticsearch observability live proof — 2026-06-30

## Scope

This document records the live Docker Compose proof for:

```text
OBS-VENDOR-7 — End-to-end vendor proof (Elasticsearch/OpenSearch first)
```

The proof validates the path:

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

## Environment

```text
Repository: jakbuczarnecki/intergrax
Branch: development
Date: 2026-06-30
Elasticsearch URL: http://127.0.0.1:9200
Elasticsearch index: intergrax-lkw-observability
```

## Commands

Start full local stack with all overlays:

```bat
applications\local_workspace_application\scripts\run-local-docker-all.bat
```

Run proof helper for selected run:

```bat
applications\local_workspace_application\scripts\run-elasticsearch-observability-proof.bat run_d28d5f36f5ca4240b8693ae46eaa5946
```

## Health checks

LKW health:

```json
{"status":"ok"}
```

Elasticsearch health:

```json
{
  "cluster_name": "docker-cluster",
  "status": "yellow",
  "number_of_nodes": 1,
  "number_of_data_nodes": 1,
  "active_primary_shards": 1,
  "active_shards": 1,
  "unassigned_shards": 1,
  "active_shards_percent_as_number": 50.0
}
```

The yellow status is acceptable for the local single-node proof because the primary shard is active and the unassigned shard is the replica.

## Run proof

```text
run_id=run_d28d5f36f5ca4240b8693ae46eaa5946
records=24
```

Observed timeline includes the required vendor proof events:

```text
tool_requested   agent_id=local_search  tool_id=rag.retrieve
tool_completed   agent_id=local_search  tool_id=rag.retrieve
task_completed
```

## Duplicate check

```text
Duplicate check: duplicate groups = 0
```

Result: **PASS**.

## Safety-key check

```text
Safety check: 0 forbidden keys
```

Result: **PASS**.

The inspector safety check validates document keys against canonical `FORBIDDEN_EXPORT_CONTENT_FIELDS` from the runtime export boundary. No raw prompt/content/chunks/tool args/secrets/full paths were observed by the readback guardrail.

## Combined proof result

```text
Proof result: PASS
```

## Conclusion

OBS-VENDOR-7 live proof passed for Elasticsearch/OpenSearch-compatible backend.

The proof confirms:

- LKW exported policy-sanitized observability documents to Elasticsearch.
- Documents were read back by real `run_id`.
- `tool_requested` and `tool_completed` appeared for `rag.retrieve`.
- Duplicate check returned `duplicate groups = 0`.
- Safety-key check returned `0 forbidden keys`.
- No runtime/LKW vendor SDK branching was introduced by this proof.

## Out of scope preserved

No changes were made for:

```text
retry/backoff
batching
dead-letter
auth/TLS
dashboards
index templates
mapping changes
Langfuse/Phoenix/Arize
runtime export logic
provider transport logic
```
