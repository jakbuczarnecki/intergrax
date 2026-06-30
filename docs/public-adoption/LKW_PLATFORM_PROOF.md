# Intergrax Platform Proof — Local Knowledge Workspace

This walkthrough is the fastest practical way to see Intergrax working as a platform, not just as an architecture document.

It uses the **Local Knowledge Workspace (LKW)** as a real Tier-3 application built on the Intergrax harness. You will start the local stack, execute a real LKW run, inspect the exported runtime events in Elasticsearch, and open the same run in Kibana.

> This is a local proof path for technical evaluation. It is not a production readiness, security, compliance, SLA, or commercial-use claim. See [`COLLABORATION.md`](../../COLLABORATION.md) and [`LICENSE`](../../LICENSE).

---

## What this proves

This proof shows a real Intergrax application run flowing through multiple platform layers:

```text
User / evaluator
→ LKW HTTP API
→ Tier-3 application host
→ Nexus / task runner
→ Tier-2 local_search agent
→ Tier-0 rag.retrieve tool
→ runtime events
→ ObservabilityExportEnvelope
→ ObservabilityExportPolicy
→ ElasticsearchObservabilityIntegration
→ Elasticsearch index
→ Kibana Discover / CLI inspector
```

You should see:

- a real `run_id` returned by the LKW API
- runtime events indexed in Elasticsearch
- `tool_requested` and `tool_completed` for `rag.retrieve`
- Kibana Discover showing the run timeline
- CLI proof helper returning `duplicate_check=0`
- CLI proof helper returning `safety_check=passed`

---

## What you will run

The proof stack includes:

```text
local_workspace   LKW API host, port 8020
qdrant            local vector store
ollama            local LLM service used by the LKW stack
otel-collector    local OTLP proof service still present in the base stack
elasticsearch     policy-safe observability document backend, port 9200
kibana            visual UI over Elasticsearch, port 5601
```

The Elasticsearch/Kibana overlay lives here:

```text
applications/local_workspace_application/docker/docker-compose.elasticsearch.yml
```

The platform proof helper lives here:

```text
applications/local_workspace_application/scripts/run-elasticsearch-observability-proof.bat
```

---

## Prerequisites

Recommended local environment:

```text
Windows + PowerShell
Docker Desktop
Docker Compose
Python 3.12
uv
```

The commands below use Windows paths because LKW is currently easiest to verify on the maintainer's Windows local stack. Linux/macOS equivalents exist for some helper scripts, but this proof path is documented first for the currently validated local workflow.

---

## Step 1 — Start the local platform proof stack

From repository root:

```powershell
applications\local_workspace_application\scripts\run-local-docker-all.bat
```

This script runs the base LKW Docker Compose file plus all optional overlays in:

```text
applications/local_workspace_application/docker/
```

For this proof, that includes the Elasticsearch/Kibana overlay.

Expected compose services include:

```text
local_workspace
qdrant
ollama
otel-collector
elasticsearch
kibana
```

---

## Step 2 — Check service health

In a second terminal, from repository root:

```powershell
curl -s http://127.0.0.1:8020/health
```

Expected:

```json
{"status":"ok"}
```

Check Elasticsearch:

```powershell
curl -s http://127.0.0.1:9200/_cluster/health
```

Expected local single-node result is usually `green` before data is indexed or `yellow` after the first index is created with an unassigned replica. For this local proof, `yellow` is acceptable when the primary shard is active.

Open Kibana:

```text
http://127.0.0.1:5601
```

Expected: Kibana home page loads.

---

## Step 3 — Run the proof helper before the first LKW run

```powershell
applications\local_workspace_application\scripts\run-elasticsearch-observability-proof.bat
```

Before any LKW run has exported observability documents, it may print:

```text
No observability index found yet.
This is expected before the first LKW run exports Elasticsearch observability documents.
```

That is a valid pre-run state. Elasticsearch is up, but the index is not created until the first export.

---

## Step 4 — Execute a real LKW run

Submit a real LKW request:

```powershell
$body = @{
  message = "Find documents about local workspace observability proof"
  capability = "local.workspace.search"
  metadata = @{
    proof = "LKW_PLATFORM_PROOF"
  }
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8020/v1/local_workspace/run" `
  -ContentType "application/json" `
  -Body $body
```

Expected response shape:

```text
run_id
state
answer
agent_id
metadata
```

Copy the returned `run_id`.

Example validated run:

```text
run_id=run_d28d5f36f5ca4240b8693ae46eaa5946
```

---

## Step 5 — Validate the run through the CLI proof helper

Replace the example with your own `run_id`:

```powershell
applications\local_workspace_application\scripts\run-elasticsearch-observability-proof.bat run_d28d5f36f5ca4240b8693ae46eaa5946
```

Expected output includes a timeline like:

```text
tool_requested   local_search   rag.retrieve
tool_completed   local_search   rag.retrieve
task_completed
```

Expected proof summary:

```text
Duplicate check: duplicate groups = 0
Safety check: 0 forbidden keys
Proof result: PASS
```

A validated proof run produced:

```text
run_id=run_d28d5f36f5ca4240b8693ae46eaa5946
records=24
duplicate_check=0
safety_check=passed
proof_result=PASS
```

Full recorded proof artifact:

```text
applications/local_workspace_application/docs/ELASTICSEARCH_OBSERVABILITY_PROOF_2026_06_30.md
```

---

## Step 6 — Create the Kibana data view

Open Kibana:

```text
http://127.0.0.1:5601
```

Go to:

```text
Stack Management → Kibana → Data Views → Create data view
```

Use:

```text
Name: intergrax-lkw-observability
Index pattern: intergrax-lkw-observability
Timestamp field: @timestamp
```

Save the data view.

---

## Step 7 — Inspect the run in Kibana Discover

Go to:

```text
Analytics → Discover
```

Select data view:

```text
intergrax-lkw-observability
```

Set time range to a wide enough window, for example:

```text
Last 24 hours
```

Filter by your run id:

```text
intergrax.run_id: "run_d28d5f36f5ca4240b8693ae46eaa5946"
```

Expected: Kibana shows the indexed runtime documents for that run.

A validated proof run showed:

```text
Documents: 24
```

---

## Step 8 — Recommended Kibana columns

In Discover, add these columns:

```text
@timestamp
intergrax.run_id
intergrax.event_id
intergrax.event_type
intergrax.agent_id
intergrax.tool_id
intergrax.capability
intergrax.status
```

Useful KQL filters:

```text
intergrax.event_type: "tool_requested"
intergrax.event_type: "tool_completed"
intergrax.tool_id: "rag.retrieve"
intergrax.agent_id: "local_search"
```

Save the Discover view as:

```text
LKW Observability Run Timeline
```

---

## Optional screenshots

Screenshots are intentionally not required for the proof to pass, but they are useful for public review and demos.

Recommended screenshot paths:

```text
docs/public-adoption/assets/lkw-platform-proof/kibana-discover-run-timeline.png
docs/public-adoption/assets/lkw-platform-proof/proof-helper-pass.png
docs/public-adoption/assets/lkw-platform-proof/kibana-data-view.png
```

Suggested captions:

```text
Kibana Discover showing 24 indexed LKW runtime documents for one run_id.
CLI proof helper returning duplicate_check=0 and safety_check=passed.
Kibana Data View configured for intergrax-lkw-observability with @timestamp.
```

Before committing screenshots, verify they do not include secrets, private documents, raw prompts, raw chunks, full local file paths, tokens, or personal data.

---

## What this proves architecturally

This proof is intentionally small, but it exercises a real platform path:

| Platform concern | What the proof demonstrates |
|------------------|-----------------------------|
| Tier-3 application host | LKW exposes a real HTTP product boundary |
| Task intake | `POST /v1/local_workspace/run` creates a real task/run |
| Nexus/task runner | The task flows through the platform execution path |
| Agent routing | `local.workspace.search` reaches the `local_search` agent path |
| Tool execution | `rag.retrieve` emits `tool_requested` and `tool_completed` events |
| Observability spine | Runtime events become exportable observability envelopes |
| Export policy | Raw content/prompt/chunks/tool args/secrets/full paths are not exported by default |
| Vendor integration | Elasticsearch receives policy-safe documents via the vendor integration path |
| Visual inspection | Kibana can inspect real run timelines by `run_id` |
| Repeatable proof | CLI helper verifies duplicate and safety conditions |

---

## What this does not prove yet

This proof deliberately does not claim:

```text
production security posture
production auth/TLS
production Elasticsearch hardening
index lifecycle management
retry/backoff hardening
dead-letter export handling
throughput or load testing
dashboards as code
multi-vendor semantic mapping
commercial-use permission
```

Those are separate hardening and adoption tracks.

---

## Troubleshooting

### Kibana shows no results

Usually this is a time range issue.

Set the Discover time picker to:

```text
Last 24 hours
```

or click:

```text
Search entire time range
```

### Kibana cannot see the index

Check Elasticsearch indices:

```powershell
curl -s "http://127.0.0.1:9200/_cat/indices?v"
```

Expected:

```text
intergrax-lkw-observability
```

If the index does not exist, execute a real LKW run first.

### Proof helper says no index exists

That is expected before the first LKW run. Run Step 4, then run the helper again with the returned `run_id`.

### Proof helper fails for a selected run_id

Check:

```powershell
applications\local_workspace_application\scripts\inspect-elasticsearch-observability.bat --list-runs
```

Then use one of the listed run ids.

---

## Related docs

- LKW build/deploy runbook: [`applications/local_workspace_application/docs/BUILD_AND_DEPLOY.md`](../../applications/local_workspace_application/docs/BUILD_AND_DEPLOY.md)
- Kibana guide: [`applications/local_workspace_application/docs/KIBANA_OBSERVABILITY.md`](../../applications/local_workspace_application/docs/KIBANA_OBSERVABILITY.md)
- Recorded Elasticsearch proof: [`applications/local_workspace_application/docs/ELASTICSEARCH_OBSERVABILITY_PROOF_2026_06_30.md`](../../applications/local_workspace_application/docs/ELASTICSEARCH_OBSERVABILITY_PROOF_2026_06_30.md)
- LKW product-validation narrative: [`docs/product-validation/LOCAL_KNOWLEDGE_WORKSPACE_ALPHA.md`](../product-validation/LOCAL_KNOWLEDGE_WORKSPACE_ALPHA.md)
- Observability architecture: [`docs/architecture/OBSERVABILITY.md`](../architecture/OBSERVABILITY.md)
- Observability plan: [`docs/plan/OBSERVABILITY.md`](../plan/OBSERVABILITY.md)
