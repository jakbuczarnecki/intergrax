# Intergrax Platform Proof — Local Knowledge Workspace

This document is the **guided reviewer path** for verifying that Intergrax works as a real platform, not only as architecture documentation.

It uses the **Local Knowledge Workspace (LKW)** application as the proof target. You will start the local proof stack, run one proof event into Sentry, run one LKW workflow into Elasticsearch/Kibana, and verify the results in the browser.

> Scope: local technical proof only. This is not a production readiness, security, compliance, SLA, hosting, or commercial-use claim. See [`COLLABORATION.md`](../../COLLABORATION.md) and [`LICENSE`](../../LICENSE).

---

## What you are proving

A reviewer should be able to verify three things without reading the whole repository:

```text
1. LKW starts as a real Tier-3 Intergrax application.
2. LKW emits platform observability records into Elasticsearch/Kibana.
3. LKW emits controlled problem signals into local Sentry.
```

The important platform path is:

```text
LKW HTTP API
→ Tier-3 application host
→ Intergrax runtime / agent path
→ platform observability envelope
→ export policy / safety filtering
→ vendor integration
→ local proof backend UI
```

The proof intentionally uses local Docker services. You do **not** need an external Sentry account, SaaS DSN, cloud account, or hosted service.

---

## What will be started locally

The canonical local stack starts all current proof backends:

```text
local_workspace   LKW API host              http://127.0.0.1:8020
elasticsearch     Observability documents   http://127.0.0.1:9200
kibana            Elasticsearch UI          http://127.0.0.1:5601
sentry-nginx      Local Sentry UI           http://127.0.0.1:9000
qdrant            Local vector store
ollama            Local LLM service
otel-collector    Local OTLP proof service
```

First start can take several minutes, especially the local Sentry stack.

---

## Before you start

Recommended environment:

```text
Docker Desktop / Docker Compose
Python 3.12
uv
PowerShell on Windows
```

Run commands from the repository root.

On Windows the repository root looks like:

```text
D:\Projekty\intergrax
```

---

## Step 1 — Hard reset and start the full local proof stack

Use this when you want a clean, reproducible proof from scratch.

Windows:

```bat
applications\local_workspace_application\scripts\hard-reset-local-docker-all.bat
```

What this script does:

```text
1. stops the local proof stack
2. removes Docker volumes and orphan containers
3. removes generated local Sentry proof runtime state
4. starts the full proof stack again with up --build
```

It does **not** delete source files, `.env`, committed local Relay credentials, sample documents, or Docker images.

If you only want to start without cleaning state, use:

```bat
applications\local_workspace_application\scripts\run-local-docker-all.bat
```

Linux/macOS startup helper:

```bash
chmod +x applications/local_workspace_application/scripts/run-local-docker-all.sh
applications/local_workspace_application/scripts/run-local-docker-all.sh
```

---

## Step 2 — Wait until the stack is ready

Open a second terminal and run:

```bat
applications\local_workspace_application\scripts\run-local-docker-all.bat ps -a
```

You are looking for this general state:

```text
local_workspace      Up ... healthy   0.0.0.0:8020->8020/tcp
sentry-web           Up ... healthy
sentry-relay         Up
sentry-nginx         Up               0.0.0.0:9000->80/tcp
sentry-bootstrap     Exited (0)
sentry-upgrade       Exited (0)
elasticsearch        Up               0.0.0.0:9200->9200/tcp
kibana               Up               0.0.0.0:5601->5601/tcp
```

One-shot setup containers such as `sentry-bootstrap`, `sentry-upgrade`, and `sentry-snuba-bootstrap` are expected to show `Exited (0)`.

Then check LKW health:

```powershell
curl -i http://127.0.0.1:8020/health
```

Expected result:

```text
HTTP/1.1 200 OK
{"status":"ok"}
```

---

## Step 3 — Prove controlled problem export into local Sentry

Run one controlled Sentry proof event:

```bat
applications\local_workspace_application\scripts\run-sentry-observability-proof.bat --run-id lkw-sentry-live-001 --correlation-id lkw-sentry-live-001
```

Expected result in the terminal:

```text
proof_result=PASS
backend=sentry
sentry_mode=local_docker
safety_check=passed
sentry_event_sent=true
```

This means LKW successfully emitted a controlled platform problem signal through the shared observability path into the local Sentry backend.

---

## Step 4 — Open the generated Sentry issue

Open local Sentry in your browser:

```text
http://127.0.0.1:9000
```

Login:

```text
email:    admin@intergrax.local
password: proof-local-only
```

Open the LKW proof organization/project directly:

```text
http://127.0.0.1:9000/organizations/intergrax-local/issues/?project=2
```

If the issue list is empty, wait a few seconds and refresh. The local consumer may need a short moment after Kafka topics are created.

Expected issue details include tags like:

```text
intergrax.problem_kind        lkw.proof_controlled_failure
intergrax.problem_error_code  LKW_PROOF_CONTROLLED_FAILURE
intergrax.run_id              lkw-sentry-live-001
intergrax.correlation_id      lkw-sentry-live-001
intergrax.provider_id         sentry
intergrax.record_type         problem_signal
```

Expected architectural meaning:

```text
LKW controlled failure
→ platform ProblemReporter
→ ObservabilityExportPolicy
→ Sentry vendor provider
→ local Sentry issue
```

---

## Step 5 — Run a real LKW workflow for Elasticsearch/Kibana

The Elasticsearch proof is based on a real LKW run. Submit one LKW request:

PowerShell:

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

Copy the returned `run_id`.

Example shape:

```text
run_id=run_d28d5f36f5ca4240b8693ae46eaa5946
state=completed
agent_id=local_search
```

---

## Step 6 — Validate Elasticsearch observability for that run

Replace `<run_id>` with the value from Step 5:

```bat
applications\local_workspace_application\scripts\run-elasticsearch-observability-proof.bat <run_id>
```

Expected result:

```text
Proof result: PASS
run_id=<run_id>
elasticsearch_url=http://127.0.0.1:9200
elasticsearch_index=intergrax-lkw-observability
duplicate_check=0
safety_check=passed
```

This proves that the LKW run produced policy-safe observability documents in Elasticsearch.

---

## Step 7 — Open Kibana and inspect the run timeline

Open Kibana:

```text
http://127.0.0.1:5601
```

Create a data view if Kibana asks for one:

```text
Name: intergrax-lkw-observability
Index pattern: intergrax-lkw-observability
Timestamp field: @timestamp
```

Then open:

```text
Analytics → Discover
```

Set the time range to:

```text
Last 24 hours
```

Search for your run:

```text
intergrax.run_id: "<run_id>"
```

Expected records include event types such as:

```text
tool_requested
tool_completed
task_completed
```

Useful columns to add:

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

---

## Expected pass criteria

| Area | How to verify | Expected result |
|------|---------------|-----------------|
| LKW health | `curl -i http://127.0.0.1:8020/health` | HTTP 200 and `{"status":"ok"}` |
| Sentry proof helper | `run-sentry-observability-proof.bat` | `proof_result=PASS`, `sentry_event_sent=true` |
| Sentry UI | `http://127.0.0.1:9000/organizations/intergrax-local/issues/?project=2` | Issue visible with `lkw.proof_controlled_failure` tags |
| Elasticsearch proof helper | `run-elasticsearch-observability-proof.bat <run_id>` | `Proof result: PASS`, `duplicate_check=0`, `safety_check=passed` |
| Kibana UI | `http://127.0.0.1:5601` Discover | Documents visible for the selected `run_id` |

---

## If something is not visible yet

Use one check at a time.

### Sentry UI opens but no issue appears

Run:

```bat
applications\local_workspace_application\scripts\run-local-docker-all.bat ps -a
```

If `sentry-events-consumer` is not `Up`, inspect it:

```bat
applications\local_workspace_application\scripts\run-local-docker-all.bat logs --tail=200 sentry-events-consumer
```

The local stack is configured to restart this consumer on failure because the Kafka `ingest-events` topic may be created shortly after the first consumer start.

### LKW is not healthy

Run:

```bat
applications\local_workspace_application\scripts\run-local-docker-all.bat logs --tail=200 local_workspace
```

### Sentry login

Use exactly:

```text
admin@intergrax.local
proof-local-only
```

### Wrong Sentry organization/project

Use the direct URL:

```text
http://127.0.0.1:9000/organizations/intergrax-local/issues/?project=2
```

The default Sentry organization may show an `internal` project. The LKW proof project is `lkw-proof` in organization `intergrax-local`.

---

## What this proves architecturally

| Platform concern | What the proof demonstrates |
|------------------|-----------------------------|
| Tier-3 application boundary | LKW exposes a real HTTP product boundary |
| Runtime execution | A real LKW request creates a run through the platform path |
| Agent/tool path | `local.workspace.search` reaches the `local_search` / retrieval path |
| Observability envelope | Runtime records are exported through platform observability envelopes |
| Export safety | Proof helpers check duplicate records and forbidden/sensitive keys |
| Vendor integration | Elasticsearch/Kibana and Sentry are reached through provider integrations |
| Problem signaling | Controlled LKW failures become Sentry problem issues |

---

## Current reviewer UX note

The Sentry proof is already one-script after stack startup:

```text
run-sentry-observability-proof.bat
```

The Elasticsearch proof currently requires one manual LKW request to obtain a `run_id`, then one proof helper call. The intended next UX improvement is an all-in-one Elasticsearch proof helper that creates the LKW run and validates Elasticsearch/Kibana output in a single command.

---

## Related docs

- Sentry local proof details: [`applications/local_workspace_application/docs/SENTRY_OBSERVABILITY.md`](../../applications/local_workspace_application/docs/SENTRY_OBSERVABILITY.md)
- Kibana/Elasticsearch proof details: [`applications/local_workspace_application/docs/KIBANA_OBSERVABILITY.md`](../../applications/local_workspace_application/docs/KIBANA_OBSERVABILITY.md)
- LKW outreach entry points: [`docs/public-adoption/OUTREACH_KIT.md`](OUTREACH_KIT.md)
