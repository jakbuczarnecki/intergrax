# Intergrax Platform Proof — Local Knowledge Workspace

This is the guided reviewer path for verifying that Intergrax works as a real platform, not only as architecture documentation.

You will verify the proof from this document only:

```text
1. Start the local proof stack.
2. Verify the stack is ready.
3. Emit one controlled problem signal into local Sentry.
4. Open the generated Sentry issue in the browser.
5. Run one real LKW workflow and validate Elasticsearch output.
6. Open Kibana and inspect the generated run timeline.
```

> Scope: local technical proof only. This is not a production readiness, security, compliance, SLA, hosting, or commercial-use claim. See [`COLLABORATION.md`](../../COLLABORATION.md) and [`LICENSE`](../../LICENSE).

---

## What this proves

A reviewer should be able to verify three platform behaviors without reading the whole repository:

```text
1. LKW starts as a real Tier-3 Intergrax application.
2. LKW emits policy-safe observability records into Elasticsearch/Kibana.
3. LKW emits controlled problem signals into local Sentry.
```

The important path is:

```text
LKW HTTP API
→ Tier-3 application host
→ Intergrax runtime / agent path
→ platform observability envelope
→ export policy / safety filtering
→ vendor integration
→ local proof backend UI
```

The proof uses local Docker services only. You do not need an external Sentry account, SaaS DSN, cloud account, or hosted service.

---

## Local services used by the proof

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

## Prerequisites

Recommended local environment:

```text
Docker Desktop / Docker Compose
Python 3.12
uv
PowerShell on Windows
```

Run all commands from the repository root.

---

## Step 1 — Hard reset and start the full local proof stack

Use this for a clean proof from scratch.

Windows:

```bat
applications\local_workspace_application\scripts\hard-reset-local-docker-all.bat
```

What this script does:

```text
1. stops the local proof stack
2. removes Docker volumes and orphan containers
3. removes generated local Sentry proof runtime state
4. starts the full proof stack in detached mode with up -d --build
```

Expected terminal behavior:

```text
The script should finish and return you to the command prompt.
It should not leave you watching a live stream of Sentry/Kafka/Elasticsearch logs.
The last lines should point you to the next ps -a check.
```

It does not delete source files, `.env`, committed local Relay credentials, sample documents, or Docker images.

If you only want to start without cleaning state, use:

```bat
applications\local_workspace_application\scripts\run-local-docker-all.bat up -d --build
```

Linux/macOS startup helper:

```bash
chmod +x applications/local_workspace_application/scripts/run-local-docker-all.sh
applications/local_workspace_application/scripts/run-local-docker-all.sh up -d --build
```

---

## Step 2 — Verify the stack is ready

Run:

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

Expected terminal result:

```text
proof_result=PASS
backend=sentry
sentry_mode=local_docker
safety_check=passed
sentry_event_sent=true
```

This means LKW emitted a controlled platform problem signal through the shared observability path into the local Sentry backend.

---

## Step 4 — Open the generated Sentry issue

Open local Sentry:

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

Expected issue tags include:

```text
intergrax.problem_kind        lkw.proof_controlled_failure
intergrax.problem_error_code  LKW_PROOF_CONTROLLED_FAILURE
intergrax.run_id              lkw-sentry-live-001
intergrax.correlation_id      lkw-sentry-live-001
intergrax.provider_id         sentry
intergrax.record_type         problem_signal
```

If the issue list is empty, wait a few seconds and refresh.

---

## Step 5 — Prove Elasticsearch/Kibana observability in one command

Windows:

```bat
applications\local_workspace_application\scripts\run-lkw-elasticsearch-proof.bat
```

Linux/macOS:

```bash
chmod +x applications/local_workspace_application/scripts/run-lkw-elasticsearch-proof.sh
applications/local_workspace_application/scripts/run-lkw-elasticsearch-proof.sh
```

What this helper does:

```text
1. executes a real LKW run through POST /v1/local_workspace/run
2. captures the returned run_id
3. validates Elasticsearch records for that run_id
4. runs duplicate and safety checks
5. prints the Kibana URL and Discover filter to use in the browser
```

Expected result:

```text
Proof result: PASS
run_id=<generated_run_id>
kibana_url=http://127.0.0.1:5601
elasticsearch_validation=passed
```

Copy the printed `run_id`. You will use it in Kibana.

---

## Step 6 — Open Kibana and inspect the run timeline

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

Set time range to:

```text
Last 24 hours
```

Search for the run using the filter printed by the helper:

```text
intergrax.run_id: "<generated_run_id>"
```

Expected event types include:

```text
tool_requested
tool_completed
task_completed
```

Useful columns:

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
| Elasticsearch/Kibana proof helper | `run-lkw-elasticsearch-proof.bat` | `Proof result: PASS`, generated `run_id`, `elasticsearch_validation=passed` |
| Kibana UI | `http://127.0.0.1:5601` Discover | Documents visible for the selected `run_id` |

---

## Troubleshooting one step at a time

### Sentry UI opens but no issue appears

Run:

```bat
applications\local_workspace_application\scripts\run-local-docker-all.bat ps -a
```

If `sentry-events-consumer` is not `Up`, inspect it:

```bat
applications\local_workspace_application\scripts\run-local-docker-all.bat logs --tail=200 sentry-events-consumer
```

### LKW is not healthy

```bat
applications\local_workspace_application\scripts\run-local-docker-all.bat logs --tail=200 local_workspace
```

### Elasticsearch proof helper fails

```powershell
curl -i http://127.0.0.1:8020/health
curl -i http://127.0.0.1:9200/_cluster/health
```

Then inspect one service at a time:

```bat
applications\local_workspace_application\scripts\run-local-docker-all.bat logs --tail=200 local_workspace
```

```bat
applications\local_workspace_application\scripts\run-local-docker-all.bat logs --tail=200 elasticsearch
```

---

## Reviewer shortcut summary

After prerequisites are installed, the intended fast path is:

```bat
applications\local_workspace_application\scripts\hard-reset-local-docker-all.bat
applications\local_workspace_application\scripts\run-sentry-observability-proof.bat --run-id lkw-sentry-live-001 --correlation-id lkw-sentry-live-001
applications\local_workspace_application\scripts\run-lkw-elasticsearch-proof.bat
```

Then open:

```text
Sentry: http://127.0.0.1:9000/organizations/intergrax-local/issues/?project=2
Kibana: http://127.0.0.1:5601
```

---

## Related docs

- Sentry local proof details: [`applications/local_workspace_application/docs/SENTRY_OBSERVABILITY.md`](../../applications/local_workspace_application/docs/SENTRY_OBSERVABILITY.md)
- Kibana/Elasticsearch proof details: [`applications/local_workspace_application/docs/KIBANA_OBSERVABILITY.md`](../../applications/local_workspace_application/docs/KIBANA_OBSERVABILITY.md)
- LKW outreach entry points: [`docs/public-adoption/OUTREACH_KIT.md`](OUTREACH_KIT.md)
