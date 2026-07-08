# Intergrax Platform Proof — Local Knowledge Workspace

This is the guided reviewer path for verifying that Intergrax works as a real platform.

Use this document as the source of truth. Follow the steps in order. A reviewer should not need to inspect raw Docker output or infer what to check from long logs.

---

## What this proves

```text
1. LKW starts as a real Tier-3 Intergrax application.
2. LKW emits policy-safe observability records into Elasticsearch/Kibana.
3. LKW emits controlled problem signals into local Sentry.
4. LKW persists indexed local knowledge across a non-destructive restart.
```

Local proof endpoints:

```text
LKW API         http://127.0.0.1:8020
Elasticsearch   http://127.0.0.1:9200
Kibana          http://127.0.0.1:5601
Sentry UI       http://127.0.0.1:9000
```

---

## Prerequisites

```text
Docker Desktop / Docker Compose
Python 3.12
uv
PowerShell on Windows
```

Run all commands from the repository root.

---

## Step 1 — Start a clean local proof stack

Windows:

```bat
applications\local_workspace_application\scripts\hard-reset-local-docker-all.bat
```

Expected result:

```text
The script finishes and returns to the command prompt.
It launches Docker startup separately instead of keeping this terminal inside a live startup stream.
The last lines point to the next status check.
```

Do not wait on Docker progress output manually. Step 2 is the readiness check.

---

## Step 2 — Verify the stack with the proof status checker

Run:

```bat
applications\local_workspace_application\scripts\check-lkw-platform-proof-status.bat
```

Expected result:

```text
proof_status=PASS
next_step=run-sentry-observability-proof
```

If you get:

```text
proof_status=WAIT
```

wait 30-60 seconds and run the same command again.

Then verify LKW health:

```powershell
curl -i http://127.0.0.1:8020/health
```

Expected result:

```text
HTTP/1.1 200 OK
{"status":"ok"}
```

The raw Docker status command is only for troubleshooting, not for the normal reviewer path.

---

## Step 3 — Emit one controlled problem signal into local Sentry

Run:

```bat
applications\local_workspace_application\scripts\run-sentry-observability-proof.bat --run-id lkw-sentry-live-001 --correlation-id lkw-sentry-live-001
```

Expected result:

```text
proof_result=PASS
backend=sentry
sentry_mode=local_docker
safety_check=passed
sentry_event_sent=true
```

---

## Step 4 — Open the generated Sentry issue

Open local Sentry:

```text
http://127.0.0.1:9000
```

Use this local proof login:

```text
email:    admin@intergrax.local
password: proof-local-only
```

Open the LKW proof project directly:

```text
http://127.0.0.1:9000/organizations/intergrax-local/issues/?project=2
```

Expected: one issue is visible for `lkw-sentry-live-001`.

Useful tags to check:

```text
intergrax.problem_kind        lkw.proof_controlled_failure
intergrax.problem_error_code  LKW_PROOF_CONTROLLED_FAILURE
intergrax.run_id              lkw-sentry-live-001
intergrax.correlation_id      lkw-sentry-live-001
```

---

## Step 5 — Run the Elasticsearch/Kibana proof helper

Windows:

```bat
applications\local_workspace_application\scripts\run-lkw-elasticsearch-proof.bat
```

Expected result:

```text
Proof result: PASS
run_id=<generated_run_id>
kibana_url=http://127.0.0.1:5601
elasticsearch_validation=passed
```

Copy the printed `run_id`.

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

Open:

```text
Analytics -> Discover
```

Set time range:

```text
Last 24 hours
```

Use the filter printed by the helper:

```text
intergrax.run_id: "<generated_run_id>"
```

Expected event types:

```text
tool_requested
tool_completed
task_completed
```

---

## Step 7 — Verify persistent local knowledge after restart

Run:

```bat
applications\local_workspace_application\scripts\run-lkw-persistence-proof.bat
```

Expected result:

```text
proof_result=PASS
proof_kind=persistent_vector_storage
restart_mode=non_destructive
volumes_removed=false
before_restart_results=1
after_restart_results=1
reindexed_after_restart=false
```

This proof:

- indexes a marker document into the local vector store,
- verifies search before restart,
- restarts LKW and Qdrant without deleting Docker volumes,
- searches again without reindexing.

`PASS` means indexed local knowledge survived the restart.

Latest recorded live result: [`LKW_5_PERSISTENCE_VERIFICATION.md`](../../applications/local_workspace_application/docs/LKW_5_PERSISTENCE_VERIFICATION.md).

Do not use hard-reset-local-docker-all between the before/after search. Hard reset removes volumes and invalidates this persistence proof.

---

## Reviewer shortcut

```bat
applications\local_workspace_application\scripts\hard-reset-local-docker-all.bat
applications\local_workspace_application\scripts\check-lkw-platform-proof-status.bat
applications\local_workspace_application\scripts\run-sentry-observability-proof.bat --run-id lkw-sentry-live-001 --correlation-id lkw-sentry-live-001
applications\local_workspace_application\scripts\run-lkw-elasticsearch-proof.bat
applications\local_workspace_application\scripts\run-lkw-persistence-proof.bat
```

Then open:

```text
Sentry: http://127.0.0.1:9000/organizations/intergrax-local/issues/?project=2
Kibana: http://127.0.0.1:5601
```

---

## Token optimization claim guardrails

For token-optimization proof wording and claim boundaries, see [`TOKEN_OPTIMIZATION_CLAIMS.md`](TOKEN_OPTIMIZATION_CLAIMS.md).
