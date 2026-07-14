# Intergrax Platform Proof — Local Knowledge Workspace

**Prerequisite:** Complete the [README Quick start](../../README.md#quick-start) (lab host, ~5 min) before this platform proof.

This document is the guided reviewer path. Structured ProofReceipt documents persisted through the platform DocumentStore are the source of truth for proof outcomes. Follow the steps in order. A reviewer should not need to inspect raw Docker output or infer what to check from long logs.

---

## What this proves

```text
1. LKW starts as a real Tier-3 Intergrax application.
2. LKW emits policy-safe observability records into Elasticsearch/Kibana.
3. LKW emits controlled problem signals into local Sentry.
4. LKW persists indexed local knowledge across a non-destructive restart.
5. LKW enqueues and executes background ingest jobs through the real platform message-bus / TaskQueue path with a local provider in the proof stack.
6. LKW records structured proof evidence through ProofReceiptStore into a real MongoDB DocumentStore vendor and exposes it for reviewer inspection through Mongo Express.
```

Local proof endpoints:

```text
LKW API         http://127.0.0.1:8020
Elasticsearch   http://127.0.0.1:9200
Kibana          http://127.0.0.1:5601
Kafka UI        http://127.0.0.1:8085
Mongo Express   http://127.0.0.1:8086
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

## Step 8 — Verify background task / queue platform proof

Run:

```bat
applications\local_workspace_application\scripts\run-lkw-background-task-proof.bat
```

The helper is idempotent: it starts or refreshes the combined Kafka + MongoDB overlay stack before running the proof, even if the full proof stack was already started in Step 1.

Expected result:

```text
proof_result=PASS
proof_kind=platform_background_task
task_name=lkw.background_ingest.v1
message_bus_provider=kafka
enqueue_mode=real_provider
worker_execution=asynchronous
task_status=SUCCEEDED
task_result_available=true
handler_resolved=true
worker_runtime_received=true
index_ingested=1
search_results=<n>
evidence_marker_found=true
kafka_ui_url=http://127.0.0.1:8085
kafka_topics=intergrax.tasks,intergrax.task-events,intergrax.task-status,intergrax.task-results
mock_queue=false
inmemory_bypass=false
direct_handler_call=false
direct_indexer_call=false
run_id=<generated_run_id>
correlation_id=<generated_correlation_id>
task_id=<task_id>
marker=<proof_marker>
collection_id=local_workspace
proof_receipt_recorded=true
proof_receipt_verified=true
proof_receipt_store=platform
document_store_provider=mongodb
proof_receipt_id=<generated_proof_id>
proof_receipt_run_id=<generated_run_id>
proof_receipt_result=PASS
proof_receipt_query_verified=true
mongo_express_url=http://127.0.0.1:8086
markdown_source_of_truth=false
direct_mongodb_write=false
direct_pymongo_from_lkw=false
```

Acceptance notes:

- `search_results` must be greater than or equal to `1`; the exact value may vary between runs.
- `run_id`, `correlation_id`, `task_id`, and `marker` are generated per run. Copy them if you want to inspect Kafka UI manually.

This proof:

- enqueues a real `LkwBackgroundIngestJob` through platform `message_bus.enqueue`,
- routes the `TaskRequest` through a **real local message bus provider** in the proof stack (Kafka in Docker),
- executes the registered handler through the platform worker path **asynchronously** (enqueue returns before work completes),
- inspects lifecycle through provider-neutral `message_bus.get_status` / `message_bus.get_result`,
- verifies indexed content through `local.workspace.search` after the task succeeds,
- constructs a `ProofReceipt` from the actual live evidence,
- stores it through `ProofReceiptStore` → platform `DocumentStore` → `MongoDBDocumentStoreIntegration`,
- verifies read-back and query before printing final `proof_result=PASS`,
- fails if receipt persistence is unavailable.

**Platform proof guardrails — this step is not satisfied by:**

- mocks or fake queue implementations,
- in-memory-only or synchronous in-process bypasses,
- unit-test-only handler invocation without the live `message_bus.*` tool surface,
- calling `local.workspace.index` directly while skipping enqueue / queue / worker lifecycle.

The proof stack must include a configured `message_bus` integration, a running broker/queue backend, a worker consumer for `lkw.background_ingest.v1`, and a live MongoDB document store for receipt persistence. LKW remains the proof workload; platform owns contracts, tools, provider adapters, worker execution, and receipt storage.

Open Kafka UI:

```text
http://127.0.0.1:8085
```

Topics to inspect:

```text
intergrax.tasks
intergrax.task-events
intergrax.task-status
intergrax.task-results
```

Expected in Kafka UI:

- `TaskRequest` message exists for the printed `run_id` / `correlation_id` in `intergrax.tasks`,
- lifecycle events `task.enqueued`, `task.started`, `task.succeeded`, `task.result_stored` exist in `intergrax.task-events`,
- status/result records exist in `intergrax.task-status` / `intergrax.task-results`.

---

## Step 9 — Inspect the structured ProofReceipt in Mongo Express

After Step 8 prints `proof_receipt_recorded=true` and `proof_receipt_verified=true`, inspect the persisted receipt.

### Open Mongo Express

```text
http://127.0.0.1:8086
```

### Select

```text
database: intergrax_proofs
collection: proof_receipts
```

### Find the receipt

Use the values printed by Step 8:

```text
proof_receipt_id
proof_receipt_run_id
task_id
```

Each stored document is a MongoDB row mapped from `DocumentRecord`. The `data` field contains the full `ProofReceipt` JSON (`schema_version`, `proof_id`, `proof_kind`, `application_id`, `result`, `run_id`, `correlation_id`, `task_id`, `provider_evidence`, `domain_evidence`, `guardrails`, `metadata`, `recorded_at`). Partition and row keys are derived as:

```text
partition_key = proof_receipts/local_workspace
row_key       = proof/platform_background_task/<run_id>
```

### Reviewer checks

Verify in the stored `data` object:

```text
schema_version = intergrax.proof_receipt.v1
application_id = local_workspace
proof_kind = platform_background_task
result = PASS
run_id matches Step 8
correlation_id matches Step 8
task_id matches Step 8
provider_evidence.message_bus_provider = kafka
provider_evidence.worker_execution = asynchronous
provider_evidence.task_status = SUCCEEDED
domain_evidence.task_name = lkw.background_ingest.v1
domain_evidence.search_results >= 1
domain_evidence.evidence_marker_found = true
guardrails.mock_queue = false
guardrails.inmemory_bypass = false
guardrails.direct_handler_call = false
guardrails.direct_indexer_call = false
guardrails.direct_mongodb_write = false
guardrails.direct_pymongo_from_lkw = false
guardrails.markdown_source_of_truth = false
```

### Authority

The MongoDB `ProofReceipt` is the source of truth for this run. This markdown page explains how to execute and inspect the proof but does not store the authoritative result.

### Historical example (non-authoritative)

Older markdown-only closeout blocks are retained only as examples. They are not the live source of truth once receipt recording is enabled.

```text
example_only=true
recorded=2026-07-09
message_bus_provider=kafka
worker_execution=asynchronous
```

---

## Reviewer shortcut

```bat
applications\local_workspace_application\scripts\hard-reset-local-docker-all.bat
applications\local_workspace_application\scripts\check-lkw-platform-proof-status.bat
applications\local_workspace_application\scripts\run-sentry-observability-proof.bat --run-id lkw-sentry-live-001 --correlation-id lkw-sentry-live-001
applications\local_workspace_application\scripts\run-lkw-elasticsearch-proof.bat
applications\local_workspace_application\scripts\run-lkw-persistence-proof.bat
applications\local_workspace_application\scripts\run-lkw-background-task-proof.bat
```

Then open:

```text
Sentry:        http://127.0.0.1:9000/organizations/intergrax-local/issues/?project=2
Kibana:        http://127.0.0.1:5601
Kafka UI:      http://127.0.0.1:8085
Mongo Express: http://127.0.0.1:8086
```

1. Inspect Kafka lifecycle using `run_id` / `correlation_id`.
2. Inspect MongoDB receipt using `proof_receipt_id` / `proof_receipt_run_id`.

Kafka topics to inspect:

```text
intergrax.tasks
intergrax.task-events
intergrax.task-status
intergrax.task-results
```

Expected in Kafka UI:

- `TaskRequest` message exists for `run_id` / `correlation_id` in `intergrax.tasks`,
- lifecycle events `task.enqueued`, `task.started`, `task.succeeded`, `task.result_stored` exist in `intergrax.task-events`,
- status/result records exist in `intergrax.task-status` / `intergrax.task-results`.

---

## Token optimization claim guardrails

For token-optimization proof wording and claim boundaries, see [`TOKEN_OPTIMIZATION_CLAIMS.md`](TOKEN_OPTIMIZATION_CLAIMS.md).
