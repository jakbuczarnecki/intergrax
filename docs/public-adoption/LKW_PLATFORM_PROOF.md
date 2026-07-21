# Intergrax Platform Proof — Local Knowledge Workspace

**Prerequisite:** Complete the [README Quick start](../../README.md#quick-start) (lab host, ~5 min) before this platform proof.

This document is the guided reviewer path for two proof categories:

```text
Core Platform Proof
Optional Operating-System Interaction Proofs
```

The Core Platform Proof verifies platform and application
capabilities that are not specific to one interaction client.

Operating-system interaction proofs are optional extensions.
A reviewer runs only the optional proof that matches the
reviewer's operating system.

Skipping an OS-specific interaction proof does not invalidate
the Core Platform Proof.

MongoDB-backed ProofReceipt documents are authoritative.
Terminal output is reviewer convenience.
Markdown is execution and inspection guidance.

A reviewer should not need to inspect raw Docker output or infer what to check from long logs.

---

## Core platform claims

```text
1. LKW starts as a real Tier-3 Intergrax application.
2. LKW emits policy-safe observability records into Elasticsearch/Kibana.
3. LKW emits controlled problem signals into local Sentry.
4. LKW persists indexed local knowledge across a non-destructive restart.
5. LKW enqueues and executes background-ingest jobs through the real platform message-bus and worker path.
6. LKW records structured proof evidence through ProofReceiptStore into a real MongoDB DocumentStore.
7. LKW runs through Intergrax Application Hosting as a real foreground process, including readiness, single-instance ownership, graceful stop, lock release and supervised restart.
8. A file created in a watched folder is automatically indexed through the real Kafka/worker path, remains searchable after a non-destructive restart and produces a verified ProofReceipt.
```

## Optional operating-system interaction claims

These proofs validate one concrete operating-system client
adapter. They are not required to complete the Core Platform Proof.

```text
Windows optional interaction:
  implemented and live-certified

Linux optional interaction:
  implemented, not live-certified

macOS optional interaction:
  implemented, not live-certified
```

Windows claim (optional):

```text
A real Windows PowerShell wrapper launches the shared Python
interaction client, which sends work through
/v1/interactions/intake into the shared LKW executor and Nexus
path, performing real index and search work.
```

Shared interaction architecture:

```text
Windows PowerShell wrapper ─┐
Linux shell wrapper ────────┼→ invoke-lkw-interaction.py
macOS shell wrapper ────────┘

Windows BAT ─┐
Linux SH ────┼→ run-lkw-os-interaction-proof.py
macOS SH ────┘
```

Frozen OS identities:

```text
windows / lkw.windows_powershell / windows_powershell / windows_powershell
linux   / lkw.linux_shell        / linux_shell        / posix_sh
macos   / lkw.macos_shell        / macos_shell        / posix_sh
```

Implementation is shared. Live certification remains OS-specific.
Linux and macOS paths are implemented but not live-certified until
PROOF-PORTABILITY-1D.

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

## Choose your operating system

### Windows

Shared core entrypoint implemented and live-tested.

Complete the Core Platform Proof through the shared Python runner
(Windows BAT launcher) or the numbered core steps below.

Optional Windows PowerShell interaction proof remains separate.

The Windows interaction proof is not required for core completion.

### Linux

Shared core entrypoint implemented.

The Linux launcher invokes the same Python core-proof runner.

Linux core execution is not live-certified until
PROOF-PORTABILITY-1D.

The optional Linux interaction client and proof runner are
implemented under PROOF-PORTABILITY-1C and are not live-certified
until PROOF-PORTABILITY-1D.

Do not run Windows .bat or Windows PowerShell interaction steps.

### macOS

Shared core entrypoint implemented.

The macOS launcher invokes the same Python core-proof runner.

macOS core execution is not live-certified until
PROOF-PORTABILITY-1D.

The optional macOS interaction client and proof runner are
implemented under PROOF-PORTABILITY-1C and are not live-certified
until PROOF-PORTABILITY-1D.

Do not run Windows .bat or Windows PowerShell interaction steps.

---

## Operating-system proof status

| Operating system | Core reviewer path                       | Optional OS interaction proof                         | Certification status       |
| ---------------- | ---------------------------------------- | ----------------------------------------------------- | -------------------------- |
| Windows          | Shared Python runner through Windows BAT | Shared Python interaction proof through Windows BAT   | Core live-certified        |
| Linux            | Shared Python runner through Linux SH    | Shared Python interaction proof through Linux SH      | Implemented, not certified |
| macOS            | Shared Python runner through macOS SH    | Shared Python interaction proof through macOS SH      | Implemented, not certified |

```text
Implemented:
  executable repository path exists and is contract-tested

Live-certified:
  the path has completed a real run on that OS and produced
  its required persisted evidence

Planned:
  no implementation or certification claim is made
```

---

## Core prerequisites

```text
Docker Desktop or Docker Engine with Docker Compose
Python 3.12
uv
repository checkout
```

## Current reviewer-command requirements

```text
Windows:
  Docker, Python 3.12 and uv
  Windows PowerShell required only for the optional Windows
  interaction proof

Linux:
  Docker Engine/Desktop with Compose, Python 3.12, uv,
  POSIX /bin/sh

macOS:
  Docker Desktop with Compose, Python 3.12, uv,
  POSIX /bin/sh
```

This describes the current reviewer command wrappers only.
It does not imply that the hosting engine itself is Windows-only.

Run all commands from the repository root.

---

## Recommended one-command Core Platform Proof

### Windows

```bat
applications\local_workspace_application\scripts\run-lkw-core-platform-proof-windows.bat
```

### Linux

```sh
./applications/local_workspace_application/scripts/run-lkw-core-platform-proof-linux.sh
```

### macOS

```sh
./applications/local_workspace_application/scripts/run-lkw-core-platform-proof-macos.sh
```

```text
All three launchers invoke the same Python implementation.

The launchers contain no proof workload or acceptance logic.

The numbered Core Steps 1–13 below define the proof phases,
expected evidence and manual inspection boundaries.

A reviewer using the one-command entrypoint does not run the
optional OS interaction proof as part of core completion.

The shared core entrypoint was delivered by PROOF-PORTABILITY-1B.
Shared OS interaction client/proof plumbing was delivered by
PROOF-PORTABILITY-1C.
Linux/macOS live certification remains PROOF-PORTABILITY-1D.
```

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

## Step 10 — Run the Application Hosting proof

Run:

```bat
applications\local_workspace_application\scripts\run-lkw-hosting-proof.bat
```

Optional deterministic reviewer command:

```bat
applications\local_workspace_application\scripts\run-lkw-hosting-proof.bat --run-id lkw-hosting-live-001 --correlation-id lkw-hosting-live-001
```

The helper is idempotent: it starts or refreshes only MongoDB and Mongo Express before running the proof. It does not start the Docker LKW application, Kafka, Redis, Qdrant, Elasticsearch, or Sentry. The accepted APP-HOST-8C/8D live tests own the real hosted LKW processes.

Expected result:

```text
proof_result=PASS
proof_kind=platform_application_hosting
proof_tests_passed=3

foreground_ready=true
real_index_before_restart=true
instance_conflict_verified=true
first_process_remained_ready=true

foreground_clean_stop=true
foreground_shutdown_reason=<signal.sigterm|signal.sigbreak>
replacement_process_ready=true
instance_lock_released=true
replacement_clean_stop=true

restart_requested=true
first_instance_id=<actual first ID>
second_instance_id=<actual second ID>
instance_id_changed=true

first_attempt_exit_kind=restart_requested
first_attempt_cleanup_verified=true
first_lease_released=true
first_context_closed=true

stopped_events_verified=true
restart_events_verified=true

second_instance_ready=true
real_index_after_restart=true

profile_digest=<sha256 digest>
definition_digest=<sha256 digest>
profile_digest_preserved=true
definition_digest_preserved=true

final_exit_kind=clean_stop
final_cleanup_verified=true
final_lease_released=true
final_context_closed=true
final_lock_reacquired=true

proof_receipt_recorded=true
proof_receipt_verified=true
proof_receipt_query_verified=true
proof_receipt_store=platform
document_store_provider=mongodb

proof_receipt_id=<generated_proof_id>
proof_receipt_run_id=<generated_run_id>
proof_receipt_result=PASS
correlation_id=<correlation_id>

mongo_express_url=http://127.0.0.1:8086

inmemory_receipt_store=false
direct_mongodb_write=false
direct_pymongo_from_lkw=false
markdown_source_of_truth=false
manual_evidence_injection=false
```

This proof:

- starts only MongoDB and Mongo Express,
- runs the exact accepted APP-HOST-8C/8D live tests,
- collects structured JUnit evidence,
- records one `ProofReceipt`,
- verifies write/read/query,
- prints `PASS` only after verification.

A green unit test alone is not APP-HOST-8E acceptance.

The reviewer command must complete with `proof_result=PASS` and a verified MongoDB-backed `ProofReceipt`.

This step does not claim Windows Service, systemd, launchd, reboot persistence, service-manager installation, production HA, multi-node supervision, crash recovery, or restart exhaustion.

---

## Step 11 — Inspect the Application Hosting ProofReceipt in Mongo Express

After Step 10 prints `proof_receipt_recorded=true` and `proof_receipt_verified=true`, inspect the persisted hosting receipt.

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

Use the values printed by Step 10:

```text
proof_receipt_id
proof_receipt_run_id
proof_kind = platform_application_hosting
```

Each stored document is a MongoDB row mapped from `DocumentRecord`. The `data` field contains the full `ProofReceipt` JSON. Partition and row keys are derived as:

```text
partition_key = proof_receipts/local_workspace
row_key       = proof/platform_application_hosting/<run_id>
```

### Reviewer checks

Verify in the stored `data` object:

```text
schema_version = intergrax.proof_receipt.v1
application_id = local_workspace
proof_kind = platform_application_hosting
result = PASS
run_id matches Step 10
correlation_id matches Step 10
provider_evidence.foreground_execution = real_subprocess
provider_evidence.supervisor = HostedApplicationSupervisor
provider_evidence.engine = HostedApplicationEngine
provider_evidence.instance_guard = FileHostedApplicationInstanceGuard
provider_evidence.evidence_source = pytest_junit_properties
provider_evidence.selected_live_tests = 3
provider_evidence.receipt_document_store_provider = mongodb
domain_evidence.foreground_ready = true
domain_evidence.real_index_before_restart = true
domain_evidence.instance_conflict_verified = true
domain_evidence.first_process_remained_ready = true
domain_evidence.foreground_clean_stop = true
domain_evidence.instance_lock_released = true
domain_evidence.restart_requested = true
domain_evidence.first_instance_id != domain_evidence.second_instance_id
domain_evidence.instance_id_changed = true
domain_evidence.first_attempt_exit_kind = restart_requested
domain_evidence.first_attempt_cleanup_verified = true
domain_evidence.first_lease_released = true
domain_evidence.first_context_closed = true
domain_evidence.restart_events_verified = true
domain_evidence.second_instance_ready = true
domain_evidence.real_index_after_restart = true
domain_evidence.profile_digest_preserved = true
domain_evidence.definition_digest_preserved = true
domain_evidence.final_exit_kind = clean_stop
domain_evidence.final_cleanup_verified = true
domain_evidence.final_lease_released = true
domain_evidence.final_context_closed = true
domain_evidence.final_lock_reacquired = true
guardrails.mock_hosting_runtime = false
guardrails.fake_supervisor = false
guardrails.fake_engine = false
guardrails.fake_instance_guard = false
guardrails.inmemory_receipt_store = false
guardrails.direct_mongodb_write = false
guardrails.direct_pymongo_from_lkw = false
guardrails.markdown_source_of_truth = false
guardrails.manual_evidence_injection = false
```

### Authority

The MongoDB ProofReceipt is the authoritative outcome for the Application Hosting proof run.

The pytest JUnit file is temporary evidence transport.

The terminal summary is a reviewer convenience.

This markdown page is the execution and inspection guide.

None of those replaces the persisted ProofReceipt.

---

## Step 12 — Run the File Watcher E2E proof

Run:

```bat
applications\local_workspace_application\scripts\run-lkw-file-watcher-e2e-proof.bat
```

The command:

```text
starts the required stack
warms the real search/embedding path
creates the file only after baseline
uses no manual indexing command
verifies exact source_ref
restarts the mechanism non-destructively
verifies checkpoint restore
records and verifies ProofReceipt
```

Expected core PASS fields:

```text
proof_result=PASS
proof_kind=file_watcher_persistent_search
embedding_warmup_completed=true
reviewer_rerun_required=false
trigger=filesystem_create
manual_index_command=false
direct_enqueue=false
message_bus_provider=kafka
worker_execution=asynchronous
vector_store_provider=qdrant
persistent_index=true
watcher_checkpoint_ready=true
watcher_restored_after_restart=true
task_topic_increased=true
source_ref_found_before_restart=true
restart_mode=non_destructive
volumes_removed=false
source_file_modified_after_index=false
reindexed_after_restart=false
duplicate_enqueue_after_restart=false
source_ref_found_after_restart=true
proof_receipt_recorded=true
proof_receipt_verified=true
proof_receipt_query_verified=true
proof_receipt_store=platform
document_store_provider=mongodb
proof_receipt_result=PASS
proof_receipt_application_id=local_workspace
proof_receipt_task=LKW.7C2
markdown_source_of_truth=false
direct_mongodb_write=false
direct_pymongo_from_lkw=false
manual_evidence_injection=false
```

---

## Step 13 — Inspect the File Watcher ProofReceipt in Mongo Express

After Step 12 prints `proof_receipt_recorded=true` and `proof_receipt_verified=true`, inspect the persisted file-watcher receipt.

Open:

```text
http://127.0.0.1:8086
```

Select:

```text
database:
  intergrax_proofs

collection:
  proof_receipts
```

Document identity:

```text
partition_key =
  proof_receipts/local_workspace

row_key =
  proof/file_watcher_persistent_search/<run_id>
```

Reviewer checks:

```text
schema_version = intergrax.proof_receipt.v1
application_id = local_workspace
proof_kind = file_watcher_persistent_search
result = PASS
run_id matches printed proof_receipt_run_id
provider_evidence.message_bus_provider = kafka
provider_evidence.worker_execution = asynchronous
provider_evidence.enqueue_trigger = filesystem_create
provider_evidence.checkpoint_restore_verified = true
provider_evidence.vector_store_provider = qdrant
provider_evidence.persistent_index = true
provider_evidence.document_store_provider = mongodb
provider_evidence.task_topic_increased = true
provider_evidence.duplicate_enqueue_after_restart = false
domain_evidence.embedding_warmup_completed = true
domain_evidence.reviewer_rerun_required = false
domain_evidence.source_ref_found_before_restart = true
domain_evidence.source_ref_found_after_restart = true
domain_evidence.source_file_modified_after_index = false
domain_evidence.reindexed_after_restart = false
guardrails.manual_index_command = false
guardrails.direct_enqueue = false
guardrails.direct_indexer_call = false
guardrails.direct_ingest_call = false
guardrails.direct_mongodb_write = false
guardrails.direct_pymongo_from_lkw = false
guardrails.markdown_source_of_truth = false
```

### Authority

The MongoDB-backed ProofReceipt is authoritative.

Terminal output is a reviewer convenience.

This markdown page is the execution and inspection guide.

---

## Core Platform Proof completion

The Core Platform Proof is complete when the required core steps
have produced their expected PASS results and authoritative
ProofReceipt records.

No operating-system interaction proof is required for core
completion.

---

# Optional operating-system interaction proofs

This section is optional.

A PASS result extends the evidence set with one OS-specific client
adapter proof recorded through the shared Python interaction proof
runner.

Its omission does not invalidate the Core Platform Proof.

Run the matching OS launcher only on that operating system.

## Windows users — Optional W1: Run the Windows PowerShell interaction proof

Windows optional interaction remains implemented and live-certified.

Run:

```bat
applications\local_workspace_application\scripts\run-lkw-windows-interaction-proof.bat
```

Optional deterministic reviewer command:

```bat
applications\local_workspace_application\scripts\run-lkw-windows-interaction-proof.bat --run-id lkw-windows-interaction-live-001 --correlation-id lkw-windows-interaction-live-001
```

The public BAT delegates to `run-lkw-os-interaction-proof.py --os-family windows`.

The shared runner starts only MongoDB and Mongo Express.

The live test starts hosted LKW itself.

The live test invokes the thin PowerShell wrapper, which launches
the shared Python interaction client.

The shared client calls only `/v1/interactions/intake`.

A green static/unit test alone does not close live certification.

Expected result:

```text
proof_result=PASS
proof_kind=platform_windows_interaction
proof_tests_passed=1
os_family=windows
adapter_invoked=true
adapter_id=lkw.windows_powershell
client_runtime=python
wrapper_runtime=windows_powershell
powershell_runtime=Windows PowerShell
transport=http
intake_endpoint=/v1/interactions/intake
interaction_surface=lab_json
interaction_channel=lab
hosted_ready=true
index_executed=true
index_state=completed
index_task_id=<actual ID>
index_run_id=<actual ID>
search_executed=true
search_state=completed
search_task_id=<actual ID>
search_run_id=<actual ID>
task_ids_distinct=true
run_ids_distinct=true
graceful_stop=true
cleanup_verified=true
proof_receipt_recorded=true
proof_receipt_verified=true
proof_receipt_query_verified=true
proof_receipt_store=platform
document_store_provider=mongodb
proof_receipt_id=<generated ID>
proof_receipt_run_id=<run ID>
proof_receipt_result=PASS
correlation_id=<correlation ID>
mongo_express_url=http://127.0.0.1:8086
direct_run_endpoint=false
direct_task_executor_call=false
direct_nexus_call=false
fake_interaction_service=false
new_platform_interaction_adapter=false
generic_os_hosting_adapter=false
service_installation=false
manual_evidence_injection=false
inmemory_receipt_store=false
direct_mongodb_write=false
direct_pymongo_from_lkw=false
markdown_source_of_truth=false
```

---

## Windows users — Optional W2: Inspect the Windows Interaction ProofReceipt

After Optional W1 prints `proof_receipt_recorded=true` and `proof_receipt_verified=true`, inspect the persisted Windows interaction receipt.

Open:

```text
http://127.0.0.1:8086
```

Select:

```text
database:
  intergrax_proofs

collection:
  proof_receipts
```

Filter:

```text
proof_kind = platform_windows_interaction
run_id = <printed proof_receipt_run_id>
```

Document identity:

```text
partition_key = proof_receipts/local_workspace
row_key = proof/platform_windows_interaction/<run_id>
```

Reviewer checks:

```text
schema_version = intergrax.proof_receipt.v1
application_id = local_workspace
proof_kind = platform_windows_interaction
result = PASS
provider_evidence.os_family = windows
provider_evidence.os_adapter = lkw.windows_powershell
provider_evidence.client_runtime = python
provider_evidence.wrapper_runtime = windows_powershell
provider_evidence.intake_endpoint = /v1/interactions/intake
provider_evidence.intake_service = InteractionIntakeService
provider_evidence.execution_boundary = LocalWorkspaceTaskExecutor
provider_evidence.orchestrator = NexusLoop
domain_evidence.adapter_invoked = true
domain_evidence.interaction_surface = lab_json
domain_evidence.interaction_channel = lab
domain_evidence.powershell_runtime = Windows PowerShell
domain_evidence.index_executed = true
domain_evidence.index_state = completed
domain_evidence.index_task_id is non-empty
domain_evidence.index_run_id is non-empty
domain_evidence.search_executed = true
domain_evidence.search_state = completed
domain_evidence.search_task_id is non-empty
domain_evidence.search_run_id is non-empty
domain_evidence.task_ids_distinct = true
domain_evidence.run_ids_distinct = true
domain_evidence.graceful_stop = true
domain_evidence.cleanup_verified = true
guardrails.direct_run_endpoint = false
guardrails.direct_task_executor_call = false
guardrails.direct_nexus_call = false
guardrails.fake_interaction_service = false
guardrails.new_platform_interaction_adapter = false
guardrails.generic_os_hosting_adapter = false
guardrails.service_installation = false
guardrails.direct_mongodb_write = false
guardrails.markdown_source_of_truth = false
```

### Authority

MongoDB ProofReceipt is authoritative.

JUnit is temporary evidence transport.

Terminal output is reviewer convenience.

Markdown is the execution and inspection guide.

---

## Linux users — Optional interaction proof

Status: implemented, not live-certified

Linux has a thin shell wrapper and shares the Python interaction
client and OS interaction proof runner.

Run only on Linux:

```sh
./applications/local_workspace_application/scripts/run-lkw-linux-interaction-proof.sh
```

Frozen identity:

```text
os_family=linux
proof_kind=platform_linux_interaction
adapter_id=lkw.linux_shell
source=linux_shell
wrapper_runtime=posix_sh
```

Do not substitute the Windows PowerShell proof.

A Linux ProofReceipt can only be produced by a real successful run
on Linux. Source-code existence is not live evidence.

Linux optional interaction is not live-certified until
PROOF-PORTABILITY-1D.

---

## macOS users — Optional interaction proof

Status: implemented, not live-certified

macOS has a thin shell wrapper and shares the Python interaction
client and OS interaction proof runner.

Run only on macOS:

```sh
./applications/local_workspace_application/scripts/run-lkw-macos-interaction-proof.sh
```

Frozen identity:

```text
os_family=macos
proof_kind=platform_macos_interaction
adapter_id=lkw.macos_shell
source=macos_shell
wrapper_runtime=posix_sh
```

Do not substitute the Windows PowerShell proof.

A macOS ProofReceipt can only be produced by a real successful run
on macOS. Source-code existence is not live evidence.

macOS optional interaction is not live-certified until
PROOF-PORTABILITY-1D.

---

## Core reviewer shortcuts

Recommended one-command Core Platform Proof launchers:

```bat
applications\local_workspace_application\scripts\run-lkw-core-platform-proof-windows.bat
```

```sh
./applications/local_workspace_application/scripts/run-lkw-core-platform-proof-linux.sh
./applications/local_workspace_application/scripts/run-lkw-core-platform-proof-macos.sh
```

### Legacy/manual Windows phase commands

```bat
applications\local_workspace_application\scripts\hard-reset-local-docker-all.bat
applications\local_workspace_application\scripts\check-lkw-platform-proof-status.bat
applications\local_workspace_application\scripts\run-sentry-observability-proof.bat --run-id lkw-sentry-live-001 --correlation-id lkw-sentry-live-001
applications\local_workspace_application\scripts\run-lkw-elasticsearch-proof.bat
applications\local_workspace_application\scripts\run-lkw-persistence-proof.bat
applications\local_workspace_application\scripts\run-lkw-background-task-proof.bat
applications\local_workspace_application\scripts\run-lkw-hosting-proof.bat
applications\local_workspace_application\scripts\run-lkw-file-watcher-e2e-proof.bat
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
3. Inspect the Application Hosting receipt using `proof_receipt_id` / `proof_receipt_run_id` and `proof_kind=platform_application_hosting`.
4. Inspect the File Watcher receipt using `proof_receipt_id` / `proof_receipt_run_id` and `proof_kind=file_watcher_persistent_search`.

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

## Optional Windows reviewer shortcut

```bat
applications\local_workspace_application\scripts\run-lkw-windows-interaction-proof.bat
```

Inspect the Windows Interaction receipt using `proof_receipt_id` / `proof_receipt_run_id` and `proof_kind=platform_windows_interaction`.

---

## Token optimization claim guardrails

For token-optimization proof wording and claim boundaries, see [`TOKEN_OPTIMIZATION_CLAIMS.md`](TOKEN_OPTIMIZATION_CLAIMS.md).
