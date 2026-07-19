# LKW.7 File Watcher E2E Verification

**Status: Closed**

## Purpose

Document the accepted live proof that a file created in a watched folder is automatically indexed through the real Kafka/worker path, remains searchable after a non-destructive restart, and produces a verified MongoDB-backed ProofReceipt.

## Claim being proved

```text
filesystem create after watcher baseline
  → watcher metadata diff
  → Kafka background-ingest task
  → asynchronous worker
  → LocalIndexerAgent / rag.ingest_document
  → persistent Qdrant
  → exact source_ref before restart
  → non-destructive restart
  → exact source_ref after restart
  → no duplicate enqueue
  → checkpoint restore proven
  → ProofReceipt persisted and verified in MongoDB
```

## One-command reviewer execution

```bat
applications\local_workspace_application\scripts\run-lkw-file-watcher-e2e-proof.bat
```

The reviewer does not manually call an index endpoint.

The reviewer does not manually enqueue a job.

The reviewer does not run the command twice to warm the model.

## Proof topology

```text
Compose merge:
  docker-compose.yml
  docker-compose.kafka.yml
  file-watcher-e2e.compose.yml
  docker-compose.mongodb.yml

Services:
  local_workspace
  lkw-background-worker
  lkw-file-watcher
  lkw-kafka / lkw-kafka-topics / lkw-kafka-ui
  lkw-redis
  qdrant
  ollama
  lkw-mongodb / lkw-mongo-express
```

## Cold-start warm-up

Before Kafka baseline counts and proof-file creation, the Python runner issues bounded
`POST /v1/local_workspace/run` requests with `capability=local.workspace.search` and
`proof_phase=embedding_warmup`. Warm-up succeeds only when diagnostics show
`used=true` and `reason=retrieve_complete`. Zero hits are acceptable. The warm-up
does not index or enqueue. Successful runs print `embedding_warmup_completed=true`
and `reviewer_rerun_required=false`.

## Baseline-before-file invariant

The watcher checkpoint must be ready before the proof document is created. Pre-existing
files are baseline-only and are not enqueued as `created` at sidecar start.

## Kafka evidence

`task_count_after_file > task_count_before_file` on `intergrax.tasks`.
After non-destructive restart of watcher/worker/backend/Qdrant,
`task_count_after_restart == task_count_before_restart`.

## Search/source_ref evidence

Exact `container_source_path` appears in `lkw.search_summary.v1.source_refs`
before and after restart.

## Persistent-index restart evidence

Restart mode is non-destructive (`volumes_removed=false`). Search still returns the
same source after restart without reindexing the unchanged file.

## Checkpoint-restore evidence

Graceful watcher stop emits sidecar JSON with `restored_from_checkpoint=true` and
`final_checkpoint_saved=true`. The watcher is resumed before receipt recording.

## Source-file immutability evidence

Host-side size and `modified_time_ns` are captured after index and after restart.
`source_file_modified_after_index=false` is required.

## Duplicate-enqueue negative control

Unchanged source after restart must not increase the Kafka task topic count.

## ProofReceipt mapping

```text
proof_kind = file_watcher_persistent_search
application_id = local_workspace
run_id = proof marker
task_id = None
correlation_id = None
```

LKW maps in-memory live evidence into a platform `ProofReceipt` and records it through
`ProofReceiptStore` → `DocumentStore` → `MongoDBDocumentStoreIntegration`.
LKW does not write MongoDB directly.

## Mongo Express inspection

```text
http://127.0.0.1:8086
database: intergrax_proofs
collection: proof_receipts
partition_key = proof_receipts/local_workspace
row_key = proof/file_watcher_persistent_search/<run_id>
```

## Authority and source of truth

The MongoDB-backed ProofReceipt is the authoritative result.

Terminal output is a reviewer convenience.

This markdown document is the execution and interpretation guide.

The latest-run block is non-authoritative.

## Known boundaries

```text
change identity is metadata-based, not content hashing
watcher uses polling, not native filesystem events
first fresh baseline does not enqueue pre-existing files
deletion-only changes do not remove indexed vectors
one sidecar configuration maps to one tenant/workspace/collection
OS-service packaging remains outside LKW.7
```

Closing LKW.7 does not close these separate future concerns.

## Latest accepted live run

Non-authoritative reviewer convenience

```text
recorded_at_utc: 2026-07-19T12:19:18Z
proof_result: PASS
proof_receipt_id: local_workspace:file_watcher_persistent_search:LKW_FILE_WATCHER_E2E_20260719T121918Z_140fd6a9
proof_receipt_run_id: LKW_FILE_WATCHER_E2E_20260719T121918Z_140fd6a9
proof_receipt_result: PASS
marker: LKW_FILE_WATCHER_E2E_20260719T121918Z_140fd6a9
container_source_path: /data/user_docs/lkw_file_watcher_e2e_20260719T121918Z_140fd6a9.txt
task_count_before_file: 4
task_count_after_file: 5
task_count_before_restart: 5
task_count_after_restart: 5
search_results_before_restart: 4
search_results_after_restart: 4
watcher_restored_after_restart: true
source_file_modified_after_index: false
duplicate_enqueue_after_restart: false
embedding_warmup_completed: true
reviewer_rerun_required: false
```
