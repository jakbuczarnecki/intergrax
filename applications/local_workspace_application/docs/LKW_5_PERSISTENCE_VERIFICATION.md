# LKW.5 persistence verification — 2026-07-07

## Current status

```text
LKW.5A — canonical LKW data-home settings contract: CLOSED
LKW.5B — repo-dev persistence env defaults aligned with data-home layout: CLOSED
LKW.5C — persistent vector storage contract guardrails: CLOSED
LKW.5D — persistent storage platform proof helper + public proof step: CLOSED
LKW.5E — persistent storage live proof: PASSED
LKW.5 — LKW_DATA_HOME + persistent vector storage: CLOSED IN SCOPE / PERSISTENCE PROOF PASSED
```

## Verified proof path

The LKW.5 live proof verified that indexed local knowledge survives a non-destructive restart of the local LKW stack:

```text
index proof document
-> search before restart
-> docker compose restart local_workspace qdrant
-> search after restart without reindex
-> proof_result=PASS
```

The restart was intentionally non-destructive. The proof did **not** use `hard-reset-local-docker-all`, `docker compose down -v`, or any volume-removal command.

## Latest live proof result

Run date: **2026-07-07**

Command:

```bat
applications\local_workspace_application\scripts\run-lkw-persistence-proof.bat
```

Result:

```text
proof_result=PASS
proof_kind=persistent_vector_storage
restart_mode=non_destructive
volumes_removed=false
tenant_id=lkw-persistence-proof
workspace_id=lkw-persistence-proof
collection_id=lkw-persistence-proof
marker=LKW_PERSISTENCE_PROOF_20260707121815
before_restart_results=1
after_restart_results=1
reindexed_after_restart=false
```

Detailed observed output:

```text
LKW persistent vector storage proof
Repository root: D:\Projekty\intergrax
LKW base URL: http://127.0.0.1:8020
marker=LKW_PERSISTENCE_PROOF_20260707121815

proof_document_host_path=D:\Projekty\intergrax\applications\local_workspace_application\sample_docs\lkw_persistence_proof_20260707121815.txt
proof_document_container_path=/data/user_docs/lkw_persistence_proof_20260707121815.txt

Phase 1/5: waiting for LKW health before indexing...
lkw_health=ok

Phase 2/5: indexing proof document...
index_signal_count=1

Phase 3/5: searching before non-destructive restart...
before_restart_results=1

Phase 4/5: non-destructive restart of local_workspace and qdrant...
[+] Restarting 2/2
 ✔ Container intergrax_lkw-local_workspace-1  Started
 ✔ Container intergrax_lkw-qdrant-1           Started
restart_mode=non_destructive
volumes_removed=false

Phase 5/5: waiting for LKW health after restart...
lkw_health=ok

Searching again without reindexing...
after_restart_results=1

proof_result=PASS
proof_kind=persistent_vector_storage
restart_mode=non_destructive
volumes_removed=false
tenant_id=lkw-persistence-proof
workspace_id=lkw-persistence-proof
collection_id=lkw-persistence-proof
marker=LKW_PERSISTENCE_PROOF_20260707121815
before_restart_results=1
after_restart_results=1
reindexed_after_restart=false
```

## Acceptance

- [x] Canonical data-home settings contract exists.
- [x] Repo-dev persistence env defaults align with the derived data-home layout.
- [x] Qdrant is protected as the persistent local vector-store default.
- [x] `inmemory` remains an explicit dev/test fallback only.
- [x] Docker Compose uses persistent `qdrant_data:/qdrant/storage` storage.
- [x] Public platform proof includes a persistence proof step.
- [x] Persistence helper uses a non-destructive restart.
- [x] Live proof passed with `before_restart_results=1` and `after_restart_results=1` without reindexing.

## Platform propagation classification

No shared platform runtime/provider changes were required.

This closeout is application-local and reviewer-runbook focused:

- LKW owns the proof workload and public reviewer step.
- Qdrant remains the local-first baseline vector store.
- Future platform work can generalize this pattern under vector-store portability / provider-switch proofs.
- Production hardening remains out of scope for this LKW.5 closeout.
