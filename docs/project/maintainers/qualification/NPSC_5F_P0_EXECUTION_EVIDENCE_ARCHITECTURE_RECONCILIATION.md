# NPSC-5F/P0 — Execution Evidence Architecture Reconciliation (Qualification)

> **Task:** `NPSC-5F/P0`  
> **Architecture:** [`NPSC_5F_EXECUTION_EVIDENCE_REPLAY_OBSERVABILITY_ARCHITECTURE.md`](../architecture/NPSC_5F_EXECUTION_EVIDENCE_REPLAY_OBSERVABILITY_ARCHITECTURE.md)  
> **NPSC-5E Final:** `fabdcfe931dfd3a0b22d35cbf06ac94b2b0176f7`  
> **START_HEAD / START_ORIGIN:** `fabdcfe931dfd3a0b22d35cbf06ac94b2b0176f7`  
> **Drift gate** `fabdcfe..origin/development`: **empty** (no A–F drift)

## P0 verdict

**PASS / INVENTORY QUALIFIED** — existing observability/evidence architecture inventoried; enterprise gaps recorded as blockers; no production code changes.

## Ownership matrix

| Concern | Existing owner | Desired owner | Conflict? |
| ------- | -------------- | ------------- | --------- |
| Event identity (`EventId`) | `RuntimeEvent` + persistence acceptance (`reconcile_idempotent_event_acceptance`) | Same | NO |
| Durable evidence persistence | `RuntimeEventPersistence` implementations | Same (+ future typed mandatory classes) | NO |
| Event ordering | `ExecutionEventPosition` per `(tenant, run)` | Same; clarify task API | NO |
| Lineage | `ExecutionLineagePersistence` | Same | NO |
| Checkpoints | R2 / `LongRunningCoordinator` | Same | NO |
| Terminal truth | `ExecutionTerminalService` | Same | NO |
| Redaction/export | `ObservabilityExportEnvelope` / `export_boundary` | Same; journal path must conform | **GAP** (bypass) |
| Reconstruction | `execution_reconstruction`, `asof_projection`, DIAG read stack | Same | NO |
| Metrics | Downstream projections | Same | NO |
| Diagnostics | `DIAGNOSTICS.md` plane | Same | NO |
| Event transport | `RuntimeEventBus` | Same (not durability authority) | NO |

## Gap matrix

| Capability | Existing | Enterprise-ready | Gap |
| ---------- | -------: | --------------: | --- |
| Durable append | Yes (SQLite/memory/doc) | Partial | Bus fail-open |
| Idempotency | Yes | Yes | — |
| Content conflict | Yes (`reconcile_*`) | Yes | — |
| Scoped ordering | Per-run | Partial | Task list semantics |
| Cross-process read | SQLite file/shared store | Yes | — |
| Complete run read | Partial | No | `build_unified_run_journal` limit |
| Tenant isolation (read) | Yes | Yes | — |
| Tenant routing (write) | Partial | No | explicit ≠ event tenant |
| Schema evolution | Partial | Partial | Unknown persisted version |
| Redaction export | Canonical path yes | No | journal_export bypass |
| Reconstruction | Yes (DIAG) | Partial | Completeness model |
| As-of | `load_positioned_*` + projection | Partial | Public API planned |
| Bitemporal correction | K/E slices | Partial | Full E+K+VT planned |
| Gap detection | As-of prefix | Partial | Full run journal |
| Corruption detection | Identity claim (doc store) | Partial | No chain hash |
| Query access control | Host layer | Partial | Product-specific |
| Backpressure | Fail-open bus | No | Typed policies missing |
| Concurrency | SQLite transactional | Yes | — |

## Historical audit re-qualification

| Finding | Classification | Evidence |
| ------- | -------------- | -------- |
| Fail-open evidence durability on persistence error | **STILL PRESENT** | `event_bus.py` `_store_event` except+log continue |
| EventId-only idempotency without content equivalence | **FIXED** | `reconcile_idempotent_event_acceptance`; `test_event_id_persistence_semantics.py` |
| Journal export bypassing `ObservabilityExportEnvelope` | **STILL PRESENT** | `journal_export.serialize_runtime_event` → `model_dump` |
| Silently truncated full-run journal reads | **STILL PRESENT** | `build_unified_run_journal` single `limit` |
| Tenant routing divergence | **STILL PRESENT** | `resolve_event_tenant_id`; P0 test documents mismatch |
| Run-local position as task-global ordering | **STILL PRESENT** | `sqlite_runtime_event_store.list_for_task` ORDER BY `execution_position` |

## P0 tests

Module: `tests/unit/runtime/architecture/test_npsc5f_p0_execution_evidence_architecture_reconciliation.py`

| Requirement | Coverage |
| ----------- | -------- |
| EventId content conflict | Blocked (integrity error) |
| Tenant routing | Gap documented |
| Complete run read | Truncation without marker |
| Redaction | Raw export in journal path |
| Ordering scope | Per-run position restarts |
| Concurrent positions | SQLite uniqueness |
| Unknown schema | Validating wrapper rejects |
| Cross-tenant read | Blocked |
| Lineage ownership | Static — no mutation in evidence roots |
| Execution control bypass | Static — zero hits in events/observability |
| Process boundary | Second SQLite instance reads same DB |
| Persistence failure | Bus fail-open |
| Stream completeness API | No `is_complete` / `next_cursor` on unified journal |

## Enterprise blockers

### P0

- Cross-tenant **write** routing mismatch (index tenant vs event tenant) — provenance/audit risk
- Journal export raw payload bypass — secret/PII exfiltration risk
- Bus **fail-open** on mandatory persist failure — audit completeness risk

### P1

- Silent truncation in `build_unified_run_journal`
- Missing stream completeness markers on full-run API
- Task-scoped ordering contract misleading for multi-run tasks

### P2

- Operator UX for paginated journal export defaults (`limit=2000`)
- Non-critical projection lag documentation

## Regression commands (P0 session)

```text
uv run pytest tests/unit/runtime/architecture/test_npsc5e_final_recovery_plane_qualification_and_freeze.py -q
uv run pytest tests/unit/runtime/diagnostics/test_dg001_lineage_read_integration_r1_final_qualification.py -q
uv run pytest tests/unit/runtime/architecture/test_npsc5d_final_multi_agent_governance_qualification.py -q
uv run pytest tests/unit/runtime/architecture/test_npsc5f_p0_execution_evidence_architecture_reconciliation.py -q
uv run pytest tests/unit/runtime/events/test_observability_persistence_conformance.py tests/unit/runtime/events/test_event_id_persistence_semantics.py tests/unit/runtime/events/test_unified_run_journal.py -q
uv run ruff check tests/unit/runtime/architecture/test_npsc5f_p0_execution_evidence_architecture_reconciliation.py
uv run pyright tests/unit/runtime/architecture/test_npsc5f_p0_execution_evidence_architecture_reconciliation.py
```

## Proposed next

**NPSC-5F/R1** — Canonical durable evidence contract & persistence hardening (bus tiers, tenant equality, fail-closed classes).
