# NPSC-5F/R2 — Journal Completeness & Ordering

> **Status:** IMPLEMENTATION COMPLETE (not FROZEN — await R2 Final qualification)

## Purpose

Close OBS-04 (silent full-journal truncation) and OBS-06 (task-level misuse of run-local `ExecutionEventPosition`) with typed, provider-neutral read contracts.

## Provenance

- Baseline: NPSC-5F/R1 Final `455c09f342f995ac0a6fcb03ffef2f4d3e36a447`
- P0 reconciliation: `7811371da1069b661987b050a4c9bf42c02bda69`

## Parallel-session ownership

- **Session A (this R2):** unified run journal read semantics, pagination/completeness, run-local ordering, task grouping API.
- **Session B/C/D:** must not regress R2 query contracts without reconciliation.

## Current defect (pre-R2)

- `build_unified_run_journal(..., limit=N)` returned a plain list — partial vs complete indistinguishable.
- `list_for_task` ordered by `execution_position` across runs (invalid global chronology).

## Journal read model

| API | Role |
| --- | ---- |
| `read_run_journal_page` | Bounded page; explicit `is_complete` / `next_cursor` |
| `load_complete_run_journal` | Multi-page complete load or typed failure |
| `build_unified_run_journal` | Complete canonical journal (wrapper over loader) |
| `load_positioned_run_journal_through` | Unchanged prefix/as-of authority |

## Completeness contract

- Pages use `limit+1` peek to prove exhaustion (exact-limit safe).
- Complete loader raises `JournalReadLimitExceededError` when `max_events` exceeded — never silent truncation.

## Pagination contract

- `RunJournalReadPage`: `events`, `is_complete`, `next_cursor`.
- `RunJournalContinuationCursor`: `tenant_id`, `run_id`, `exclusive_after`, `snapshot_through`.

## Cursor scope

- Cursor valid only for matching tenant + run; mismatch → `JournalCursorScopeMismatchError`.

## Full journal semantics

- `build_unified_run_journal` / `load_complete_run_journal` return full snapshot-bounded history or fail closed.

## Exact-limit / N+1 semantics

- `count == page_size` with no peek row → `is_complete=True`.
- `count == page_size + 1` → partial page + cursor.

## Concurrent append semantics

- **Snapshot boundary:** first page freezes `snapshot_through`; later appends excluded from that pagination sequence.

## Run-local ordering

- `ExecutionEventPosition` orders within `(tenant_id, run_id)` only.

## Task-level ordering decision

- No fake task-global chronology.
- `list_positioned_for_task_grouped_by_run` + stable `(run_id, execution_position)` flatten order for `list_for_task`.

## Store parity

- InMemory, SQLite, DocumentBacked, Validating — shared contract (`after` on run lists, grouped task reads).

## Consumer migration

- `journal_export` uses `read_run_journal_page` (bounded preview).
- Debug `build_unified_run_journal` uses complete loader.

## Interaction with R1

- R1 durability/tenant integrity unchanged; read extensions only on persistence query surface.

## Deferred R3

- Export redaction / `ObservabilityExportEnvelope` alignment — unchanged.

## Deferred R4

- Bitemporal reconstruction enhancements — prefix helper preserved.

## Scale follow-up

- **OWNER = SESSION C:** document partition full-scan for very large runs; consider store-native continuation without full partition materialization.

## Regression matrix

- `tests/unit/runtime/architecture/test_npsc5f_r2_journal_completeness_ordering.py`
- P0 / R1 Final gates, unified journal, journal export, as-of prefix suites.

## Static quality

- `ruff` + `pyright` on changed production paths required before R2 Final.

## Final verdict

**NPSC-5F/R2: IMPLEMENTATION COMPLETE** — proceed to R2 Final qualification and freeze.
