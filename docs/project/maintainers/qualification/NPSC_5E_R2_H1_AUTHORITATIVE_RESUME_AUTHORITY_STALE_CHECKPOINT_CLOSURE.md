# NPSC-5E/R2-H1 — Authoritative Resume Authority & Stale Checkpoint Closure

**Status:** `PASS`

**Date:** 2026-09-10

**Branch:** `development`

**R2 implementation baseline:** `37276a7e2755847b088fc91da76dee0f96824172`

---

## Purpose

Close R2 correction gaps:

1. Checkpoint historical authority cannot become effective resume authority.
2. Stale checkpoint detection uses durable monotonic store ordering, not timestamp alone.

---

## Authority formula

```text
checkpoint authority = historical constraint / provenance only
current authority    = authoritative capability source (Task.execution_authority)
effective resume authority = narrow(current, historical) — monotonic narrowing only
```

Fail closed:

```text
checkpoint authority exists + current authority unavailable → REJECT_AUTHORITY
malformed checkpoint authority provenance → REJECT_MALFORMED / REJECT_AUTHORITY
```

---

## Canonical seams

| Seam | Role |
| ---- | ---- |
| `Task.execution_authority` | Authoritative current authority on resume |
| `narrow_resume_execution_authority` | Pure deterministic narrowing helper |
| `validate_checkpoint_resume_authority` | Eligibility gate wired into `evaluate_checkpoint_resume_eligibility` |
| `resolve_resume_execution_authority` | Coordinator restore narrowing (no checkpoint rehydration) |

No `CheckpointAuthorityResolver` / `ResumeAuthorityEngine` introduced.

---

## Stale ordering

| Key | Contract |
| --- | -------- |
| `TaskCheckpoint.store_sequence` | Populated from SQLite `rowid` on save/load |
| `SQLiteTaskCheckpointStore.get_latest` | `ORDER BY rowid DESC` |
| Timestamp | Auxiliary; equal/missing timestamps defer to `store_sequence` |

---

## Qualification gate

`tests/unit/runtime/architecture/test_npsc5e_r2_h1_authority_stale_checkpoint_closure.py`

| Area | Result |
| ---- | ------ |
| Current authority `None` + checkpoint authority | `REJECT_AUTHORITY` |
| Current narrower than checkpoint | Current wins |
| Checkpoint narrower than current | Historical bound preserved |
| Checkpoint unrestricted + current restricted | Current restriction wins |
| Checkpoint restricted + current unrestricted | Historical bound preserved |
| Both unrestricted | PASS |
| Malformed snapshot authority | BLOCKED |
| Stale earlier timestamp | BLOCKED |
| Same timestamp different revision | BLOCKED (`store_sequence`) |
| Missing timestamp | BLOCKED via `store_sequence` |
| Latest identical checkpoint | ALLOW |
| Stale resume token | BLOCKED |
| Stale write race (late old timestamp) | Canonical latest = higher `rowid` |
| AST: no `restored.execution_authority` fallback | PASS |
| R2 original gate | PASS |
| R1 final regression | PASS |

---

## Next

`NPSC-5E/R2 Final` — Checkpoint & Durable Resume Qualification and Freeze
