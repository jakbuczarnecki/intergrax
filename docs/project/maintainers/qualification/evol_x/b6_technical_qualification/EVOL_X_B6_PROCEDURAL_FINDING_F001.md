# F-001 — Windows worktree / pytest basetemp procedural finding

**Finding ID:** F-001  
**Gate:** G-06  
**Severity:** MEDIUM (procedural — not product defect)

## Summary

First independent replay attempt on Windows used pytest basetemp inside (or defaulting within) a **linked git worktree** (`wt-tes`). P03 integrity tests construct nested temporary repositories; basetemp placement caused widespread false FAIL and `EVIDENCE_INTEGRITY_FAIL` in Mode B.

## Invalid attempt (retained)

| Artifact | Observation |
|----------|-------------|
| `a2-pytest.log` | Many failures in `test_evol_x_p03_integrity.py` |
| `a2-mode-b.log` | Outcome swapped errors (archived=passed, replay=failure) |

## Corrected replay (valid G-06 evidence)

| Change | Detail |
|--------|--------|
| Basetemp | `d:\Projekty\intergrax\build\pytest-b6-tes-replay` (primary repo `build/`, outside nested worktree) |
| Logs | `a2-pytest-v2.log`, `a2-mode-b-v2.log` |
| Result | 290 passed; Mode B OK |

## Normative alignment

[EVOL_X_B6_INDEPENDENT_AUDIT_PACKAGE.md](../EVOL_X_B6_INDEPENDENT_AUDIT_PACKAGE.md) Mode B section — auditors on Windows must avoid worktree-local basetemp for P03 replay.

## Disposition

**CLOSED (remediated)** for technical qualification; historical invalid logs **must remain** in session history.
