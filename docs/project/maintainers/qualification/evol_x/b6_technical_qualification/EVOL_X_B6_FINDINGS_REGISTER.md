# EVOL-X B6 — findings register (consolidated A5)

**Status:** Technical consolidation · **not** independent B6 acceptance

| Finding ID | Severity | Gate / norm | Evidence SHA | Expected | Actual | Disposition | Remediation | Owner |
|------------|----------|-------------|--------------|----------|--------|-------------|-------------|-------|
| F-001 | **MEDIUM (procedural)** | G-06 | TES `3c4a641b…`, FINAL `bbf3e33b…` | Independent replay exit 0 | First Windows attempt: pytest basetemp inside linked worktree → integrity FAIL | **CLOSED (remediated)** — v2 replay PASS; history preserved | Use basetemp outside nested worktree (documented in B6 package) | Implementer / auditor procedure |
| F-002 | **LOW (evidence gap)** | G-01 / CI | Run `41337077334` | Local JUnit copy optional | Artifact re-downloaded 2026-11-02; SHA256 `69c6a9ad9721b950008888dbf0d68852a35cc9e8b9df0ab5e3d895fabed15ea4`; 290 cases, 0 failures | **CLOSED (non-blocking)** | Byte proof recorded in OD-EVOL-17 technical review; copy stays gitignored | Operator / CI |
| F-003 | **HIGH (formal)** | G-02 | OD-EVOL-11 | Named independent auditor | Appointment approved; **identity unassigned** | **OPEN** | Operator records auditor id in register | Operator |
| F-004 | **HIGH (formal)** | G-07 | — | Signed B6 + `B6_ACCEPTANCE_PUBLICATION_SHA` | No signed record | **OPEN** | W7-01 independent acceptance | Named auditor |
| F-005 | **HIGH (formal)** | G-08 / FRZ-EVOL-07 | TES behavioral evidence | W7-03 Outcome A or B + auth | Not adjudicated by agent | **OPEN** | Independent auditor + operator (OD-EVOL-12 path) | Auditor + operator |
| F-006 | **HIGH (formal)** | G-09 | OD-EVOL-12 | FRZ-08/10 promotion decision | PENDING | **OPEN** | W7-03 batch | Operator |
| F-007 | **HIGH (formal)** | G-10 | OD-EVOLX-08 | Parent EVOL-X closure signature | PENDING | **OPEN** | W7-04 | Operator |
| F-008 | **INFO** | G-03 | Scoped FINAL | M-09H NOT_MEASURED | M-09H NOT_MEASURED in metrics | **N/A (PASS)** | None for scoped path | — |
| F-009 | **INFO** | Post-TES drift | `development` after TES | No VL3 tree change | Only qualification **scripts** changed | **ACCEPTED** | None for qualified population | — |

## Contradiction resolution

- **Invalid vs valid G-06:** Primary evidence = exit codes and logs (`a2-mode-b.log` FAIL vs `a2-mode-b-v2.log` OK); not resolved by agent majority — v2 matches B6 package Windows guidance.
- **Agent technical PASS vs formal B6:** Explicitly separated; no finding closed by Cursor orchestration for G-07.

## Consolidator disposition

No unsupported technical PASS claims remain for G-01, G-04, G-05, G-06 (v2), G-13, scoped G-03/G-14. Formal blockers F-003..F-007 remain. **OD-EVOL-18** records operator acceptance of technical work and prepared ENT intake; it does **not** close F-003..F-007.
