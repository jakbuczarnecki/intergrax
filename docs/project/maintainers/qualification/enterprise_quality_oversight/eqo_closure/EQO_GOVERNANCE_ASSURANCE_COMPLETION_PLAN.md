# EQO — Governance and assurance completion

**Spans:** `EQO-W0` (bind), `EQO-W2` (train tickets), `EQO-X` (governance gaps), `EQO-E` (EDER), closure matrix **C16–C19**.

## W0 governance bind (execute first)

| Task | Deliverable | Blocker if fail |
|------|-------------|-----------------|
| W0-01 | `EQO_PIN_BIND.json` | stale pins → RV-01 |
| W0-02 | `EQO_ARTIFACT_RECOVERY_REGISTRY.jsonl` | missing tickets block W2-12, W2-05-E |
| W0-03 | `pr680_governance_fast.md` | blocks W2-01 / WB01 merge path |
| W0-04 | A4-007 / N-0003 row | blocks W2-10, WB05-DIS |
| W0-05 | inventory + this tree | blocks W4-01, E-02 |

## Train ticket verification map (W2)

| Task | Governance concern |
|------|---------------------|
| W2-01 | execution authority / #680 governance-fast |
| W2-08 | material governance path validator (#673) |
| W2-13 | F-3, G-9, Part G — **WAIT_FOR_OPERATOR** (OD-EQO-04) |

## X-task governance closure

| Task | Item | Close condition |
|------|------|-----------------|
| X-03 | A0-007, IIAC IN_FORCE | RB-2 + N9 (band G), else `BLOCKED` with evidence |
| X-04 | A5-007 RRX counter | read both pins pre-FCS; PT-Q05 owns zeroing |
| X-08 | impact BLOCKS (5 findings) | `BLOCKS_ARCH_FREEZE` verified or G-band downgrade |

## Assurance engine (WB11 + ENT-AUDIT)

- WB11 executes skip/xfail census (A7-037) and PROD-Q evidence gaps (A2-023/024).
- Full KR register settlement: `EQO_TEST_RED_GREEN_SETTLEMENT_PLAN.md` (X-06).
- ENT-AUDIT FULL coordination: A5-015 — **one** non-certifying FULL at FCS, not duplicate PT-Q13.

## Operator decision package (read-only for executor)

Decisions OD-EQO-01..10 govern FCS timing, #680, record-2 scope, lokal eqo-b, WB02 write set, EDER owners. Executor records **compliance state** in closure matrix; does not re-open resolved OD-REM / DP items (plan §3).

## Acceptance

Governance band complete for pre-FCS when: W0-05 done; W2 evidence rows exist or `WAIT_FOR_OPERATOR`/`BLOCKED` with ID; X-03/X-08 not silently deferred; C16–C19 matrix rows updated at each CLS (W9 owns final).