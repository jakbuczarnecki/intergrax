# OD-PDC-ARCHF-01 — ARCH-FREEZE WBS amendment mode

| Field | Value |
| --- | --- |
| **Decision ID** | OD-PDC-ARCHF-01 |
| **Stage** | PDC-X |
| **Status** | **PENDING** |
| **Precedent** | MCP-SERVER-X P0 (T-IGX-035-A): header gate note only, no WBS row |

## Question

How is PDC-X recorded as an ARCH-FREEZE predecessor in `ARCH_FREEZE_WBS_AND_DEPENDENCY_DAG.md`?

## Recommended option (A), applied in PDC-P0

- A note-only program-gate line in the WBS header, plus the roadmap §3 and §5 requirement `PDC-X = CLOSED`.
- The task count stays **54**. The generator `verify_arch_freeze_plan_traceability.py` and `EXPECTED_WBS_TASKS` are unchanged.
- Closure is checked at `T-B8-ROADMAP` through the roadmap §5 ledger.

## Alternate option (B)

- Add a dedicated predecessor task row. This requires editing the generator rows, bumping `EXPECTED_WBS_TASKS` to 55, and extending the DAG with an edge before `T-B7-PFP-ACTIVATE` / `T-B7-LOCK-FREEZE-SHA`.

## Operator record

_Pending._ Option A is applied in T-IGX-036-A, subject to this decision.
