# Plan review — T-IGX-030-C (no-boundary `pool.submit` failure terminalization)

| Field | Value |
|---|---|
| Reviewer session id | `hist-rev-030c` |
| Assurance | `process_enforced` |
| Review SHA | `fe09375177bd0fa0d9ebb661b7f9bf64e2f1ed25` |
| Plan | `governance/tickets/plans/PLAN_T-IGX-030-C.md` |
| Ticket | `governance/tickets/T-IGX-030-C.md` |
| Date | 2026-11-08 |

## Digest verification

Recomputed digest:

`55d4c9e6e7cc05b27b794e24d71ef1c7ac02a6c4d3026ea838a3deeb007e0103`

Matches ticket plan digest.

## Defect confirmation at review SHA

In `intergrax/runtime/nexus/tools/invoker.py`, branch `dep_boundary is None` (`~1905–1942`): `ext_op.mark_running()` then `future = self._execution_pool.submit(...)` is not wrapped in `try/except BaseException` with `ext_op.mark_failed()` before re-raise. The boundary branch (`~1973–1979`) already terminalizes submit failures. Matches H01-F015 / plan scope C3.

## Allowed files

Exactly two implementation paths:

- `intergrax/runtime/nexus/tools/invoker.py`
- `tests/unit/runtime/nexus/tools/test_tool_life_x_c0_no_boundary_submit_failure_terminal.py` (NEW)

Hard dependency on T-IGX-030-A test helpers documented; at `fe0937517`, 030-A is merged on `development`.

## Gate weakening

No swallowed `Cancelled`/`BaseException` without re-raise (only submit guard). N1 pins `bind_future` residual unfixed. No invoker policy test weakening.

## ACT-N11

Single product file change; expected ACT-N11 red; not relaxed.

## File-disjoint overlap

Disjoint from 030-B (error code / recovery), 032-A (skills wiring), 033-A (CLI). `invoker.py` imports `RuntimeErrorCode` via alias; commit order either way is acceptable per plan §7.

## Adopted master-plan decisions

DR-7 / Q-A1 mechanical restore (C3+C4). Not part of 021-B/C stage.

## Verdict

**PASS**
