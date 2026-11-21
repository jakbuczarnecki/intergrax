# EQO-PKG-A4-006 — implementation evidence (WAVE-01)

| Field | Value |
| --- | --- |
| Package | `EQO-PKG-A4-006` |
| Root cause | `EQOA-A4-006` / `EDER-F-0115` |
| Base SHA | `a0598ba3e4f4f903eb4da3db8ea012d5cbf71225` (`origin/development`) |
| Branch | `eqo/w1-a4-006` |
| Worktree | `D:\Projekty\_wt\intergrax\eqo-w1-a4006` |
| Ticket | `governance/tickets/T-IGX-055-A.md` |

## Change summary

- `EMITTER_FILES`: removed stale `intergrax/agents/uaep.py`; added `intergrax/runtime/nexus/uaep/uaep_executor.py`; retained `intergrax/runtime/middleware/trace_middleware.py`.
- Missing configured emitter path still returns exit code 1 (fail closed).
- Unit test `test_emitter_files_includes_uaep_executor` pins `EMITTER_FILES` to `intergrax/runtime/nexus/uaep/uaep_executor.py` (not a swappable peer path).
- `RuntimeEvent(` without `tenant_id=` within the 600-character window still returns exit code 1.

## Scoped limitation

This package does **not** survey or gate other `RuntimeEvent(` constructors anywhere else in the repository; only the two configured emitter files remain in scope.

## Verification

- Pytest (2026-11-21): `uv run pytest tests/unit/scripts/test_check_runtime_event_tenant_propagation.py -q --basetemp=D:/Projekty/_wt/intergrax/_scratch/eqo-w1-a4006/pytest` — **3 passed**, 0 failed.
- Gate CLI: `python scripts/maintenance/check_runtime_event_tenant_propagation.py` — exit 0 after emitter rebind.
