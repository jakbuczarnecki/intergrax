# T-IGX-041-G — PDC-X P7 repair of domain D7 (Background and Scheduling): IA-D7 findings F-D7-01, 02, 04, 06, 07, 08, 10, 11, 12, 13

## Scope

Product repair of existing D7 code, by Operator option B in `DELEGATION-PDC-P7-REPAIR.md` (2026-11-15), after the independent audit IA-D7 (`PDC_P7_D7_AUDIT_REPORT.md`, audited SHA `c08efe46a74a20595468d8d8bf06eab2dca5f573`, verdict PARTIAL, 3 HIGH and 10 MEDIUM findings, row PDC-D7-R014 FAIL). In scope: one failing resume no longer aborts its claimed batch (F-D7-01); the recurrence chain survives a crash, lease expiry and store errors, ends explicitly for a target that can no longer resume, and an occurrence is recorded `skipped` rather than `completed` (F-D7-02, F-D7-13); occurrence identity is tenant-qualified (F-D7-04); `run_at_utc` is validated and normalized to UTC (F-D7-10); recurrence is bounded and cannot be minted through plain `resume_metadata` (F-D7-11, with F-D7-18 and F-D7-21 as by-products); terminal queue states are sticky, claims use conditional writes where the store offers them, and back-off rows no longer hide fresh work (F-D7-06, 07, 08); the two BACKGROUND_TASKS documents are reconciled with the final state (F-D7-12, last). Declared exceptions, not repaired (reason and reopen condition in the plan): F-D7-03 (worker drain, missing feature versus claim), F-D7-05 (not composed in any host), F-D7-09 (enqueue identity not adopted), and the LOW/INFO items for the debt register. Tests are written first with negative cases. Hotspot paths (`intergrax/contracts/`, `intergrax/applications/_shared/`): none. Product code: yes. This ticket is not a certification verdict; the IIAC activation stays stale until the P8-04 rebind and the EBH-7 identity gate stays red until the final re-pin (known red classes); the TENANT-X pin is T-IGX-041-I. Batch: `B-P7-D7`. Plan: `PLAN_T-IGX-041-G.md`.

## Plan digest

Plan digest: 034f6e48f686921faf7a1db5943ec264cd952c318f15e000db2e8b730387b675

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-041-G.md`](plans/PLAN_T-IGX-041-G.md).

## Approved

(pending; the plan is signed by a delegate session under `DELEGATION-PDC-P7-REPAIR.md`, which covers T-IGX-041-A..I; the implementer never writes this section)

## Test contract

```text
uv run pytest tests/qualification/pdc_x/d7_repair tests/qualification/pdc_x/d7 tests/qualification/bg_01 tests/qualification/sched_01 tests/qualification/pdc_x/xd/test_xd_05_host_background.py tests/qualification/pdc_x/xd/test_xd_06_background_security.py -q
uv run ruff check <touched python files>
uv run ruff format --check <touched python files>
uv run pyright <touched python files>
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

Consumers of the changed ports (`tests/qualification/state_x`, `applications/local_workspace_application/tests/workspaces/test_connected_source_recovery.py`) are run as a separate step and reported; see the plan.

## Out of scope

- F-D7-03, F-D7-05, F-D7-09 and every LOW/INFO finding (declared exceptions or debt register, see the plan); every other domain.
- `scripts/governance/`, `.github/`, `intergrax/contracts/`, `intergrax/applications/_shared/`, the colocated worker, `worker_runtime.py`, `wiring.py`, the TENANT-X gate and pin, the IIAC activation record, the EBH-7 pins, OD signatures, certification.

## Lifecycle state

plan_pending
