# T-IGX-041-G — PDC-X P7 repair of domain D7 (Background and Scheduling): IA-D7 findings F-D7-01, 02, 04, 06, 07, 08, 10, 11, 12, 13

## Scope

Product repair of existing D7 code, by Operator option B in `DELEGATION-PDC-P7-REPAIR.md` (2026-11-15), after the independent audit IA-D7 (`PDC_P7_D7_AUDIT_REPORT.md`, audited SHA `c08efe46a74a20595468d8d8bf06eab2dca5f573`, verdict PARTIAL, 3 HIGH and 10 MEDIUM findings, row PDC-D7-R014 FAIL). In scope: one failing resume no longer aborts its claimed batch (F-D7-01); the recurrence chain survives a crash, lease expiry and store errors, ends explicitly for a target that can no longer resume, and an occurrence is recorded `skipped` rather than `completed` (F-D7-02, F-D7-13); occurrence identity is tenant-qualified (F-D7-04); `run_at_utc` is validated and normalized to UTC (F-D7-10); recurrence is bounded and cannot be minted through plain `resume_metadata` (F-D7-11, with F-D7-18 and F-D7-21 as by-products); terminal queue states are sticky, claims use conditional writes where the store offers them, and back-off rows no longer hide fresh work (F-D7-06, 07, 08); the two BACKGROUND_TASKS documents are reconciled with the final state (F-D7-12, last). Declared exceptions, not repaired (reason and reopen condition in the plan): F-D7-03 (worker drain, missing feature versus claim), F-D7-05 (not composed in any host), F-D7-09 (enqueue identity not adopted), and the LOW/INFO items for the debt register. Tests are written first with negative cases. Hotspot paths (`intergrax/contracts/`, `intergrax/applications/_shared/`): none. Product code: yes. This ticket is not a certification verdict; the IIAC activation stays stale until the P8-04 rebind and the EBH-7 identity gate stays red until the final re-pin (known red classes); the TENANT-X pin is T-IGX-041-I. Batch: `B-P7-D7`. Plan: `PLAN_T-IGX-041-G.md`.

## Plan digest

Plan digest: 034f6e48f686921faf7a1db5943ec264cd952c318f15e000db2e8b730387b675

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-041-G.md`](plans/PLAN_T-IGX-041-G.md).

## Approved

Approved: operator-delegate @ 2026-11-15T07:56:30+01:00
Delegation: docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-P7-REPAIR.md@8f4b139c1f5352c9861872f0b5077996b8f0f891

Delegate checks (scope of `DELEGATION-PDC-P7-REPAIR.md`, all passed before this line was written; read at the batch branch, implementer commit `bacab00bb`; delegation file read from `origin/development`, last changed by `8f4b139c1`):

- The plan digest recomputed with `scripts/governance/plan_digest.py` equals the digest above (`034f6e48f686921faf7a1db5943ec264cd952c318f15e000db2e8b730387b675`).
- The write set of the plan (8 product files under `intergrax/runtime/long_running/` and `intergrax/queueing/providers/document_store/`, `tests/qualification/pdc_x/d7_repair/`, two existing d7 test files, 2 documents) equals the write set of row `B-P7-D7` in `PDC_IMPLEMENTATION_WBS.md` and the batch table in `pdc_x_wbs.py`, with one difference of wording: the row lists `tests/qualification/pdc_x/d7/_fencing_worker.py` unconditionally while the plan lists it "only if it calls `occurrence_schedule_id`". At this base it does not (grep: callers are only `test_d7_recurrence.py`, `test_d7_fencing_two_process.py` and frozen IIAC evidence scripts), so the row is a superset that adds no edit.
- Hotspots: no path under `intergrax/applications/_shared/` or `intergrax/contracts/` in the write set; they appear only under "Not edited". The implementer commit changes nothing under `intergrax/`, `scripts/governance/` or `.github/` (planning only).
- All 23 IA-D7 findings are accounted for: in scope F-D7-01, 02, 04, 06, 07, 08, 10, 11, 12, 13 (by-products F-D7-18, 21, 22); excluded F-D7-03 (missing feature versus claim, named by the delegation), F-D7-05 and F-D7-09 (no production caller or composition, adoption gap), F-D7-14, 15, 16, 17, 19, 20 (LOW, debt register), F-D7-23 (INFO); severities match the audit table; each exclusion has a reason and a reopen condition.
- No gate is weakened or deleted: the plan only adds tests (new directory), additive port methods with defaults and a new status value, and states that fakes of the cited suites stay byte-identical. Existing tests are touched only in the two listed d7 files (identity-format lines of the new `occurrence_schedule_id(..., tenant_id=...)` signature, plus assertions encoding the dead-task defect, "never relaxed, only re-pointed", to be named in the PR). Concern for the auditor, not a blocker: that second category is described by kind, not by assertion; the re-audit should diff those two files.
- No registry, scheduler, engine, trust, policy or audit authority is added (the existing `LongRunningScheduler` is repaired in place); nothing under `scripts/governance/` or `.github/`; the plan states it is not a certification verdict and certifies nothing.
- F-D7-07 fallback clause ("narrowing needs a plan amendment signed like the plan, not an implementer decision"): consistent with "write-set amendments are not delegated", because it keeps the decision away from the implementer and does not change the write set (same file). This signature does not pre-authorize the narrowing. F-D7-07 is a real defect, not one of the exclusion kinds the delegation names (no production caller, missing feature versus claim); an amendment that excludes it would be a scope change that returns to the Operator, and a write-set change would too.
- The implementer session (planner lane) and this delegate session are different sessions; the delegate edited nothing except this approval block and the lifecycle state. This signature is not a certification verdict.
- Merge gate (as the delegation requires): product pull requests of this ticket merge under authorizations 1 and 3 of `DELEGATION-PDC-STANDING.md` and decision 8 of `DELEGATION-PDC-WAVE2.md`: green CI, only the known red classes (IIAC activation until the P8-04 rebind, EBH-7 identity until the final re-pin), `headRefOid` equal to the verified SHA.

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

plan_approved
