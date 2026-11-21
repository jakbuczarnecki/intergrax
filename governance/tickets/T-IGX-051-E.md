# T-IGX-051-E — RW train node T-AF2-RW-RECOVERY: background and recovery (EX-L-06, EX-D7-01, EX-D4-01, EX-L-03, EX-D1-07, EX-D7-03, PDC-D7-R015, G-20)

## Scope

Product-path ticket of the RW product train (branch `arch-freeze/rw-train`; RW-PLAN v13 table T12, lane 3; plan of record section E row "REMEDIATION — background / recovery" and section G). Drafted for the ARCH-FREEZE wave packet; facts read at `origin/development` `2727a2c2364a97492202b931375772a400279a9b` (product tree equal to AUDIT_BASELINE_2 `7960ae7f6`) and compared with the train head `05699b633` (no difference on any write-set path). The binding scope is RW-PLAN v13 appendix A1 block `T-AF2-RW-RECOVERY (T-IGX-051-E)`; the plan's write set equals it (differences: none, see `INPUT_T-IGX-051-E.md`).

- Part A (firm): `DocumentStoreTaskWorker.stop()` keeps a still-running consumer referenced and `start()` refuses while it lives, so a later `start()` can no longer run two consumers (EX-L-06).
- Part B (firm): `recover_interrupted_running` becomes a per-row compare-and-set on a conditional store, so a row finished by a concurrent writer is not overwritten (EX-L-03). A worker lease or heartbeat is not part of this ticket.
- Part C (firm): a product active-work controller over `WorkerRuntime` and the document-store worker (new `intergrax/background_tasks/active_work.py`), an intake gate and in-flight count on `WorkerRuntime`, a runner seam that hands the controller to every hosted engine, and an optional field on the queue worker wiring (EX-D7-01, EX-D4-01). The engine and the shutdown executor already call the controller; their edit is not expected to be needed.
- Part D (firm): the durable projection-input compare covers `revision_state`, `activated_at` and the bindings as they are, instead of neutralizing them (EX-D1-07). If the full compare turns a pinned existing test red, the implementer stops and reports; the compare is not weakened again.
- Part E (firm): a new `intergrax/runtime/long_running/uncertain_reconciler.py` that defines the schedule-source port and reconciles UNCERTAIN schedules exactly once with a recorded double-execution check; the scheduler runs it as an isolated tick phase (PDC-D7-R015).
- Part F (firm): a bounded, process-local release count per schedule in `scheduler.py`; after the cap the claim is skipped with a stable error code instead of being released every tick (G-20). `OverflowError` mapping is already present and is pinned by a control test.
- Part G (firm): the two A1 producer sites (`retention_sweep.py`, `worker_handler.py`) carry `TaskRequest.identity` (EX-D7-03, DS-EXC option "durable producers set an identity").

Write set (exact lists in the plan, "Write set"; equals `RW_PLAN.md` appendix A1 block `T-AF2-RW-RECOVERY`): 12 firm product paths (the glob `intergrax/background_tasks/*` with one new file under it, one further new product module, 10 edited paths; `engine.py` and `shutdown.py` may end unedited), 5 firm new test paths, 0 conditional paths, no docs. Five new test files hold 8 negative-test rows (NT-RW-REC-01..08 = RW-PLAN NT-ROW-109..116) in 9 test functions. No edit of an existing test, of `intergrax/contracts/**`, of the schedule stores (`store.py`, `postgresql_task_checkpoint_store.py`, `scheduled_resume.py`), of `registry_projection_descriptor.py`, of gate scripts, `.github/`, `pyproject.toml`, `uv.lock`, IIAC or PDC-X records. Product commits go only to branch `arch-freeze/rw-train`.

Dependencies and overlaps (plan section 12): lane 3, after PT-WA (merged into the train) and PT-WB; **serialized after PT-FC2 on `intergrax/applications/_shared/queue_worker_wiring.py`** (hard precedence "PT-FC2 -> T-AF2-RW-RECOVERY"; the only shared path); the `queue_worker_wiring.py` edit must not start before PT-FC2 has merged into the train. ACT-N11: every product path above breaks the IIAC activation until the single train-level re-pin; this ticket does not perform it and must not merge outside the wave. This ticket edits nothing under `intergrax/contracts/`; the wave-packet contract typing check (`rw-plan/work/typ05/measure_typ05.py` plus `typ_measure.py`) is expected to report nothing new (OD-AF-11).

Open questions that can change the write set (plan section 14; draft defaults keep A1 unchanged): Q-RC-01 (schedule stores need a listing/resolve port for the reconciler to reach real UNCERTAIN rows; `postgresql_task_checkpoint_store.py` is also a PT-WB conditional path), Q-RC-02 (full durable compare versus the production flow; descriptor decode), Q-RC-03 (host composition of the controller), Q-RC-04 (EX-D7-03 names every durable enqueue site; A1 holds 2 of the 5 open sites). Until these are ruled, PDC-D7-R015, EX-D1-07 and EX-D7-03 have explicit residuals and are not complete by this ticket.

Prerequisites before implementation starts (not assumed): the wave packet approved by the Operator and the approval block of this ticket written by the signing delegate (OD-AF-22), PT-FC2 merged into the train for the queue wiring edit, the product-path authorization of the train, the rulings on the questions above.

No PASS, closure, FRZ promotion, stage certification or independent verification is claimed anywhere in this ticket or its plan. Register labels quoted in the plan are quoted with attribution. Closure of each ID is an independent acceptance at the freeze SHA.

## Plan digest

Plan digest: 40040bf51bd46e10f0e5c89b7d235eda4fb516a2c1796ad17ec9bbd81820dc4c

Canonical plan: `governance/tickets/plans/PLAN_T-IGX-051-E.md` (drafted at `.tmp/session/arch-freeze/w3-drafts/rw-recovery/PLAN_T-IGX-051-E.md`; the plan is copied unchanged into `governance/tickets/plans/` when the ticket is committed). The digest was computed with `scripts/governance/plan_digest.py` (copy extracted from `origin/development`, run with `python -I`) on the draft; it must be recomputed after any edit of the plan.

## Approved

Approved: operator-delegate @ 2026-11-17T15:17:24+01:00
Delegation: docs/project/maintainers/qualification/arch_freeze_program/RW_TRAIN_DELEGATION_RECORD_4.md@5ba8c4928a06345d6885a59cc2519a793de73caf
Delegate session: local_64ab068c-841e-4021-9874-5d5ba65f95ff

## Test contract

Nothing was run when this ticket was drafted. One process at a time, one `uv run pytest` per verification step with multiple paths, under the lock `D:/Projekty/_wt/_locks/pytest.lock` (the creator removes it); exact scopes in plan section 10:

- Red-first (S0): the five new test files on a detached worktree at the pre-change train commit. Expected: all 8 negative-test rows red, the controls (the different-slug leg of NT-RW-REC-05 and the `OverflowError` control of NT-RW-REC-07) green. The log is kept for the auditor.
- After the change (S2): all new tests green.
- Regression (S1): the RW-PLAN section (e) row for this ticket: `tests/unit/queueing`, `tests/unit/hosting`, `tests/unit/background_tasks`, `tests/unit/runtime/long_running`, `tests/qualification/pdc_x/xd`. The plan also proposes nine further read-only regression nodes (recovery, enqueue identity, scheduler, durable store, the five LKW worker test files); they need orchestrator acceptance (Q-RC-07). No pass is claimed for any of them.
- Static (S3): `uv run ruff check` and `uv run pyright` on the changed Python files. Pre-push: `uv run python -m scripts.governance.igx_gov ticket`.
- No broad or diagnostic suite, no Docker, no vendor integration, no production configuration change (C10/C15 freeze). The independent audit is a separate session at the exact wave head.

Commits touching `intergrax/`, `agents/`, `applications/` carry the trailer `Ref: governance/tickets/T-IGX-051-E.md`; no AI-tool attribution anywhere.

## Out of scope

- A worker lease or heartbeat on queue rows (the register's remedy for crash versus write in flight), a durable release counter, parking a schedule in `UNCERTAIN` after the cap, any listing or resolve API on the SQLite or PostgreSQL schedule stores.
- Wiring the controller into the document-store worker compositions (`intergrax/runtime/vendor_knowledge/sync_runtime.py`, `applications/local_workspace_application/workspaces/sync_runtime.py`); re-pointing XD-05 (`tests/qualification/pdc_x/xd/test_xd_05_host_background.py`) at the product class.
- The three `TaskRequest` constructor sites outside A1 (`intergrax/runtime/task/queued_host_task_execution_adapter.py`, `intergrax/runtime/vendor_knowledge/sync_jobs.py`, `intergrax/tools/providers/message_bus/service.py`).
- EX-D4-02 (in-process supervision only, OS child watch `NOT_CLAIMED`), EX-D7-02 (retry policy at the composition sites; `ACCEPTABLE_SCOPED_WITH_EVIDENCE`, T-AF2-RW-GATES), EX-L-10 beyond its G-20 component (T-AF2-RW-GATES), `PDC-D7-R022` and the D7 documentation fix.
- The IIAC re-pin, the EBH-7 pin, FRZ status promotion, stage closure, GitHub settings and rulesets, validator code, `scripts/`, `.github/`, `pyproject.toml`, `uv.lock`, the roadmap, the FRZ checklist, PDC-X and IIAC record directories, Docker or vendor work.

## Lifecycle state

plan_approved

## Implementer session

Implementer session: local_eaf9ef72-d1c1-44d0-acbc-fac05f02edf7
