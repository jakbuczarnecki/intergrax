# PLAN T-IGX-041-G — PDC-X P7 repair of domain D7 (Background and Scheduling): IA-D7 findings F-D7-01, 02, 04, 06, 07, 08, 10, 11, 12, 13

Parent: PDC-X campaign, repair loop 1 after the P7 independent audits. Origin: `PDC_P7_D7_AUDIT_REPORT.md` (IA-D7, audited SHA `c08efe46a74a20595468d8d8bf06eab2dca5f573`, verdict PARTIAL, 3 HIGH, 10 MEDIUM, 7 LOW, 3 INFO; row PDC-D7-R014 FAIL). Authority: Operator option B in `DELEGATION-PDC-P7-REPAIR.md` (repair real defects and test hygiene; "no production caller" and "missing feature versus claim" findings are declared exceptions, not repairs). Batch row: `B-P7-D7` in `PDC_IMPLEMENTATION_WBS.md`. This plan is written by the planner lane and is **unsigned** (`plan_pending`). It is not a certification verdict and certifies nothing. Implementer and signer are different sessions; the implementer never writes `Approved:` or `Delegation:` lines.

Base: `origin/development` at `1c864ea19` (the D7 product and test paths are byte-identical to the audited SHA per the audit method, step 2). Every decision below was checked by reading the owner code at this base; the audit probes were **not** re-run by the planner. The implementer's red tests are the reproduction.

## Problem

D7 delivers recurrence, a retry budget, dead-lettering and a durable queue that pass 52 d7 tests, `bg_01` (22), `sched_01` (14) and XD-05/06, yet none of the failure paths is exercised, and several are broken:

1. `LongRunningScheduler._process_due_schedules` (`scheduler.py:170-176`) has `try/finally` and no `except`: one raising resume (or a `StaleClaimError` from `complete_claim`) escapes the loop, the sibling claims of the batch stay RUNNING, become UNCERTAIN at the next `claim_due`, and are never re-claimed. The batch is not tenant-scoped, so one tenant's failing task strands other tenants' due schedules.
2. The recurrence successor is materialized after the occurrence finished (`scheduler.py:176`, `_materialize_next_occurrence` swallowing every exception at `:217-223`). A crash or lease expiry while the occurrence runs, a crash between `complete_claim` and materialization, or one store error ends the chain silently; the `finally` also keeps minting rows for dead tasks.
3. The occurrence id `rec_<rule_id>_<stamp>` (`recurrence.py:382-385`) is the global primary key without a tenant component, and a conflict is swallowed (`:477-478`): another tenant's rule can pre-empt a chain.
4. `ScheduledResume.run_at_utc` is an unvalidated string compared lexicographically in the stores (`scheduled_resume.py`, `store.py` `claim_due`/`list_due`), and `_ensure_utc` (`scheduler.py:389-392`) does not convert a non-UTC aware `now`.
5. `recurrence_rule` / `recurrence_occurrence_utc` are not in the forbidden-metadata set (`scheduled_resume_metadata.py`), so `LongRunningScheduler.schedule_resume` can start an unvalidated chain; `RecurrenceRule` has no interval or catch-up bound and every fired occurrence re-scans the missed range.
6. Occurrences whose target checkpoint is no longer resumable are completed without executing and recorded `completed` (`scheduler.py:186-198`) while the chain keeps minting rows.
7. `DocumentStoreTaskQueue` (`document_store_task_queue.py`): terminal states are not sticky (no status check in `mark_*`, `retry_or_dead_letter`, `_settle_against_budget`), the exclusion is a per-instance `threading.RLock` (`:54`) although the store contract offers compare-and-set, and `claim_pending` inspects only the first `max(4*limit, 32)` index rows (`:206`) so back-off rows hide fresh work.
8. The two BACKGROUND_TASKS documents still describe the pre-B-D7 state.

## In-scope findings (IN) — verified by code read at the base

| Finding | Sev | Evidence at base | Deliverable group |
| --- | --- | --- | --- |
| F-D7-01 batch abort | HIGH | `scheduler.py:170-176` (no `except`), `_execute_resume` raises at `:341`, `complete_claim` raises `StaleClaimError` at `:343` | S1 |
| F-D7-02 chain ends silently | HIGH | `scheduler.py:176`, `:217-223`; `store.py` `claim_due` converts expired RUNNING to UNCERTAIN, nothing re-materializes | S2 |
| F-D7-04 occurrence id without tenant | MED | `recurrence.py:382-385`, `:462-478` | S3 |
| F-D7-10 `run_at_utc` not validated | MED | `ScheduledResume` has no validator; `scheduler.py:126`, `:389-392`; `store.py` and PG `run_at_utc <= ?` | S4 |
| F-D7-11 unbounded recurrence, metadata smuggling | MED | `scheduled_resume_metadata.py` key set; `recurrence.py:74`, `:252-275`, `:388-411` | S5 |
| F-D7-13 silent no-op occurrences | MED | `scheduler.py:186-198`; target is one `(task_id, resume_token)` | S2 (defined together with the chain end) |
| F-D7-06 terminal states not sticky | MED | `document_store_task_queue.py:244-357` | Q1 |
| F-D7-08 back-off rows block fresh work | MED | `:206`, `:220-221` | Q2 |
| F-D7-07 per-instance lock | MED | `:54`, `:61`, `:204`; `ConditionalDocumentStore` (`put_if_absent`, `replace_if_match`, `delete_if_match`) is implemented by the in-memory, MongoDB, PostgreSQL and Cassandra adapters, so a bounded store-level mechanism exists | Q3 |
| F-D7-12 stale documents | MED | B.2.1, B.2.3, ADOPTION GAP in the architecture doc; A.1 row in the plan doc | D1 (last) |

Closed as by-products inside files already in the write set (no separate deliverable, each needs a test): F-D7-18 (reject an unsatisfiable cron and cap `interval_seconds` at rule validation, with S5) and F-D7-21 (docstring wording "wall time plus gap length", with S5). F-D7-22 (weak negative tests) is closed by the tests of this plan.

## Excluded findings (declared exceptions for the P9 exception register)

| Id | Sev | Reason | Reopen condition |
| --- | --- | --- | --- |
| F-D7-03 graceful worker shutdown / host drain | HIGH | Missing feature versus claim: no class implements `HostedApplicationActiveWorkController`; `WorkerRuntime` has no stop or drain API. Named by the delegation as not repaired in PDC-X. Row PDC-D7-R014 stays FAIL in the audit record and is rated PASS_SCOPED only by the Operator's option-B authorization | Operator commissions the worker-side drain feature (a controller over `WorkerRuntime` and `DocumentStoreTaskWorker` plus re-pointing XD-05 at the product class) |
| F-D7-05 recurrence, retry budget, DLQ not reachable from a production composition | MED | "Exists but no production caller or composition" (F-04 family); `wiring.py` has no recurrence parameter, the three `DocumentStoreTaskWorker` sites pass no `retry_policy` | A host composition that enables recurrence or a retry policy is commissioned (then the S and Q groups here become reachable and need a host-level test) |
| F-D7-09 enqueue identity not adopted | MED | Wiring and adoption gap (no producer sets `identity=`, principal and trace unused); adding producers is new composition, not a defect fix | Producers adopt `TaskRequest.identity` at the real enqueue sites (Operator decision) |
| F-D7-14, 15, 16, 17, 19, 20 | LOW | LOW goes to the debt register of the P9 package unless closed as a by-product. F-D7-20 (`stop()` abandons a running thread) belongs with F-D7-03; F-D7-19 (tenant-scoped `cancel`) is a port signature change used by fakes of cited suites | Next D7 batch or the debt register owner |
| F-D7-23 (INFO) | INFO | No action | — |

F-D7-07 fallback: if the implementer proves that a compare-and-set claim cannot be made bounded and reviewable inside `document_store_task_queue.py`, the finding moves to the table above with the reopen condition "multi-instance claim fencing over a conditional store"; this narrowing needs a plan amendment signed like the plan (it is not an implementer decision).

## Deliverables (tests first; each fix lands only after its pinned test is shown red for the stated reason)

All new tests go to a new directory `tests/qualification/pdc_x/d7_repair/` (real SQLite store and a real `LongRunningScheduler`, an in-memory `ConditionalDocumentStore` for the queue). Every group has positive and negative cases; red logs are kept under `.tmp/proof/pdc-x-p7/041-G/`.

### S — scheduler and recurrence (`scheduler.py`, `recurrence.py`, `scheduled_resume.py`, `scheduled_resume_metadata.py`, `scheduler_claim.py`, `store.py`, `postgresql_task_checkpoint_store.py`)

- **S1 (F-D7-01).** Each claim of a batch is processed in isolation: an exception from one claim is logged with the `schedule_id` and does not stop the loop; `StaleClaimError` at completion is caught separately (the lease was lost, another owner decides the outcome; no successor action). `CancelledError` still propagates. The same isolation applies per checkpoint in `_process_expired_human_timeouts` (same defect class, same file). Tests: three due claims where the first resume raises, the sibling resumes still run and end `completed` in the same tick; a claim of another tenant in the batch is unaffected; `StaleClaimError` path; human-timeout loop with a raising executor. Negative: a failing executor leaves its own claim non-completed (at-most-once is kept, never silently completed).
- **S2 (F-D7-02, F-D7-13).** New order inside `_process_claimed_schedule` for a recurring entry: (1) eligibility checks (checkpoint present, materialization-eligible, resumable) — read-only; (2) if not eligible, the occurrence is recorded `SKIPPED` (new additive `ScheduledResumeStatus.SKIPPED`), **no successor is created**, and the chain end is logged at WARNING with the reason; (3) if eligible, the successor is materialized **before** the resume executes (strict: errors propagate); (4) execute; (5) `complete_claim`. Consequences: a crash or lease expiry while the occurrence runs leaves the successor already pending; the chain no longer depends on a post-execution step; dead tasks stop minting rows. If step 3 fails with a store error, the claim is returned with the new port method `release_claim` (status PENDING, fence kept, owner cleared) so the occurrence is retried at the next tick, and the failure is logged at ERROR; an invalid persisted rule (`RecurrenceError`) ends the occurrence `SKIPPED` with an ERROR log (fail closed, nothing executed). Port additions on `ScheduledResumePersistence` are additive with defaults: `skip_claim(claim)` defaults to `complete_claim(claim)` and `release_claim(claim)` defaults to `NotImplementedError`, which the scheduler treats as "leave the claim to its lease" (logged). The `SQLiteTaskCheckpointStore` implements both fence-validated; the PostgreSQL store receives the same two methods and the same SQL shape (parity only, not executable here; covered by a structural test, C10/C15 freeze unchanged). Fakes of the cited suites (`sched_01`, `state_x`) stay byte-identical and keep working through the defaults.
  - **Target semantics (F-D7-13), declared:** a recurrence rule targets exactly one `(task_id, resume_token)`. Once that checkpoint is superseded, terminal, missing or not resumable, the occurrence is `SKIPPED` and the chain ends explicitly. Retargeting a rule to the newest checkpoint is a new feature and is not built here. One-shot (non-recurring) entries keep today's behaviour (`completed`).
  - Tests: crash while an occurrence runs (claim by a dying owner, survivor ticks later) leaves a pending successor; the **two-process** variant reuses the helper pattern of `test_d7_fencing_two_process.py` without editing it; `schedule()` failing once during materialization releases the claim and the next tick succeeds without a duplicate successor; a re-paused target ends the chain with exactly one `skipped` row and no pending row; dead task stops materialization; invalid persisted rule. Overlap note to pin: with the successor created before execution, a second scheduler process may claim occurrence N+1 while N still runs; the test records the observed behaviour (host execution admission for the deterministic checkpoint run/attempt identity) and the implementer documents it in the architecture doc; if the test shows a double execution, the implementer stops and reports (a guard needs a plan amendment).
- **S3 (F-D7-04).** `occurrence_schedule_id` gets a required keyword-only `tenant_id` and yields `rec_<tenant digest>_<task digest>_<rule_id>_<stamp>` style ids (tenant and task qualified by a fixed-length hash, `rule_id` unchanged, deterministic, no tenant or task text in the id). `RecurrenceMaterializer.start()` keeps its return type; a new additive `start_checked()` returns a typed result (`STARTED`, `ALREADY_EXISTS`, `NO_OCCURRENCE`) so a conflict is distinguishable from "no occurrence". Tests: two tenants with the same `rule_id` and start both get their chain (old behaviour: second `start()` returned `None`); same tenant restart is `ALREADY_EXISTS`; pinned literal id for a fixed input; ids contain neither tenant nor task. The two existing d7 test files that call `occurrence_schedule_id(rule.rule_id, …)` are updated to pass `tenant_id` (identity-format change only, assertions stay as strict; see Write set). Compatibility note: pre-change persisted `rec_<rule>_<stamp>` rows would not be recognised as the same occurrence; recurrence is not composed in any production host (F-D7-05), so no production rows exist; stated in the architecture doc.
- **S4 (F-D7-10).** `ScheduledResume.run_at_utc` is validated and normalized at construction: aware ISO-8601 required (naive, empty and non-ISO are rejected), converted to UTC with the existing `isoformat()` shape; `tick()` converts an aware non-UTC `now` with `astimezone(UTC)`; `claim_due` and `list_due` of both stores normalize `before_utc_iso` through one helper defined next to the model and reject non-ISO input. A malformed **legacy** row must not block a batch: when the store reads a row that fails validation it moves that row to `UNCERTAIN` with a log and continues. Tests: `2026-12-01T05:00:00-05:00` (10:00Z) does not fire at 08:30Z, `2026-12-01T10:00:00+02:00` (08:00Z) fires; `''`, `not-a-time` and naive values rejected; non-UTC `now`; legacy malformed row does not poison `claim_due`.
- **S5 (F-D7-11, by-products F-D7-18, F-D7-21).** `LongRunningScheduler.schedule_resume` rejects `recurrence_rule` and `recurrence_occurrence_utc` in `resume_metadata`; the store-level persistence acceptance accepts those keys only on a self-consistent occurrence row (embedded rule validates, tenant, task and token equal the row, `schedule_id` equals the canonical id, `run_at_utc` equals the occurrence instant), so a hand-built row cannot mint a chain and the materializer still can. `RecurrenceRule` gets named bounds: `MIN_INTERVAL_SECONDS`, `MAX_INTERVAL_SECONDS` (rejects the `OverflowError` case) and `MAX_CATCH_UP`; a cron that has no occurrence from `start_utc` is rejected at validation; `plan_next_occurrences` scans the missed range once per materialization and the per-tick work is bounded by `MAX_MISFIRE_SCAN` and `MAX_CATCH_UP` (measured by iteration count, not wall clock). The module docstring is reworded to "wall time plus gap length". The numeric values of the three constants are chosen by the implementer so that all existing d7 tests stay valid and are recorded in the PR description; no existing test is edited for this. Tests: smuggling through `schedule_resume` rejected, hand-built inconsistent row rejected at persistence, valid materializer rows still persist, bounds rejected at the boundary values, `0 0 30 2 *` rejected, `10**15` seconds rejected, catch-up drain iteration count bounded.

### Q — `DocumentStoreTaskQueue` (`document_store_task_queue.py`)

- **Q1 (F-D7-06).** No transition leaves a terminal status (`SUCCEEDED`, `FAILED`, `DEAD_LETTERED`): `mark_succeeded`, `mark_failed`, `mark_dead_lettered`, `retry_or_dead_letter` (returns the existing terminal status) and the claim path are no-ops on a terminal row, with a WARNING log; non-terminal transitions are unchanged. Tests: 4 transitions × 3 terminal statuses (12 negative cases) plus the stale `dead_letter` field assertion; positive cases for RUNNING → each terminal.
- **Q2 (F-D7-08).** `claim_pending` must find ready work behind any number of backed-off rows. Mechanism chosen by the implementer within these constraints: bounded work and memory per call, no unbounded scan of a tenant, rows already in the old index at upgrade still drain (fallback), and the mechanism is explained in the PR (for example a delayed index keyed by availability with promotion at claim time, or cursor paging with an explicit scan bound). Tests: 40 and 200 tasks failed with a one-hour back-off, then a fresh task: `claim_pending(limit=4)` returns it; back-off rows are not claimed early; tenant filter intact; legacy-index fallback.
- **Q3 (F-D7-07).** When `document_store` is a `ConditionalDocumentStore`, claim (PENDING → RUNNING) and the terminal transitions use `replace_if_match` with a bounded retry, and idempotent enqueue uses `put_if_absent` on the idempotency row with orphan repair (a reservation whose task row is missing is repaired, never returned as a live handle). With a plain `DocumentStore` the behaviour stays the per-instance lock and is documented as single-instance. Tests: two queue instances over one in-memory store (with a write delay to widen the window) claim 40 tasks with no task claimed twice; concurrent idempotent enqueue yields one task; plain-store path unchanged; crash between reservation and task write.

### D — documentation (last, after all code deliverables are green)

- **D1 (F-D7-12).** `docs/project/architecture/BACKGROUND_TASKS.md` (B.2.1, B.2.3, the ADOPTION GAP list, and a limits statement) and `docs/project/maintainers/plans/BACKGROUND_TASKS.md` (A.1 table) are rewritten to the **final** state: recurrence, retry budget, dead-letter and enqueue identity delivered as library capabilities **not composed in any host** (F-D7-05, F-D7-09); the multi-process fencing proof exists (SQLite scope; PostgreSQL excluded by C10/C15); the durable queue limit (conditional store required for multi-instance safety, F-D7-07 outcome); the recurrence target semantics and chain-end rule (S2); the overlap note; the declared exceptions F-D7-03, 05, 09 with reopen conditions. No sentence may overstate, and none may describe a limit that was closed.

## Write set (exact; nothing else changes)

Product (`intergrax/` only, no hotspot: nothing under `intergrax/contracts/` or `intergrax/applications/_shared/`):

- `intergrax/runtime/long_running/scheduler.py`
- `intergrax/runtime/long_running/recurrence.py`
- `intergrax/runtime/long_running/scheduled_resume.py`
- `intergrax/runtime/long_running/scheduled_resume_metadata.py`
- `intergrax/runtime/long_running/scheduler_claim.py`
- `intergrax/runtime/long_running/store.py`
- `intergrax/runtime/long_running/postgresql_task_checkpoint_store.py`
- `intergrax/queueing/providers/document_store/document_store_task_queue.py`

Tests:

- `tests/qualification/pdc_x/d7_repair/` (new directory, new files only)
- `tests/qualification/pdc_x/d7/test_d7_recurrence.py` and `tests/qualification/pdc_x/d7/test_d7_fencing_two_process.py` (and `_fencing_worker.py` only if it calls `occurrence_schedule_id`): only the lines that follow the new `occurrence_schedule_id(..., tenant_id=...)` signature, plus any assertion that encodes the defect "the chain keeps minting rows for a dead task" (to be named in the PR; never relaxed, only re-pointed to the declared behaviour of S2)

Documentation:

- `docs/project/architecture/BACKGROUND_TASKS.md`
- `docs/project/maintainers/plans/BACKGROUND_TASKS.md`

Governance documents of this batch: `governance/tickets/T-IGX-041-G.md`, this plan, `PDC_IMPLEMENTATION_WBS.md`, `PDC_DEPENDENCY_DAG.md` and the batch table in `scripts/qualification/pdc_x_wbs.py` (row `B-P7-D7`, generated; unrelated rows are not regenerated).

Not edited: `scripts/governance/`, `.github/`, `intergrax/contracts/`, `intergrax/applications/_shared/`, the colocated worker, `worker_runtime.py`, `wiring.py`, `task_queue.py`, the Celery providers, `intergrax/hosting/`, the TENANT-X gate and its pin, the IIAC activation record, the EBH-7 pins and allowlist, the cited `bg_01`, `sched_01`, `state_x`, XD-05 and XD-06 files, and the fakes inside them.

## Tenant carriers (for T-IGX-041-I)

New or changed `tenant_id` parameters and fields in `intergrax/` add carriers to the TENANT-X current-head inventory: the new `tenant_id` argument of `occurrence_schedule_id`, any `tenant_id` on the new `start_checked()` result type, and none on `release_claim`/`skip_claim` (they take the claim). The implementer reads the carrier count before and after with the inventory helper of `tests/qualification/tenant_x/` (a throwaway script under `.tmp/session/041-G/`; the heavy TENANT-X gate test itself is not part of this ticket's contract) and records the exact delta in the PR description so T-IGX-041-I can account for it. This ticket does not touch the pin.

## Size and proposed split

One ticket, one write set. The S group (about 8 files plus 6 to 8 test files) and the Q group (one file plus 3 test files) are independent in code; the documentation depends on both. If the single pull request is not reviewable, the proposed split is **HIGH-first in two sequential PRs under the same ticket id**: PR-1 = S1, S2, S3, S4, S5 (both HIGH findings and the scheduler/recurrence MEDIUMs); PR-2 = Q1, Q2, Q3 and D1 (documents last, reflecting the merged state of PR-1). Both carry exactly one `Ref: governance/tickets/T-IGX-041-G.md` trailer per commit. The split does not change the write set or the signature.

## Protocol

1. Token-budget preflight (read scope, edit scope, tests) before each phase. Phase 2 starts only on the orchestrator's message after the delegate signature.
2. Tests first, shown red for the stated reason, then the fix, one group at a time; one `uv run pytest` at a time with narrow paths; output under `.tmp/proof/pdc-x-p7/041-G/` (this supersedes the generic `.tmp/proof/pdc-x-p5/B-P7-D7/` path in the generated WBS evidence line); scratch under `.tmp/session/041-G/`. No file-level `# ruff: noqa`; no skipped, xfailed or narrowed test.
3. `uv run ruff check`, `uv run ruff format --check` and `uv run pyright` on every touched Python file.
4. Commits: exactly one `Ref: governance/tickets/T-IGX-041-G.md` trailer, no AI attribution. Set `branch.<name>.remote`/upstream to `origin/development` before the first push and push with an explicit refspec; open a PR against `development`; never merge (the merge belongs to the orchestrator under the delegation's merge gate).
5. The implementer never certifies, never writes an `Approved:`/`Delegation:` line, never amends this plan; a write-set or scope change returns to the signer through the orchestrator.

## Acceptance

- Every IN finding has a pinned test that was red before its fix and is green after, including the negative cases named above; the exact defect repros of the audit (3-claim batch with a raising first resume, crash while running, two tenants with one rule id, non-UTC `run_at_utc`, smuggled recurrence, terminal overwrite, 32+ back-off rows, two queue instances) are among them.
- The cited D7 suites stay green with no assertion weakened; only the identity-format updates named in the Write set differ.
- No gate weakened or deleted; no registry, scheduler, engine, trust, policy or audit authority is added (the existing `LongRunningScheduler` and ledger are repaired in place; the new port methods are additive with defaults).
- `git diff --stat origin/development...HEAD` lists only the Write set; the two documents match the final code state, name every declared exception with its reopen condition, and contain no statement that the capabilities are composed in a host.
- The PR description lists: chosen constant values (S5), the Q2 mechanism, the overlap observation (S2), the TENANT-X carrier delta, and the exclusions.

## Test contract

```text
uv run pytest tests/qualification/pdc_x/d7_repair tests/qualification/pdc_x/d7 tests/qualification/bg_01 tests/qualification/sched_01 tests/qualification/pdc_x/xd/test_xd_05_host_background.py tests/qualification/pdc_x/xd/test_xd_06_background_security.py -q
uv run ruff check <touched python files>
uv run ruff format --check <touched python files>
uv run pyright <touched python files>
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

Consumers of the changed ports are run as a second, separate step only if the first command and their measured runtime allow it (`tests/qualification/state_x` for the schedule-store fake, `applications/local_workspace_application/tests/workspaces/test_connected_source_recovery.py` for the queue); a consumer that cannot be run is reported, not dropped silently.

## Known red classes this change does not hide

The change edits product paths, so the IIAC activation record (ACT-N11) stays stale until the P8-04 rebind and the EBH-7 executable-identity gate stays red until the campaign's final re-pin; both are known red classes of the delegation's merge gate. The TENANT-X carrier count is not re-pinned here (T-IGX-041-I). This plan is a repair plan and **not a certification verdict**: it claims no FRZ change, no closure and no production readiness, and a fresh IA-D7 re-audit at the repair SHA decides the domain verdict.

## Out of scope

- The excluded findings above (F-D7-03, 05, 09 and the LOW/INFO items) and every other domain.
- Graceful drain controllers, host composition of recurrence or retry policy, enqueue producers, retargeting of recurrence rules, Celery/Kafka/RabbitMQ identity and DLQ, PostgreSQL execution or Docker proof.
- `scripts/governance/`, `.github/`, `intergrax/contracts/`, `intergrax/applications/_shared/`, the TENANT-X pin and gate, the EBH-7 re-pin, the IIAC activation, OD signatures, certification.
