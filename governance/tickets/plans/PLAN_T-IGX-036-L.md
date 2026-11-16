# PLAN T-IGX-036-L — PDC-P5 batch B-D7: D7 Background & Scheduling

Parent: PDC-X implementation WBS (`PDC_IMPLEMENTATION_WBS.md`, batch **B-D7**: recurrence on SchedulerLedger, dead-letter state, identity on enqueue, two-process fencing). Decision record: ADR-PDC-07 (Accepted). Requirements: PDC-D7-R006, R007, R008, R009, R012, R013, R016.

This batch changes product code. It is implemented in an isolated worktree and **does not merge before MCP-SERVER-X P8 is published** (OD-PDC-SEQ-01). A merge invalidates the IIAC activation (ACT-N11), so PDC-P7-00 re-activates it.

## Deliverables

1. **Recurrence (R006-R008, OD-PDC-SCOPE-05 = A).** A `RecurrenceRule` (interval or cron, IANA time zone, misfire policy `skip | fire_once | catch_up(n)`) materializes the next one-shot `ScheduledResume` on the existing `SchedulerLedger` after each occurrence. Storage stays in UTC.
2. **Dead letter (R009, R013).** `DEAD_LETTERED` becomes a terminal task state in the queue contract, with a typed reason and attempt count, reached when the retry budget is exhausted. Providers may map it to a native DLQ.
3. **Identity on enqueue (R016).** `TaskRequest` carries canonical execution identity (attempt/execution id, principal, trace correlation).
4. **Fencing (R012).** A two-process test over SQLite proves no duplicate fire across schedulers.

Design constraints: No second scheduler and no execution engine change (UER is FROZEN; extend through the boundary only). Each occurrence is admitted through canonical execution like any scheduled resume. A provider scheduler is not adopted as platform authority. Registry convergence (R022) is not in this batch.

## Write set

`intergrax/queueing/`, `intergrax/background_tasks/`, `intergrax/runtime/long_running/`, `intergrax/runtime/background_execution/`, `tests/qualification/pdc_x/d7/`

Nothing outside this set changes. Any need to write elsewhere stops the batch and goes to the orchestrator, who is the single integration owner for the shared hotspots (`intergrax/applications/_shared/`, `intergrax/contracts/`).

## Protocol

1. Reuse the canonical mechanism named in ADR-PDC-07.
2. Write the tests first, under `tests/qualification/pdc_x/d7/`.
3. One `uv run pytest` per verification step; ruff and pyright on touched files.
4. Commit with `Ref: governance/tickets/T-IGX-036-L.md`, push to a branch, open a PR against `development`, and **do not merge**.
5. Record the implementation SHA; the publication SHA is recorded after the merge.
6. The implementer never certifies the batch and never writes the `Approved:` line.

## Acceptance

- Every requirement row of the batch has a passing test at the implementation SHA.
- Layer boundaries hold (`intergrax/` does not import `agents/` or `applications/`).
- No new registry, scheduler or authority beside the existing owner.
- Existing suites for the touched packages stay green (no weakened gate).
- Test log under `.tmp/proof/pdc-x-p5/B-D7/`.

## Test contract

```text
uv run pytest tests/qualification/pdc_x/d7 -q
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

The implementer adds one more bounded `uv run pytest` over the existing unit suites of the touched packages and records it in the checkpoint.

## Out of scope

Other batches; the `ollama` regression (B-REG-P); FRZ changes; OD signatures; merges; IIAC activation.
