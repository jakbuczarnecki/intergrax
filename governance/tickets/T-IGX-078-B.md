# T-IGX-078-B - AF-CLOSE GA.2R RW plan repair

## Owner

AF-CLOSE, task GA.2R.

## Scope

Repair the ARCH-FREEZE RW plan so that RW-PLAN-AUDIT round 2 (GA.2) can run: fix the plan generator and the phase check 3 verifier, regenerate the plan from a reviewed bindings file, publish the cited historical plan versions as a binding record, re-measure red-first evidence for the rows named in the audit, and open ownership tickets T-IGX-078-C, T-IGX-078-D and T-IGX-078-E. The exact write set is in the canonical plan.

Classification: K3.3 tooling component, OP-A item 4. GA.1 D1 and audit finding H5 are routed to K3.3 in the AF-CLOSE triage notes; GA.2R executes that component (orchestrator GO GA.2R, choices C and D).

No gate is set by this ticket.

## Plan digest

Plan digest: c03c9b2f1275079495a7fd77ecbc08b6e8234721569a229aaa9b55617cecc2e9

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-078-B.md`](plans/PLAN_T-IGX-078-B.md).

## Test contract

Run one command at a time.

- `uv run pytest tests/unit/governance/test_build_rw_plan.py tests/unit/governance/test_verify_rw_plan_phase_check3.py tests/unit/qualification/test_run_campaign_b_w2_alpha.py -q`
- `uv run ruff check scripts/governance/build_rw_plan.py scripts/governance/verify_rw_plan_phase_check3.py tests/unit/governance/test_build_rw_plan.py tests/unit/governance/test_verify_rw_plan_phase_check3.py`
- `uv run python scripts/governance/verify_rw_plan_phase_check3.py`
- `uv run python scripts/governance/verify_arch_freeze_plan_traceability.py`
- `sha256sum -c SOURCES.sha256` in `docs/project/maintainers/qualification/arch_freeze_program/`

## Out of scope

`intergrax/`, `agents/`, `applications/`, new lane tests, the FRZ acceptance matrix, MEP-AF2, the RED-CLASSES decision record, older RED_FIRST records, digest-bound plan files other than this ticket's plan, any `Approved:` line, any Operator disposition, push, PR and merge.

## Implementer session

Implementer session: AF-CLOSE GA.2R implementer session (worktree branch `claude/clever-newton-6ded37`). It does not write the `Approved:` line.

## Approved

_(pending: a commissioned delegate after an independent REVIEW_PASS; the implementing session does not write this line)_

## Lifecycle state

plan_pending
