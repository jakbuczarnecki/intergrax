# P1-SIGN — BLOCKED (designated signing session required)

| Field | Value |
|---|---|
| Status | **BLOCKED** |
| Date | 2026-11-08 |
| Orchestrator | hist-cert Phase 1 slice (this session) |
| Prerequisite reviews | All four `PLAN_REVIEW.md` verdicts **PASS** (030-B, 030-C, 032-A, 033-A) |

## Why blocked

P1-SIGN requires a **new designated signing session** that is not the plan reviewers, not the implementer, and not this orchestrator. This session performed the four plan reviews (`process_enforced` assurance) but cannot fabricate `Approved:` delegate lines.

No `Approved:` lines were written to:

- `governance/tickets/T-IGX-030-B.md`
- `governance/tickets/T-IGX-030-C.md`
- `governance/tickets/T-IGX-032-A.md`
- `governance/tickets/T-IGX-033-A.md`

## Required actions for the signer session

1. Recompute each plan digest; refuse on mismatch (digests at `fe0937517` listed in ticket files and in `audits/plan-*/PLAN_REVIEW.md`).
2. Publish a new delegation record under `historical_certification_campaign/delegations/` (Amendment 01 form), one commission or four scoped commissions as Operator policy requires.
3. Write digest-bound lines, form:

   `Approved: operator-delegate <session-id> @ <ISO-8601 tz> | Delegation: docs/project/maintainers/qualification/historical_certification_campaign/delegations/<file>.md@<40-hex commit sha>`

4. Verify:

   ```text
   <venv>/python scripts/governance/verify_plan_approval.py --plan governance/tickets/plans/PLAN_T-IGX-030-B.md governance/tickets/T-IGX-030-B.md
   ```

   (repeat per ticket; see `verify_plan_approval.py --help`.)

5. Target branch for publication: `hist-cert/plan-approvals` or extend `hist-cert/plan-reviews` per Operator merge policy.

## Gate impact

**Phase 3 product lanes (C1–C4) must not start** for a ticket until that ticket has a verified `Approved:` line (P1-SIGN terminal). P2-AGREE publication does not unblock signing.
