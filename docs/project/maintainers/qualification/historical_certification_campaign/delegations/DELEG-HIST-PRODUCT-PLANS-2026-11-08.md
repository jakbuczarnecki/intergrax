# DELEG-HIST-PRODUCT-PLANS-2026-11-08 — delegation record (TRANSCRIBED from Operator Build approval + campaign commission)

Provenance: NOT signed by the Operator personally. Transcription of the Operator's approval of the Historical certification closure master plan in Cursor Build (2026-11-08) and the P1-SIGN commission in the campaign master plan (`HISTORICAL_CERTIFICATION_FINAL_MASTER_PLAN.md`, section P1-SIGN). The Operator wrote in Polish in chat; English-only rule applies here. Assurance: process_enforced.
Per IIAC Amendment 01 R-6 the commission transfers the plan-approval scope to the designated session; S-1 is struck, so this record is a transcription, not a condition of validity.

## Operator authority (English rendering)
- The Operator approved the HIST-01..04 closure **master plan** in Cursor Build (technical choices sections A.1–A.3, P1-SIGN prerequisites).
- Four independent plan reviews (`audits/plan-030b`, `plan-030c`, `plan-032a-r2`, `plan-033a-r2`) recorded verdict **PASS** at base `fe0937517` (published on branch `hist-cert/plan-reviews` / cherry-picked to `hist-cert/plan-approvals`).
- P1-SIGN names a **new** designated signing session to write digest-bound `Approved:` delegate lines for the four HELD product tickets only.

## Scope of THIS delegation (plan approval only)
- Decision: for each ticket below, if the session is satisfied with the bound plan digest (recomputed; refuse on mismatch), **APPROVE THE PLAN** (procedural gateway `Approved:` + `plan_approved` lifecycle). The session decides on its own judgment; it may REJECT any ticket.
- **Not delegated:** IIAC policy activation or any `Approved IIAC policy activation …` line; FRZ promotion; roadmap / checklist edits; implementation; audit; test runs; merge; PASS / closure claims; approval of any ticket outside the four listed digests.

### Tickets and bound plan digests (at `fe0937517`)

| Ticket | Plan file | Plan digest |
|--------|-----------|-------------|
| T-IGX-030-B | `governance/tickets/plans/PLAN_T-IGX-030-B.md` | `8c650422f0ece0e26254ba63d2b5a9675d61ea9e8a131d0e4212cbd8679a3378` |
| T-IGX-030-C | `governance/tickets/plans/PLAN_T-IGX-030-C.md` | `55d4c9e6e7cc05b27b794e24d71ef1c7ac02a6c4d3026ea838a3deeb007e0103` |
| T-IGX-032-A | `governance/tickets/plans/PLAN_T-IGX-032-A.md` | `2628d1a834d2fcf25f47c55b0d60355811898c9e60337306b557f152d7ba7446` |
| T-IGX-033-A | `governance/tickets/plans/PLAN_T-IGX-033-A.md` | `ae167672d4075e3e4dc190ac93c78a937d7ad0c435dfc3971e615143e4273775` |

## Designated session
- Name: **hist-sign-plans** (Cursor agent session `hist-sign-plans`), a NEW session, dedicated solely to P1-SIGN plan approvals, not a continuation, not an implementation session (R-2).
- Must not be: the Phase 1 orchestrator that authored plan reviews or `P1_SIGN_HANDOFF.md`; sessions `plan-rev-b`, `plan-rev-c`, `plan-rev-032`, `plan-rev-033` (or any author of the four `PLAN_REVIEW.md` files); any future P3 implementer (`hist-c1-030b` … `hist-c4-033a`) or auditor of these batches; MASTER; cert-031a / cert-031a2 (separate ticket).
- Session statement: this session was started fresh for this brief only; it read the four `PLAN_REVIEW.md` verdicts (all PASS), the master plan P1-SIGN gate, and `P1_SIGN_HANDOFF.md`; it authored no plan review and no product-path commit.
- Expires: none stated by the Operator. R-5: the session must act immediately after being commissioned.

## Required form of the resulting lines (disclosed delegate form, S-2)
Per ticket, under `## Approved` in `governance/tickets/T-IGX-<id>.md`:

`Approved: operator-delegate hist-sign-plans @ <ISO-8601 with timezone> | Delegation: docs/project/maintainers/qualification/historical_certification_campaign/delegations/DELEG-HIST-PRODUCT-PLANS-2026-11-08.md@<40-hex sha of the commit that carries the published copy>`

Verification gate (each ticket):

`uv run python scripts/governance/verify_plan_approval.py --plan governance/tickets/plans/PLAN_T-IGX-<id>.md governance/tickets/T-IGX-<id>.md`
