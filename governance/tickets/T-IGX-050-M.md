# T-IGX-050-M — ARCH-FREEZE node T-AF2-REGISTER-UPDATE-1: Operator decisions and facts of 2026-11-16/17 in the tracked registers

## Scope

Documentation and data only. Records in the tracked ARCH-FREEZE registers the Operator decisions and facts that so far exist only in session records of the ARCH-FREEZE orchestrator: rows in the Operator decision register (OD-AF-11, Q-RW-19, Q-WA-05, OD-AF-22, the packet 0, packet 1 and packet 2 approvals and rulings, the standing PR rules OD-AF-20/21/21a, the one-time delegate authorizations for T-IGX-050-B and T-IGX-050-L, and the DP-54 ticket assignment), one known-red row (the two `tests/acceptance/llm_routing` nodes), and four post-ingest rows in the Enterprise Debt & Exceptions Register (Q-WA-03, the DUP-X item for Q-WA-07, the A4-007 seams b1/b2 residual of PT-WB, and DP-54 with ticket T-IGX-052-E). Two OD rows (OD-AF-11, DS-5) and one stale field of `reconciliation.json` are corrected because the new facts contradict them. Out of scope: every product path, `scripts/`, `tests/`, `.github/`, `pyproject.toml`, `uv.lock`, the roadmap, the FRZ checklist, every `IIAC_*` file, the DUP-X and closed-world gap registers, `qualification/README.md`. No PASS, FRZ promotion, stage closure, IIAC closure or independent-verification claim.

## Plan digest

Plan digest: 1aab70f6c15a429416a6e44c6280e228f60429375a230c1f2228f1f373f3ba20

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-050-M.md`](plans/PLAN_T-IGX-050-M.md). The digest was computed with `scripts/governance/plan_digest.py`; it must be recomputed after any edit of the plan.

## Approved

Approved: operator-delegate @ 2026-11-17T13:50:05+01:00
Delegation: docs/project/maintainers/qualification/arch_freeze_program/DELEGATION_RECORD_OD-AF-20_STANDING.md@29990c2baa74cf530658ce4a20f93db2e3bd4b8c
Delegate session: local_64ab068c-841e-4021-9874-5d5ba65f95ff

## Implementer

Implementer session: local_4d0f892f-3cea-47ba-ad96-0d490411236c

The review and the approval line must come from sessions that are not the implementer (standing rules OD-AF-20 and OD-AF-21a: documentation and ticket files only).

## Test contract

One process at a time, under the shared lock `D:/Projekty/_wt/_locks/pytest.lock` (at most 15 minutes):

- Parse check of `known_red_register.v1.json`, `reconciliation.json` and `register.jsonl`; the four new EDER rows checked against `register_schema.json`.
- `python -m scripts.governance.lint_iiac_closure_regex` once (default targets), result reported as measured.
- Ticket validators (`verify_ticket_lifecycle`, `verify_plan_approval`) on this ticket, if they run without extra dependencies.
- Reviewer check: `git diff --stat origin/development...HEAD` lists exactly the seven paths of the plan's write set.

## Out of scope

Code under `intergrax/`, `agents/`, `applications/`; scripts, tests, workflows; dependency files; IIAC scripts and records; the Q-RW-10 owner changes of KR-17/18/19; the AR-45/AR-50 trial outcome; the DP-47 data flip; Docker, vendor integration, production configuration.

## Lifecycle state

plan_approved
