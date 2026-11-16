# OD-MCP-P7-P8-AUTH-01 — standing authorizations for the MCP-SERVER-X P7 and P8 campaign

| Field | Value |
| --- | --- |
| **Decision ID** | OD-MCP-P7-P8-AUTH-01 |
| **Status** | **APPROVED** |
| **Tickets** | `governance/tickets/T-IGX-035-H.md` (P7), `governance/tickets/T-IGX-035-I.md` (P8) |
| **Valid** | from signature until the MCP-SERVER-X final closure PR is merged or the operator revokes it in chat; no numeric expiry stated by the operator |

## Question

Does the operator grant the standing authorizations below so that the orchestrator can run P7 and P8 end to end without asking again, under the decision rules and automatic stop conditions below?

## Standing authorizations

- **SA-1 Branches:** create branches `mcp-p7-*` and `mcp-p8-*` from `origin/development`, set the upstream to `origin/development`, push only with an explicit refspec; no other branch names.
- **SA-2 Sessions:** start separate sessions for test authors (at most four at a time, disjoint test files, own worktrees), tester, independent auditor, correctors, certifier and terminal verifier. Roles are never reused across implementer, auditor and certifier.
- **SA-3 Pull requests and merge:** open PRs against `development`; merge **only with a merge commit** (no squash, no rebase, no auto-merge), **only after every required check has finished green** on the exact PR head (`ci-smoke`, `governance-fast`, `evol-x-b4-preflight`, `ent-audit-fast`, `ent-audit-pr-gate`), after comparing the head SHA with the pushed SHA. Pending is not green.
- **SA-4 Tests:** bounded qualification and audit-replay packages, one `pytest` process at a time; no broad diagnostic suites (C10/C15 restrictions stay in force: no Docker, no new vendors, no production configuration).
- **SA-5 Records:** commit documentation, tests and records for the two tickets with the `Ref:` trailer; delete merged P7/P8 branches and worktrees after each merge.

## Decision rules

- **DR-1** `REQ-MCP-OBS-01` stays a residual (no product change in P7).
- **DR-2** Audit findings: LOW/MEDIUM with an owner may be accepted as scoped limitations (documented). HIGH/BLOCKER go to the correction loop, at most three rounds, then stop.
- **DR-3** FRZ: evidence notes at existing IDs only; no PASS promotion; no new family.
- **DR-4** Any change in an ACT-N11 product path: stop (needs an independent delta audit and a two-phase IIAC rebind before merge).
- **DR-5** Red or failing required CI: fix only documentation, tests and records inside the ticket scope; anything else: stop.
- **DR-6** No waivers, no retroactive approvals; every process deviation is recorded as a finding.

## Automatic stops (the orchestrator asks the operator)

Product-path change (DR-4); audit BLOCKER/HIGH after the third round; red CI outside DR-5; a finding without an owner; SHA or role ambiguity; any request for a new FRZ family, FRZ promotion or ARCH-FREEZE action.

## Operator line

**Provenance:** signed by an operator-designated signing session under the delegation record `docs/project/maintainers/qualification/IIAC_OD_DELEGATION_MCP_P7_P8_AUTH_2026-11-12.md@bddcbb293f53812bebb0134b2278d8f038a86825` (IIAC Amendment 01, commission given in chat 2026-11-12). The line below is in the plain form because it is written by that session; it is not claimed to be operator-written.

```text
Approved: OD-MCP-P7-P8-AUTH-01 @ 2026-11-12T09:07:52+01:00
```

**Explicit non-actions:** does not close MCP-SERVER-X (see `OD-MCP-P8-CLOSE-01`), does not close ARCH-FREEZE, does not promote FRZ, does not authorize production configuration, Docker or new vendors, does not authorize any product-path change, and does not waive OPEN-BP-01.
