# DELEGATION-SFX-CLOSEOUT — Operator commission for plan-approval signing (T-IGX-037-F)

| Field | Value |
| --- | --- |
| **Record ID** | DELEGATION-SFX-CLOSEOUT |
| **Stage** | SCENARIO-FACTORY-X (SFX) |
| **Basis** | IIAC Amendment 01, rules R-1..R-6, as summarized in `governance/tickets/README.md` §Approved. Precedent: `DELEGATION-SFX-PRE-FREEZE.md`. |
| **Date** | 2026-11-13 |
| **Provenance** | The SFX closeout implementer session transcribed this record. It is not operator-signed. Assurance: `process_enforced`. |

## Commission (transcription; English rendering of the operator's Polish chat)

The implementer asked who writes the `Approved:` line of `T-IGX-037-F`. The options were the operator personally, or a delegate in a separate session. The operator answered **"Sesja delegata"** (English rendering: "A delegate session"). Earlier the operator had asked the campaign to continue "with proper authorization of ticket T-IGX-037-F".

## Scope of the delegation

- A dedicated delegate session may write `Approved: operator-delegate @ <ISO8601>`, followed on the next line by `Delegation: docs/project/maintainers/qualification/scenario_factory_x/operator_decisions/DELEGATION-SFX-CLOSEOUT.md@<commit sha>`, in **`governance/tickets/T-IGX-037-F.md` only**.
- The delegate decides independently and may refuse. Before writing the line it confirms all of the following:
  - The `Plan digest:` equals `scripts/governance/plan_digest.py` run on `governance/tickets/plans/PLAN_T-IGX-037-F.md`.
  - The plan is documentation-only. The diff of the candidate changes no path under `intergrax/`, `agents/`, `applications/`, `scripts/`, `agent-env/`, `.claude/`, `.cursor/` or `governance/agent-control-plane/`. The only test file touched is `tests/unit/docs/test_public_readme_contract.py`.
  - The plan does not authorize SFX-P4-B, SFX-P5 or later, accepts no ADR, changes no FRZ, grants no `SFX_PRE_IMPLEMENTATION_READY`, sets no `PRE_IMPLEMENTATION_READY_SHA`, and writes no operator signature.
  - The audits recorded in the ticket exist and the latest one has no open BLOCKER or HIGH finding.
- **Not delegated:** OD signatures, ADR acceptance, IIAC activation, FRZ promotion, merges, `GATE-SFX-ENTRY`, `SFX_PRE_IMPLEMENTATION_READY`, any SFX-P4-B or SFX-P5+ ticket, and any other ticket.

## Separation

- The implementer session, its reviewer and auditor sessions never write this line.
- The delegate session does not implement or audit the ticket.
