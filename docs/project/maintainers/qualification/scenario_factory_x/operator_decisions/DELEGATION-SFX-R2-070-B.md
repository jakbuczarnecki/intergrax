# DELEGATION-SFX-R2-070-B — Operator commission for plan approval (T-IGX-070-B)

| Field | Value |
| --- | --- |
| **Delegation ID** | DELEGATION-SFX-R2-070-B |
| **Stage** | SCENARIO-FACTORY-X (SFX) R2 documentation corrective |
| **Campaign** | IGX-SFX-R2-FINAL-CLOSURE-001 |
| **Ticket** | `T-IGX-070-B` |
| **Plan path** | `governance/tickets/plans/PLAN_T-IGX-070-B.md` |
| **Expected plan digest** | `0cf2a40a9b5faec7f9f5d89ed13658c930007bbb30ecba74451e199bf5fe85a7` |
| **Historical corrective candidate (reference)** | `7daac655226f0f5abd4200617983d592a5b9bc34` |
| **Basis** | IIAC Amendment 01, rules R-1..R-6, as summarized in `governance/tickets/README.md` §Approved |
| **Date** | 2026-11-23 |
| **Source of commission** | Operator chat commission **IGX-SFX-R2-BOOTSTRAP-001** (English directive; operator language Polish) |
| **Transcription session** | Parent orchestrator Cursor host session `local_ac06fc97-1ab8-49b5-b1a0-d2ea0ebd2385` |
| **Assurance** | `process_enforced` — delegate may refuse if verification fails |

## Operator commission (verbatim transcription)

As the Integrax Operator, the operator commissioned a **new, independent, dedicated operator-delegate session** for the sole purpose of reviewing, approving and publishing the governance authorization for `T-IGX-070-B`, associated plan `governance/tickets/plans/PLAN_T-IGX-070-B.md`, at corrective candidate `7daac655226f0f5abd4200617983d592a5b9bc34` with expected plan digest `0cf2a40a9b5faec7f9f5d89ed13658c930007bbb30ecba74451e199bf5fe85a7`.

This is a commission to an independent delegate, **not permission for the existing implementer to approve its own work**.

The delegate must independently verify that the plan matches the approved documentation-only scope and the expected digest. The delegate may refuse approval if verification fails.

The operator also authorized the existing Cursor delivery campaign, **after lawful approval is published**, to perform the complete technical lifecycle: reconciliation → validation → independent review → commit → push → PR → CI → scoped corrections → permitted merge → remote verification, within the explicitly authorized scope. No additional operator permission is required for routine Git operations within that scope.

## Scope of the delegation

- A dedicated delegate session may write exactly one approval block in **`governance/tickets/T-IGX-070-B.md` only**, using strict grammar from `scripts/governance/ticket_approval.py`.
- Scope authorized: **documentation corrections and approval bootstrap only** for SFX R2 corrective dossier (T-IGX-070-B).
- Before writing the line the delegate independently confirms:
  - `Plan digest:` equals `scripts/governance/plan_digest.py` on `governance/tickets/plans/PLAN_T-IGX-070-B.md` (`0cf2a40a9b5faec7f9f5d89ed13658c930007bbb30ecba74451e199bf5fe85a7`).
  - The plan is documentation-only; it does not authorize product implementation, canonical R2 adoption, HO-2 or SEF-X authority changes, ADR/ECAP acceptance, FRZ/IIAC promotion, or GitHub security changes.
  - The delegate session differs from the implementer session recorded on the ticket.
- The delegate may **refuse** approval if any check fails.

## Independence rules (R-1..R-6)

1. Fresh host session dedicated exclusively to this approval task (not a resumed implementation session).
2. Dedicated operator-delegate role; actual host-provided session identity on the approval line.
3. No implementation or audit of T-IGX-070-B by the delegate.
4. No operator identity fabrication and no invented host session IDs.
5. The SFX implementer session (`Implementer session:` on the ticket) must differ from `Delegate session:`.
6. This transcription record (G1) must be a strict Git ancestor of the approval commit (G2).

## Explicit prohibited operations

The delegate and this record **do not** authorize:

- Adoption of Scenario Factory R2 as canonical architecture.
- SFX product implementation.
- HO-2 or SEF-X authority changes.
- ADR/ECAP acceptance.
- FRZ/IIAC promotion.
- Modification of GitHub security settings.
- Bypassing hooks, CI or branch protection.
- Changes to `igx_integrator.py`, its guards, protected governance policy or branch settings to force publication.

**This is not authorization for R2 product implementation.**  
`CANONICAL_R2_APPROVED = NO` and `IMPLEMENTATION_AUTHORIZED = NO` remain in force.

## Separation

- The SFX implementer session, its reviewer and auditor sessions never write the `Approved:` line for T-IGX-070-B.
- The delegate session does not implement SFX corrections, does not run as the independent SFX reviewer, and does not merge the SFX PR.