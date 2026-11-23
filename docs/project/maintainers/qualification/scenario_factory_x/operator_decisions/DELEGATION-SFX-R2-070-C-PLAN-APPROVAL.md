# DELEGATION-SFX-R2-070-C-PLAN-APPROVAL — Operator commission for plan approval (T-IGX-070-C)

| Field | Value |
| --- | --- |
| **Delegation ID** | DELEGATION-SFX-R2-070-C-PLAN-APPROVAL |
| **Stage** | SCENARIO-FACTORY-X (SFX) R2 canonical adoption (Option B) |
| **Campaign** | IGX-SFX-R2-ADOPTION-FINAL-AUTH-003 |
| **Ticket** | `T-IGX-070-C` |
| **Plan path** | `governance/tickets/plans/PLAN_T-IGX-070-C.md` |
| **Expected plan digest** | `7c039ad2c00db38b6a1e0ac8f0feca204cfd39c4805efc584c4aaa92752e9925` (via `scripts/governance/plan_digest.py`) |
| **Adoption candidate (reference)** | `891e59c01ff2e3bd0d9e69e464bffeb22486e962` |
| **Basis** | IIAC Amendment 01, rules R-1..R-6, as summarized in `governance/tickets/README.md` §Approved |
| **Date** | 2026-11-23 |
| **Source of commission** | Operator chat commission **IGX-SFX-R2-ADOPTION-FINAL-AUTH-003** (English directive; operator language Polish), section 0 and G2 |
| **Assurance** | `process_enforced` — delegate may refuse if verification fails |

## Operator commission (transcription)

As the Integrax Operator, the operator commissioned a **new, independent, dedicated Plan Approval Delegate session** for the sole purpose of reviewing, verifying digest, and recording governance authorization for `T-IGX-070-C` and plan `governance/tickets/plans/PLAN_T-IGX-070-C.md`, under **Option B** (R2 as versioned overlay on R1; documentation/governance only).

This is a commission to an independent delegate, **not permission for the SFX R2 adoption implementer to approve its own work**.

The delegate must independently verify that the plan matches the approved Option B scope, does **not** authorize SFX product implementation, and that `Plan digest:` equals `scripts/governance/plan_digest.py` on the plan file. The delegate may refuse approval if verification fails.

## Scope of the delegation

- A dedicated delegate session may write exactly one approval block in **`governance/tickets/T-IGX-070-C.md` only**, using strict grammar from `scripts/governance/ticket_approval.py`.
- Scope authorized: **plan approval for documentation-only R2 canonical overlay adoption** (T-IGX-070-C).
- Before writing the line the delegate independently confirms:
  - Plan digest matches `7c039ad2c00db38b6a1e0ac8f0feca204cfd39c4805efc584c4aaa92752e9925`.
  - The plan is documentation-only; it does not authorize SFX-P5+, delegated multi-submit, HO-2 relaxation, SEF CON-N-01 change, ADR/ECAP acceptance, FRZ/IIAC promotion, production admission, or product code.
  - The delegate session differs from the implementer session on this campaign.
- The delegate may **refuse** approval if any check fails.

## Independence rules (R-1..R-6)

1. Fresh host session dedicated exclusively to this approval task (not a resumed implementation session).
2. Dedicated operator-delegate role; actual host-provided session identity on the approval line.
3. No implementation, reconciliation, publication, or G6 audit of T-IGX-070-C by the delegate.
4. No operator identity fabrication and no invented host session IDs.
5. The SFX R2 adoption implementer session must differ from `Delegate session:`.
6. This transcription record (G1) must be a strict Git ancestor of the approval commit (G2).

## Explicit prohibited operations

The delegate and this record **do not** authorize:

- SFX product implementation or SFX-P5+ work.
- HO-2 or SEF-X authority changes; ADR-SEF acceptance.
- Clearing ARCH-FREEZE, GATE-SFX-ENTRY, SCENARIO-GATE, or PROD-Q.
- Bypassing hooks, CI, or branch protection.
- Copying approval from T-IGX-070-B or backdating records.

`CANONICAL_R2_APPROVED = NO` and `IMPLEMENTATION_AUTHORIZED = NO` remain until OD signature and lawful publication complete.

## Separation

- The adoption implementer, its reviewer, OD signing delegate, and G6 auditor never write the `Approved:` line for T-IGX-070-C.
- The plan-approval delegate does not sign OD-SFX-R2-CANONICAL-ADOPTION, does not push adoption overlay commits, and does not merge.