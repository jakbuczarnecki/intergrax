# PLAN T-IGX-070-C — IGX-SFX-R2-CANONICAL-ADOPTION-001 (Option B)

**Ticket:** T-IGX-070-C  
**Parent inputs:** T-IGX-070-A (R2 dossier), T-IGX-070-B (corrective publication)

## Objective

Publish R2 as a **versioned architectural overlay** on binding R1 in `SFX-MASTER-PLAN.md`, synchronize publication SSOT in the qualification hub, and record enterprise-quality findings — **without** product implementation, HO-2 relaxation, SEF canonical amendment, or production admission authority.

## Edit scope

- `governance/tickets/plans/SFX-MASTER-PLAN.md` (§18 overlay + cross-references only)
- `docs/project/maintainers/qualification/scenario_factory_x/SFX_INDEX.md`
- `docs/project/maintainers/qualification/scenario_factory_x/SFX_R2_{AMENDMENT,CORRECTIVE,PUBLICATION_HANDOFF}.md`
- `docs/project/maintainers/qualification/scenario_factory_x/r2/SFX_R2_INDEPENDENT_AUDIT_MATRIX.md`
- `docs/project/maintainers/qualification/scenario_factory_x/operator_decisions/OD-SFX-R2-CANONICAL-ADOPTION.md`
- This ticket pair; optional delegation record for plan approval

## Acceptance

1. Publication pins: PR #756 @ `67b127f3`, PR #757 @ `7ddd532`, reviewed head `e2b2a2b7`, merge-tree equivalence for SFX paths.
2. Historical SHAs and audit findings preserved; adoption adds trace rows only.
3. Master plan R1 text unchanged except additive §18, §3.2.1, §9/§10/§12 R2 pointers.
4. `IMPLEMENTATION_AUTHORIZED = NO`; `ARCH-FREEZE` / `GATE-SFX-ENTRY` unchanged.
5. Operator decision record prepared; signature not forged.

## Out of scope

Product code; `ASF-SEF-MASTER-PLAN.md` binding edits; IIAC/ENT-AUDIT remediation; T-IGX-070-A `Approved:` inference from 070-B.