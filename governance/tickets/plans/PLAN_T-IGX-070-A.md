# PLAN T-IGX-070-A — SFX R2 documentation amendment

**Status:** PROPOSED — pending operator `Approved:` line  
**Ticket:** [`T-IGX-070-A.md`](../T-IGX-070-A.md)  
**Classification:** documentation only · OD-SFX-04 / pre-ARCH-FREEZE design delta

## Objective

Transform R1 into an implementation-ready **R2 proposal** supporting Business Problem → … → Validated POC/MVP, with manual, assisted, and bounded autonomous execution modes, without violating ARCH-FREEZE or duplicating platform authorities.

## Steps

1. Reconcile authoritative R1 artifacts and code surface at pinned HEAD; record drift vs external blueprint.
2. Publish R2 dossier package under `docs/project/maintainers/qualification/scenario_factory_x/`.
3. Extend gap register, WBS overlay, requirements proposal, gap analysis §R2, UX/metrics appendices.
4. Propose (not merge) `SFX-MASTER-PLAN` R2 amendment and SEF cross-track package.
5. Run documentation link integrity test; publish independent review matrix.

## Edit scope

| Path | Action |
| --- | --- |
| `docs/project/maintainers/qualification/scenario_factory_x/SFX_R2_*` | create |
| `docs/project/maintainers/qualification/scenario_factory_x/r2/**` | create |
| `docs/project/maintainers/qualification/scenario_factory_x/proposals/**` | create |
| `SFX_INDEX.md`, `SFX_GAP_REGISTER.md`, `SFX_WBS.md`, `SFX_GAP_ANALYSIS.md`, `SFX_UX_DESIGN.md`, `design/SFX_METRICS.md` | PROPOSED appendices only |
| `governance/tickets/T-IGX-070-A.md`, this plan | create |

## Acceptance

- Every new REQ-R2-* maps to WBS unit, test family, evidence artifact, gate.
- HO-2 remains binding in canonical R1; R2 autonomous submit is **PROPOSED** cross-track only.
- Verdict states: `DOCUMENTATION_PREPARED` only (not `CANONICAL_R2_APPROVED`).