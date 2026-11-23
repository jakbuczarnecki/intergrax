# PROPOSED amendment to SFX-MASTER-PLAN (R1 → R2 overlay)

**Status:** **MERGED into** `governance/tickets/plans/SFX-MASTER-PLAN.md` **§18** (T-IGX-070-C) — **effective when** [`OD-SFX-R2-CANONICAL-ADOPTION.md`](../operator_decisions/OD-SFX-R2-CANONICAL-ADOPTION.md) **is signed**  
**Ticket:** T-IGX-070-A

## Frontmatter change (PROPOSED)

```yaml
revision: R2-PROPOSED
r1_authority: preserved
amendment_dossier: docs/project/maintainers/qualification/scenario_factory_x/SFX_R2_AMENDMENT_DOSSIER.md
```

## Section insertions (summary)

| § | Insert |
| --- | --- |
| After §3.2 HO-2 | §3.2.1 **PROPOSED** delegated grant path — reference `SFX_R2_SEF_CROSS_PROGRAM_AMENDMENT.md`; HO-2 remains until OD |
| §7.4 interfaces | +6 PROPOSED schemas (mission, envelope, iteration, verdict, compare, source intake) |
| §7.9 lifecycle | Clarify mission state **projections** vs runtime FSM |
| §8 UX | Mission Dashboard per `SFX_R2_UX_MISSION_EXTENSION.md` |
| §9 requirements | + REQ-R2-001..018 via merged traceability after approval |
| §10 WBS | R2-W3..W8 overlay table from `SFX_R2_WBS_DAG_AMENDMENT.md` |
| §12 REF | REF-1 first vertical slice + autonomous loop acceptance |
| New §17 | Commercialization milestones (sandbox boundaries) |

## Explicit non-changes

- Option B placement, tier boundaries, SEF ownership, runtime authority.
- GATE-SFX-ENTRY / GATE-SFX-FULL definitions.
- GEN seal rules.

## Adoption procedure

1. Operator signs OD-SFX-R2-02.
2. Publish amended plan as **R2** with digest + SHA ladder row.
3. Merge requirements JSON after conflict check on REQ IDs.
4. SEF parallel track for CON-N-01 / HO-2 amendment.