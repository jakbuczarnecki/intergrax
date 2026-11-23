# T-IGX-070-A — SCENARIO-FACTORY-X R2: documentation and architecture amendment (docs only)

## Scope

Prepare a complete **R2 amendment dossier** for Scenario Factory: autonomous mission model, R1→R2 delta, proposed contracts, SFX↔SEF cross-program amendment, WBS/DAG overlay, UX/metrics/commercialization extensions, REF/GEN evaluation plan, security qualification plan, operator decision register, and independent review matrix. **Does not** replace canonical R1 authority, accept ADRs, sign operator decisions, authorize implementation, or modify SEF-X authority.

**Input:** independent strategic blueprint `INTEGRAX_SFX_R2_STRATEGIC_BLUEPRINT_2026-11-23.md` (external proposal; reconciled at repo HEAD).

Plan: [`PLAN_T-IGX-070-A.md`](plans/PLAN_T-IGX-070-A.md).

## Baseline

| Field | Value |
| --- | --- |
| Audit HEAD (start) | `b4146cbfab9858c058fc44df2ad43d8670d4925d` (`development`) |
| Canonical SFX authority | R1 [`SFX-MASTER-PLAN.md`](plans/SFX-MASTER-PLAN.md) — unchanged binding text |
| Pre-freeze closure | `SFX_PRE_FREEZE_DOCS_CLOSED` (T-IGX-037-F) — preserved |

## Out of scope

- Product code in `agents/`, `applications/`, `intergrax/` (no SFX-P5+).
- SEF-X plan/ADR acceptance; HO-2 binding change without cross-program OD.
- ARCH-FREEZE promotion, GATE-SFX-ENTRY, GATE-SEF-CONSUME satisfaction claims.
- Reading or revealing sealed GEN content.

## Test contract

```text
uv run pytest tests/unit/docs/test_public_documentation_link_integrity.py -q
```

## Lifecycle state

plan_pending

## Adoption note (T-IGX-070-C)

Deliverables for 070-A are published on `development` @ `7ddd53266d32ce47f059456e110e414b92248453`. Canonical overlay text is in `SFX-MASTER-PLAN.md` §18. **`Approved:` for this ticket is not inferred from T-IGX-070-B.** Close via legitimate plan approval only.

## Deliverables

Hub: [`docs/project/maintainers/qualification/scenario_factory_x/SFX_R2_AMENDMENT_DOSSIER.md`](../docs/project/maintainers/qualification/scenario_factory_x/SFX_R2_AMENDMENT_DOSSIER.md)