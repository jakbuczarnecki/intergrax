# EVOL-X Change Safety — Architecture (multi-layer feature)

**Status:** W1 architecture freeze (draft ADR)  
**Pair:** [plan/EVOL_X_CHANGE_SAFETY.md](../plan/EVOL_X_CHANGE_SAFETY.md)  
**Stage:** EVOL-X — Enterprise Evolvability & Change Safety Certification  
**START_HEAD:** `5a4e922ffacabef0ee24eb676e5c8705c20ef4a1`

## Purpose

Deliver **change-time** impact analysis, consumer closure, regression selection, workflow/deploy **assessment**, and passive diagnostics **without** a second release, governance, or execution authority.

## St zones (OD-EVOL-01 recommendation — pending approval)

| Zone | Path | Role |
|------|------|------|
| **P** | `intergrax/` (ship) | Existing runtime/domain; optional **single** additive AD port if OD-EVOL-05b Option 2 |
| **Q** | `testing_support/change_safety/` | Typed contracts, discovery, graph, selection, assessors, dossier |
| **T** | `tests/qualification/evol_x/` | Adapters to COMPAT-X, QUAL-X, DUP-X, STATE-X, AD harness |
| **S** | `scripts/qualification/evol_x_*.py` | Workflow CLI, evidence publication |
| **D** | `docs/.../evol_x_p03_independent_replay/` | Protocol 03 dossier (W6) |

## Component ownership (10-question screen summary)

| Component | Zone | Duplicate authority? |
|-----------|------|-------------------|
| Closed-world inventory | Q | No — adapts COMPAT-X discovery |
| Semantic graph builder | Q | No — slice + provenance vs CTL-11 SSOT |
| Contract diff at change | Q | No — consumes COMPAT policy |
| Suite selection | Q | No — consumes QUAL-X P5 |
| Workflow matrix | Q | No — reads profile resolution types |
| Deploy assessor | Q | No — read-only AD facts |
| Precondition port (optional) | P | **Extends** AD rollback — not parallel executor |
| Change dossier | Q | No — passive vs TRACE-X runtime |

## Import boundaries

- IB-1: production tiers ↛ `tests`, `testing_support`, `scripts` (extend existing gate).
- IB-4: only `tests/qualification/evol_x/adapters/**` imports `tests.qualification.*`.

## Domain dependencies

- **COMPAT-X** policy; **QUAL-X** Pop-A manifest; **STATE-X** checkpoint semantics; **Agent Distribution** activation paths; **Post-freeze governance** frozen-core tables.

## W1 outcome

Architecture frozen on paper; **no** `testing_support/change_safety/` code until W2 after OD signatures.

## Related W1 artifacts

Hub: [EVOL_X_MASTER_IMPLEMENTATION_AND_CERTIFICATION_PLAN.md](../../maintainers/qualification/EVOL_X_MASTER_IMPLEMENTATION_AND_CERTIFICATION_PLAN.md)
