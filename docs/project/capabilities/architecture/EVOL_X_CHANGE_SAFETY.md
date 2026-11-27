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

## Cross-cutting ownership (EQO-W4, `EDER-F-0239`)

One line per area (as of 2026-11-26 @ `646e63953`); "not applicable" lines state the reason.

| Area | Statement |
| --- | --- |
| Tenant & Security | Not applicable at runtime: EVOL-X is a passive change-time analysis and qualification toolchain (`testing_support/change_safety/`, `tests/qualification/evol_x/`, `scripts/qualification/evol_x_*.py`) that processes repository artifacts and introduces no runtime tenant or authorization bypass; production tiers remain strictly isolated from Q/T/S zones (IB-1). |
| Configuration | Configured strictly via CLI flags, checked-in change-safety policies, and consumed COMPAT-X / QUAL-X Pop-A manifests; no runtime environment or production config mutation. |
| Observability | Passive change dossier emission only (Zone Q/D); explicitly does not duplicate or write into `TRACE-X` runtime traces or production telemetry. |
| Failure & Recovery | Fail-closed static/qualification execution: discovery, diff, or suite-selection errors abort the change-safety assessment with non-zero exit status and never auto-approve deploy or rollback; optional AD rollback precondition port (Zone P) extends canonical Agent Distribution rather than creating a parallel recovery engine. |
| Tests | Qualification adapters and negative boundary gates live under `tests/qualification/evol_x/` (Zone T), consuming COMPAT-X, QUAL-X, DUP-X, STATE-X, and Agent Distribution harnesses. |
