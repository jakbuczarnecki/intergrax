# EVOL-X Change Safety — Implementation plan (multi-layer feature)

**Status:** W1 — planning only  
**Pair:** [architecture/EVOL_X_CHANGE_SAFETY.md](../architecture/EVOL_X_CHANGE_SAFETY.md)

## Waves (canonical — do not split)

| Wave | Scope | Production delta |
|------|-------|------------------|
| **W1** | Architecture freeze, baseline, specs, OD packs | **0** |
| **W2** | Contracts, IB gates, closed-world engine, graph, CI profile stub | 0 |
| **W3** | COMPAT adapter, QUAL selection, oracle | 0 |
| **W4** | Workflow, deploy assessor, dossier, triggers, CLI | **0 or 1 module** (OD-EVOL-05b) |
| **W5** | S1–S8 experiments, FRZ tests | 0 |
| **W6** | Metrics, VL3, evidence | 0 |
| **W7** | Independent closure, roadmap/checklist | 0 |

## W1 completed deliverables

Listed in [EVOL_X_W1_INDEPENDENT_ARCHITECTURE_AUDIT_REPORT.md](../../maintainers/qualification/evol_x/EVOL_X_W1_INDEPENDENT_ARCHITECTURE_AUDIT_REPORT.md).

## Domain plan rows (future)

When W2 starts, add rows to owning domain plans:

- **Applications** — profile/graph migration alignment (N-04).
- **Agent Distribution** — optional rollback precondition (OD-EVOL-05b).
- **Architecture / CTL** — capability graph adapter boundary only.

## Entry gate for W2

See checklist below. **W2 is not auto-authorized** after publication @ `54e0de377`.

## Blockers (reconciled)

| Blocker | Status |
|---------|--------|
| Remote publication @ `54e0de377` | **Resolved** (reviewer confirmed) |
| AF-6 operator signatures | **Open** |
| AF-8 independent architecture audit | **Open** |
| OD-EVOL-10 CONSERVATIVE_FULL manifest | **Open** |
| N-18 | Classified — not W1 regression; see N-18 annex |
| N-04 | N4-A proposed — operator disposition |

Full checklist: [EVOL_X_W2_PREREQUISITES.md](../../maintainers/qualification/evol_x/EVOL_X_W2_PREREQUISITES.md).
