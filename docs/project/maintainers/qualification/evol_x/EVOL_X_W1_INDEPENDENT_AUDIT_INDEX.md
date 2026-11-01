# EVOL-X W1 — Independent audit index (exact SHA)

**Purpose:** Deterministic reading order for external reviewer. **Does not** constitute independent acceptance.

## SHA chain

| Role | Full SHA | Commit message (subject) |
|------|----------|---------------------------|
| **START_HEAD** | `5a4e922ffacabef0ee24eb676e5c8705c20ef4a1` | (EBH-7 pin baseline) |
| **W1_DOCUMENTATION_SHA** | `9f9cc8b573d2629c140b3a58bf239e8a92d0b1d0` | docs(EVOL-X): publish W1 architecture freeze and qualification annexes |
| **LOCAL_HEAD** | `cac5dfd91f8a98c1183e6864a93bc74ddc41b725` | docs(EVOL-X): record W1_DOCUMENTATION_SHA on ladder |
| **origin/development** (at check) | `5a4e922ffacabef0ee24eb676e5c8705c20ef4a1` | Remote **behind** local by 2 commits |

**Verify locally:** `git merge-base --is-ancestor 5a4e922ffacabef0ee24eb676e5c8705c20ef4a1 HEAD` → exit 0.

## Hub and master

| ID | Path |
|----|------|
| HUB | [EVOL_X_MASTER_IMPLEMENTATION_AND_CERTIFICATION_PLAN.md](../EVOL_X_MASTER_IMPLEMENTATION_AND_CERTIFICATION_PLAN.md) |
| LADDER | [EVOL_X_SHA_LADDER.md](../EVOL_X_SHA_LADDER.md) |
| Cursor master v2 | `.cursor/plans/evol-x_master_plan_93e203ad.plan.md` (local IDE; not on origin) |

## Architecture / plan pair

| Path |
|------|
| [docs/project/capabilities/architecture/EVOL_X_CHANGE_SAFETY.md](../../../capabilities/architecture/EVOL_X_CHANGE_SAFETY.md) |
| [docs/project/capabilities/plan/EVOL_X_CHANGE_SAFETY.md](../../../capabilities/plan/EVOL_X_CHANGE_SAFETY.md) |

## W1 annexes (`evol_x/`)

| Topic | File |
|-------|------|
| Normative matrix | [EVOL_X_NORMATIVE_REQUIREMENTS_MATRIX.md](EVOL_X_NORMATIVE_REQUIREMENTS_MATRIX.md) |
| Drift RD-xx | [EVOL_X_DOCUMENT_DRIFT_REGISTER.md](EVOL_X_DOCUMENT_DRIFT_REGISTER.md) |
| Closed-world baseline | [W1_DISCOVERY_BASELINE.md](W1_DISCOVERY_BASELINE.md) |
| Exclusions | [W1_EXCLUSION_REGISTER.md](W1_EXCLUSION_REGISTER.md) |
| N-01..N-18 | [EVOL_X_N_VERIFICATION_REGISTER_W1.md](EVOL_X_N_VERIFICATION_REGISTER_W1.md) |
| N-18 detail | [EVOL_X_N18_COLLECTION_CLASSIFICATION_W1.md](EVOL_X_N18_COLLECTION_CLASSIFICATION_W1.md) |
| N-04 detail | [EVOL_X_N04_GRAPH_VERSION_RECONCILIATION_W1.md](EVOL_X_N04_GRAPH_VERSION_RECONCILIATION_W1.md) |
| AF-1..8 | [EVOL_X_AF_GATE_MATRIX_W1.md](EVOL_X_AF_GATE_MATRIX_W1.md) |
| Enforcement A | [EVOL_X_ENFORCEMENT_BOUNDARY_W1.md](EVOL_X_ENFORCEMENT_BOUNDARY_W1.md) |
| Closed-world B | [EVOL_X_CLOSED_WORLD_GATE_SPEC_W1.md](EVOL_X_CLOSED_WORLD_GATE_SPEC_W1.md) |
| FRZ-08/10 C | [EVOL_X_FRZ_08_10_CLOSURE_POLICY_W1.md](EVOL_X_FRZ_08_10_CLOSURE_POLICY_W1.md) |
| Contracts freeze | [EVOL_X_CONTRACT_SPEC.md](EVOL_X_CONTRACT_SPEC.md) |
| S1–S8 + metrics | [EVOL_X_EXPERIMENT_AND_METRIC_SPEC.md](EVOL_X_EXPERIMENT_AND_METRIC_SPEC.md) |
| Operator decisions | [EVOL_X_OPERATOR_DECISION_PACK_W1.md](EVOL_X_OPERATOR_DECISION_PACK_W1.md) |
| W2 entry | [EVOL_X_W2_PREREQUISITES.md](EVOL_X_W2_PREREQUISITES.md) |
| W1 implementer report | [EVOL_X_W1_INDEPENDENT_ARCHITECTURE_AUDIT_REPORT.md](EVOL_X_W1_INDEPENDENT_ARCHITECTURE_AUDIT_REPORT.md) |
| **This index** | [EVOL_X_W1_INDEPENDENT_AUDIT_INDEX.md](EVOL_X_W1_INDEPENDENT_AUDIT_INDEX.md) |

## Session evidence (gitignored)

`.tmp/session/evol-x-w1/pytest-collect-qual_x.log`, `pytest-collect-gate-unit.log` — reproduce on same HEAD.

## Link check

All paths above are repo-relative from `docs/project/maintainers/qualification/evol_x/` — **resolve** on `cac5dfd91`.
