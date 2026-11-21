# EQO-W3-WB01 — Governance & execution authority (read-only)

**Pin:** `28c27cb896d02e0185b9fa02047411fa152f6469`  
**Band:** C read-only + isolated pytest (no product edits).

## Finding scope

| ID | Probe |
|----|-------|
| EQOA-A1-021 | ENTERPRISE-FREEZE-CONTROL-01 independent audit artefact |
| EQOA-A4-015 | RuntimeContext production_mode reachability |
| EQOA-A4-044 | Host-task routing vs production_mode (#680 path) |
| EQOA-A5-003 | GOV-X1 gate-weakening review |
| EQOA-A7-030 | Delegated decision audit obligations |

## Read-only facts @ TRAIN_PIN

1. **#680 / T-IGX-043-A** is **not** merged into `28c27cb` (W2-01). Governance-fast failure class recorded in `pr680_governance_fast.md` (W0-03).
2. **T-IGX-044-F / #673** merge `9b7eca0e9` is on pin (W2-08); material-governance validators exist under `scripts/governance/` and `tests/unit/governance/test_iiac_*.py`.
3. **IIAC activation:** `pytest tests/unit/governance/test_iiac_activation_gate.py` → **1 failed** (`test_canonical_policy_is_in_force_and_verifies`), 23 passed, 1 skipped. Failure: `IIAC policy activation: FAIL` — consistent with EQOA-A0-007 / X-03 (`NOT_IN_FORCE` on train tree).

## Disposition (pre-AUD)

| Finding | DIS (interim) |
|---------|----------------|
| A4-044, A4-015 (680 cluster) | `BLOCKED` until #680 merges or exact blocker |
| A1-021, A5-003, A7-030 | `ACCEPTED_SCOPED_LIMITATION` pending session E — doc/dossier gap |
| IIAC cluster (A0-007 related) | `BLOCKED` pasmo G — RB-2 before IN_FORCE |

Evidence log: `pytest_wb01_readonly.log`.