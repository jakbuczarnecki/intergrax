# EVOL-X — SHA ladder (W1 + plan publication)

## Publication facts (@ verification 2026-11-01)

| Check | Result |
|-------|--------|
| `git rev-parse origin/development` | **`b4fca4bc4…`** @ plan publication (2026-11-01) |
| W1 documentation chain | Unchanged through **W1_AUDIT_REMEDIATION_SHA** |
| Plan v3 publication | Adds **PLAN_PUBLICATION_SHA** row below (same commit as `EVOL_X_FINAL_COMPLETION_MASTER_PLAN_v3.md`) |

## W1 commit chain (ancestor order)

| Order | Role | Full SHA | Subject |
|-------|------|----------|---------|
| 0 | **START_HEAD** / authorized stage baseline | `5a4e922ffacabef0ee24eb676e5c8705c20ef4a1` | EBH-7 pin baseline (pre-EVOL-X docs) |
| 1 | **W1_DOCUMENTATION_SHA** | `9f9cc8b573d2629c140b3a58bf239e8a92d0b1d0` | publish W1 architecture freeze and qualification annexes |
| 2 | **W1_LADDER_UPDATE_SHA** | `cac5dfd91f8a98c1183e6864a93bc74ddc41b725` | record W1_DOCUMENTATION_SHA on ladder |
| 3 | **W1_RECONCILIATION_PUBLICATION_SHA** | `54e0de377644fe19156a2f8497efa6cd10d7fdd2` | W1 reconciliation — N-18/N-04, audit index, AF/OD register |
| 4 | **W1_AUDIT_REMEDIATION_SHA** | `e48cae037fc022da0a99da66314d17dd6abbe296` | Independent audit remediation F-01/F-02/F-03 |

**Independent audit snapshot (documentation):** pin **`e48cae037`** (includes F-batch) or **`54e0de377`** (pre–F-batch tip) — record which in AF-8 worksheet.

## Implementation baseline (audit @ plan publication; not qualification evidence)

| Role | SHA | Notes |
|------|-----|--------|
| **IMPLEMENTATION_PARTIAL_SHA** | `8848857741f4b04e5b476dca3c47f7d0a60bf672` | Partial W2–W6 code on `development` at architecture audit; **not** TEST_EXECUTION_SHA |

## Plan publication (execution authority)

| Role | SHA | Notes |
|------|-----|--------|
| **PLAN_PUBLICATION_SHA** | `b4fca4bc413a650b2e5a8dd8826669a9025864c5` | Final completion master plan v3 + ladder (docs only). Resolve anytime: `git log -1 --format=%H -- docs/project/maintainers/qualification/EVOL_X_FINAL_COMPLETION_MASTER_PLAN_v3.md` |

**SHA lifecycle (no circular manifest self-hash):** `TEST_EXECUTION_SHA` is recorded **inside** evidence `MANIFEST.json`; `EVIDENCE_PUBLICATION_SHA` is **`git rev-parse HEAD`** on the evidence bundle checkout only. Require `merge-base --is-ancestor TEST_EXECUTION_SHA EVIDENCE_PUBLICATION_SHA`.

## Later stage roles (post-plan execution; not started)

| Role | SHA | Status |
|------|-----|--------|
| IMPLEMENTATION_SHA | — | NOT STARTED (W2+ closure commit) |
| TEST_EXECUTION_SHA | — | NOT STARTED |
| EVIDENCE_PUBLICATION_SHA | — | NOT STARTED (git-derived; not MANIFEST self-field) |
| B6_ACCEPTANCE_PUBLICATION_SHA | — | NOT STARTED |
| FORMAL_CLOSURE_PUBLICATION_SHA | — | NOT STARTED |

**Note:** W1 documentation commits are **descendants** of START_HEAD (`merge-base --is-ancestor 5a4e922 HEAD`).

**Operator entry:** `OD-EVOLX-01` — PENDING (Batch B blocked until W0).

**Prior stage:** EBH-7 — see `EBH_7_SHA_LADDER.md`.

**Execution authority:** [`EVOL_X_FINAL_COMPLETION_MASTER_PLAN_v3.md`](EVOL_X_FINAL_COMPLETION_MASTER_PLAN_v3.md)

**W7 SSOT ladder (roles TECH/FINAL/B6):** [`evol_x/EVOL_X_SHA_LADDER.md`](evol_x/EVOL_X_SHA_LADDER.md)
