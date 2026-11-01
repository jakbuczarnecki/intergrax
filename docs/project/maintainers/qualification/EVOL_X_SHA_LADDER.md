# EVOL-X — SHA ladder (W1)

## Publication facts (@ verification 2026-11-01)

| Check | Result |
|-------|--------|
| `git rev-parse origin/development` | `54e0de377…` until F-batch **pushed**; local HEAD may be **`e48cae037…`** |
| Production delta in W1 chain | **None** (22 documentation paths only) |

## W1 commit chain (ancestor order)

| Order | Role | Full SHA | Subject |
|-------|------|----------|---------|
| 0 | **START_HEAD** / authorized stage baseline | `5a4e922ffacabef0ee24eb676e5c8705c20ef4a1` | EBH-7 pin baseline (pre-EVOL-X docs) |
| 1 | **W1_DOCUMENTATION_SHA** | `9f9cc8b573d2629c140b3a58bf239e8a92d0b1d0` | publish W1 architecture freeze and qualification annexes |
| 2 | **W1_LADDER_UPDATE_SHA** | `cac5dfd91f8a98c1183e6864a93bc74ddc41b725` | record W1_DOCUMENTATION_SHA on ladder |
| 3 | **W1_RECONCILIATION_PUBLICATION_SHA** | `54e0de377644fe19156a2f8497efa6cd10d7fdd2` | W1 reconciliation — N-18/N-04, audit index, AF/OD register |
| 4 | **W1_AUDIT_REMEDIATION_SHA** | `e48cae037fc022da0a99da66314d17dd6abbe296` | Independent audit remediation F-01/F-02/F-03 |

**Independent audit snapshot (documentation):** pin **`e48cae037`** (includes F-batch) or **`54e0de377`** (pre–F-batch tip) — record which in AF-8 worksheet.

## Later stage roles (not W1)

| Role | SHA | Status |
|------|-----|--------|
| IMPLEMENTATION_SHA | — | NOT STARTED (W2+) |
| TEST_EXECUTION_SHA | — | NOT STARTED |
| EVIDENCE_PUBLICATION_SHA | — | NOT STARTED |
| B6_ACCEPTANCE_PUBLICATION_SHA | — | NOT STARTED |
| FORMAL_CLOSURE_PUBLICATION_SHA | — | NOT STARTED |

**Note:** W1 documentation commits are **descendants** of START_HEAD (`merge-base --is-ancestor 5a4e922 HEAD`).

**Operator entry:** `OD-EVOLX-01` — PENDING.

**Prior stage:** EBH-7 — see `EBH_7_SHA_LADDER.md`.
