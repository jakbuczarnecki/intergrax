# DUP-X — C0 Baseline Audit Record

| Field | Value |
| --- | --- |
| **Mandate** | Wave A entry — git gate |
| **Repository** | `jakbuczarnecki/intergrax` |
| **Origin** | `https://github.com/jakbuczarnecki/intergrax.git` |
| **Branch** | `development` |
| **EXPECTED_START_HEAD** | `8245b7471379414dc24385d1e770ccb56a7b0e2d` |
| **QUALIFICATION_HEAD** | `8245b7471379414dc24385d1e770ccb56a7b0e2d` |
| **AUDITED_HEAD** | `8245b7471379414dc24385d1e770ccb56a7b0e2d` |

## C0-01 Git identity

| Check | Result |
| --- | --- |
| Remote `origin` → jakbuczarnecki/intergrax | **PASS** |
| Branch `development` | **PASS** |
| `HEAD == EXPECTED_START_HEAD` | **PASS** |
| Ancestor `8245b747…` → HEAD | **PASS** (0 commits delta) |
| Untracked pollution | `platform_proofs/test_*` only — **out of DUP-X evidence scope** |

## C0-02 Evidence freshness

| Label | SHA | Role |
| --- | --- | --- |
| **AUDITED_HEAD** | `8245b7471379414dc24385d1e770ccb56a7b0e2d` | EBH-6 handoff baseline · DUP-X Wave A entry |
| **IMPLEMENTATION_SHA (Wave A–D pack)** | **PENDING OPERATOR COMMIT** | See [`DUP_X_PUBLICATION_MANIFEST.md`](DUP_X_PUBLICATION_MANIFEST.md) · Commit A+B |
| **EVIDENCE_PUBLICATION_SHA** | **PENDING OPERATOR COMMIT** | Manifest external record — **not** this doc’s commit alone |
| **TEST_RUN_HEAD (@ WIP)** | `8245b7471379414dc24385d1e770ccb56a7b0e2d` | pytest @ HEAD with uncommitted pack — log: `.tmp/proof/dup-x-wave-d/run.log` |
| **Publication ladder SSOT** | [`DUP_X_PUBLICATION_MANIFEST.md`](DUP_X_PUBLICATION_MANIFEST.md) | Update SHA fields after operator Commit C + optional re-run |

## C0-03 Canonical scope

| Unit | SSOT |
| --- | --- |
| Responsibility families | **15** — `tests/qualification/dup_x/_closed_world_inventory.py` |
| DUP classes | **DUP-01..14** — `DUP_X_CLASS_COVERAGE_MATRIX.md` |
| Pre-audit candidates | **DUPX-CAND-01..15** — `_findings_register.py` |
| EBH-6 deferrals | Class **D** selectors · duplicate census → **DUP-X PRIMARY** |

## C0 EXIT

**Baseline accepted for DUP-X closed-world inventory @ `8245b7471379414dc24385d1e770ccb56a7b0e2d`.**

**Status:** `DUP-X — WAVE A/D PACK IN PROGRESS — NOT CLOSED`

**Remaining mandatory gates (not C0 failures):** OD-DUPX-02 · OD-DUPX-01 · Protocol DUP-X-03 · OD-DUPX-08.
