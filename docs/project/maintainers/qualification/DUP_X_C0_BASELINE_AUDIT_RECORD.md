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
| **IMPLEMENTATION_SHA (Wave A–D pack)** | `2ee22ae4b42d5e684f40d55056a6f9b84130b8ec` | Commit **A** — qual SSOT + CONFIG-X discovery union |
| **EVIDENCE_PUBLICATION_SHA** | `f0214533b347f3dfbbd729d96ef860869148dc0f` | Commit **B** — narrative pack (**not** manifest-only Commit C) |
| **TEST_EXECUTION_SHA** | `5c97fb679a3220ff7a156ca6ab7fd7fe8c5b5176` | `@` Commit C HEAD · **26 passed** · `.tmp/proof/dup-x-wave-d/run.log` |
| **Publication ladder SSOT** | [`DUP_X_PUBLICATION_MANIFEST.md`](DUP_X_PUBLICATION_MANIFEST.md) | External SHA ladder — dossier does not self-reference Commit C |

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
