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

## C0-04 Final evidence reconciliation (@ `0159c2a18b39547bcd60823489c1a4bea1229b38`)

| Check | Result |
| --- | --- |
| Branch `development` | **PASS** |
| **CURRENT_PUBLICATION_HEAD** | `0159c2a18b39547bcd60823489c1a4bea1229b38` (`git rev-parse HEAD`) |
| Ancestor `8245b7471379414dc24385d1e770ccb56a7b0e2d` → HEAD | **PASS** |
| **ORIGINAL_AUDITED_HEAD** | `8245b7471379414dc24385d1e770ccb56a7b0e2d` (unchanged Wave A anchor) |

| Label | SHA | Role |
| --- | --- | --- |
| **IMPLEMENTATION_SHA** (first pub **A**) | `2ee22ae4b42d5e684f40d55056a6f9b84130b8ec` | Historical Wave A–D qual pack |
| **EVIDENCE_PUBLICATION_SHA** (first pub **B**) | `f0214533b347f3dfbbd729d96ef860869148dc0f` | Historical narrative pack |
| **TEST_EXECUTION_SHA** (first pub **C**) | `5c97fb679a3220ff7a156ca6ab7fd7fe8c5b5176` | **26 passed** — `.tmp/proof/dup-x-wave-d/run.log` |
| **WAVE1_PUBLICATION_START** | `2325460ff7f1f0290f95dae03145e30e20e0d991` | Wave 1 operator entry |
| **WAVE1_IMPLEMENTATION_SHA** (**A**) | `44759ed4db69a16fc07b837d9ed87d29a235163f` | Wave 1 qual SSOT |
| **WAVE1_EVIDENCE_PUBLICATION_SHA** (**B**) | `3a2d47d2c66fe60a4443d9800678ae2681a7f688` | Wave 1 narrative |
| **WAVE1_TEST_EXECUTION_SHA** (**C**) | `f188a496f969c99e66ca9d409cd29ddceeec8f0d` | **36 passed** primary B5 attestation — `.tmp/proof/dup-x-wave-1-post-publish/run.log` |
| **WAVE1_FINAL_EVIDENCE_RECONCILIATION_SHA** (**D**) | `0159c2a18b39547bcd60823489c1a4bea1229b38` | Doc-only reconciliation @ **CURRENT_PUBLICATION_HEAD** |

Reconciliation replay (agent session): `tests/qualification/dup_x/` → **36 passed** · `.tmp/proof/dup-x-03-final-evidence/run.log`

## C0 EXIT

**Baseline accepted for DUP-X closed-world inventory @ `8245b7471379414dc24385d1e770ccb56a7b0e2d`.**

**Status:** `DUP-X — FINAL EVIDENCE RECONCILED @ CURRENT_PUBLICATION_HEAD — NOT CLOSED`

**Remaining mandatory gates (not C0 failures):** OD-DUPX-01 · Protocol DUP-X-03 independent **B6 PASS** · OD-DUPX-08. (**OD-DUPX-02 APPROVED — 2026-10-29**.)

## C0-05 Block I parent-closure qual (@ `0e19bc28a6dc7700c633e9f2179974bb9238fbe6`)

| Label | SHA | Role |
| --- | --- | --- |
| **BLOCK_I_START** | `5b607c698f17b4fb1ca678d0c6204131e2971834` | Pre-Block-I publication HEAD |
| **BLOCK_I_IMPLEMENTATION_SHA** (**E**) | `053040ef758f4cf94b2844a08abe5a56244b666b` | Qual hardening |
| **BLOCK_I_EVIDENCE_PUBLICATION_SHA** (**F**) | `0e19bc28a6dc7700c633e9f2179974bb9238fbe6` | Evidence + Protocol/OD templates |
| **QUALIFICATION_EXECUTION_SHA** | `0e19bc28a6dc7700c633e9f2179974bb9238fbe6` | **40 passed** — `.tmp/proof/dup-x-parent-closure-l3/run.log` |
| **BLOCK_I_RECONCILIATION_SHA** (**G**) | `ffc724c4c2aa325231cd7ac54df5788f66cfc7f7` | Manifest + C0-05 + protocol ladder |
| **INDEPENDENT_AUDIT_BASELINE_SHA** (recommended) | `0e19bc28a6dc7700c633e9f2179974bb9238fbe6` | Qual + evidence before manifest Commit **G** |

**Status:** `DUP-X — READY FOR INDEPENDENT PROTOCOL 03 B6 — NOT CLOSED`

## C0-06 B4 NPSC negative boundary (@ `6ccddd3c40a88377e39962a300984d3b81f5c220`)

| Label | SHA | Role |
| --- | --- | --- |
| **B4_PUBLICATION_START** | `3feaee3d11d65d08a7860715e26bb38e763fcd40` | Pre-B4 publication HEAD |
| **B4_IMPLEMENTATION_SHA** | `100ab6b52dfdc4b86423601100bcc738c45a3b0c` | B4 negative boundary qual gates |
| **B4_EVIDENCE_PUBLICATION_SHA** | `6ccddd3c40a88377e39962a300984d3b81f5c220` | Delta matrix + evidence record + NPSC cross-ref |
| **QUALIFICATION_EXECUTION_SHA** | `6ccddd3c40a88377e39962a300984d3b81f5c220` | **44 passed** — `.tmp/proof/dup-x-b4-publication/dup_x-full.log` |
| **INDEPENDENT_AUDIT_BASELINE_SHA** (recommended) | `6ccddd3c40a88377e39962a300984d3b81f5c220` | Full DUP-X qual + B4 evidence @ replay |
| **B4_RECONCILIATION_SHA** | `a804a4319cbe644da757e196f9692c9d55070a67` | Protocol 03 final handoff |

**Status:** `DUP-X B4 — PUBLISHED / READY FOR FINAL INDEPENDENT B6 AUDIT — NOT CLOSED`

## C0-07 Final closure Block A + B-01 replay (@ `a7d83925b375b5f08eb0f175ced2c17fa026bba9`)

| Label | SHA | Role |
| --- | --- | --- |
| **FINAL_CLOSURE_START** | `f0928167aadc8f871dccadda039aa77b27f2f904` | Pre–final-closure HEAD |
| **FINAL_CLOSURE_BLOCK_A_SHA** | `cdf269fc9186f47737ed747babc2f9d831b1e713` | Portable qual paths · operator/NPSC docs |
| **FINAL_CLOSURE_QUAL_ADJUSTMENT_SHA** | `a7d83925b375b5f08eb0f175ced2c17fa026bba9` | Dossier gate wording + **44 passed** replay |
| **QUALIFICATION_EXECUTION_SHA** | `a7d83925b375b5f08eb0f175ced2c17fa026bba9` | `.tmp/proof/dup-x-final-b6-replay/dup_x-full.log` |
| **INDEPENDENT_AUDIT_BASELINE_SHA** (recommended) | `a7d83925b375b5f08eb0f175ced2c17fa026bba9` | External B6 replay pin |
| **FINAL_CLOSURE_RECONCILIATION_SHA** | `f8d267cc5f516d7ed45978758225dcdb971ad9ff` | Handoff + C0-07 ladder |
| **FINAL_CLOSURE_CURRENT_PUBLICATION_HEAD** | `fcc0930b9d4b627b80e6a0296984f41e1fc0589b` | SHA literal sync (post-reconciliation) |

**Status:** `DUP-X — READY FOR EXTERNAL B6 — NOT CLOSED`

## C0-08 Independent Protocol DUP-X-03 B6 (@ `a7d83925b375b5f08eb0f175ced2c17fa026bba9`)

| Label | SHA / value | Role |
| --- | --- | --- |
| **INDEPENDENT_ACCEPTANCE_BASELINE_SHA** | `a7d83925b375b5f08eb0f175ced2c17fa026bba9` | Qual + evidence replay pin |
| **PRE_ACCEPTANCE_PUBLICATION_HEAD** | `1f782aaa9017b31b2e68a1490d186a1e3fa055cd` | Doc-only commits after qual pin (reviewed) |
| **B6 verdict** | **ISSUED — PASS (SCOPED)** | External auditor **ChatGPT** — not Cursor |
| **ACCEPTANCE_PUBLICATION_SHA** | *(commit publishing PASS record)* | See manifest |
| **NPSC-4.2 exclusion** | **ACCEPT** (DUP-X scoped) | Allowlist gate **FAIL** unchanged |

**Status:** `DUP-X — INDEPENDENTLY ACCEPTED / OPERATOR CLOSURE PENDING`
