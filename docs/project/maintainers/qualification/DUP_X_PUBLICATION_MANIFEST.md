# DUP-X — Publication Manifest (operator external record)

| Field | Value |
| --- | --- |
| **Purpose** | Publication SHA ladder @ operator commits A/B/C (+ optional D) |
| **Branch** | `development` |
| **Manifest author** | Operator-authorized publication session |
| **Operator commit / push** | **AUTHORIZED** |

## Evidence chain (@ Commit C)

| Label | Value | Notes |
| --- | --- | --- |
| **BASELINE_HEAD** | `8245b7471379414dc24385d1e770ccb56a7b0e2d` | `HEAD == origin/development` @ publication start |
| **AUDITED_HEAD** | `8245b7471379414dc24385d1e770ccb56a7b0e2d` | Matches `_findings_register.DUP_X_AUDITED_HEAD` |
| **IMPLEMENTATION_SHA** | `2ee22ae4b42d5e684f40d55056a6f9b84130b8ec` | Commit **A** — `tests/qualification/dup_x/**` + CONFIG-X discovery |
| **EVIDENCE_PUBLICATION_SHA** | `f0214533b347f3dfbbd729d96ef860869148dc0f` | Commit **B** — qualification narrative pack (excludes this manifest) |
| **TEST_EXECUTION_SHA** | `5c97fb679a3220ff7a156ca6ab7fd7fe8c5b5176` | Commit C HEAD @ publication pytest replay |
| **WAVE1_IMPLEMENTATION_HEAD** | `2325460ff7f1f0290f95dae03145e30e20e0d991` | Wave 1 qual/doc gates @ `git rev-parse HEAD` (uncommitted WIP allowed) |
| **MANIFEST_PUBLICATION_SHA** | `5c97fb679a3220ff7a156ca6ab7fd7fe8c5b5176` | Commit **C** — ladder only; **not** EVIDENCE_PUBLICATION_SHA |

**External record:** dossier/C0 reference **IMPLEMENTATION_SHA** and **EVIDENCE_PUBLICATION_SHA** above; do not equate **EVIDENCE_PUBLICATION_SHA** to Commit C.

## Operator commit split (executed)

| Commit | SHA | Paths |
| --- | --- | --- |
| **A** | `2ee22ae4b42d5e684f40d55056a6f9b84130b8ec` | `tests/qualification/dup_x/**` + `tests/qualification/config_x/_config_x_discovery.py` |
| **B** | `f0214533b347f3dfbbd729d96ef860869148dc0f` | All `DUP_X_*.md` **except** this manifest (pre-SHA-final) |
| **C** | `5c97fb679a3220ff7a156ca6ab7fd7fe8c5b5176` | This manifest + SHA ladder in C0 / parent dossier |
| **D** (evidence) | *(Commit D)* | **TEST_EXECUTION_SHA** attestation + publication validation table |

## Wave 1 evidence chain (@ publication Commit C)

| Label | Value | Notes |
| --- | --- | --- |
| **WAVE1_PUBLICATION_START** | `2325460ff7f1f0290f95dae03145e30e20e0d991` | `HEAD == origin/development` @ Wave 1 publication start |
| **WAVE1_IMPLEMENTATION_SHA** | `44759ed4db69a16fc07b837d9ed87d29a235163f` | Wave 1 Commit **A** — `tests/qualification/dup_x/**` only |
| **WAVE1_EVIDENCE_PUBLICATION_SHA** | `3a2d47d2c66fe60a4443d9800678ae2681a7f688` | Wave 1 Commit **B** — `DUP_X_*.md` **except** this manifest |
| **WAVE1_TEST_EXECUTION_SHA** | `f188a496f969c99e66ca9d409cd29ddceeec8f0d` | HEAD @ post-publish `uv run pytest tests/qualification/dup_x/ -q` |
| **WAVE1_MANIFEST_PUBLICATION_SHA** | `f188a496f969c99e66ca9d409cd29ddceeec8f0d` | Wave 1 Commit **C** — ladder only; **not** WAVE1_EVIDENCE_PUBLICATION_SHA |
| **CURRENT_PUBLICATION_HEAD** | `0159c2a18b39547bcd60823489c1a4bea1229b38` | Commit **D** · final evidence reconciliation |

**Historical (first publication @ AUDITED_HEAD `8245b747`):** rows in **Evidence chain (@ Commit C)** above — preserved unchanged.

## Wave 1 operator commit split (executed)

| Commit | SHA | Paths |
| --- | --- | --- |
| **A** | `44759ed4db69a16fc07b837d9ed87d29a235163f` | `tests/qualification/dup_x/**` |
| **B** | `3a2d47d2c66fe60a4443d9800678ae2681a7f688` | All `DUP_X_*.md` **except** this manifest |
| **C** | `f188a496f969c99e66ca9d409cd29ddceeec8f0d` | This manifest + Wave 1 SHA ladder |
| **D** (final evidence reconciliation) | `0159c2a18b39547bcd60823489c1a4bea1229b38` | Doc-only reconciliation · **CURRENT_PUBLICATION_HEAD** · **FINAL_PUBLICATION_HEAD** |

## Validation (final evidence reconciliation @ CURRENT_PUBLICATION_HEAD)

| Check | Result |
| --- | --- |
| **CURRENT_PUBLICATION_HEAD** | `0159c2a18b39547bcd60823489c1a4bea1229b38` |
| Command | `uv run pytest tests/qualification/dup_x/ -q` |
| Exit code | **0** |
| Tests | **36 passed** |
| Parent closure gates | **10 passed** — `test_dup_x_parent_closure_mechanical_gates.py` |
| Log | `.tmp/proof/dup-x-03-final-evidence/run.log` |

## Final evidence reconciliation commit (operator doc-only)

**Purpose:** Single operator commit capturing DUP-X-03 final evidence reconciliation (no qual SSOT change in this slice).

| Path | Action |
| --- | --- |
| `docs/project/maintainers/qualification/DUP_X_C0_BASELINE_AUDIT_RECORD.md` | C0-04 SHA ladder + reconciliation status |
| `docs/project/maintainers/qualification/DUP_X_PROTOCOL_03_INDEPENDENT_AUDIT_PACKAGE.md` | B5 dual attestation · B1–B6 handoff table |
| `docs/project/maintainers/qualification/DUP_X_PARENT_CERTIFICATION_DOSSIER.md` | Commit **D** · final status |
| `docs/project/maintainers/qualification/DUP_X_PUBLICATION_MANIFEST.md` | This section + validation @ HEAD |
| `docs/project/maintainers/qualification/DUP_X_CLOSURE_MATRIX.md` | Protocol row reconciliation note |
| `docs/project/maintainers/qualification/DUP_X_NPSC_4_2_DUPX_EXCLUSION_RECORD.md` | NPSC-4.2 ten-module exclusion record |

**Excluded from this commit:** `tests/qualification/dup_x/**` (frozen @ Wave 1 Commit **A** `44759ed4…`); NPSC test allowlists; production code.

## Validation (publication @ TEST_EXECUTION_SHA)

| Check | Result |
| --- | --- |
| Command | `uv run pytest tests/qualification/dup_x/ -q` |
| Exit code | **0** |
| Tests | **26 passed** |
| Duration | ~78s (see log) |
| Log | `.tmp/proof/dup-x-wave-d/run.log` |
| **TEST_EXECUTION_SHA** | `5c97fb679a3220ff7a156ca6ab7fd7fe8c5b5176` |

## Validation (Wave D prep @ WIP)

| Check | Result |
| --- | --- |
| Command | `uv run pytest tests/qualification/dup_x/ -q` |
| Exit code | **0** |
| Tests | **26 passed** |
| Log (prep) | same log path (overwritten @ publication replay) |
| **TEST_RUN_HEAD (prep)** | `8245b7471379414dc24385d1e770ccb56a7b0e2d` (WIP tree) |

## Mandatory docs gate (`test_dup_x_exit_mandatory_docs_present`)

Source: `tests/qualification/dup_x/test_dup_x_mechanical_exit_gates.py` — paths committed in **B**.

## Validation (Wave 1 prep @ WAVE1_PUBLICATION_START)

| Check | Result |
| --- | --- |
| Command | `uv run pytest tests/qualification/dup_x/ -q` |
| Exit code | **0** |
| Tests | **36 passed** (prep @ WIP; not publication attestation) |
| Log | `.tmp/proof/dup-x-wave-1/run.log` |
| **WAVE1_PUBLICATION_START** | `2325460ff7f1f0290f95dae03145e30e20e0d991` |

## Validation (Wave 1 publication @ WAVE1_TEST_EXECUTION_SHA)

| Check | Result |
| --- | --- |
| Command | `uv run pytest tests/qualification/dup_x/ -q` |
| Exit code | **0** |
| Tests | **36 passed** |
| Duration | ~76s (see log) |
| Log | `.tmp/proof/dup-x-wave-1-post-publish/run.log` |
| **WAVE1_TEST_EXECUTION_SHA** | `f188a496f969c99e66ca9d409cd29ddceeec8f0d` |
| **TEST_EXEC @ attestation** | `f188a496f969c99e66ca9d409cd29ddceeec8f0d` (Commit C HEAD — pytest replay) |

## Publication boundaries (unchanged)

- **NPSC-4.2** optional / not DUP-X L2 primary blocker — [`DUP_X_NPSC_4_2_DUPX_EXCLUSION_RECORD.md`](DUP_X_NPSC_4_2_DUPX_EXCLUSION_RECORD.md); no allowlist change in Wave 1.
- **DUPX-GAP-*** **PASS @ Wave 1** (evidence) — parent stage still **NOT CLOSED** until OD-DUPX-08 + Protocol PASS.
- Parent dossier: **NOT CLOSED** · no independent PASS · no global **FRZ-*** PASS claim.

## Operator verdict

**DUP-X — FINAL EVIDENCE RECONCILED @ `0159c2a18b39547bcd60823489c1a4bea1229b38` — READY FOR INDEPENDENT PROTOCOL 03 B6**

Qual + narrative on `development`; evidence SHAs A/B recorded; **TEST_EXECUTION_SHA** finalized after publication pytest.
