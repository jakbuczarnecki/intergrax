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
| **WAVE1_TEST_EXECUTION_SHA** | *(pending publication pytest)* | HEAD @ post-publish `uv run pytest tests/qualification/dup_x/ -q` |
| **WAVE1_MANIFEST_PUBLICATION_SHA** | *(this Commit C)* | Ladder only; **not** WAVE1_EVIDENCE_PUBLICATION_SHA |

**Historical (first publication @ AUDITED_HEAD `8245b747`):** rows in **Evidence chain (@ Commit C)** above — preserved unchanged.

## Wave 1 operator commit split (executed)

| Commit | SHA | Paths |
| --- | --- | --- |
| **A** | `44759ed4db69a16fc07b837d9ed87d29a235163f` | `tests/qualification/dup_x/**` |
| **B** | `3a2d47d2c66fe60a4443d9800678ae2681a7f688` | All `DUP_X_*.md` **except** this manifest |
| **C** | *(this commit)* | This manifest + Wave 1 SHA ladder |
| **D** (optional) | *(if needed)* | **WAVE1_TEST_EXECUTION_SHA** attestation after publication pytest |

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
| Exit code | *(pending)* |
| Tests | *(pending — record from log)* |
| Log | `.tmp/proof/dup-x-wave-1-post-publish/run.log` |
| **WAVE1_TEST_EXECUTION_SHA** | *(pending)* |

## Publication boundaries (unchanged)

- **NPSC-4.2** optional / not DUP-X L2 primary blocker — documented in [`DUP_X_FINDINGS_REGISTER.md`](DUP_X_FINDINGS_REGISTER.md) § NPSC-4.2 exclusion; no allowlist change in Wave 1.
- **DUPX-GAP-*** **PASS @ Wave 1** (evidence) — parent stage still **NOT CLOSED** until OD-DUPX-08 + Protocol PASS.
- Parent dossier: **NOT CLOSED** · no independent PASS · no global **FRZ-*** PASS claim.

## Operator verdict

**DUP-X — WAVE 1 IMPLEMENTATION COMPLETE — PUBLISHED / READY FOR INDEPENDENT PROTOCOL 03 AUDIT**

Qual + narrative on `development`; evidence SHAs A/B recorded; **TEST_EXECUTION_SHA** finalized after publication pytest.
