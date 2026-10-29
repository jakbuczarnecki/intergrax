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
| **TEST_EXECUTION_SHA** | **PENDING POST-PUSH PYTEST** | Record after `uv run pytest tests/qualification/dup_x/ -q` @ publication HEAD — optional Commit **D** |
| **MANIFEST_PUBLICATION_SHA** | *(this commit)* | Commit **C** — ladder only; **not** EVIDENCE_PUBLICATION_SHA |

**External record:** dossier/C0 reference **IMPLEMENTATION_SHA** and **EVIDENCE_PUBLICATION_SHA** above; do not equate **EVIDENCE_PUBLICATION_SHA** to Commit C.

## Operator commit split (executed)

| Commit | SHA | Paths |
| --- | --- | --- |
| **A** | `2ee22ae4b42d5e684f40d55056a6f9b84130b8ec` | `tests/qualification/dup_x/**` + `tests/qualification/config_x/_config_x_discovery.py` |
| **B** | `f0214533b347f3dfbbd729d96ef860869148dc0f` | All `DUP_X_*.md` **except** this manifest (pre-SHA-final) |
| **C** | *(record at commit)* | This manifest + SHA ladder in C0 / parent dossier |
| **D** (optional) | TBD | **TEST_EXECUTION_SHA** + publication pytest log reference |

## Validation (Wave D prep @ WIP)

| Check | Result |
| --- | --- |
| Command | `uv run pytest tests/qualification/dup_x/ -q` |
| Exit code | **0** |
| Tests | **26 passed** |
| Log (prep) | `.tmp/proof/dup-x-wave-d/run.log` |
| **TEST_RUN_HEAD (prep)** | `8245b7471379414dc24385d1e770ccb56a7b0e2d` (WIP tree) |

**Publication replay:** re-run pytest after push @ `TEST_EXECUTION_SHA` HEAD; update Validation table + optional Commit **D**.

## Mandatory docs gate (`test_dup_x_exit_mandatory_docs_present`)

Source: `tests/qualification/dup_x/test_dup_x_mechanical_exit_gates.py` — paths committed in **B**.

## Publication boundaries (unchanged)

- **NPSC-4.2** optional / not DUP-X L2 primary blocker (see dossier evidence commands).
- **DUPX-GAP-*** remain **OPEN**.
- Parent dossier: **NOT CLOSED** · no independent PASS · no global **FRZ-*** PASS claim.

## Operator verdict

**DUP-X — PUBLISHED / READY FOR INDEPENDENT PROTOCOL 03 AUDIT**

Qual + narrative on `development`; evidence SHAs A/B recorded; **TEST_EXECUTION_SHA** finalized after publication pytest.
