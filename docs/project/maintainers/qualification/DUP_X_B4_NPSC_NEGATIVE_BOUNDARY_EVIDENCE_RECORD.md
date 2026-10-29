# DUP-X — B4 NPSC negative boundary evidence record

| Field | Value |
| --- | --- |
| **Record type** | Independent B4 — orchestration backend ten-module boundary |
| **Publication status** | **PUBLISHED** — qual + evidence commits on `development` |
| **B4_PUBLICATION_START** | `3feaee3d11d65d08a7860715e26bb38e763fcd40` |
| **B4_IMPLEMENTATION_SHA** | `100ab6b52dfdc4b86423601100bcc738c45a3b0c` |
| **B4_EVIDENCE_PUBLICATION_SHA** | `6ccddd3c40a88377e39962a300984d3b81f5c220` |
| **QUALIFICATION_EXECUTION_SHA** | `a7d83925b375b5f08eb0f175ced2c17fa026bba9` |
| **INDEPENDENT_AUDIT_BASELINE_SHA** (recommended) | `a7d83925b375b5f08eb0f175ced2c17fa026bba9` |
| **Branch** | `development` |
| **Production code delta** | **0** |
| **NPSC gate modified** | **No** |

## Artifacts

| Artifact | Path |
| --- | --- |
| Ten-module matrix + B4 rationale | [`DUP_X_RESPONSIBILITY_DELTA_NPSC_ORCHESTRATION_BACKEND.md`](DUP_X_RESPONSIBILITY_DELTA_NPSC_ORCHESTRATION_BACKEND.md) |
| NPSC exclusion register | [`DUP_X_NPSC_4_2_DUPX_EXCLUSION_RECORD.md`](DUP_X_NPSC_4_2_DUPX_EXCLUSION_RECORD.md) |
| Adversarial negative gates | `tests/qualification/dup_x/test_dup_x_b4_npsc_orchestration_backend_negative_boundary.py` |

## Test execution (@ `QUALIFICATION_EXECUTION_SHA`)

| Command | Exit | Result | Log |
| --- | --- | --- | --- |
| `uv run pytest tests/qualification/dup_x/test_dup_x_b4_npsc_orchestration_backend_negative_boundary.py -q` | **0** | **4 passed** | `.tmp/proof/dup-x-b4-publication/b4-negative-gates.log` |
| `uv run pytest tests/unit/runtime/architecture/test_npsc4_2_residual_compatibility_gate.py::test_npsc42_orchestration_backend_access_confined_to_allowlist -q` | **1** | **1 failed** (expected) — **10** modules · allowlist unchanged | `.tmp/proof/dup-x-b4-publication/npsc-allowlist.log` |
| `uv run pytest tests/qualification/dup_x/ -q` | **0** | **44 passed** | `.tmp/proof/dup-x-final-b6-replay/dup_x-full.log` |

## B4 disposition (implementer — not independent PASS)

| Item | Disposition |
| --- | --- |
| **B4 (DUP-X scoped)** | **PUBLISHED** — negative boundary gates green @ qual SHA |
| **NPSC-4.2 suite PASS** | **Not claimed** — `test_npsc42_orchestration_backend_access_confined_to_allowlist` remains **FAIL** |
| **Remaining formal gates** | External **B6 PASS** · **OD-DUPX-08** · roadmap §5 (not CLOSED) |

## Agent forbidden

No **B6 PASS**, no **OD-DUPX-08**, no roadmap §5 CLOSED, no NPSC allowlist weakening.
