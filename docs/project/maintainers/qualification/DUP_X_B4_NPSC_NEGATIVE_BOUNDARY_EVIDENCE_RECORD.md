# DUP-X — B4 NPSC negative boundary evidence record

| Field | Value |
| --- | --- |
| **Record type** | Independent B4 — orchestration backend ten-module boundary |
| **QUALIFICATION_BASELINE** | `0e19bc28a6dc7700c633e9f2179974bb9238fbe6` |
| **Evidence authoring HEAD** | `3feaee3d11d65d08a7860715e26bb38e763fcd40` (local `development` @ record time — **not committed** unless operator authorizes) |
| **Branch** | `development` |
| **Production code delta** | **0** (default) — qual + docs only in this slice |
| **NPSC gate modified** | **No** |

## Artifacts

| Artifact | Path |
| --- | --- |
| Ten-module matrix + B4 rationale | [`DUP_X_RESPONSIBILITY_DELTA_NPSC_ORCHESTRATION_BACKEND.md`](DUP_X_RESPONSIBILITY_DELTA_NPSC_ORCHESTRATION_BACKEND.md) |
| NPSC exclusion register | [`DUP_X_NPSC_4_2_DUPX_EXCLUSION_RECORD.md`](DUP_X_NPSC_4_2_DUPX_EXCLUSION_RECORD.md) |
| Adversarial negative gates | `tests/qualification/dup_x/test_dup_x_b4_npsc_orchestration_backend_negative_boundary.py` |

## Test execution (@ Evidence authoring HEAD)

| Command | Result |
| --- | --- |
| `uv run pytest tests/qualification/dup_x/test_dup_x_b4_npsc_orchestration_backend_negative_boundary.py -q` | **4 passed** — `.tmp/session/dup-x-b4-npsc-negative-boundary/pytest.log` |
| `uv run pytest tests/unit/runtime/architecture/test_npsc4_2_residual_compatibility_gate.py::test_npsc42_orchestration_backend_access_confined_to_allowlist -q` | **1 failed** (expected) — exactly **10** modules · allowlist unchanged |
| `uv run pytest tests/qualification/dup_x/ -q` | **44 passed** @ evidence HEAD — `.tmp/session/dup-x-b4-npsc-negative-boundary/dup_x-full.log` |

## B4 disposition (implementer — not independent PASS)

| Item | Disposition |
| --- | --- |
| **B4 PASS (DUP-X scoped)** | **READY** — negative boundary gates green; no production bypass found |
| **NPSC-4.2 test PASS** | **Not claimed** — separate allowlist program |
| **Remaining real blocker** | **None for DUP-X B4** · external B6 must accept NPSC exclusion + replay negative gates @ baseline SHA |

## Agent forbidden

No **B6 PASS**, no **OD-DUPX-08**, no roadmap §5 CLOSED, no NPSC allowlist weakening.
