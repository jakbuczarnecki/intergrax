# DUP-X — Protocol 03 Independent Audit Package

| Field | Value |
| --- | --- |
| **Protocol** | DUP-X-03 (mirrors EBH-6 Protocol 03) |
| **ECAP** | [ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md](ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md) **B1–B6** |
| **Execution status** | **DUP-X B4 PUBLISHED — READY FOR FINAL INDEPENDENT B6** — **not PASS** |
| **AUDITED_HEAD** | `8245b7471379414dc24385d1e770ccb56a7b0e2d` |
| **INDEPENDENT_AUDIT_BASELINE_SHA** (recommended) | `6ccddd3c40a88377e39962a300984d3b81f5c220` |
| **QUALIFICATION_EXECUTION_SHA** | `6ccddd3c40a88377e39962a300984d3b81f5c220` — **44 passed** |
| **CURRENT_PUBLICATION_HEAD** | `e04052eac6bad86d02729eee0b5b83037a91ad8e` |

## Preconditions (@ handoff)

| # | Gate | Status |
| --- | --- | --- |
| 1 | C0 git baseline | **RECORDED** — [`DUP_X_C0_BASELINE_AUDIT_RECORD.md`](DUP_X_C0_BASELINE_AUDIT_RECORD.md) |
| 2 | Closed-world inventory | **RECORDED** — 15 families · 14 classes |
| 3 | Findings + candidates | **RECORDED** — 0 open gaps @ Wave 1 · 0 unresolved DUPLICATE/BLOCKER |
| 4 | Operator OD-DUPX-02 | **APPROVED — 2026-10-29** |
| 5 | Operator OD-DUPX-01 | **OPEN** |
| 6 | Primary L2 pytest (B5) | **44 passed** @ `6ccddd3c40a88377e39962a300984d3b81f5c220` — `.tmp/proof/dup-x-b4-publication/dup_x-full.log` · historical **40** @ Block I · **36** @ Wave 1 · **26** @ first pub |
| 6c | B4 negative boundary | **4 passed** @ same SHA — `.tmp/proof/dup-x-b4-publication/b4-negative-gates.log` |
| 7 | Extended replay (optional) | NPSC-4.2: **`test_npsc42_orchestration_backend_access_confined_to_allowlist` FAIL** (10 modules) — **known · not DUP-X PASS claim** — `.tmp/proof/dup-x-b4-publication/npsc-allowlist.log` · see B4 matrix |

## Auditor checklist (B1–B6)

1. **B1 Scope completeness** — inventory families + DUP-01..14 matrix + CAND-01..15  
2. **B2 Evidence integrity** — SHA match C0 record vs `git rev-parse` @ audit  
3. **B3 Authority** — factory-first selection; registry_v2 delta; harness projection read-only  
4. **B4 Bypass / compat / NPSC ten-module boundary** — COMPAT-R3 lab fence; B4 negative gates; [`DUP_X_RESPONSIBILITY_DELTA_NPSC_ORCHESTRATION_BACKEND.md`](DUP_X_RESPONSIBILITY_DELTA_NPSC_ORCHESTRATION_BACKEND.md) · NPSC allowlist gate **still FAIL**  
5. **B5 Qual replay** — **44 passed** @ `6ccddd3c40a88377e39962a300984d3b81f5c220` (`.tmp/proof/dup-x-b4-publication/dup_x-full.log`); historical attestation in C0-04..06  
6. **B6 Verdict** — **PASS** or **REJECT** (auditor only) → future `DUP_X_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md`

## ECAP B1–B6 final handoff (@ B4 publication qual SHA)

| Block | Auditor focus | Implementer package |
| --- | --- | --- |
| **B1** | Scope — 15 families · DUP-01..14 · CAND-01..15 · Tier-3 disposition | **READY** — inventory, matrix, register @ baseline |
| **B2** | SHA ladder vs `git rev-parse` @ `6ccddd3c40a88377e39962a300984d3b81f5c220` | **READY** — C0-04..06 + [`DUP_X_PUBLICATION_MANIFEST.md`](DUP_X_PUBLICATION_MANIFEST.md) |
| **B3** | Factory-first · registry_v2 · harness projection · Class D matrix | **READY** — Wave B gates + delta dossiers |
| **B4** | COMPAT-R3 · tools_base · **NPSC 10-module responsibility** | **PUBLISHED** — negative gates **4/4 PASS** @ qual SHA · matrix in responsibility delta · **NPSC token gate FAIL** (explicit backlog) |
| **B5** | `tests/qualification/dup_x/` full replay | **READY** — **44 passed** @ qual SHA |
| **B6** | Independent PASS / REJECT | **OPEN** — auditor only → [`DUP_X_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md`](DUP_X_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md) |

## Inputs bundle

| Input | Path |
| --- | --- |
| Inventory | `DUP_X_CLOSED_WORLD_INVENTORY.md` |
| Class matrix | `DUP_X_CLASS_COVERAGE_MATRIX.md` |
| Findings | `DUP_X_FINDINGS_REGISTER.md` |
| Closure matrix | `DUP_X_CLOSURE_MATRIX.md` |
| Delta dossiers | `DUP_X_RESPONSIBILITY_DELTA_*.md` |
| Python SSOT | `tests/qualification/dup_x/` |
| Pytest log (primary) | `.tmp/proof/dup-x-b4-publication/dup_x-full.log` |
| Pytest log (B4 negative) | `.tmp/proof/dup-x-b4-publication/b4-negative-gates.log` |
| Pytest log (NPSC allowlist — expected FAIL) | `.tmp/proof/dup-x-b4-publication/npsc-allowlist.log` |
| B4 responsibility delta | `DUP_X_RESPONSIBILITY_DELTA_NPSC_ORCHESTRATION_BACKEND.md` |
| B4 evidence record | `DUP_X_B4_NPSC_NEGATIVE_BOUNDARY_EVIDENCE_RECORD.md` |
| NPSC exclusion | `DUP_X_NPSC_4_2_DUPX_EXCLUSION_RECORD.md` |

## Handoff commands

```powershell
git checkout development
git merge-base --is-ancestor 8245b7471379414dc24385d1e770ccb56a7b0e2d HEAD
git checkout 6ccddd3c40a88377e39962a300984d3b81f5c220
New-Item -ItemType Directory -Force -Path .tmp/proof/dup-x-b4-publication
uv run pytest tests/qualification/dup_x/test_dup_x_b4_npsc_orchestration_backend_negative_boundary.py -q 2>&1 | Tee-Object -FilePath .tmp/proof/dup-x-b4-publication/b4-negative-gates.log
uv run pytest tests/qualification/dup_x/ -q 2>&1 | Tee-Object -FilePath .tmp/proof/dup-x-b4-publication/dup_x-full.log
uv run pytest tests/unit/runtime/architecture/test_npsc4_2_residual_compatibility_gate.py::test_npsc42_orchestration_backend_access_confined_to_allowlist -q 2>&1 | Tee-Object -FilePath .tmp/proof/dup-x-b4-publication/npsc-allowlist.log
```

**Agent forbidden:** issue PASS record · OD-DUPX-08 · declare DUP-X CLOSED.
