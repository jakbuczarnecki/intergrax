# DUP-X — Protocol 03 Independent Audit Package

| Field | Value |
| --- | --- |
| **Protocol** | DUP-X-03 (mirrors EBH-6 Protocol 03) |
| **ECAP** | [ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md](ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md) **B1–B6** |
| **Execution status** | **FINAL EVIDENCE RECONCILED — READY FOR INDEPENDENT REVIEW** — **not PASS** |
| **AUDITED_HEAD** | `8245b7471379414dc24385d1e770ccb56a7b0e2d` |
| **CURRENT_PUBLICATION_HEAD** | `0159c2a18b39547bcd60823489c1a4bea1229b38` |

## Preconditions (@ handoff)

| # | Gate | Status |
| --- | --- | --- |
| 1 | C0 git baseline | **RECORDED** — [`DUP_X_C0_BASELINE_AUDIT_RECORD.md`](DUP_X_C0_BASELINE_AUDIT_RECORD.md) |
| 2 | Closed-world inventory | **RECORDED** — 15 families · 14 classes |
| 3 | Findings + candidates | **RECORDED** — 0 open gaps @ Wave 1 · 0 unresolved DUPLICATE/BLOCKER |
| 4 | Operator OD-DUPX-02 | **OPEN** |
| 5 | Operator OD-DUPX-01 | **OPEN** |
| 6 | Primary L2 pytest (B5) | **Wave 1 primary:** **36 passed** @ `f188a496f969c99e66ca9d409cd29ddceeec8f0d` — `.tmp/proof/dup-x-wave-1-post-publish/run.log` · **reconciliation replay:** **36 passed** @ `0159c2a18b39547bcd60823489c1a4bea1229b38` — `.tmp/proof/dup-x-03-final-evidence/run.log` |
| 6b | Historical L2 (first publication) | **26 passed** @ `5c97fb679a3220ff7a156ca6ab7fd7fe8c5b5176` — `.tmp/proof/dup-x-wave-d/run.log` |
| 7 | Extended replay (optional) | EBH-6 B2: **17 passed**; NPSC-4.2: **26 passed / 1 failed** (`test_npsc42_orchestration_backend_access_confined_to_allowlist`) — [`DUP_X_NPSC_4_2_DUPX_EXCLUSION_RECORD.md`](DUP_X_NPSC_4_2_DUPX_EXCLUSION_RECORD.md) · **excluded from DUP-X mandatory** |

## Auditor checklist (B1–B6)

1. **B1 Scope completeness** — inventory families + DUP-01..14 matrix + CAND-01..15  
2. **B2 Evidence integrity** — SHA match C0 record vs `git rev-parse` @ audit  
3. **B3 Authority** — factory-first selection; registry_v2 delta; harness projection read-only  
4. **B4 Bypass / compat** — COMPAT-R3 lab fence; tools_base import gate  
5. **B5 Qual replay** — primary: **36** @ Wave 1 `WAVE1_TEST_EXECUTION_SHA`; historical **26** @ first-pub `TEST_EXECUTION_SHA`; reconciliation log @ `CURRENT_PUBLICATION_HEAD`  
6. **B6 Verdict** — **PASS** or **REJECT** (auditor only) → future `DUP_X_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md`

## ECAP B1–B6 handoff (@ final evidence reconciliation)

| Block | Auditor focus | Agent reconciliation status |
| --- | --- | --- |
| **B1** | Scope — 15 families · DUP-01..14 · CAND-01..15 | **READY** — inventory + matrix committed |
| **B2** | SHA ladder vs `git rev-parse` | **READY** — [`DUP_X_C0_BASELINE_AUDIT_RECORD.md`](DUP_X_C0_BASELINE_AUDIT_RECORD.md) C0-04 + [`DUP_X_PUBLICATION_MANIFEST.md`](DUP_X_PUBLICATION_MANIFEST.md) |
| **B3** | Factory-first · registry_v2 · harness projection | **READY** — Wave B/C qual gates green @ HEAD |
| **B4** | COMPAT-R3 · tools_base | **READY** — closure matrix **RATIFIED** / **PASS** rows |
| **B5** | Qual replay | **READY** — 36 primary + 26 historical attested; reconciliation **36 passed** @ `0159c2a…` |
| **B6** | Independent PASS | **OPEN** — auditor only; OD-DUPX-02 · OD-DUPX-01 remain **OPEN** |

## Inputs bundle

| Input | Path |
| --- | --- |
| Inventory | `DUP_X_CLOSED_WORLD_INVENTORY.md` |
| Class matrix | `DUP_X_CLASS_COVERAGE_MATRIX.md` |
| Findings | `DUP_X_FINDINGS_REGISTER.md` |
| Closure matrix | `DUP_X_CLOSURE_MATRIX.md` |
| Delta dossiers | `DUP_X_RESPONSIBILITY_DELTA_*.md` |
| Python SSOT | `tests/qualification/dup_x/` |
| Pytest log (primary) | `.tmp/proof/dup-x-wave-1-post-publish/run.log` |
| Pytest log (reconciliation) | `.tmp/proof/dup-x-03-final-evidence/run.log` |
| NPSC exclusion | `DUP_X_NPSC_4_2_DUPX_EXCLUSION_RECORD.md` |

## Handoff commands

```powershell
git checkout development
git merge-base --is-ancestor 8245b7471379414dc24385d1e770ccb56a7b0e2d HEAD
New-Item -ItemType Directory -Force -Path .tmp/proof/dup-x-03-final-evidence
uv run pytest tests/qualification/dup_x/ -q 2>&1 | Tee-Object -FilePath .tmp/proof/dup-x-03-final-evidence/run.log
uv run pytest tests/qualification/dup_x/test_dup_x_parent_closure_mechanical_gates.py -q 2>&1 | Tee-Object -Append -FilePath .tmp/proof/dup-x-03-final-evidence/run.log
# Optional extended replay (NPSC may fail @ HEAD — document; not DUP-X blocker):
uv run pytest tests/qualification/ebh_6/test_ebh_6_b2_layer_closed_world_integrity.py -q
uv run pytest tests/unit/runtime/architecture/test_npsc4_2_residual_compatibility_gate.py -q
```

**Agent forbidden:** issue PASS record · OD-DUPX-08 · declare DUP-X CLOSED.
