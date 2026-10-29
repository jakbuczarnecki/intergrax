# DUP-X — Protocol 03 Independent Audit Package

| Field | Value |
| --- | --- |
| **Protocol** | DUP-X-03 (mirrors EBH-6 Protocol 03) |
| **ECAP** | [ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md](ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md) **B1–B6** |
| **Execution status** | **ISSUED — PASS (SCOPED)** @ `a7d83925b…` — parent **NOT CLOSED** until **OD-DUPX-08** |
| **AUDITED_HEAD** | `8245b7471379414dc24385d1e770ccb56a7b0e2d` |
| **INDEPENDENT_AUDIT_BASELINE_SHA** (recommended) | `a7d83925b375b5f08eb0f175ced2c17fa026bba9` |
| **QUALIFICATION_EXECUTION_SHA** | `a7d83925b375b5f08eb0f175ced2c17fa026bba9` — **44 passed** |
| **CURRENT_PUBLICATION_HEAD** | `1f782aaa9017b31b2e68a1490d186a1e3fa055cd` (pre–B6 PASS publication) |

## Preconditions (@ handoff)

| # | Gate | Status |
| --- | --- | --- |
| 1 | C0 git baseline | **RECORDED** — [`DUP_X_C0_BASELINE_AUDIT_RECORD.md`](DUP_X_C0_BASELINE_AUDIT_RECORD.md) |
| 2 | Closed-world inventory | **RECORDED** — 15 families · 14 classes |
| 3 | Findings + candidates | **RECORDED** — 0 open gaps @ Wave 1 · 0 unresolved DUPLICATE/BLOCKER |
| 4 | Operator OD-DUPX-02 | **APPROVED — 2026-10-29** |
| 5 | Operator OD-DUPX-01 | **OPEN — NOT TRIGGERED** |
| 6 | Primary L2 pytest (B5) | **44 passed** @ `a7d83925b375b5f08eb0f175ced2c17fa026bba9` — `.tmp/proof/dup-x-final-b6-replay/dup_x-full.log` |
| 6c | B4 negative boundary | **4 passed** — `.tmp/proof/dup-x-final-b6-replay/b4-negative-gates.log` |
| 7 | Extended replay (optional) | NPSC-4.2 allowlist test **FAIL** (10 modules) — `.tmp/proof/dup-x-final-b6-replay/npsc-allowlist-expected-fail.log` — **not** DUP-X NPSC PASS |

## Auditor checklist (B1–B6)

1. **B1 Scope completeness** — inventory families + DUP-01..14 matrix + CAND-01..15  
2. **B2 Evidence integrity** — SHA match C0 record vs `git rev-parse` @ audit  
3. **B3 Authority** — factory-first selection; registry_v2 delta; harness projection read-only  
4. **B4 Bypass / compat / NPSC ten-module boundary** — COMPAT-R3 lab fence; B4 negative gates; [`DUP_X_RESPONSIBILITY_DELTA_NPSC_ORCHESTRATION_BACKEND.md`](DUP_X_RESPONSIBILITY_DELTA_NPSC_ORCHESTRATION_BACKEND.md) · NPSC allowlist gate **still FAIL**  
5. **B5 Qual replay** — **44 passed** @ `a7d83925b375b5f08eb0f175ced2c17fa026bba9` (`.tmp/proof/dup-x-final-b6-replay/dup_x-full.log`); historical attestation in C0-04..07  
6. **B6 Verdict** — **PASS** or **REJECT** (auditor only) → future `DUP_X_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md`

## ECAP B1–B6 final handoff (@ `a7d83925b375b5f08eb0f175ced2c17fa026bba9`)

| Block | Auditor focus | Implementer package |
| --- | --- | --- |
| **B1** | Scope — 15 families · DUP-01..14 · CAND-01..15 · Tier-3 disposition | **READY** |
| **B2** | SHA ladder vs `git rev-parse` @ pin | **READY** — C0-04..07 + manifest |
| **B3** | Factory-first · registry_v2 · harness projection · Class D matrix | **READY** |
| **B4** | COMPAT-R3 · NPSC ten-module matrix · negative gates | **READY** — NPSC allowlist **FAIL** documented |
| **B5** | Full `tests/qualification/dup_x/` replay | **READY** — **44 passed** @ pin |
| **B6** | Independent PASS / REJECT | **ISSUED — PASS (SCOPED)** — [`DUP_X_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md`](DUP_X_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md) |

## Inputs bundle

| Input | Path |
| --- | --- |
| Inventory | `DUP_X_CLOSED_WORLD_INVENTORY.md` |
| Class matrix | `DUP_X_CLASS_COVERAGE_MATRIX.md` |
| Findings | `DUP_X_FINDINGS_REGISTER.md` |
| Closure matrix | `DUP_X_CLOSURE_MATRIX.md` |
| Delta dossiers | `DUP_X_RESPONSIBILITY_DELTA_*.md` |
| Python SSOT | `tests/qualification/dup_x/` |
| Pytest log (primary) | `.tmp/proof/dup-x-final-b6-replay/dup_x-full.log` |
| Pytest log (L3) | `.tmp/proof/dup-x-final-b6-replay/l3-parent.log` |
| Pytest log (B4 negative) | `.tmp/proof/dup-x-final-b6-replay/b4-negative-gates.log` |
| Pytest log (NPSC allowlist — expected FAIL) | `.tmp/proof/dup-x-final-b6-replay/npsc-allowlist-expected-fail.log` |
| B4 responsibility delta | `DUP_X_RESPONSIBILITY_DELTA_NPSC_ORCHESTRATION_BACKEND.md` |
| B4 evidence record | `DUP_X_B4_NPSC_NEGATIVE_BOUNDARY_EVIDENCE_RECORD.md` |
| NPSC exclusion | `DUP_X_NPSC_4_2_DUPX_EXCLUSION_RECORD.md` |

## Handoff commands

```powershell
git switch development
git merge-base --is-ancestor a7d83925b375b5f08eb0f175ced2c17fa026bba9 HEAD
# Optional exact pin: git worktree add ../intergrax-dupx-a7d839 a7d83925b375b5f08eb0f175ced2c17fa026bba9
New-Item -ItemType Directory -Force -Path .tmp/proof/dup-x-final-b6-replay
uv run pytest tests/qualification/dup_x/test_dup_x_b4_npsc_orchestration_backend_negative_boundary.py -q 2>&1 | Tee-Object -FilePath .tmp/proof/dup-x-final-b6-replay/b4-negative-gates.log
uv run pytest tests/qualification/dup_x/test_dup_x_parent_closure_mechanical_gates.py tests/qualification/dup_x/test_dup_x_mechanical_exit_gates.py -q 2>&1 | Tee-Object -FilePath .tmp/proof/dup-x-final-b6-replay/l3-parent.log
uv run pytest tests/qualification/dup_x/ -q 2>&1 | Tee-Object -FilePath .tmp/proof/dup-x-final-b6-replay/dup_x-full.log
uv run pytest tests/unit/runtime/architecture/test_npsc4_2_residual_compatibility_gate.py::test_npsc42_orchestration_backend_access_confined_to_allowlist -q 2>&1 | Tee-Object -FilePath .tmp/proof/dup-x-final-b6-replay/npsc-allowlist-expected-fail.log
```

**B6:** [`DUP_X_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md`](DUP_X_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md) — **ISSUED — PASS (SCOPED)** (external auditor **ChatGPT**).

**Agent forbidden:** OD-DUPX-08 · declare DUP-X **CLOSED** · start **ROADMAP-REPLAY-X**.
