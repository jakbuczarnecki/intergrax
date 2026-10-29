# DUP-X — Protocol 03 Independent Audit Package

| Field | Value |
| --- | --- |
| **Protocol** | DUP-X-03 (mirrors EBH-6 Protocol 03) |
| **ECAP** | [ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md](ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md) **B1–B6** |
| **Execution status** | **READY FOR INDEPENDENT REVIEW** — **not PASS** |
| **AUDITED_HEAD** | `8245b7471379414dc24385d1e770ccb56a7b0e2d` |

## Preconditions (@ handoff)

| # | Gate | Status |
| --- | --- | --- |
| 1 | C0 git baseline | **RECORDED** — [`DUP_X_C0_BASELINE_AUDIT_RECORD.md`](DUP_X_C0_BASELINE_AUDIT_RECORD.md) |
| 2 | Closed-world inventory | **RECORDED** — 15 families · 14 classes |
| 3 | Findings + candidates | **RECORDED** — 2 gaps · 0 unresolved DUPLICATE/BLOCKER |
| 4 | Operator OD-DUPX-02 | **OPEN** |
| 5 | Operator OD-DUPX-01 | **OPEN** |
| 6 | Primary L2 pytest log | `.tmp/proof/dup-x-wave-d/run.log` — **`tests/qualification/dup_x/` → 26 passed** @ HEAD |
| 7 | Extended replay (optional) | EBH-6 B2: **17 passed**; NPSC-4.2 allowlist: **1 failed** @ HEAD (pre-existing drift — **not** DUP-X regression; track under architecture gate, not DUP-X closure) |

## Auditor checklist (B1–B6)

1. **B1 Scope completeness** — inventory families + DUP-01..14 matrix + CAND-01..15  
2. **B2 Evidence integrity** — SHA match C0 record vs `git rev-parse` @ audit  
3. **B3 Authority** — factory-first selection; registry_v2 delta; harness projection read-only  
4. **B4 Bypass / compat** — COMPAT-R3 lab fence; tools_base import gate  
5. **B5 Qual replay** — primary: `tests/qualification/dup_x/` (log); optional: EBH-6 B2; NPSC-4.2 noted separately if still failing @ audit SHA  
6. **B6 Verdict** — **PASS** or **REJECT** (auditor only) → future `DUP_X_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md`

## Inputs bundle

| Input | Path |
| --- | --- |
| Inventory | `DUP_X_CLOSED_WORLD_INVENTORY.md` |
| Class matrix | `DUP_X_CLASS_COVERAGE_MATRIX.md` |
| Findings | `DUP_X_FINDINGS_REGISTER.md` |
| Closure matrix | `DUP_X_CLOSURE_MATRIX.md` |
| Delta dossiers | `DUP_X_RESPONSIBILITY_DELTA_*.md` |
| Python SSOT | `tests/qualification/dup_x/` |
| Pytest log | `.tmp/proof/dup-x-wave-d/run.log` |

## Handoff commands

```powershell
git checkout development
git merge-base --is-ancestor 8245b7471379414dc24385d1e770ccb56a7b0e2d HEAD
New-Item -ItemType Directory -Force -Path .tmp/proof/dup-x-wave-d
uv run pytest tests/qualification/dup_x/ -q 2>&1 | Tee-Object -FilePath .tmp/proof/dup-x-wave-d/run.log
# Optional extended replay (NPSC may fail @ HEAD — document; not DUP-X blocker):
uv run pytest tests/qualification/ebh_6/test_ebh_6_b2_layer_closed_world_integrity.py -q
uv run pytest tests/unit/runtime/architecture/test_npsc4_2_residual_compatibility_gate.py -q
```

**Agent forbidden:** issue PASS record · OD-DUPX-08 · declare DUP-X CLOSED.
