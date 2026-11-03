# ENT-AUDIT-X — one-pass agent handoff (Protocol 03 / B6)

**Agent disposition:** `ENT-AUDIT-X — IMPLEMENTATION PARTIAL / EXACT BLOCKERS`  
**Do not treat as:** B6 PASS, OD approval, FRZ CLOSED, ARCH-FREEZE.

## Integration summary (R0–R4 partial)

- Fail-closed: baseline promotion requires closed evidence bundle dir; `build_evidence_manifest` requires certifying execution manifest; VBN domain report validation + SHA pin; FRZ digest uses per-check mapping; launchers reject unknown modes; CLI `promote-baseline` requires `--evidence-dir`.
- Qual harness: `tests/qualification/ent_audit_x/_manifest_fixtures.py`; unit/integration **93 passed** (gate matrix 85/85 opt-in via `ENT_AUDIT_X_GATE_MATRIX=1`).
- Script: `scripts/enterprise_audit/build_evidence_bundle.py` (C050 partial).

## External blockers (exact)

| Blocker | Detail |
| --- | --- |
| TEST_EXECUTION_SHA | Not pinned post-integration; ladder TES still historical `0cee41a8…` |
| L3 FULL 85/85 | Real semantic gates + `coverage_label==FULL` + exit 0 not demonstrated @ TES |
| ADV31 | Adversarial matrix not re-run @ TES |
| EVIDENCE_PUBLICATION_SHA / B6_ACCEPTANCE_PUBLICATION_SHA | Unset |
| Independent B6 | Protocol 03 dossier + operator OD records required |

## Reproduce qual slice @ IMPLEMENTATION_SHA

```powershell
uv run pytest tests/qualification/ent_audit_x/unit tests/qualification/ent_audit_x/integration -q `
  --ignore=tests/qualification/ent_audit_x/integration/test_semantic_gate_execution.py
```

## Matrix C (unchanged dual-axis baseline from closure SSOT)

Implementation axis remains majority **NOT_STARTED** / **PARTIAL**; qualification axis **NOT_QUALIFIED** except prior **C001/C002** doc-registry entries until §7 runs @ pinned TES.
