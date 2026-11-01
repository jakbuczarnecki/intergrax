# EVOL-X E2 — Registry verification & rebind procedure

## Published registry (TECH bundle @ HEAD)

**Path:** `docs/project/maintainers/qualification/evol_x/evol_x_p03_independent_replay/evol_m09_drill_case_registry.v1.json`

| Check | Result @ prep time |
|-------|---------------------|
| C1, C2, C3 present ordered | PASS |
| Distinct `changed_path` | PASS |
| `input_digest` matches `sha256(file @ TES)` per case | PASS @ TES `dcd7ae6e…` |
| `bound_test_execution_sha` | **`6fec9ec387089dfcee1039089ba5ad901fab39d6` (stale vs active TES)** |
| `registry_digest` (TECH bytes) | `e941fd5c75b04f70c3cd99cc41f00f4857c80f4b489932689035ebfdce23c056` |
| `population_digest` (TECH registry) | `7a2c9a3617f5512741e8032ecfb40ef34356dd041ddc219293203486df779212` |

## Rebind staging (FINAL evidence only)

**File:** [artifacts/evol_m09_drill_case_registry.v1.rebind_staging.json](./artifacts/evol_m09_drill_case_registry.v1.rebind_staging.json)

| Field | Staged value |
|-------|----------------|
| `bound_test_execution_sha` | `dcd7ae6e264d435994c61162e0d41aba25f71283` |
| Cases C1–C3 | Unchanged paths, `input_digest`, `expected_cause_identities` vs TECH |

**Staging digest (prep publication):** run at operator FINAL assembly:

```powershell
uv run python -c "from pathlib import Path; from testing_support.change_safety.evidence import p03; b=Path('docs/project/maintainers/qualification/evol_x/e2_m09h_human_drill_prep/artifacts/evol_m09_drill_case_registry.v1.rebind_staging.json').read_bytes(); print(p03.sha256_hex(b))"
```

Expected after write: compute at commit time in [EVOL_X_E2_PREPARATION_RECORD.md](./EVOL_X_E2_PREPARATION_RECORD.md).

## Rebind procedure (no circular SHA)

1. Human drill completes on genuine measurements @ TES checkout.
2. On **EVIDENCE_FINAL** commit branch: copy staging registry → `evol_x_p03_independent_replay/evol_m09_drill_case_registry.v1.json`.
3. Add `evol_m09_human_drill.v2.json`, worksheet, update `evol_metrics.v1.json` M-09H → 3/3 MEASURED PASS with `population_digest` from **rebound** registry.
4. Regenerate `ARTIFACT_DIGESTS.json` + `MANIFEST.json` artifact_paths (include drill + worksheet + registry).
5. `source_commit_sha` fields in new artifacts = TES; evidence HEAD = FINAL publication SHA (ancestor chain via normal P03 rules).
6. Do **not** amend TECH SHA `9f875fc62d1f6b2231b31c843ae0428a2d871031`.

## Verification commands

```powershell
# input_digest @ TES
uv run python -c "
import json, hashlib, subprocess
from pathlib import Path
from testing_support.change_safety.evidence import p03
TES='dcd7ae6e264d435994c61162e0d41aba25f71283'
reg=json.loads(Path(p03.BUNDLE_REL,p03.M09_DRILL_CASE_REGISTRY_NAME).read_text())
for c in reg['cases']:
    raw=subprocess.run(['git','show',f'{TES}:{c[\"changed_path\"]}'],capture_output=True).stdout
    ok=hashlib.sha256(raw).hexdigest()==c['input_digest']
    print(c['case_id'], 'input_digest', ok)
"

# Schema
uv run python -c "
import json
from pathlib import Path
from testing_support.change_safety.contracts.artifacts import M09DrillCaseRegistry
from testing_support.change_safety.evidence import p03
for label, rel in [('TECH', Path(p03.BUNDLE_REL)/p03.M09_DRILL_CASE_REGISTRY_NAME), ('STAGING', Path('docs/project/maintainers/qualification/evol_x/e2_m09h_human_drill_prep/artifacts/evol_m09_drill_case_registry.v1.rebind_staging.json'))]:
    M09DrillCaseRegistry.model_validate(json.loads(rel.read_text()))
    print(label, 'schema OK')
"
```
