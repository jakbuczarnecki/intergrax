# EVOL-X E2 — Operator execution runbook (M-09H human drill)

> **BLOCKED @ cause preflight:** Human execution is **not authorized** until [EVOL_X_E2_CAUSE_ORACLE_PREFLIGHT_RECORD.md](./EVOL_X_E2_CAUSE_ORACLE_PREFLIGHT_RECORD.md) disposition is **PASS / HUMAN DRILL AUTHORIZED**. The steps below remain the prepared procedure after **E-CORR-M09H-ORACLE**.

**Policy:** OD-EVOL-08-M1 · **TES:** `dcd7ae6e264d435994c61162e0d41aba25f71283`

## Before the human session

1. Clean checkout @ TES (or documented descendant for tool only — evidence publication uses FINAL HEAD):

```powershell
cd d:\Projekty\intergrax
git fetch origin development
git checkout dcd7ae6e264d435994c61162e0d41aba25f71283
uv sync --extra dev-ci --frozen
```

2. Run [EVOL_X_E2_REGISTRY_VERIFICATION.md](./EVOL_X_E2_REGISTRY_VERIFICATION.md) pre-flight (operator). Resolve any **STOP** before involving the evaluator.

3. Confirm evaluator ≠ implementer; record `evaluator_id` / `evaluator_display_name` (not the Cursor agent). **OD-EVOL-11** auditor name is still **not** recorded — B6 remains blocked until operator updates [EVOL_X_OPERATOR_DECISION_REGISTER_W1.md](../EVOL_X_OPERATOR_DECISION_REGISTER_W1.md).

4. Do **not** give the evaluator: P03 `evol_m09_drill_case_registry.v1.json`, [AUDITOR_ONLY/EVOL_X_E2_ORACLE_ANSWER_KEY.md](./AUDITOR_ONLY/EVOL_X_E2_ORACLE_ANSWER_KEY.md), or honeytoken sheet before submissions.

## Session flow (C1 → C2 → C3 → adversarial)

For each case, in order:

| Step | Actor | Action |
|------|-------|--------|
| T1 | Operator | Run qualified tool @ TES on seed path; record UTC start/stop → `tool_seconds` |
| T2 | Operator | Build **complete diagnostic packet** (tool JSON + case cover sheet); hand to evaluator |
| T3 | Evaluator | Start **attribution** clock when packet is complete |
| T4 | Evaluator | Submit diagnosis (cause identity list) on sealed form / worksheet **before** any oracle |
| T5 | Evaluator | Stop attribution clock at submission |
| T6 | Operator | Compare submission to oracle **after** seal; record `correctness_disposition` |
| T7 | Operator | Store evidence refs (paths under `.tmp/session/evol-x-e2-m09h/<case-id>/`) |

Then run [EVOL_X_E2_ADVERSARIAL_CONTROL_PROCEDURE.md](./EVOL_X_E2_ADVERSARIAL_CONTROL_PROCEDURE.md) once (after all three diagnoses).

## Qualified tool (EVOL-M-09T-equivalent per case)

Output directory (session only):

```powershell
New-Item -ItemType Directory -Force -Path .tmp/session/evol-x-e2-m09h/tool-output | Out-Null
```

Emit diagnostic JSON (repeat per case with `$seed` / `$caseId`):

```powershell
$tes = "dcd7ae6e264d435994c61162e0d41aba25f71283"
$caseId = "C1"
$seed = "intergrax/contracts/execution_identity.py"
$out = ".tmp/session/evol-x-e2-m09h/tool-output/$caseId.json"
uv run python -c @"
import json, time
from pathlib import Path
from testing_support.change_safety.discovery.closed_world import build_closed_world_inventory
from testing_support.change_safety.graph.impact import assess_path_change_impact
from testing_support.change_safety.diagnostics.dossier import build_change_dossier

tes = '$tes'
seed = '$seed'
case_id = '$caseId'
t0 = time.perf_counter()
inv = build_closed_world_inventory(source_commit_sha=tes)
impact = assess_path_change_impact(inv, changed_paths=frozenset({seed}))
dossier = build_change_dossier(
    graph_digest=impact.graph_digest,
    inventory_digest=inv.inventory_digest,
    impact=impact,
    created_at_utc='1970-01-01T00:00:00+00:00',
)
elapsed = time.perf_counter() - t0
direct = [n.model_dump(mode='json') for n in impact.affected if n.kind == 'DIRECT']
payload = {
    'case_id': case_id,
    'test_execution_sha': tes,
    'seed_changed_path': seed,
    'tool_seconds': round(elapsed, 3),
    'impact': impact.model_dump(mode='json'),
    'dossier': dossier.model_dump(mode='json'),
}
Path('$out').write_text(json.dumps(payload, indent=2), encoding='utf-8')
print(f'tool_seconds={elapsed:.3f} wrote $out')
"@
```

**C2 seed:** `testing_support/change_safety/graph/impact.py` · **C3 seed:** `testing_support/change_safety/contracts/enums.py`

**Acceptance:** each `tool_seconds` ≤ **120** (record actual; do not round down).

## Diagnostic packet (evaluator-facing)

Per case, merge:

1. Cover: [evaluator_packets/CASE_C*_DIAGNOSTIC_PACKET.md](./evaluator_packets/)
2. Tool output JSON from `.tmp/session/evol-x-e2-m09h/tool-output/<case>.json` (impact + dossier sections Q1–Q6 as emitted)

Attribution clock (**OD-EVOL-08-M1**): starts when evaluator confirms packet complete; stops at diagnosis submission **before** oracle disclosure.

## After measurements

1. Fill [artifacts/evol_m09_human_drill_worksheet.v1.template.json](./artifacts/evol_m09_human_drill_worksheet.v1.template.json) with genuine values (replace every `PENDING_HUMAN_EXECUTION`).
2. Compute worksheet SHA-256 → `worksheet_digest`.
3. Build `evol_m09_human_drill.v2.json` from genuine timings (see [EVOL_X_E2_FINAL_EVIDENCE_ASSEMBLY.md](./EVOL_X_E2_FINAL_EVIDENCE_ASSEMBLY.md)).
4. Do **not** commit FINAL evidence until operator assembly checklist passes.

## Return to development branch

```powershell
git checkout development
```
