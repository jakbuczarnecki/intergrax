# NPSC-4.2-H3 — Raw NexusLoop Import Ownership Reconciliation

**Task:** NPSC-4.2-H3  
**Verdict:** PASS  
**Branch:** `development`

## Offending importers

| File | Raw NexusLoop import | Role | Allowlisted | True composition owner? |
| --- | ---: | --- | ---: | ---: |
| `intergrax/applications/_shared/compensation_side_effect_wiring.py` | YES (before fix) | U2 compensation admission wiring | NO | NO |
| `intergrax/applications/_shared/production_delegated_subtask_child_execution_wiring.py` | YES (before fix) | U4 child execution binding factory | NO | NO |

Gate: `test_npsc42_raw_nexus_imports_remain_composition_owner_allowlist`.

## Classification

| File | Classification |
| --- | --- |
| `compensation_side_effect_wiring.py` | **INVALID_CONCRETE_DEPENDENCY** |
| `production_delegated_subtask_child_execution_wiring.py` | **INVALID_CONCRETE_DEPENDENCY** |

## Ownership reasoning

### `compensation_side_effect_wiring.py`

- Does **not** construct or assemble `NexusLoop`.
- Consumed three Nexus-owned surfaces (`execution_budget_ledger_factory`, `run_budget`, `execution_lineage_persistence`) and forwarded them into `build_runtime_compensation_side_effect_execution`.
- That is runtime-stack intake wiring, not Nexus composition ownership.

### `production_delegated_subtask_child_execution_wiring.py`

- Docstring names a composition root for **child execution port binding**, not for **NexusLoop** construction.
- Optional `nexus_loop` parameter only read `run_budget`; no production caller passed `nexus_loop`.
- Budget alignment belongs at the caller that already owns Nexus run budget — not via a concrete `NexusLoop` parameter on this adapter.

Neither file satisfies the allowlist bar: conscious, limited concrete Nexus wiring at the harness/platform composition boundary.

## Chosen fix

| File | Fix |
| --- | --- |
| `compensation_side_effect_wiring.py` | Remove `NexusLoop` import; accept `ExecutionBudgetLedgerFactory`, `RunBudget`, and `ExecutionLineagePersistence` as explicit composition inputs (same types already required by `build_runtime_compensation_side_effect_execution`). |
| `production_delegated_subtask_child_execution_wiring.py` | Remove `NexusLoop` import and `nexus_loop` parameter; retain `run_budget` / `ledger` for composition-owner-supplied budget alignment. |

## Port reused / created

| | |
| --- | --- |
| **NEW PORT CREATED** | NO |
| **REUSED PORT / CONTRACT** | Existing runtime builder intake: `build_runtime_compensation_side_effect_execution` keyword contract; `RunBudget`, `ExecutionBudgetLedgerFactory`, `ExecutionLineagePersistence` |

### Why existing ports were sufficient

No new Protocol was required:

- Compensation wiring only needed the same three dependencies the runtime builder already declares — exposing them at the Tier-3 wiring boundary removes the illicit `NexusLoop` façade without duplicating Nexus API.
- Child execution wiring already accepted `run_budget` directly; the unused `nexus_loop` shortcut was redundant concrete coupling.

## Allowlist change

**NO** — offenders were not legitimate composition owners; allowlist remains minimal and literal.

## Production bypass scan (report only)

Scanned `intergrax/applications/_shared/**` and `applications/**/host/**` for raw `NexusLoop`, orchestration-backend access, and local bootstrap bypass patterns.

- After fix: unauthorized raw `NexusLoop` imports under `_shared` = **0** (gate-enforced).
- Remaining raw imports in `_shared` match `_ALLOWED_RAW_NEXUS_IMPORTERS` only.
- No new direct child-execution or orchestration bypass introduced by this task.
- Unrelated pre-existing host/runtime seams outside this task scope were not modified.

## Test evidence

- `tests/unit/runtime/architecture/test_npsc4_2_residual_compatibility_gate.py` — PASS (full file)
- Frozen matrix suites per session run (EE-A1, EE-A2, NPSC-4.2, NPSC-5E Final, NPSC-5F Final, W5-H1, W5-H1-FIX1, EE-B1.1) — see commit session log
- `tests/unit/runtime/execution/test_compensation_side_effect_admission.py` — updated U2 wiring test (explicit invoker-only call)

## Final verdict

**PASS** — unauthorized raw `NexusLoop` imports in shared application wiring eliminated; ownership model preserved; no duplicate Nexus/scheduling; no allowlist inflation.
