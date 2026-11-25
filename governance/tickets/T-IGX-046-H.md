# T-IGX-046-H — PT-G01 G2-only: registry bypass maintenance gate AST (EC-02/P1)

## Scope

ARCH-FREEZE slice **G2_ONLY** under master-plan node **PT-G01** (gate-meaning change, NONPRODUCT). Operator decision **OD-EC02-P1** assigns this ticket ID for the first implementable PT-G01 batch: rewrite `scripts/maintenance/check_agent_registry_bypass.py` from regex to AST enforcement of **DP-14** (`intergrax.integrations.contracts` leaves only under `agents/`; deny other `intergrax.integrations.*` and `intergrax.tools.*`). Add `tests/unit/maintenance/test_check_agent_registry_bypass_ast.py`.

Finding context: EQOA-A4-004 / X-A4-004 — six historical contract-leaf imports false-positive under the regex gate at the ARCH-FREEZE train baseline (`68db9c0ef6207b023388eecc460644698a92b895` read for EC-02/P1). Closure of the finding is not claimed by this ticket draft.

**Not authority for this ticket:** `T-IGX-046-F` on `development` (DB-C documentation scope and approval).

**Sibling scaffold:** `T-IGX-046-F` on `arch-freeze/rw-train` remains the broader PT-G01 W2 node; this ticket carries only the G2 slice.

**G1 / Q-WA-01:** `scripts/maintenance/check_intergrax_no_applications_imports.py` is unchanged; fifteen product paths outside G1 scan roots remain **OWNER_SCOPED_RESIDUAL**.

Sensitivity: NONPRODUCT. Target branch: `arch-freeze/rw-train`.

Implementation write set (firm, post-`plan_approved`):

```text
scripts/maintenance/check_agent_registry_bypass.py
tests/unit/maintenance/test_check_agent_registry_bypass_ast.py
```

Governance write set (this batch):

```text
governance/tickets/T-IGX-046-H.md
governance/tickets/plans/PLAN_T-IGX-046-H.md
```

Nothing in this ticket claims PASS, FRZ promotion, independent verification, or EQO-X-01 closure.

## Plan digest

Plan digest: de29ed025120a756eff4fe4302198c304d3fbf4e8c9ed4bfca2c272538149374

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-046-H.md`](plans/PLAN_T-IGX-046-H.md). The digest was computed with `scripts/governance/plan_digest.py`; it must be recomputed after any edit of the plan.

## Test contract

Nothing was run when this ticket was published to governance (governance-prep only). After `plan_approved`, one process at a time under `D:/Projekty/_wt/_locks/pytest.lock`; exact order and commands are in plan section 6:

- `tests/unit/maintenance/test_check_agent_registry_bypass_ast.py` (all AST ALLOW/DENY rows and syntax fail-closed via `check_imports_in_tree` on `tmp_path` fixtures)
- `tests/unit/agents/test_wa_agent_import_rule.py::test_wa_13_agents_tree_uses_only_integration_contract_leaves`
- `tests/unit/agents/test_wa_agent_import_rule.py::test_wa_17_old_gate_pattern_matches_only_contract_leaf_files`
- `uv run python scripts/maintenance/check_agent_registry_bypass.py` (expect exit `0` after implementation)
- `uv run python scripts/maintenance/check_intergrax_no_applications_imports.py` (regression only; script unchanged)
- `uv run python -m scripts.governance.igx_gov ticket` before push

`intergrax.applications` import negatives are not part of this test contract (G1 obligation).

## Out of scope

Product paths (`intergrax/`, `agents/`, `applications/`), `scripts/maintenance/check_intergrax_no_applications_imports.py`, `.github/`, `pyproject.toml`, `uv.lock`, FRZ and roadmap, WB05, ENT-AUDIT, EQO closure matrix, EBH-3 scan extension, seam ledger, other PT-G01 scripts, EC-02/P2 and EC-03.

## Approved

Approved: operator @ 2026-11-24T18:00:00+01:00

Authorization: Enterprise Campaign V3 §19 (S-W03a). Operator instructed execution of the canonical write-set `T-IGX-046-H` (AST registry bypass gate, DP-14) on `arch-freeze/rw-train`; decision recorded on 2026-11-24 in the campaign session.

## Lifecycle state

plan_approved — S-W03a implementation authorized 2026-11-24 per §19; G6 / independent PASS not claimed by this ticket.