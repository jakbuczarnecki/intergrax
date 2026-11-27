# PLAN_T-IGX-090-D — Wave 2 Round 2 Pre-Freeze SFX & `SCENARIO-READINESS-X` Prep (`production delta = 0`)

| Field | Value |
|---|---|
| **Ticket** | [`governance/tickets/T-IGX-090-D.md`](../T-IGX-090-D.md) |
| **Authority** | `OP-EC-01` (`OD-W2-SFX-PREP=A`, Operator approved `2026-11-26T17:26:01+01:00`) |
| **Branch** | `ec-w2/sfx-rdy-prep-090d` (isolated worktree) |
| **Product Code Delta** | **`0`** (docs, proposed JSON design schemas, and governance ticket/plan files only) |

---

## 1. Objective

Execute the approved `production delta = 0` pre-freeze Wave 6 preparation package under `OP-EC-01` / `OD-W2-SFX-PREP=A`:

1. Remediate all 7 MEDIUM + 5 LOW independent-review (`W2-SFX-REV`) findings across `ADR-SFX-001`..`ADR-SFX-006` while keeping all 6 ADR statuses **`Proposed`**.
2. Close `WT-12` across all 13 companion R1 proposed JSON schemas (`docs/project/maintainers/qualification/scenario_factory_x/design/schemas/*.schema.json`) by adding CF-7 ordering / set-semantics descriptions to all 30 undeclared arrays and tightening the 4 sort-only arrays.
3. Publish the permanent independent review record (`SFX_ADR_001_006_INDEPENDENT_REVIEW_RECORD.md`, including `GEN_SET.json` scratch-relocation custody hygiene note) and the Wave 6 Entry Operator Decision Packet (`SFX_WAVE6_ENTRY_DECISION_PACKET.md`).
4. Scaffold `SCENARIO-READINESS-X` P0 canonical execution plan (`governance/tickets/plans/PLAN_T-IGX-RDY-A.md`) and draft ticket (`governance/tickets/T-IGX-RDY-A.md`, `Lifecycle state: plan_pending`, NO fabricated post-freeze `Approved:` line) so Wave 6 starts with zero cold-start delay upon `ARCH-FREEZE = CLOSED`.
5. Record this executing plan (`PLAN_T-IGX-090-D.md`) and ticket (`T-IGX-090-D.md`).

---

## 2. Explicit Edit Scope (Zero Overlap with `AF-CLOSE` or `W2-B1/B2/B3`)

### Modified Files (19)
- `docs/project/technical/adr/entries/2026-11-12/ADR-SFX-001.md`
- `docs/project/technical/adr/entries/2026-11-12/ADR-SFX-002.md`
- `docs/project/technical/adr/entries/2026-11-12/ADR-SFX-003.md`
- `docs/project/technical/adr/entries/2026-11-12/ADR-SFX-004.md`
- `docs/project/technical/adr/entries/2026-11-12/ADR-SFX-005.md`
- `docs/project/technical/adr/entries/2026-11-12/ADR-SFX-006.md`
- 13 companion R1 schemas under `docs/project/maintainers/qualification/scenario_factory_x/design/schemas/`:
  1. `sfx.business_problem_statement.v1.schema.json`
  2. `sfx.discovery_record.v1.schema.json`
  3. `sfx.business_scenario_specification.v1.schema.json`
  4. `sfx.scenario_blueprint.v1.schema.json`
  5. `sfx.integration_setup_plan.v1.schema.json`
  6. `sfx.scenario_test_plan.v1.schema.json`
  7. `sfx.scenario_run_report.v1.schema.json`
  8. `sfx.scenario_value_report.v1.schema.json`
  9. `sfx.scenario_catalog_entry.v1.schema.json`
  10. `sfx.sfx_session_state.v1.schema.json`
  11. `sfx.scenario_lifecycle_record.v1.schema.json`
  12. `sfx.scenario_qualification_record.v1.schema.json`
  13. `sfx.scenario_admission_dossier.v1.schema.json`

### Created Files (6)
- `docs/project/maintainers/qualification/scenario_factory_x/SFX_ADR_001_006_INDEPENDENT_REVIEW_RECORD.md`
- `docs/project/maintainers/qualification/scenario_factory_x/SFX_WAVE6_ENTRY_DECISION_PACKET.md`
- `governance/tickets/plans/PLAN_T-IGX-RDY-A.md`
- `governance/tickets/T-IGX-RDY-A.md`
- `governance/tickets/plans/PLAN_T-IGX-090-D.md`
- `governance/tickets/T-IGX-090-D.md`

---

## 3. Hard Exclusions & Non-Claims

- **Zero product delta:** NO edits to `intergrax/`, `agents/`, `applications/`, `scripts/`, `.github/`, `tests/`.
- **Zero overlap with other tracks:** NO edits to `PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md`, `PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md`, `arch_freeze_program/`, `enterprise_debt_register/`, `eqo_closure/`, or any `W2-B1/B2/B3` files.
- **Zero `pytest` execution:** Deterministic schema/doc/governance checks only so pytest lock stays free for parallel test writers.
- **No unauthorized declarations:** NEVER declare `CLOSED`, `IIAC PASS`, `FRZ PASS`, `G6 PASS`, or `SFX_PRE_IMPLEMENTATION_READY`. All 6 `ADR-SFX-*` remain `Proposed`. No `git push`, no merge to `development`.

---

## 4. Verification Contract (Serial Execution, Zero `pytest`)

1. Deterministic Python validation of all 13 edited `.schema.json` files (`json.loads` + `jsonschema.Draft202012Validator.check_schema` if installed).
2. `uv run python scripts/docs/check_docs_domain_pairs.py`
3. `uv run python -m scripts.governance.plan_digest governance/tickets/plans/PLAN_T-IGX-RDY-A.md` and `governance/tickets/plans/PLAN_T-IGX-090-D.md`
4. `uv run python -m scripts.governance.verify_plan_approval governance/tickets/T-IGX-090-D.md --plan governance/tickets/plans/PLAN_T-IGX-090-D.md`
5. `git diff --name-only` and `git status --short` confirming exact write-set compliance and `production delta = 0`.
