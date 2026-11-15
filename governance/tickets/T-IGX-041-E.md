# T-IGX-041-E — PDC-X batch B-P7-D5: repair of the D5 Evaluation domain after the independent audit IA-D5 (loop 1)

## Scope

Repair ticket of domain D5 (Evaluation) under Operator option B (`DELEGATION-PDC-P7-REPAIR.md`), after the independent audit `PDC_P7_D5_AUDIT_REPORT.md` (verdict FAIL, row `PDC-D5-R015`). In scope: tenant isolation of online evaluation observations (F-D5-01: legacy registry surface limited to the untenanted namespace, tenant stamped by `record_user_satisfaction` and `record_shadow_observation`, tenant bound by the three hosts that hold one, id collision), `ExperimentSession.run` (F-D5-07: missing `mint_run_id` import and the governance identity admission hook behind it), evaluation evidence status and comparison coverage (F-D5-04), a certification of another agent can no longer authorize a promotion (F-D5-03, strengthening only), MVP promotion gate G0/G2 symbol checks and anchored authorization lines (F-D5-02, strengthening only), contract regex hygiene (F-D5-11 part) and the R017 boundary documentation with an honest exception paragraph (F-D5-08). Excluded with reason and reopen condition (exception or debt register of the P9 closure package): F-D5-05 and F-D5-06 (not implemented, not wired), the Operator-decision remainders of F-D5-02 and F-D5-03, and the LOW and INFO findings listed in the plan. Product code: yes. Hotspot files named exactly in the plan: `intergrax/contracts/evaluation_contracts.py` (regex lines only) and three files under `intergrax/applications/_shared/` (one call each). This ticket is not a certification verdict. Plan: `PLAN_T-IGX-041-E.md`.

## Plan digest

Plan digest: 1bf69a9c023c331d721a227d17fdf673a8bdec2c6896fef9c85e045b36bfdf75

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-041-E.md`](plans/PLAN_T-IGX-041-E.md).

## Approved

Approved: operator-delegate @ 2026-11-15T08:55:35+01:00
Delegation: docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-P7-REPAIR.md@8f4b139c1f5352c9861872f0b5077996b8f0f891

Delegate checks (scope of `DELEGATION-PDC-P7-REPAIR.md` read at `origin/development`, last changing commit `8f4b139c1`; signed at the batch branch, implementer commit `3cc320856`; all passed before this line was written):

- The plan digest recomputed with `scripts/governance/plan_digest.py` equals the digest above (`1bf69a9c023c331d721a227d17fdf673a8bdec2c6896fef9c85e045b36bfdf75`).
- The plan write set (8 product files, 4 hotspot files, `tests/qualification/pdc_x/d5_repair/`, 2 docs: 15 entries) equals the write set of batch row `B-P7-D5` in `PDC_IMPLEMENTATION_WBS.md` and in `scripts/qualification/pdc_x_wbs.py`. The implementer commit changes only the ticket, the plan, the WBS, the DAG and the batch register `pdc_x_wbs.py`; nothing under `scripts/governance/` or `.github/`.
- Hotspots: `intergrax/contracts/evaluation_contracts.py` ("regex lines only") and `intergrax/applications/_shared/{harness_host_runtime,runtime_config_bridge,scenario_runtime_baseline}.py` ("one call each") are named by exact file; no directory wildcard. The three hosts hold a tenant at the call site (`tenant_id` parameter of `build_harness_host_runtime`, `request.tenant_id`, `resolved_tenant_id`); `policy_wiring.py` uses only `domain_fragments` and is not edited.
- In-scope findings (F-D5-01, F-D5-07, F-D5-04, F-D5-03 part, F-D5-02 part, F-D5-08, F-D5-11/12 by-products) and excluded findings (F-D5-05, F-D5-06, remainders of F-D5-01/02/03, F-D5-09..18) are all listed with class, reason and reopen condition. Exclusions are of the allowed kinds: missing feature versus claim (F-D5-05), F-04 family (F-D5-06, F-D5-03 wiring), Operator decision (F-D5-02 binding, F-D5-03 comparison contract), LOW/INFO debt.
- F-D5-01 remainder read in code: `test_r2_eval_05_tenant_verdict_not_tenant_aware_at_this_stage` asserts `"tenant_id" not in OnlineEvaluationObservation.model_fields`, so the exclusion is truthful; `TenantOnlineEvaluationObservation` already carries `tenant_id`; `policy_wiring.py:154` never reads the registry.
- No gate weakened or deleted. `check_mvp_promotion_gates.py` read in full: G0/G2 symbol tuples are `()`, so naming symbols can only turn OK into FAIL (all named symbols exist at the base: `main`, `build_parser`, `run_doctor`, `run_command`, `create_lab_application`, `ResiliencePolicy`, `default_resilience_policy`, `AutonomyLevel`, `resolve_effective_autonomy`, `tool_allowed_for_autonomy`, `main`); reading `Approved:`/`Delegation:` from the `## Approved` section only narrows today's whole-file match, so it can only turn GRANTED into a denial; G1, the advisory status and the exit codes stay. Other fixes only deny more (`_run_status` FAILED/UNAVAILABLE, coverage and human-pending breaches, certification subject mismatch, `fullmatch` regexes). Legacy registry surface narrows to the untenanted namespace; `get_observation` may return only the caller's own row. Every item has red-first and negative tests; no existing test is edited (new directory only; a failing existing test stops the batch).
- The in-scope fixes match the audit report `PDC_P7_D5_AUDIT_REPORT.md` (F-D5-01..08, reproductions and suggested scopes). The `mint_run_id` import is missing at `intergrax/experiments/workflow.py:163` and `intergrax/debug/app.py` uses `admit_certified_internal_harness_root_governance_identity` for the same loop. The plan has a stop condition when that hook is not enough.
- No registry, scheduler, engine, trust, policy or audit authority is added; the plan states it is not a certification verdict. The implementer session (`plan-041e`) and this delegate session are different sessions; the delegate edited nothing except this approval block and the lifecycle state.
- Merge gate (stated as the delegation record requires): product pull request merges under authorizations 1 and 3 of `DELEGATION-PDC-STANDING.md` and decision 8 of `DELEGATION-PDC-WAVE2.md`: green CI, only the known red classes (IIAC activation staleness, EBH-7 identity gate), `headRefOid` equal to the verified SHA; additionally OD-PDC-SEQ-01 (merge only after MCP-SERVER-X P8 publication) from the WBS row. Write-set amendments are not delegated. Not decided here: the numbering of the standing authorizations was not re-verified.

## Test contract

```text
uv run pytest tests/qualification/pdc_x/d5_repair tests/qualification/pdc_x/d5 tests/qualification/pdc_x/xd/test_xd_08_evaluation.py tests/unit/experiments tests/unit/eval tests/unit/runtime/architecture/test_online_evaluation_registry.py tests/unit/runtime/architecture/test_online_evaluation.py tests/unit/runtime/architecture/test_online_evaluation_trend.py tests/unit/runtime/architecture/test_user_satisfaction.py tests/unit/applications/test_harness_evaluation_wiring.py tests/unit/applications/test_runtime_config_bridge.py tests/unit/applications/test_scenario_runtime_baseline.py tests/unit/applications/test_harness_host_runtime_llm.py tests/unit/applications/test_harness_host_prepare_before_publish.py tests/qualification/control_planes/ctrl_x/test_ctrl_x_r1_evaluation_plane.py tests/qualification/control_planes/ctrl_x/test_ctrl_x_r2_evaluation_plane.py tests/integration/runtime/adaptive/test_adaptive_shadow_run.py tests/unit/tools/providers/eval -q
uv run ruff check <touched product, script and test files>
uv run pyright <touched product, script and test files>
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python scripts/docs/check_docs_domain_pairs.py
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- Every excluded finding of the plan (F-D5-05, F-D5-06, the Operator-decision remainders of F-D5-02 and F-D5-03, F-D5-09 to F-D5-18 except the named by-products); the TENANT-X carrier re-pin (ticket `T-IGX-041-I`); `evaluation_wiring.py`, `policy_wiring.py`, the base `OnlineEvaluationObservation`, `canonical_payload_hash.py`, the requirement matrix; `scripts/governance/`, `.github/`.
- The IIAC activation and its P8-04 rebind (stays stale after these product changes, known red class), the EBH-7 re-pin (stays red until the final re-pin, known red class), OD signatures, FRZ changes, certification, the re-audit.

## Lifecycle state

plan_approved
