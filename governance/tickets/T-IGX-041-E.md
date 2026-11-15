# T-IGX-041-E — PDC-X batch B-P7-D5: repair of the D5 Evaluation domain after the independent audit IA-D5 (loop 1)

## Scope

Repair ticket of domain D5 (Evaluation) under Operator option B (`DELEGATION-PDC-P7-REPAIR.md`), after the independent audit `PDC_P7_D5_AUDIT_REPORT.md` (verdict FAIL, row `PDC-D5-R015`). In scope: tenant isolation of online evaluation observations (F-D5-01: legacy registry surface limited to the untenanted namespace, tenant stamped by `record_user_satisfaction` and `record_shadow_observation`, tenant bound by the three hosts that hold one, id collision), `ExperimentSession.run` (F-D5-07: missing `mint_run_id` import and the governance identity admission hook behind it), evaluation evidence status and comparison coverage (F-D5-04), a certification of another agent can no longer authorize a promotion (F-D5-03, strengthening only), MVP promotion gate G0/G2 symbol checks and anchored authorization lines (F-D5-02, strengthening only), contract regex hygiene (F-D5-11 part) and the R017 boundary documentation with an honest exception paragraph (F-D5-08). Excluded with reason and reopen condition (exception or debt register of the P9 closure package): F-D5-05 and F-D5-06 (not implemented, not wired), the Operator-decision remainders of F-D5-02 and F-D5-03, and the LOW and INFO findings listed in the plan. Product code: yes. Hotspot files named exactly in the plan: `intergrax/contracts/evaluation_contracts.py` (regex lines only) and three files under `intergrax/applications/_shared/` (one call each). This ticket is not a certification verdict. Plan: `PLAN_T-IGX-041-E.md`.

## Plan digest

Plan digest: 1bf69a9c023c331d721a227d17fdf673a8bdec2c6896fef9c85e045b36bfdf75

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-041-E.md`](plans/PLAN_T-IGX-041-E.md).

## Approved

(pending; the plan is signed by a delegate session under `DELEGATION-PDC-P7-REPAIR.md`; the implementer never writes the approval line)

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

plan_pending
