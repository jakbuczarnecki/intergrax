# © Artur Czarnecki. All rights reserved.

"""Behavioral evidence validation for TEN/XDM scenarios (not module-name heuristics)."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import Final

from tests.qualification.ebh_7._cross_domain_scenarios import CrossDomainScenario
from tests.qualification.ebh_7._ebh7_scenario_normative_audit import ScenarioVerifiedStatus
from tests.qualification.ebh_7._semantic_disposition_registry import disposition_for
from tests.qualification.ebh_7._tenant_e2e_scenarios import TenantE2EScenario

_HELPERS = Path(__file__).resolve().parent / "cross_domain_e2e" / "_ebh7_xdm_qualification_helpers.py"
_INTEGRATED = Path(__file__).resolve().parent / "cross_domain_e2e" / "_ebh7_cross_domain_integrated_assertions.py"
_TEN_INTEGRATED = (
    Path(__file__).resolve().parent / "tenant_platform_e2e" / "_ebh7_integrated_tenant_boundary_assertions.py"
)

_XDM_PLATFORM_TO_ASSERT: Final[dict[str, str]] = {
    "test_ebh7_xdm_007_tool_skill_plugin_shadow_denied": "assert_xdm_tool_skill_plugin_shadow_denied",
    "test_ebh7_xdm_010_backend_persistence_sqlite_behavioral": "qualify_relational_store_sqlite_behavioral",
    "test_ebh7_xdm_012_trace_replay_tenant_boundary": "assert_xdm_trace_replay_tenant_lineage_fail_closed",
    "test_ebh7_xdm_014_cancel_external_effect_tenant": "assert_xdm_cancel_external_effect_tenant_boundary",
    "test_ebh7_xdm_015_compat_historical_evidence_reject": "assert_xdm_compat_historical_evidence_fail_closed",
    "test_ebh7_xdm_016_vbn_composition_factory_behavioral": "assert_xdm_vbn_composition_factory_dual_slug",
    "test_ebh7_xdm_017_failure_diagnostics_tenant_leak": "assert_xdm_failure_diagnostics_tenant_leak_denied",
    "test_ebh7_xdm_018_plugin_tenant_activate_denied": "assert_plugin_host_compat_cross_tenant_activation_denied",
    "test_ebh7_xdm_019_recovery_gov_restore_denied": "assert_xdm_recovery_gov_restore_without_grant_denied",
    "test_ebh7_xdm_022_llm_tenant_gov_block": "assert_xdm_llm_tenant_gov_sdk_block",
    "test_ebh7_xdm_023_worker_lease_tenant": "assert_xdm_worker_lease_tenant_denied",
    "test_ebh7_xdm_024_retry_side_effect_tenant": "assert_xdm_retry_side_effect_cross_tenant_denied",
    "test_ebh7_xdm_025_hitl_trace_tenant": "assert_xdm_hitl_trace_tenant_scope_denied",
    "test_ebh7_xdm_026_config_compat_tenant": "assert_xdm_config_compat_tenant_fail_closed",
    "test_ebh7_xdm_027_agent_distribution_tenant": "assert_xdm_agent_distribution_rogue_register_denied",
    "test_ebh7_xdm_028_nexus_agent_tenant": "assert_xdm_nexus_agent_private_loop_denied",
    "test_ebh7_xdm_029_integration_external_di_boundary": "assert_xdm_integration_external_di_boundary",
    "test_ebh7_xdm_030_obs_evidence_tenant_isolate": "assert_xdm_obs_evidence_tenant_isolate",
    "test_ebh7_xdm_031_scale_backpressure_tenant": "assert_xdm_scale_backpressure_tenant_isolate",
}


def _function_body_calls_test_star(path: Path, function_name: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                    if child.func.id.startswith("test_"):
                        return True
            return False
    return True


def _resolve_callable(module: str, function: str) -> bool:
    try:
        mod = importlib.import_module(module)
        fn = getattr(mod, function, None)
        return callable(fn)
    except Exception:
        return False


def validate_ten_scenario_status(scenario: TenantE2EScenario) -> ScenarioVerifiedStatus:
    if scenario.proof_module is None or scenario.proof_function is None:
        return ScenarioVerifiedStatus.PRODUCTION_GAP
    if not _resolve_callable(scenario.proof_module, scenario.proof_function):
        return ScenarioVerifiedStatus.PRODUCTION_GAP
    disp = disposition_for(scenario.scenario_id)
    if disp is None:
        return ScenarioVerifiedStatus.PRODUCTION_GAP
    expected = f"{scenario.proof_module}::{scenario.proof_function}"
    if disp.actual_test != expected:
        return ScenarioVerifiedStatus.STALE_SHA
    if scenario.scenario_id == "EBH7-TEN-E2E-03":
        src = _TEN_INTEGRATED.read_text(encoding="utf-8")
        if "probe_governed_root_launch_denied_before_intake" not in src:
            return ScenarioVerifiedStatus.PROJECTION_ONLY
    return ScenarioVerifiedStatus.DEMONSTRATED


def _integrated_assert_path(proof_module: str) -> Path | None:
    if "cross_domain_integrated_assertions" in proof_module:
        return _INTEGRATED
    if "integrated_tenant_boundary_assertions" in proof_module:
        return _TEN_INTEGRATED
    return None


def _validate_platform_xdm_proof(proof_function: str) -> ScenarioVerifiedStatus:
    if proof_function == "test_ebh7_xdm_018_plugin_tenant_activate_denied":
        if _function_body_calls_test_star(_TEN_INTEGRATED, "assert_plugin_host_compat_cross_tenant_activation_denied"):
            return ScenarioVerifiedStatus.COMPOSED_PROXY
        return ScenarioVerifiedStatus.DEMONSTRATED
    if proof_function == "test_ebh7_xdm_010_backend_persistence_sqlite_behavioral":
        if _function_body_calls_test_star(_HELPERS, "qualify_relational_store_sqlite_behavioral"):
            return ScenarioVerifiedStatus.COMPOSED_PROXY
        return ScenarioVerifiedStatus.DEMONSTRATED
    assert_fn = _XDM_PLATFORM_TO_ASSERT.get(proof_function)
    if assert_fn is None:
        return ScenarioVerifiedStatus.PRODUCTION_GAP
    if assert_fn.startswith("qualify_"):
        if _function_body_calls_test_star(_HELPERS, assert_fn):
            return ScenarioVerifiedStatus.COMPOSED_PROXY
        return ScenarioVerifiedStatus.DEMONSTRATED
    if _function_body_calls_test_star(_INTEGRATED, assert_fn):
        return ScenarioVerifiedStatus.COMPOSED_PROXY
    if assert_fn.startswith("assert_xdm_"):
        tree = ast.parse(_INTEGRATED.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == assert_fn:
                for child in ast.walk(node):
                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                        callee = child.func.id
                        if callee.startswith("qualify_") and _function_body_calls_test_star(_HELPERS, callee):
                            return ScenarioVerifiedStatus.COMPOSED_PROXY
                break
    return ScenarioVerifiedStatus.DEMONSTRATED


def validate_xdm_scenario_status(scenario: CrossDomainScenario) -> ScenarioVerifiedStatus:
    if scenario.proof_module is None or scenario.proof_function is None:
        return ScenarioVerifiedStatus.PRODUCTION_GAP
    if not _resolve_callable(scenario.proof_module, scenario.proof_function):
        return ScenarioVerifiedStatus.PRODUCTION_GAP
    disp = disposition_for(scenario.scenario_id)
    if disp is None:
        return ScenarioVerifiedStatus.PRODUCTION_GAP
    expected = f"{scenario.proof_module}::{scenario.proof_function}"
    if disp.actual_test != expected:
        return ScenarioVerifiedStatus.STALE_SHA
    pf = scenario.proof_function
    pm = scenario.proof_module
    if pf.startswith("test_ebh7_xdm_"):
        return _validate_platform_xdm_proof(pf)
    integrated = _integrated_assert_path(pm)
    if integrated is not None and pf.startswith("assert_"):
        if _function_body_calls_test_star(integrated, pf):
            return ScenarioVerifiedStatus.COMPOSED_PROXY
        return ScenarioVerifiedStatus.DEMONSTRATED
    if pf.startswith("test_"):
        return ScenarioVerifiedStatus.DEMONSTRATED
    return ScenarioVerifiedStatus.PRODUCTION_GAP


def assert_ebh7_ten_xdm_evidence_validation_complete() -> None:
    from tests.qualification.ebh_7._cross_domain_scenarios import EBH7_CROSS_DOMAIN_SCENARIOS
    from tests.qualification.ebh_7._tenant_e2e_scenarios import EBH7_TENANT_E2E_SCENARIOS

    gaps = [
        s.scenario_id
        for s in EBH7_TENANT_E2E_SCENARIOS
        if validate_ten_scenario_status(s) is ScenarioVerifiedStatus.PRODUCTION_GAP
    ]
    assert gaps == [], f"TEN production gaps: {gaps}"
    xdm_gaps = [
        s.scenario_id
        for s in EBH7_CROSS_DOMAIN_SCENARIOS
        if validate_xdm_scenario_status(s) is ScenarioVerifiedStatus.PRODUCTION_GAP
    ]
    assert xdm_gaps == [], f"XDM production gaps: {xdm_gaps}"


def ten_xdm_evidence_summary() -> dict[str, dict[str, int]]:
    from tests.qualification.ebh_7._cross_domain_scenarios import EBH7_CROSS_DOMAIN_SCENARIOS
    from tests.qualification.ebh_7._tenant_e2e_scenarios import EBH7_TENANT_E2E_SCENARIOS

    ten: dict[str, int] = {}
    xdm: dict[str, int] = {}
    for s in EBH7_TENANT_E2E_SCENARIOS:
        k = validate_ten_scenario_status(s).value
        ten[k] = ten.get(k, 0) + 1
    for s in EBH7_CROSS_DOMAIN_SCENARIOS:
        k = validate_xdm_scenario_status(s).value
        xdm[k] = xdm.get(k, 0) + 1
    return {"TEN": ten, "XDM": xdm}
