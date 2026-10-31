# © Artur Czarnecki. All rights reserved.

"""Normative TEN/XDM scenario audit statuses for independent remediation program."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from tests.qualification.ebh_7._cross_domain_scenarios import CrossDomainScenario
from tests.qualification.ebh_7._tenant_e2e_scenarios import TenantE2EScenario

_CROSS_DOMAIN_ASSERTIONS = (
    Path(__file__).resolve().parent / "cross_domain_e2e" / "_ebh7_cross_domain_integrated_assertions.py"
)


class ScenarioVerifiedStatus(StrEnum):
    DEMONSTRATED = "DEMONSTRATED"
    COMPOSED_PROXY = "COMPOSED_PROXY"
    PROJECTION_ONLY = "PROJECTION_ONLY"
    PRODUCTION_GAP = "PRODUCTION_GAP"
    STALE_SHA = "STALE_SHA"


@dataclass(frozen=True, slots=True)
class NormativeScenarioRow:
    scenario_id: str
    verified_status: ScenarioVerifiedStatus
    remediation: str
    exit_criterion: str


_PROJECTION_ONLY_TEN: Final[frozenset[str]] = frozenset()


def _assert_function_uses_foreign_test_calls(function_name: str) -> bool:
    source = _CROSS_DOMAIN_ASSERTIONS.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                    if child.func.id.startswith("test_"):
                        return True
            return False
    return True


def normative_ten_row(scenario: TenantE2EScenario) -> NormativeScenarioRow:
    if scenario.scenario_id in _PROJECTION_ONLY_TEN:
        return NormativeScenarioRow(
            scenario_id=scenario.scenario_id,
            verified_status=ScenarioVerifiedStatus.PROJECTION_ONLY,
            remediation="EBH7-W3-TEN-03 — Governance→Execution admission chain",
            exit_criterion="Cross-tenant denied before execution side effects",
        )
    if scenario.proof_module is None:
        return NormativeScenarioRow(
            scenario_id=scenario.scenario_id,
            verified_status=ScenarioVerifiedStatus.PRODUCTION_GAP,
            remediation=f"EBH7-W3-{scenario.scenario_id}",
            exit_criterion="Integrated proof node @ production path",
        )
    if "integrated_tenant_boundary_assertions" in (scenario.proof_module or ""):
        return NormativeScenarioRow(
            scenario_id=scenario.scenario_id,
            verified_status=ScenarioVerifiedStatus.DEMONSTRATED,
            remediation="Maintain integrated assertion + mutation gate",
            exit_criterion="Disposition EXACT with orchestrated entrypoint",
        )
    return NormativeScenarioRow(
        scenario_id=scenario.scenario_id,
        verified_status=ScenarioVerifiedStatus.DEMONSTRATED,
        remediation="Reuse program — verify assertion equivalence @ HEAD",
        exit_criterion="Registry proof node matches executed chain",
    )


def normative_xdm_row(scenario: CrossDomainScenario) -> NormativeScenarioRow:
    if (
        scenario.proof_module
        and "cross_domain_e2e" in scenario.proof_module
        and scenario.proof_function
        and scenario.proof_function.startswith("test_ebh7_xdm_")
    ):
        _map = {
            "test_ebh7_xdm_010_backend_persistence_sqlite_behavioral": "qualify_relational_store_sqlite_behavioral",
            "test_ebh7_xdm_018_plugin_tenant_activate_denied": (
                "assert_plugin_host_compat_cross_tenant_activation_denied"
            ),
            "test_ebh7_xdm_012_trace_replay_tenant_boundary": "assert_xdm_trace_replay_tenant_lineage_fail_closed",
            "test_ebh7_xdm_014_cancel_external_effect_tenant": "assert_xdm_cancel_external_effect_tenant_boundary",
            "test_ebh7_xdm_015_compat_historical_evidence_reject": "assert_xdm_compat_historical_evidence_fail_closed",
            "test_ebh7_xdm_016_vbn_composition_factory_behavioral": "assert_xdm_vbn_composition_factory_dual_slug",
            "test_ebh7_xdm_017_failure_diagnostics_tenant_leak": "assert_xdm_failure_diagnostics_tenant_leak_denied",
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
        fn = _map.get(scenario.proof_function or "")
        _special = (
            "qualify_relational_store_sqlite_behavioral",
            "assert_plugin_host_compat_cross_tenant_activation_denied",
        )
        if fn in _special:
            return NormativeScenarioRow(
                scenario_id=scenario.scenario_id,
                verified_status=ScenarioVerifiedStatus.DEMONSTRATED,
                remediation="Qualification helper / integrated tenant host proof",
                exit_criterion="Production-effective path @ helper or host assert",
            )
        if fn is None:
            return NormativeScenarioRow(
                scenario_id=scenario.scenario_id,
                verified_status=ScenarioVerifiedStatus.PRODUCTION_GAP,
                remediation=f"EBH7-W3-{scenario.scenario_id} — map integrated assertion",
                exit_criterion="Integrated cross-domain proof mapped",
            )
        if _assert_function_uses_foreign_test_calls(fn):
            return NormativeScenarioRow(
                scenario_id=scenario.scenario_id,
                verified_status=ScenarioVerifiedStatus.COMPOSED_PROXY,
                remediation=f"EBH7-W3-{scenario.scenario_id} — remove test_* calls from {fn}",
                exit_criterion="Integrated assertion uses qualification helpers only",
            )
        return NormativeScenarioRow(
            scenario_id=scenario.scenario_id,
            verified_status=ScenarioVerifiedStatus.DEMONSTRATED,
            remediation="Orchestrated integrated assertion @ qualification helpers",
            exit_criterion="Combo invariant via integrated entrypoint (no test-import in assert layer)",
        )
    fn = scenario.proof_function or ""
    if fn.startswith("assert_xdm_") and _assert_function_uses_foreign_test_calls(fn):
        return NormativeScenarioRow(
            scenario_id=scenario.scenario_id,
            verified_status=ScenarioVerifiedStatus.COMPOSED_PROXY,
            remediation=f"EBH7-W3-{scenario.scenario_id} — orchestrated cross-domain scenario",
            exit_criterion="Single run executes combo invariant (no test-import composition)",
        )
    if scenario.proof_module is None:
        return NormativeScenarioRow(
            scenario_id=scenario.scenario_id,
            verified_status=ScenarioVerifiedStatus.PRODUCTION_GAP,
            remediation=f"EBH7-W3-{scenario.scenario_id}",
            exit_criterion="Integrated cross-domain proof",
        )
    return NormativeScenarioRow(
        scenario_id=scenario.scenario_id,
        verified_status=ScenarioVerifiedStatus.DEMONSTRATED,
        remediation="Verify combo components in one orchestrated flow",
        exit_criterion="Independent auditor accepts combo chain",
    )


def assert_ebh7_cross_domain_integrated_assertions_no_direct_test_calls() -> None:
    source = _CROSS_DOMAIN_ASSERTIONS.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("assert_xdm_"):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                if child.func.id.startswith("test_"):
                    raise AssertionError(f"{node.name} must not call {child.func.id} directly")


def assert_ebh7_ten_xdm_normative_audit_complete() -> None:
    from tests.qualification.ebh_7._cross_domain_scenarios import EBH7_CROSS_DOMAIN_SCENARIOS
    from tests.qualification.ebh_7._tenant_e2e_scenarios import EBH7_TENANT_E2E_SCENARIOS

    assert len(EBH7_TENANT_E2E_SCENARIOS) == 26
    assert len(EBH7_CROSS_DOMAIN_SCENARIOS) == 34
