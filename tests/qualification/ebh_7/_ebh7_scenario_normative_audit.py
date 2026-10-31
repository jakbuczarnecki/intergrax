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


_PROJECTION_ONLY_TEN: Final[frozenset[str]] = frozenset({"EBH7-TEN-E2E-03"})


def _composed_proxy_functions() -> frozenset[str]:
    source = _CROSS_DOMAIN_ASSERTIONS.read_text(encoding="utf-8")
    tree = ast.parse(source)
    proxies: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("assert_xdm_"):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                if child.func.id.startswith("test_"):
                    proxies.add(node.name)
                    break
    return frozenset(proxies)


_COMPOSED_PROXY_ASSERTIONS: Final[frozenset[str]] = _composed_proxy_functions()


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
        return NormativeScenarioRow(
            scenario_id=scenario.scenario_id,
            verified_status=ScenarioVerifiedStatus.COMPOSED_PROXY,
            remediation=f"EBH7-W3-{scenario.scenario_id} — orchestrated cross-domain scenario",
            exit_criterion="Single run executes combo invariant (no test-import composition)",
        )
    fn = scenario.proof_function or ""
    if fn in _COMPOSED_PROXY_ASSERTIONS:
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


def assert_ebh7_cross_domain_composed_proxy_inventory() -> None:
    assert _COMPOSED_PROXY_ASSERTIONS, "Expected composed proxy assertion functions"


def assert_ebh7_ten_xdm_normative_audit_complete() -> None:
    from tests.qualification.ebh_7._cross_domain_scenarios import EBH7_CROSS_DOMAIN_SCENARIOS
    from tests.qualification.ebh_7._tenant_e2e_scenarios import EBH7_TENANT_E2E_SCENARIOS

    assert len(EBH7_TENANT_E2E_SCENARIOS) == 26
    assert len(EBH7_CROSS_DOMAIN_SCENARIOS) == 34
