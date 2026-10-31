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
    from tests.qualification.ebh_7._ebh7_scenario_evidence_validation import validate_ten_scenario_status

    status = validate_ten_scenario_status(scenario)
    return NormativeScenarioRow(
        scenario_id=scenario.scenario_id,
        verified_status=status,
        remediation=f"EBH7-W3-{scenario.scenario_id}",
        exit_criterion="Proof node resolves; disposition matches; chain validated @ HEAD",
    )


def normative_xdm_row(scenario: CrossDomainScenario) -> NormativeScenarioRow:
    from tests.qualification.ebh_7._ebh7_scenario_evidence_validation import validate_xdm_scenario_status

    status = validate_xdm_scenario_status(scenario)
    return NormativeScenarioRow(
        scenario_id=scenario.scenario_id,
        verified_status=status,
        remediation=f"EBH7-W3-{scenario.scenario_id}",
        exit_criterion="Integrated path validated; helper layer free of assert-layer test_* calls",
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
