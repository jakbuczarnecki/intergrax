# © Artur Czarnecki. All rights reserved.

"""EBH-2I — final cross-subsystem public-contract rescan invariants."""

from __future__ import annotations

from pathlib import Path

import pytest

from testing_support.architecture.public_contract_boundary import (
    evaluate_contract_surface_purity,
    evaluate_contract_surface_purity_on_source,
    evaluate_public_contract_dependency_boundary,
    format_gate_failure,
)
from testing_support.architecture.public_contract_boundary.debt_registry import (
    PUBLIC_CONTRACT_DEPENDENCY_DEBT,
)
from testing_support.architecture.public_contract_boundary.models import RemovalStage

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]

_CLOSED_EBH_2_STAGES = frozenset(
    {
        RemovalStage.EBH_2E,
        RemovalStage.EBH_2F,
        RemovalStage.EBH_2G,
        RemovalStage.EBH_2H,
    }
)


def test_ebh_2i_no_expired_stage_debt_entries() -> None:
    expired = [
        entry
        for entry in PUBLIC_CONTRACT_DEPENDENCY_DEBT
        if entry.removal_stage in _CLOSED_EBH_2_STAGES
    ]
    assert not expired, "Expired EBH-2E/F/G/H debt entries must be reconciled: " + ", ".join(
        entry.finding_id for entry in expired
    )


def test_ebh_2i_public_contract_boundary_gate_passes() -> None:
    result = evaluate_public_contract_dependency_boundary(_REPO_ROOT)
    assert result.passed, format_gate_failure(result)


def test_ebh_2i_contract_surface_purity_gate_passes() -> None:
    violations = evaluate_contract_surface_purity(_REPO_ROOT)
    assert not violations, "\n".join(item.as_message() for item in violations)


def test_ebh_2i_negative_gate_detects_implementation_laundering() -> None:
    source = '''
from dataclasses import dataclass, field

@dataclass
class HostedApplicationServiceRegistry:
    _services: dict = field(default_factory=dict)

    def register(self) -> None:
        return None

    def seal(self) -> None:
        return None

    def close(self) -> None:
        return None
'''
    violations = evaluate_contract_surface_purity_on_source(source=source)
    assert any(v.rule_id == "implementation_laundering" for v in violations)


def test_ebh_2i_negative_gate_detects_type_ignore_masking() -> None:
    source = """
from typing import Protocol

class Port(Protocol):
    def run(self) -> int:
        return 1  # type: ignore[return-value]
"""
    violations = evaluate_contract_surface_purity_on_source(source=source)
    assert any(v.rule_id == "architecture_masking_type_ignore" for v in violations)


def test_ebh_2i_negative_gate_detects_dynamic_semantic_dispatch() -> None:
    source = """
from intergrax.contracts import vendor_attribute_access as attribute_access

class Hooks:
    def hooks_for_point(self, point):
        return attribute_access.optional(self, point.value)
"""
    violations = evaluate_contract_surface_purity_on_source(source=source)
    assert any(v.rule_id == "dynamic_semantic_dispatch" for v in violations)
