# © Artur Czarnecki. All rights reserved.

"""EBH-2I — final cross-subsystem public-contract rescan invariants."""

from __future__ import annotations

from pathlib import Path

import pytest

from testing_support.architecture.public_contract_boundary import (
    CONTRACT_SURFACE_PURITY_DEBT,
    ContractSurfacePurityDebtEntry,
    evaluate_contract_surface_purity,
    evaluate_contract_surface_purity_gate_on_source,
    evaluate_contract_surface_purity_on_source,
    evaluate_public_contract_dependency_boundary,
    format_contract_surface_purity_failure,
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
    result = evaluate_contract_surface_purity(_REPO_ROOT)
    assert result.passed, format_contract_surface_purity_failure(result)


def test_ebh_2i_csp_debt_registry_is_empty() -> None:
    assert CONTRACT_SURFACE_PURITY_DEBT == ()


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


def test_ebh_2i_csp_negative_unregistered_violation_fails() -> None:
    source = """
from typing import cast

X = cast(int, "not-int")
"""
    result = evaluate_contract_surface_purity_gate_on_source(source=source)
    assert not result.passed
    assert result.unregistered_violations


def test_ebh_2i_csp_negative_stale_debt_fails() -> None:
    source = "VALUE = 1\n"
    stale_entry = ContractSurfacePurityDebtEntry(
        finding_id="TEST-CSP-STALE",
        source_path="intergrax/snippet/contracts/snippet.py",
        line=99,
        rule_id="architecture_masking_cast",
        removal_stage=RemovalStage.EBH_2I,
        rationale="test stale debt detection",
    )
    result = evaluate_contract_surface_purity_gate_on_source(
        source=source,
        debt_entries=(stale_entry,),
    )
    assert not result.passed
    assert result.stale_debt_entries == (stale_entry,)


def test_ebh_2i_csp_negative_expired_debt_fails() -> None:
    source = """
from typing import cast

X = cast(int, "not-int")
"""
    expired_entry = ContractSurfacePurityDebtEntry(
        finding_id="TEST-CSP-EXPIRED",
        source_path="intergrax/snippet/contracts/snippet.py",
        line=4,
        rule_id="architecture_masking_cast",
        removal_stage=RemovalStage.EBH_2G,
        rationale="test expired debt detection",
    )
    result = evaluate_contract_surface_purity_gate_on_source(
        source=source,
        debt_entries=(expired_entry,),
    )
    assert not result.passed
    assert result.expired_debt_entries == (expired_entry,)


def test_ebh_2i_csp_negative_duplicate_debt_registration_fails() -> None:
    entry_a = ContractSurfacePurityDebtEntry(
        finding_id="DUP-ID",
        source_path="intergrax/a.py",
        line=1,
        rule_id="architecture_masking_cast",
        removal_stage=RemovalStage.QUAL_X,
        rationale="first",
    )
    entry_b = ContractSurfacePurityDebtEntry(
        finding_id="DUP-ID",
        source_path="intergrax/b.py",
        line=2,
        rule_id="architecture_masking_type_ignore",
        removal_stage=RemovalStage.QUAL_X,
        rationale="duplicate finding_id",
    )
    result = evaluate_contract_surface_purity_gate_on_source(
        source="X = 1\n",
        debt_entries=(entry_a, entry_b),
    )
    assert not result.passed
    assert result.registry_validation_errors


def test_ebh_2i_csp_negative_zero_debt_clean_source_passes() -> None:
    result = evaluate_contract_surface_purity_gate_on_source(
        source="VALUE = 1\n",
        debt_entries=(),
    )
    assert result.passed
