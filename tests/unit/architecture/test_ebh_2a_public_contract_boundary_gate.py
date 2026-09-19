# © Artur Czarnecki. All rights reserved.

"""EBH-2A — mechanical public contract dependency boundary gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from testing_support.architecture.public_contract_boundary import (
    ContractDependencyDebtEntry,
    DependencyRuleId,
    RemovalStage,
    evaluate_public_contract_dependency_boundary,
    format_gate_failure,
)
pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_contract(root: Path, relative: str, source: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def test_gate_passes_on_repository_with_registered_debt() -> None:
    result = evaluate_public_contract_dependency_boundary(_REPO_ROOT)
    assert result.passed, format_gate_failure(result)


def test_legal_canonical_contract_import_passes(tmp_path: Path) -> None:
    _write_contract(
        tmp_path,
        "intergrax/memory/contracts/sample_legal.py",
        "from intergrax.contracts.execution_phase import ExecutionPhase\n",
    )
    result = evaluate_public_contract_dependency_boundary(tmp_path, debt_entries=())
    assert result.passed, format_gate_failure(result)


def test_illegal_runtime_import_fails(tmp_path: Path) -> None:
    _write_contract(
        tmp_path,
        "intergrax/memory/contracts/sample_illegal.py",
        "from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer\n",
    )
    result = evaluate_public_contract_dependency_boundary(tmp_path, debt_entries=())
    assert not result.passed
    assert result.unregistered_violations
    assert (
        result.unregistered_violations[0].rule_id
        == DependencyRuleId.FORBIDDEN_RUNTIME_NAMESPACE
    )


def test_registered_debt_suppresses_known_violation(tmp_path: Path) -> None:
    _write_contract(
        tmp_path,
        "intergrax/contracts/sample_debt.py",
        "from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer\n",
    )
    debt = (
        ContractDependencyDebtEntry(
            finding_id="EBH2A-T-001",
            source_module="intergrax.contracts.sample_debt",
            forbidden_import_prefix="intergrax.runtime.nexus.responses.response_schema",
            rule_id=DependencyRuleId.FORBIDDEN_RUNTIME_NAMESPACE,
            removal_stage=RemovalStage.EBH_2B,
        ),
    )
    result = evaluate_public_contract_dependency_boundary(tmp_path, debt_entries=debt)
    assert result.passed, format_gate_failure(result)


def test_new_unregistered_forbidden_import_fails(tmp_path: Path) -> None:
    _write_contract(
        tmp_path,
        "intergrax/contracts/sample_new.py",
        "from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle\n",
    )
    result = evaluate_public_contract_dependency_boundary(tmp_path, debt_entries=())
    assert not result.passed
    assert any(
        v.source_module == "intergrax.contracts.sample_new"
        for v in result.unregistered_violations
    )


def test_stale_debt_entry_fails(tmp_path: Path) -> None:
    _write_contract(
        tmp_path,
        "intergrax/contracts/clean.py",
        "from intergrax.contracts.execution_phase import ExecutionPhase\n",
    )
    stale = (
        ContractDependencyDebtEntry(
            finding_id="EBH2A-T-STALE",
            source_module="intergrax.contracts.clean",
            forbidden_import_prefix="intergrax.runtime.nexus",
            rule_id=DependencyRuleId.FORBIDDEN_RUNTIME_NAMESPACE,
            removal_stage=RemovalStage.EBH_2B,
        ),
    )
    result = evaluate_public_contract_dependency_boundary(tmp_path, debt_entries=stale)
    assert not result.passed
    assert result.stale_debt_entries
    assert result.stale_debt_entries[0].finding_id == "EBH2A-T-STALE"


def test_deterministic_violation_order(tmp_path: Path) -> None:
    _write_contract(
        tmp_path,
        "intergrax/z_domain/contracts/z_mod.py",
        "from intergrax.runtime.nexus import x\n",
    )
    _write_contract(
        tmp_path,
        "intergrax/a_domain/contracts/a_mod.py",
        "from intergrax.runtime.policy import y\n",
    )
    first = evaluate_public_contract_dependency_boundary(tmp_path, debt_entries=())
    second = evaluate_public_contract_dependency_boundary(tmp_path, debt_entries=())
    assert first.unregistered_violations == second.unregistered_violations
