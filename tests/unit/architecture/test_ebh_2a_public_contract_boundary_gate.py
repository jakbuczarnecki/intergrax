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
from testing_support.architecture.public_contract_boundary.debt_validation import (
    validate_debt_registry,
)
from testing_support.architecture.public_contract_boundary.discovery import (
    discover_public_contract_modules,
    discover_public_contract_source_files,
)
from testing_support.architecture.public_contract_boundary.evaluation import (
    _debt_covers_violation,
)
from testing_support.architecture.public_contract_boundary.models import (
    ContractDependencyViolation,
)
from testing_support.architecture.public_contract_boundary.supplemental_surfaces import (
    SUPPLEMENTAL_PUBLIC_CONTRACT_SURFACES,
    SupplementalPublicContractSurface,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NO_SUPPLEMENTAL: tuple[SupplementalPublicContractSurface, ...] = ()


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
    result = evaluate_public_contract_dependency_boundary(
        tmp_path,
        debt_entries=(),
        supplemental_surfaces=_NO_SUPPLEMENTAL,
    )
    assert result.passed, format_gate_failure(result)


def test_illegal_runtime_import_fails(tmp_path: Path) -> None:
    _write_contract(
        tmp_path,
        "intergrax/memory/contracts/sample_illegal.py",
        "from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer\n",
    )
    result = evaluate_public_contract_dependency_boundary(
        tmp_path,
        debt_entries=(),
        supplemental_surfaces=_NO_SUPPLEMENTAL,
    )
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
            forbidden_import_module="intergrax.runtime.nexus.responses.response_schema",
            rule_id=DependencyRuleId.FORBIDDEN_RUNTIME_NAMESPACE,
            removal_stage=RemovalStage.EBH_2B,
        ),
    )
    result = evaluate_public_contract_dependency_boundary(
        tmp_path,
        debt_entries=debt,
        supplemental_surfaces=_NO_SUPPLEMENTAL,
    )
    assert result.passed, format_gate_failure(result)


def test_new_unregistered_forbidden_import_fails(tmp_path: Path) -> None:
    _write_contract(
        tmp_path,
        "intergrax/contracts/sample_new.py",
        "from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle\n",
    )
    result = evaluate_public_contract_dependency_boundary(
        tmp_path,
        debt_entries=(),
        supplemental_surfaces=_NO_SUPPLEMENTAL,
    )
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
            forbidden_import_module="intergrax.runtime.nexus.engine.runtime_context",
            rule_id=DependencyRuleId.FORBIDDEN_RUNTIME_NAMESPACE,
            removal_stage=RemovalStage.EBH_2B,
        ),
    )
    result = evaluate_public_contract_dependency_boundary(
        tmp_path,
        debt_entries=stale,
        supplemental_surfaces=_NO_SUPPLEMENTAL,
    )
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
    first = evaluate_public_contract_dependency_boundary(
        tmp_path,
        debt_entries=(),
        supplemental_surfaces=_NO_SUPPLEMENTAL,
    )
    second = evaluate_public_contract_dependency_boundary(
        tmp_path,
        debt_entries=(),
        supplemental_surfaces=_NO_SUPPLEMENTAL,
    )
    assert first.unregistered_violations == second.unregistered_violations


def test_exact_debt_covers_exact_import() -> None:
    violation = ContractDependencyViolation(
        source_module="intergrax.memory.contracts.chat_session",
        imported_module="intergrax.utils.time_provider",
        rule_id=DependencyRuleId.FOREIGN_DOMAIN_IMPLEMENTATION,
        source_path="intergrax/memory/contracts/chat_session.py",
        line=1,
    )
    entry = ContractDependencyDebtEntry(
        finding_id="EBH2A-T-EXACT",
        source_module="intergrax.memory.contracts.chat_session",
        forbidden_import_module="intergrax.utils.time_provider",
        rule_id=DependencyRuleId.FOREIGN_DOMAIN_IMPLEMENTATION,
        removal_stage=RemovalStage.EBH_2H,
    )
    assert _debt_covers_violation(violation, entry)


def test_exact_debt_does_not_cover_sibling_import(tmp_path: Path) -> None:
    _write_contract(
        tmp_path,
        "intergrax/memory/contracts/edge.py",
        "from intergrax.utils.time_provider import TimeProvider\n"
        "from intergrax.utils.some_new_module import NewThing\n",
    )
    debt = (
        ContractDependencyDebtEntry(
            finding_id="EBH2A-T-SIBLING",
            source_module="intergrax.memory.contracts.edge",
            forbidden_import_module="intergrax.utils.time_provider",
            rule_id=DependencyRuleId.FOREIGN_DOMAIN_IMPLEMENTATION,
            removal_stage=RemovalStage.EBH_2H,
        ),
    )
    result = evaluate_public_contract_dependency_boundary(
        tmp_path,
        debt_entries=debt,
        supplemental_surfaces=_NO_SUPPLEMENTAL,
    )
    assert not result.passed
    assert any(
        v.imported_module == "intergrax.utils.some_new_module"
        for v in result.unregistered_violations
    )


def test_broad_debt_does_not_cover_new_submodule_import(tmp_path: Path) -> None:
    _write_contract(
        tmp_path,
        "intergrax/memory/contracts/broad.py",
        "from intergrax.utils.some_new_module import NewThing\n",
    )
    debt = (
        ContractDependencyDebtEntry(
            finding_id="EBH2A-T-BROAD",
            source_module="intergrax.memory.contracts.broad",
            forbidden_import_module="intergrax.utils",
            rule_id=DependencyRuleId.FOREIGN_DOMAIN_IMPLEMENTATION,
            removal_stage=RemovalStage.EBH_2D,
        ),
    )
    result = evaluate_public_contract_dependency_boundary(
        tmp_path,
        debt_entries=debt,
        supplemental_surfaces=_NO_SUPPLEMENTAL,
    )
    assert not result.passed
    assert result.unregistered_violations


def test_uaep_is_discovered_as_public_contract_surface() -> None:
    modules = discover_public_contract_modules(_REPO_ROOT)
    assert "intergrax.agents.uaep_protocol" in modules
    paths = discover_public_contract_source_files(_REPO_ROOT)
    assert any(p.as_posix().endswith("intergrax/agents/uaep_protocol.py") for p in paths)


def test_uaep_nexus_debt_controlled_on_repository() -> None:
    result = evaluate_public_contract_dependency_boundary(_REPO_ROOT)
    uaep_violations = [
        v
        for v in result.unregistered_violations
        if v.source_module == "intergrax.agents.uaep_protocol"
    ]
    assert not uaep_violations, format_gate_failure(result)


def test_uaep_nexus_debt_removal_fails(tmp_path: Path) -> None:
    uaep_source = (_REPO_ROOT / "intergrax/agents/uaep_protocol.py").read_text(
        encoding="utf-8",
    )
    _write_contract(tmp_path, "intergrax/agents/uaep_protocol.py", uaep_source)
    uaep_surface = (
        SupplementalPublicContractSurface(
            repo_relative_path="intergrax/agents/uaep_protocol.py",
            owner_domain="agents",
            remediation_stage=RemovalStage.EBH_2B,
        ),
    )
    result = evaluate_public_contract_dependency_boundary(
        tmp_path,
        debt_entries=(),
        supplemental_surfaces=uaep_surface,
    )
    assert not result.passed
    assert any(
        v.source_module == "intergrax.agents.uaep_protocol"
        for v in result.unregistered_violations
    )


def test_missing_supplemental_surface_fails(tmp_path: Path) -> None:
    bogus = (
        SupplementalPublicContractSurface(
            repo_relative_path="intergrax/agents/does_not_exist.py",
            owner_domain="agents",
            remediation_stage=RemovalStage.EBH_2B,
        ),
    )
    result = evaluate_public_contract_dependency_boundary(
        tmp_path,
        debt_entries=(),
        supplemental_surfaces=bogus,
    )
    assert not result.passed
    assert result.registry_validation_errors


def test_duplicate_finding_id_fails_validation() -> None:
    debt = (
        ContractDependencyDebtEntry(
            finding_id="EBH2A-T-DUP-ID",
            source_module="intergrax.contracts.a",
            forbidden_import_module="intergrax.runtime.nexus",
            rule_id=DependencyRuleId.FORBIDDEN_RUNTIME_NAMESPACE,
            removal_stage=RemovalStage.EBH_2B,
        ),
        ContractDependencyDebtEntry(
            finding_id="EBH2A-T-DUP-ID",
            source_module="intergrax.contracts.b",
            forbidden_import_module="intergrax.runtime.policy",
            rule_id=DependencyRuleId.FORBIDDEN_RUNTIME_NAMESPACE,
            removal_stage=RemovalStage.EBH_2B,
        ),
    )
    errors = validate_debt_registry(debt)
    assert errors
    result = evaluate_public_contract_dependency_boundary(
        _REPO_ROOT,
        debt_entries=debt,
    )
    assert not result.passed
    assert result.registry_validation_errors


def test_duplicate_semantic_debt_entry_fails_validation() -> None:
    debt = (
        ContractDependencyDebtEntry(
            finding_id="EBH2A-T-DUP-A",
            source_module="intergrax.contracts.sample",
            forbidden_import_module="intergrax.runtime.nexus.engine.runtime_context",
            rule_id=DependencyRuleId.FORBIDDEN_RUNTIME_NAMESPACE,
            removal_stage=RemovalStage.EBH_2B,
        ),
        ContractDependencyDebtEntry(
            finding_id="EBH2A-T-DUP-B",
            source_module="intergrax.contracts.sample",
            forbidden_import_module="intergrax.runtime.nexus.engine.runtime_context",
            rule_id=DependencyRuleId.FORBIDDEN_RUNTIME_NAMESPACE,
            removal_stage=RemovalStage.EBH_2C,
        ),
    )
    errors = validate_debt_registry(debt)
    assert errors


def test_production_debt_registry_passes_validation() -> None:
    from testing_support.architecture.public_contract_boundary.debt_registry import (
        PUBLIC_CONTRACT_DEPENDENCY_DEBT,
    )

    assert not validate_debt_registry(PUBLIC_CONTRACT_DEPENDENCY_DEBT)
