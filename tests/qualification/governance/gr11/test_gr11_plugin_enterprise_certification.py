# © Artur Czarnecki. All rights reserved.

"""GR-11 — Governance plugin enterprise certification mechanical gate (G11-01..G11-20)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.qualification.governance.gr11.catalog import (
    GR11_EXTENSION_SURFACES,
    GR11_FORBIDDEN_PARENT_STATUSES,
    GR11_HISTORICAL_PLUGINABILITY_RECONCILIATION,
    GR11_IMPLEMENTATION_BRANCH_FORBIDDEN_PATTERNS,
    GR11_QUALIFICATION_STATUS,
    GR11_STRUCTURAL_PROOF_NODES,
    GR11_WEAK_BOUNDARY_SCAN_MODULES,
    Gr11DynamicRegistrationApplicability,
    Gr11QualificationStatus,
    gr11_capability_ids,
)
from tests.qualification.governance.gr12.qualification_support import (
    assert_proof_nodes_registered,
)
from tests.qualification.governance.pytest_node_integrity import (
    pytest_nodes_missing_from_collection,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]

_WEAK_BOUNDARY_FORBIDDEN = (
    "dict[str, Any]",
    "Mapping[str, Any]",
    ": Any",
    "-> Any",
    "getattr(",
    "hasattr(",
    "setattr(",
    "inspect.",
)


def test_gr11_g01_every_row_uniquely_identified() -> None:
    ids = gr11_capability_ids()
    assert len(ids) == len(set(ids))
    assert len(ids) >= 9


def test_gr11_g02_every_extensible_mechanism_exposes_platform_contract() -> None:
    for row in GR11_EXTENSION_SURFACES:
        assert row.contract.strip(), row.capability_id
        assert "intergrax." in row.contract


def test_gr11_g03_custom_structural_proof_registered_for_every_qualified_row() -> None:
    for row in GR11_EXTENSION_SURFACES:
        if row.status is not Gr11QualificationStatus.QUALIFIED:
            continue
        assert row.custom_structural_proof_nodes, row.capability_id


def test_gr11_g04_structural_proof_nodes_collectable() -> None:
    missing, proc = pytest_nodes_missing_from_collection(
        GR11_STRUCTURAL_PROOF_NODES, _REPO_ROOT
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not missing, missing


def test_gr11_g05_exactly_one_semantic_owner_per_row() -> None:
    for row in GR11_EXTENSION_SURFACES:
        assert row.semantic_owner.strip(), row.capability_id


def test_gr11_g06_exactly_one_sanctioned_composition_owner_per_row() -> None:
    for row in GR11_EXTENSION_SURFACES:
        assert row.composition_owner.strip(), row.capability_id


def test_gr11_g07_no_vendor_literal_in_governance_contract_modules() -> None:
    contract_paths = [
        _REPO_ROOT / "intergrax/contracts/runtime_execution_policy_admission.py",
        _REPO_ROOT / "intergrax/contracts/canonical_inner_governance.py",
        _REPO_ROOT / "intergrax/contracts/execution_continuation.py",
        _REPO_ROOT / "intergrax/contracts/provider_invocation_store.py",
        _REPO_ROOT / "intergrax/contracts/control_plane_mutation.py",
    ]
    vendor_markers = ("openai", "anthropic", "azure_openai", "vendor_")
    for path in contract_paths:
        text = path.read_text(encoding="utf-8").lower()
        for marker in vendor_markers:
            assert marker not in text, f"{path.name} contains vendor marker {marker!r}"


def test_gr11_g08_no_implementation_type_branch_in_listed_governance_consumers() -> (
    None
):
    consumer_modules = (
        _REPO_ROOT
        / "intergrax/runtime/governance/runtime_execution_policy_admission.py",
        _REPO_ROOT / "intergrax/runtime/governance/canonical_inner_execution_guard.py",
        _REPO_ROOT
        / "intergrax/runtime/governance/control_plane_mutation_authorization.py",
    )
    for path in consumer_modules:
        text = path.read_text(encoding="utf-8")
        for pattern in GR11_IMPLEMENTATION_BRANCH_FORBIDDEN_PATTERNS:
            assert not re.search(pattern, text), f"{path}: matched {pattern}"


def test_gr11_g09_no_reflection_semantic_plugin_dispatch_in_scan_modules() -> None:
    for rel in GR11_WEAK_BOUNDARY_SCAN_MODULES:
        path = _REPO_ROOT / rel
        text = path.read_text(encoding="utf-8")
        assert "importlib.import_module" not in text, rel
        assert "__import__(" not in text, rel


def test_gr11_g10_no_semantic_any_object_on_gr11_contract_modules() -> None:
    for rel in GR11_WEAK_BOUNDARY_SCAN_MODULES:
        if not rel.startswith("intergrax/contracts/"):
            continue
        text = (_REPO_ROOT / rel).read_text(encoding="utf-8")
        for forbidden in _WEAK_BOUNDARY_FORBIDDEN:
            assert forbidden not in text, f"{rel}: found {forbidden!r}"


def test_gr11_g11_plugins_cannot_widen_authority_by_design() -> None:
    for row in GR11_EXTENSION_SURFACES:
        assert row.may_widen_authority is False, row.capability_id


def test_gr11_g12_fail_closed_admission_regression_nodes_registered() -> None:
    assert_proof_nodes_registered(
        (
            "tests/unit/runtime/governance/test_runtime_execution_policy_admission.py::"
            "test_evaluator_unconfigured_fail_closed",
            "tests/unit/runtime/governance/test_runtime_execution_policy_admission.py::"
            "test_unavailable_adapter_fail_closed",
        )
    )


def test_gr11_g13_continuation_row_execution_owned() -> None:
    row = next(
        r for r in GR11_EXTENSION_SURFACES if r.capability_id == "GR11-CONTINUATION"
    )
    assert "Execution" in row.semantic_owner


def test_gr11_g14_provider_invocation_store_reliability_fact_owner() -> None:
    row = next(
        r
        for r in GR11_EXTENSION_SURFACES
        if r.capability_id == "GR11-PROVIDER-INVOCATION-STORE"
    )
    assert row.authority.name == "RELIABILITY_FACT"


def test_gr11_g15_control_plane_evaluator_behind_cla04_contract() -> None:
    row = next(
        r
        for r in GR11_EXTENSION_SURFACES
        if r.capability_id == "GR11-CONTROL-PLANE-EVALUATOR"
    )
    assert "ControlPlaneMutationPolicyEvaluator" in row.contract


def test_gr11_g16_decision_requirement_not_execution_authority() -> None:
    row = next(
        r
        for r in GR11_EXTENSION_SURFACES
        if r.capability_id == "GR11-DECISION-REQUIREMENT"
    )
    assert row.authority.name == "GOVERNANCE_PROPOSAL_MATERIAL"


def test_gr11_g17_qualification_status_ready_for_audit_not_self_closed() -> None:
    assert GR11_QUALIFICATION_STATUS == "READY FOR AUDIT"
    assert GR11_QUALIFICATION_STATUS not in GR11_FORBIDDEN_PARENT_STATUSES


def test_gr11_g18_dynamic_registration_classified_for_every_row() -> None:
    for row in GR11_EXTENSION_SURFACES:
        assert isinstance(
            row.dynamic_registration, Gr11DynamicRegistrationApplicability
        )


def test_gr11_g19_historical_partial_rows_reconciled_to_qualified_or_na() -> None:
    for hist in GR11_HISTORICAL_PLUGINABILITY_RECONCILIATION:
        assert hist.current_result is Gr11QualificationStatus.QUALIFIED, hist.capability


def test_gr11_g20_no_partial_gap_unknown_inside_gr11_inventory() -> None:
    allowed = {
        Gr11QualificationStatus.QUALIFIED,
        Gr11QualificationStatus.NOT_APPLICABLE,
    }
    for row in GR11_EXTENSION_SURFACES:
        assert row.status in allowed, row.capability_id
