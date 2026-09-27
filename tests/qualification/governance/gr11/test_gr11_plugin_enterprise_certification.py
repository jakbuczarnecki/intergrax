# © Artur Czarnecki. All rights reserved.

"""GR-11 — Governance plugin enterprise certification mechanical gate (G11-01..G11-20)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from tests.qualification.governance.gr11.catalog import (
    GR11_COMPOSITION_SELF_REGISTRATION_MARKERS,
    GR11_EXTENSION_SURFACES,
    GR11_FORBIDDEN_PARENT_STATUSES,
    GR11_IMPLEMENTATION_BRANCH_FORBIDDEN_PATTERNS,
    GR11_IMPLEMENTATION_BRANCH_SCAN_MODULES,
    GR11_QUALIFICATION_STATUS,
    GR11_WEAK_BOUNDARY_SCAN_MODULES,
    Gr11DynamicRegistrationApplicability,
    Gr11QualificationStatus,
    gr11_all_registered_proof_nodes,
    gr11_all_structural_replaceability_nodes,
    gr11_capability_ids,
    gr11_contract_symbols,
    gr11_row_all_proof_nodes,
)
from tests.qualification.governance.gr12.qualification_support import (
    assert_proof_nodes_registered,
)
from tests.qualification.governance.pytest_node_integrity import (
    pytest_nodes_missing_from_collection,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _resolve_type_aliases(tree: ast.Module) -> dict[str, ast.AST]:
    aliases: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value, ast.Name):
                    aliases[target.id] = node.value
    return aliases


def _annotation_uses_forbidden(
    node: ast.AST,
    aliases: dict[str, ast.AST],
) -> bool:
    if isinstance(node, ast.Name):
        if node.id in {"Any", "object"}:
            return True
        if node.id in aliases:
            return _annotation_uses_forbidden(aliases[node.id], aliases)
    if isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Name) and node.value.id in {
            "dict",
            "Dict",
            "Mapping",
        }:
            return True
    if isinstance(node, ast.Tuple):
        return any(_annotation_uses_forbidden(elt, aliases) for elt in node.elts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _annotation_uses_forbidden(
            node.left, aliases
        ) or _annotation_uses_forbidden(node.right, aliases)
    return False


def _is_semantic_extension_contract_class(node: ast.ClassDef) -> bool:
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id in {"Protocol", "ABC"}:
            return True
        if isinstance(base, ast.Attribute) and base.attr in {"Protocol", "ABC"}:
            return True
    return node.name.endswith("Port") or node.name.endswith("Store")


def _collect_contract_port_annotations(path: Path) -> list[tuple[str, ast.AST]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    collected: list[tuple[str, ast.AST]] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if not _is_semantic_extension_contract_class(node):
            continue
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if item.name.startswith("_"):
                continue
            for arg in (*item.args.args, *item.args.kwonlyargs):
                if arg.annotation is not None:
                    collected.append(
                        (f"{node.name}.{item.name}.{arg.arg}", arg.annotation)
                    )
            if item.returns is not None:
                collected.append((f"{node.name}.{item.name}:return", item.returns))
    return collected


def _assert_gr11_proof_nodes_registered(node_ids: tuple[str, ...]) -> None:
    normalized: list[str] = []
    for node_id in node_ids:
        rel, name = node_id.split("::", 1)
        if "[" in name:
            base = name.split("[", 1)[0]
            normalized.append(f"{rel}::{base}")
        else:
            normalized.append(node_id)
    assert_proof_nodes_registered(tuple(dict.fromkeys(normalized)))


def test_gr11_g01_every_row_uniquely_identified() -> None:
    ids = gr11_capability_ids()
    assert len(ids) == len(set(ids))
    assert len(ids) == 9


def test_gr11_g02_every_extensible_mechanism_exposes_platform_contract() -> None:
    for row in GR11_EXTENSION_SURFACES:
        assert row.contract.strip(), row.capability_id
        assert "intergrax." in row.contract


def test_gr11_g03_structural_replaceability_proof_for_every_qualified_row() -> None:
    for row in GR11_EXTENSION_SURFACES:
        if row.status is not Gr11QualificationStatus.QUALIFIED:
            continue
        assert row.structural_replaceability_proof_nodes, row.capability_id


def test_gr11_g04_typed_evidence_categories_populated_for_qualified_rows() -> None:
    for row in GR11_EXTENSION_SURFACES:
        if row.status is not Gr11QualificationStatus.QUALIFIED:
            continue
        assert row.authority_proof_nodes, row.capability_id
        assert row.composition_proof_nodes, row.capability_id
        assert row.negative_bypass_proof_nodes, row.capability_id
        assert row.structural_replaceability_proof_nodes, row.capability_id


def test_gr11_g05_all_registered_proof_nodes_collectable() -> None:
    missing, proc = pytest_nodes_missing_from_collection(
        gr11_all_registered_proof_nodes(), _REPO_ROOT
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not missing, missing


def test_gr11_g06_structural_replaceability_nodes_collectable() -> None:
    missing, proc = pytest_nodes_missing_from_collection(
        gr11_all_structural_replaceability_nodes(), _REPO_ROOT
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not missing, missing


def _contract_module_paths(row) -> set[Path]:
    paths = {_REPO_ROOT / row.semantic_owner_module}
    for segment in row.contract.split(";"):
        segment = segment.strip()
        if not segment.startswith("intergrax."):
            continue
        module = segment.rsplit(".", 1)[0].replace(".", "/") + ".py"
        paths.add(_REPO_ROOT / module)
    return paths


def test_gr11_g07_semantic_owner_module_mechanically_unique_per_row() -> None:
    for row in GR11_EXTENSION_SURFACES:
        paths = _contract_module_paths(row)
        for path in paths:
            assert path.is_file(), (row.capability_id, path)
        combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
        for symbol in gr11_contract_symbols(row.contract):
            assert f"class {symbol}" in combined or f"class {symbol}(" in combined, (
                row.capability_id,
                symbol,
            )


def test_gr11_g08_sanctioned_composition_owner_module_wires_contract_per_row() -> None:
    for row in GR11_EXTENSION_SURFACES:
        rel_paths = (row.composition_owner_module, *row.consumer_scan_modules)
        sources: list[str] = []
        for rel in dict.fromkeys(rel_paths):
            path = _REPO_ROOT / rel
            assert path.is_file(), (row.capability_id, rel)
            sources.append(path.read_text(encoding="utf-8"))
        combined = "\n".join(sources)
        for symbol in gr11_contract_symbols(row.contract):
            assert symbol in combined, (row.capability_id, symbol)


def test_gr11_g09_consumers_do_not_redeclare_contract_ports() -> None:
    violations: list[str] = []
    for row in GR11_EXTENSION_SURFACES:
        for symbol in gr11_contract_symbols(row.contract):
            for rel in row.consumer_scan_modules:
                path = _REPO_ROOT / rel
                assert path.is_file(), rel
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in tree.body:
                    if isinstance(node, ast.ClassDef) and node.name == symbol:
                        violations.append(f"{rel} redefines {symbol}")
    assert violations == []


def test_gr11_g10_no_vendor_literal_in_governance_contract_modules() -> None:
    contract_paths = [
        _REPO_ROOT / rel
        for rel in GR11_WEAK_BOUNDARY_SCAN_MODULES
        if rel.startswith("intergrax/contracts/")
    ]
    vendor_markers = ("openai", "anthropic", "azure_openai", "vendor_")
    for path in contract_paths:
        text = path.read_text(encoding="utf-8").lower()
        for marker in vendor_markers:
            assert marker not in text, f"{path.name} contains vendor marker {marker!r}"


def test_gr11_g11_no_implementation_type_branch_in_closed_world_consumers() -> None:
    for rel in GR11_IMPLEMENTATION_BRANCH_SCAN_MODULES:
        path = _REPO_ROOT / rel
        assert path.is_file(), rel
        text = path.read_text(encoding="utf-8")
        for pattern in GR11_IMPLEMENTATION_BRANCH_FORBIDDEN_PATTERNS:
            assert not re.search(pattern, text), f"{rel}: matched {pattern}"


def test_gr11_g12_no_reflection_semantic_plugin_dispatch_in_scan_modules() -> None:
    for rel in GR11_WEAK_BOUNDARY_SCAN_MODULES:
        path = _REPO_ROOT / rel
        text = path.read_text(encoding="utf-8")
        assert "importlib.import_module" not in text, rel
        assert "__import__(" not in text, rel


def test_gr11_g13_semantic_weak_boundary_on_contract_port_methods() -> None:
    offenders: list[str] = []
    for rel in GR11_WEAK_BOUNDARY_SCAN_MODULES:
        if not rel.startswith("intergrax/contracts/"):
            continue
        path = _REPO_ROOT / rel
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        aliases = _resolve_type_aliases(tree)
        for label, annotation in _collect_contract_port_annotations(path):
            if _annotation_uses_forbidden(annotation, aliases):
                offenders.append(f"{rel}:{label}")
        text = path.read_text(encoding="utf-8")
        for forbidden in (
            "dict[str, Any]",
            "Mapping[str, Any]",
            "getattr(",
            "hasattr(",
            "setattr(",
        ):
            if forbidden in text:
                offenders.append(f"{rel}:text:{forbidden}")
    assert offenders == [], offenders


def test_gr11_g14_authority_proof_nodes_registered_not_catalog_metadata() -> None:
    nodes: list[str] = []
    for row in GR11_EXTENSION_SURFACES:
        if row.status is not Gr11QualificationStatus.QUALIFIED:
            continue
        assert row.authority_proof_nodes, row.capability_id
        nodes.extend(row.authority_proof_nodes)
        assert row.may_widen_authority is False
    _assert_gr11_proof_nodes_registered(tuple(dict.fromkeys(nodes)))


def test_gr11_g15_composition_and_negative_proof_nodes_registered() -> None:
    composition_nodes: list[str] = []
    negative_nodes: list[str] = []
    for row in GR11_EXTENSION_SURFACES:
        if row.status is not Gr11QualificationStatus.QUALIFIED:
            continue
        composition_nodes.extend(row.composition_proof_nodes)
        negative_nodes.extend(row.negative_bypass_proof_nodes)
    _assert_gr11_proof_nodes_registered(tuple(dict.fromkeys(composition_nodes)))
    _assert_gr11_proof_nodes_registered(tuple(dict.fromkeys(negative_nodes)))


def test_gr11_g16_fail_closed_admission_regression_nodes_registered() -> None:
    assert_proof_nodes_registered(
        (
            "tests/unit/runtime/governance/test_runtime_execution_policy_admission.py::"
            "test_evaluator_unconfigured_fail_closed",
            "tests/unit/runtime/governance/test_runtime_execution_policy_admission.py::"
            "test_unavailable_adapter_fail_closed",
        )
    )


def test_gr11_g17_continuation_structural_proof_is_two_implementation_pluginability() -> (
    None
):
    row = next(
        r for r in GR11_EXTENSION_SURFACES if r.capability_id == "GR11-CONTINUATION"
    )
    assert "Execution" in row.semantic_owner
    structural = row.structural_replaceability_proof_nodes[0]
    assert "test_pluginability_two_implementations" in structural
    assert (
        "test_mp4r3_no_duplicate_continuation_lifecycle_authority"
        in (row.authority_proof_nodes[0])
    )


def test_gr11_g18_provider_invocation_store_reliability_fact_owner() -> None:
    row = next(
        r
        for r in GR11_EXTENSION_SURFACES
        if r.capability_id == "GR11-PROVIDER-INVOCATION-STORE"
    )
    assert row.authority.name == "RELIABILITY_FACT"


def test_gr11_g19_decision_requirement_not_execution_authority() -> None:
    row = next(
        r
        for r in GR11_EXTENSION_SURFACES
        if r.capability_id == "GR11-DECISION-REQUIREMENT"
    )
    assert row.authority.name == "GOVERNANCE_PROPOSAL_MATERIAL"


def test_gr11_g20_qualification_status_ready_for_audit_not_self_closed() -> None:
    assert GR11_QUALIFICATION_STATUS == "READY FOR AUDIT"
    assert GR11_QUALIFICATION_STATUS not in GR11_FORBIDDEN_PARENT_STATUSES


def test_gr11_g21_dynamic_registration_classified_and_composition_time_evidence() -> (
    None
):
    for row in GR11_EXTENSION_SURFACES:
        assert isinstance(
            row.dynamic_registration, Gr11DynamicRegistrationApplicability
        )
        if (
            row.dynamic_registration
            is not Gr11DynamicRegistrationApplicability.COMPOSITION_TIME_ONLY
        ):
            continue
        path = _REPO_ROOT / row.composition_owner_module
        text = path.read_text(encoding="utf-8")
        for marker in GR11_COMPOSITION_SELF_REGISTRATION_MARKERS:
            assert marker not in text, f"{row.capability_id}: {marker}"


def test_gr11_g22_historical_partial_rows_reconciled_to_qualified_or_na() -> None:
    from tests.qualification.governance.gr11.catalog import (
        GR11_HISTORICAL_PLUGINABILITY_RECONCILIATION,
    )

    for hist in GR11_HISTORICAL_PLUGINABILITY_RECONCILIATION:
        assert hist.current_result is Gr11QualificationStatus.QUALIFIED, hist.capability


def test_gr11_g23_no_partial_gap_unknown_inside_gr11_inventory() -> None:
    allowed = {
        Gr11QualificationStatus.QUALIFIED,
        Gr11QualificationStatus.NOT_APPLICABLE,
    }
    for row in GR11_EXTENSION_SURFACES:
        assert row.status in allowed, row.capability_id
        assert gr11_row_all_proof_nodes(row), row.capability_id


def test_gr11_g24_control_plane_evaluator_behind_cla04_contract() -> None:
    row = next(
        r
        for r in GR11_EXTENSION_SURFACES
        if r.capability_id == "GR11-CONTROL-PLANE-EVALUATOR"
    )
    assert "ControlPlaneMutationPolicyEvaluator" in row.contract
