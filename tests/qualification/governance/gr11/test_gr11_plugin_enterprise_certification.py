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
    Gr11ExtensionSurface,
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


def _gr11_contract_segments(contract: str) -> tuple[str, ...]:
    return tuple(segment.strip() for segment in contract.split(";") if segment.strip())


def _gr11_contract_segment_symbol(segment: str) -> str:
    return segment.rsplit(".", 1)[-1].split(" ")[0]


def _gr11_contract_defining_module(segment: str) -> str | None:
    segment = segment.strip()
    if not segment.startswith("intergrax."):
        return None
    module_path = segment.rsplit(".", 1)[0].replace(".", "/") + ".py"
    return module_path


def _gr11_row_closed_world_modules(row: Gr11ExtensionSurface) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                row.semantic_owner_module,
                row.composition_owner_module,
                *row.consumer_scan_modules,
            )
        )
    )


def _parse_module_ast(rel: str) -> ast.Module:
    path = _REPO_ROOT / rel
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(rel))


def _module_declares_contract_class(rel: str, symbol: str) -> bool:
    tree = _parse_module_ast(rel)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == symbol:
            return True
    return False


def _annotation_references_name(node: ast.AST | None, name: str) -> bool:
    if node is None:
        return False
    if isinstance(node, ast.Name) and node.id == name:
        return True
    if isinstance(node, ast.Attribute) and node.attr == name:
        return True
    if isinstance(node, ast.Subscript):
        return _annotation_references_name(
            node.value, name
        ) or _annotation_references_name(node.slice, name)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _annotation_references_name(
            node.left, name
        ) or _annotation_references_name(node.right, name)
    if isinstance(node, ast.Tuple):
        return any(_annotation_references_name(elt, name) for elt in node.elts)
    return False


def _tree_references_name(tree: ast.AST, name: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == name:
            return True
        if isinstance(node, ast.Attribute) and node.attr == name:
            return True
    return False


def _composition_entry_functions(tree: ast.Module) -> tuple[ast.FunctionDef, ...]:
    return tuple(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and (node.name.startswith("build_") or node.name.startswith("wire_"))
    )


def _function_injects_contract_symbol(func: ast.FunctionDef, symbol: str) -> bool:
    for arg in (*func.args.args, *func.args.kwonlyargs):
        if _annotation_references_name(arg.annotation, symbol):
            return True
    return False


def _function_composes_contract_symbol(func: ast.FunctionDef, symbol: str) -> bool:
    if _annotation_references_name(func.returns, symbol):
        return True
    for node in ast.walk(func):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == symbol:
                return True
    return False


def _function_structurally_wires_symbol(func: ast.FunctionDef, symbol: str) -> bool:
    if _function_injects_contract_symbol(func, symbol):
        return True
    return _function_composes_contract_symbol(func, symbol)


def _class_init_wires_symbol(class_def: ast.ClassDef, symbol: str) -> bool:
    for item in class_def.body:
        if not isinstance(item, ast.FunctionDef) or item.name != "__init__":
            continue
        for arg in (*item.args.args, *item.args.kwonlyargs):
            if _annotation_references_name(arg.annotation, symbol):
                return True
    return False


def _class_structurally_wires_symbol(class_def: ast.ClassDef, symbol: str) -> bool:
    if _class_init_wires_symbol(class_def, symbol):
        return True
    for item in class_def.body:
        if isinstance(item, ast.FunctionDef) and _function_structurally_wires_symbol(
            item, symbol
        ):
            return True
    return False


def _annotation_root_name(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _annotation_root_name(node.value)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _annotation_root_name(node.left) or _annotation_root_name(node.right)
    return None


def _consumer_import_aliases(rel: str) -> frozenset[str]:
    dotted = _composition_owner_dotted_module(rel)
    aliases = {dotted}
    if dotted.startswith("applications."):
        aliases.add(dotted.removeprefix("applications."))
    return frozenset(aliases)


def _composition_delegate_imports(
    row: Gr11ExtensionSurface,
) -> dict[str, str]:
    consumer_modules: dict[str, str] = {}
    for rel in row.consumer_scan_modules:
        for alias in _consumer_import_aliases(rel):
            consumer_modules[alias] = rel
    mapping: dict[str, str] = {}
    tree = _parse_module_ast(row.composition_owner_module)
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        rel = consumer_modules.get(node.module)
        if rel is None:
            continue
        for alias in node.names:
            mapping[alias.asname or alias.name] = rel
    return mapping


def _wired_delegate_modules(row: Gr11ExtensionSurface) -> frozenset[str]:
    delegates = _composition_delegate_imports(row)
    if not delegates:
        return frozenset()
    tree = _parse_module_ast(row.composition_owner_module)
    wired: set[str] = set()
    for func in _composition_entry_functions(tree):
        for arg in (*func.args.args, *func.args.kwonlyargs):
            root = _annotation_root_name(arg.annotation)
            if root and root in delegates:
                wired.add(delegates[root])
        for node in ast.walk(func):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute) and isinstance(
                node.func.value, ast.Name
            ):
                if node.func.value.id in delegates:
                    wired.add(delegates[node.func.value.id])
    return frozenset(wired)


def _composition_owner_has_build_entrypoint(row: Gr11ExtensionSurface) -> bool:
    tree = _parse_module_ast(row.composition_owner_module)
    return bool(_composition_entry_functions(tree))


def _defining_module_for_symbol(row: Gr11ExtensionSurface, symbol: str) -> str | None:
    for segment in _gr11_contract_segments(row.contract):
        if _gr11_contract_segment_symbol(segment) == symbol:
            return _gr11_contract_defining_module(segment)
    return None


def _module_exposes_registry_resolution_seam(rel: str, symbol: str) -> bool:
    tree = _parse_module_ast(rel)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef):
                continue
            if item.name.startswith("_"):
                continue
            if _annotation_references_name(item.returns, symbol):
                return True
    return False


def _module_ast_structurally_wires_symbol(tree: ast.Module, symbol: str) -> bool:
    for func in _composition_entry_functions(tree):
        if _function_structurally_wires_symbol(func, symbol):
            return True
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and _class_structurally_wires_symbol(
            node, symbol
        ):
            return True
        if isinstance(node, ast.FunctionDef) and _function_structurally_wires_symbol(
            node, symbol
        ):
            return True
    return False


def _module_structurally_wires_symbol(rel: str, symbol: str) -> bool:
    return _module_ast_structurally_wires_symbol(_parse_module_ast(rel), symbol)


def _composition_owner_direct_wires_symbol(
    row: Gr11ExtensionSurface, symbol: str
) -> bool:
    return _module_structurally_wires_symbol(row.composition_owner_module, symbol)


def _composition_owner_wires_symbol(row: Gr11ExtensionSurface, symbol: str) -> bool:
    if _composition_owner_direct_wires_symbol(row, symbol):
        return True
    for rel in _wired_delegate_modules(row):
        if _module_structurally_wires_symbol(rel, symbol):
            return True
    defining_module = _defining_module_for_symbol(row, symbol)
    if (
        defining_module is not None
        and defining_module.endswith("plugin_spi.py")
        and _composition_owner_has_build_entrypoint(row)
    ):
        for rel in row.consumer_scan_modules:
            if _module_exposes_registry_resolution_seam(rel, symbol):
                return True
    return False


def _consumer_redeclares_contract_symbols(
    row: Gr11ExtensionSurface,
) -> list[str]:
    violations: list[str] = []
    for symbol in gr11_contract_symbols(row.contract):
        for rel in row.consumer_scan_modules:
            if _module_declares_contract_class(rel, symbol):
                violations.append(f"{rel} redefines {symbol}")
    return violations


def _composition_owner_dotted_module(composition_owner_module: str) -> str:
    return composition_owner_module.replace("/", ".").removesuffix(".py")


def _imported_composition_owner_callables(
    rel: str, composition_owner_module: str
) -> frozenset[str]:
    comp_aliases = _consumer_import_aliases(composition_owner_module)
    names: set[str] = set()
    tree = _parse_module_ast(rel)
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.module not in comp_aliases:
            continue
        for alias in node.names:
            names.add(alias.asname or alias.name)
    return frozenset(names)


def _function_calls_imported_callable(
    func: ast.FunctionDef, imported_names: frozenset[str]
) -> bool:
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in imported_names:
            return True
        if isinstance(node.func, ast.Attribute) and node.func.attr in imported_names:
            return True
    return False


def _consumer_competing_composition_wiring(
    row: Gr11ExtensionSurface,
) -> list[str]:
    violations: list[str] = []
    symbols = gr11_contract_symbols(row.contract)
    for rel in row.consumer_scan_modules:
        tree = _parse_module_ast(rel)
        delegated = _imported_composition_owner_callables(
            rel, row.composition_owner_module
        )
        for func in _composition_entry_functions(tree):
            wired_symbols = tuple(
                symbol
                for symbol in symbols
                if _function_composes_contract_symbol(func, symbol)
            )
            if not wired_symbols:
                continue
            if _function_calls_imported_callable(func, delegated):
                continue
            for symbol in wired_symbols:
                violations.append(
                    f"{rel}::{func.name} competes as composition owner for {symbol}"
                )
    return violations


def test_gr11_g07_semantic_owner_module_mechanically_unique_per_row() -> None:
    for row in GR11_EXTENSION_SURFACES:
        closed = _gr11_row_closed_world_modules(row)
        for rel in closed:
            assert (_REPO_ROOT / rel).is_file(), (row.capability_id, rel)
        for segment in _gr11_contract_segments(row.contract):
            symbol = _gr11_contract_segment_symbol(segment)
            defining_module = _gr11_contract_defining_module(segment)
            assert defining_module is not None, (row.capability_id, segment)
            assert (_REPO_ROOT / defining_module).is_file(), (
                row.capability_id,
                defining_module,
            )
            assert _module_declares_contract_class(defining_module, symbol), (
                row.capability_id,
                defining_module,
                symbol,
            )
            redeclarations = tuple(
                rel
                for rel in closed
                if rel != defining_module
                and _module_declares_contract_class(rel, symbol)
            )
            assert redeclarations == (), (
                row.capability_id,
                symbol,
                redeclarations,
            )


def test_gr11_g08_sanctioned_composition_owner_module_wires_contract_per_row() -> None:
    for row in GR11_EXTENSION_SURFACES:
        comp_path = _REPO_ROOT / row.composition_owner_module
        assert comp_path.is_file(), (row.capability_id, row.composition_owner_module)
        for symbol in gr11_contract_symbols(row.contract):
            assert _composition_owner_wires_symbol(row, symbol), (
                row.capability_id,
                row.composition_owner_module,
                symbol,
            )
        competing = _consumer_competing_composition_wiring(row)
        assert competing == [], (row.capability_id, competing)


def test_gr11_g08_regression_incidental_contract_reference_not_wiring() -> None:
    symbol = "ExampleGovernancePort"
    incidental_module = ast.parse(
        f"""
def build_host_runtime():
    _doc = {symbol}
"""
    )
    assert _tree_references_name(incidental_module, symbol)
    assert not _module_ast_structurally_wires_symbol(incidental_module, symbol)
    incidental_func = incidental_module.body[0]
    assert isinstance(incidental_func, ast.FunctionDef)
    assert _tree_references_name(incidental_func, symbol)
    assert not _function_structurally_wires_symbol(incidental_func, symbol)


def test_gr11_g09_consumers_do_not_redeclare_contract_ports() -> None:
    violations: list[str] = []
    for row in GR11_EXTENSION_SURFACES:
        violations.extend(_consumer_redeclares_contract_symbols(row))
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
