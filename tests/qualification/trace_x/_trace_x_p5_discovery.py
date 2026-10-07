# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-P0-R1 structural AST provenance discovery (registry-independent)."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

_REPO_ROOT = Path(__file__).resolve().parents[3]

_PRODUCTION_SCAN_ROOTS: Final[tuple[str, ...]] = ("intergrax", "agents", "applications")
_QUALIFICATION_FIXTURE_SCAN_ROOT: Final[str] = "tests/qualification/trace_x/r1_fixtures"
_PRODUCTION_EXCLUDE_DIR_NAMES: Final[frozenset[str]] = frozenset(
    {"tests", "docs", "examples", "__pycache__", "benchmarks", "model_runtime_proof"},
)

_POLICY_BUNDLE_FIELDS: Final[frozenset[str]] = frozenset(
    {"bundle_id", "bundle_version", "bundle_digest"},
)
_POLICY_DOCUMENT_FIELDS: Final[frozenset[str]] = frozenset(
    {"policy_document_id", "revision_id", "rule_id", "decision_id", "decision_ref"},
)
_POLICY_EVIDENCE_POINTER_FIELDS: Final[frozenset[str]] = frozenset(
    {"evidence_id", "policy_revisions"},
)
_PROFILE_REVISION_PAIR: Final[frozenset[str]] = frozenset({"revision_id", "fingerprint"})
_PROFILE_EXECUTION_BINDING: Final[frozenset[str]] = frozenset(
    {"tenant_id", "execution_id", "revision_id"},
)
_PROFILE_PROVENANCE_PROJECTION: Final[frozenset[str]] = frozenset(
    {"tenant_id", "execution_id", "revision_ref", "fingerprint"},
)
_PROFILE_SCOPE_FIELDS: Final[frozenset[str]] = frozenset({"application_id", "tenant_id"})
_CONFIG_FINGERPRINT_PAIR: Final[frozenset[str]] = frozenset(
    {"configuration_fingerprint", "configuration_version"},
)
_CONFIG_BINDING_FIELDS: Final[frozenset[str]] = frozenset(
    {"configuration_fingerprint", "configuration_type", "configuration_version"},
)
_CONFIG_REQUEST_FIELDS: Final[frozenset[str]] = frozenset(
    {"configuration_fingerprint", "current_revision", "request_id"},
)


def repo_root() -> Path:
    return _REPO_ROOT


def _production_path_excluded(rel_path: Path) -> bool:
    parts = rel_path.parts
    if _PRODUCTION_EXCLUDE_DIR_NAMES.intersection(parts):
        return True
    if "docker" in parts and "runtime-context" in parts:
        return True
    if "proofs" in parts:
        return True
    if "legacy" in parts:
        return True
    return False


def _iter_production_discovery_py_files() -> list[Path]:
    paths: list[Path] = []
    for root_name in _PRODUCTION_SCAN_ROOTS:
        root = _REPO_ROOT / root_name
        if not root.is_dir():
            continue
        for py_path in root.rglob("*.py"):
            rel = py_path.relative_to(_REPO_ROOT)
            if _production_path_excluded(rel):
                continue
            paths.append(py_path)
    return paths


def _iter_qualification_fixture_py_files() -> list[Path]:
    fixture_root = _REPO_ROOT / _QUALIFICATION_FIXTURE_SCAN_ROOT
    if not fixture_root.is_dir():
        return []
    return list(fixture_root.rglob("*.py"))


def _annotation_id(node: ast.expr | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _annotation_id(node.value)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left = _annotation_id(node.left)
        return left
    return None


def _class_field_names(class_def: ast.ClassDef) -> frozenset[str]:
    names: set[str] = set()
    for node in class_def.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.FunctionDef) and node.name == "__init__":
            for stmt in ast.walk(node):
                if (
                    isinstance(stmt, ast.AnnAssign)
                    and isinstance(stmt.target, ast.Attribute)
                    and isinstance(stmt.target.value, ast.Name)
                    and stmt.target.value.id == "self"
                ):
                    names.add(stmt.target.attr)
    return frozenset(names)


def _class_references_revision_validator(class_def: ast.ClassDef) -> bool:
    for node in ast.walk(class_def):
        if isinstance(node, ast.Call):
            callee = node.func
            label = ""
            if isinstance(callee, ast.Name):
                label = callee.id
            elif isinstance(callee, ast.Attribute):
                label = callee.attr
            lowered = label.lower()
            if "validate" in lowered and (
                "effective_profile" in lowered or "effprof" in lowered
            ):
                return True
    return False


def _is_policy_provenance_class(class_def: ast.ClassDef, fields: frozenset[str]) -> bool:
    if fields & _POLICY_BUNDLE_FIELDS and len(fields & _POLICY_BUNDLE_FIELDS) >= 3:
        return True
    if {"policy_document_id", "revision_id"} <= fields:
        return True
    if {"policy_document_id", "revision_id", "rule_id"} <= fields:
        return True
    if "policy_revisions" in fields and "derivation_snapshot_id" in fields:
        return True
    if "policy_revisions" in fields:
        return True
    if fields & _POLICY_EVIDENCE_POINTER_FIELDS and fields & _POLICY_DOCUMENT_FIELDS:
        return True
    if {"kind", "evidence_id"} <= fields:
        if fields & _POLICY_BUNDLE_FIELDS or "governance" in class_def.name.lower():
            return True
    return False


def _is_profile_revision_class(class_def: ast.ClassDef, fields: frozenset[str]) -> bool:
    if _PROFILE_REVISION_PAIR <= fields:
        return True
    if _PROFILE_EXECUTION_BINDING <= fields:
        return True
    if _PROFILE_PROVENANCE_PROJECTION <= fields:
        return True
    if _PROFILE_SCOPE_FIELDS <= fields and fields <= _PROFILE_SCOPE_FIELDS | frozenset({"revision_id"}):
        return True
    if _PROFILE_SCOPE_FIELDS <= fields and len(fields) <= 2:
        return True
    if fields == frozenset({"value"}) and _class_references_revision_validator(class_def):
        return True
    if fields == frozenset({"effective_profile"}) and "Isolation" in class_def.name:
        return True
    return False


def _method_param_names(func: ast.FunctionDef | ast.AsyncFunctionDef) -> frozenset[str]:
    names = {arg.arg for arg in func.args.args}
    names.update(arg.arg for arg in func.args.kwonlyargs)
    return frozenset(names)


def _is_protocol_stub_method(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    if not func.body:
        return True
    return all(
        isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
        for stmt in func.body
    )


def _is_profile_revision_store_implementation(class_def: ast.ClassDef) -> bool:
    methods = {
        node.name: node
        for node in class_def.body
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef)
    }
    if "save" in methods and "get" in methods:
        if "revision" in _method_param_names(methods["save"]):
            return True
    if "pin" in methods and "get" in methods:
        return "binding" in _method_param_names(methods["pin"]) or "tenant_id" in _method_param_names(
            methods["get"],
        )
    return False


def _is_profile_revision_protocol(class_def: ast.ClassDef) -> bool:
    method_names = {
        node.name
        for node in class_def.body
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef)
    }
    if {"save", "get"} <= method_names:
        for node in class_def.body:
            if isinstance(node, ast.FunctionDef) and node.name == "save":
                if "revision" in _method_param_names(node):
                    return True
    if "pin" in method_names and "get" in method_names:
        for node in class_def.body:
            if isinstance(node, ast.FunctionDef) and node.name == "pin":
                if "binding" in _method_param_names(node):
                    return True
    if "admit_root_execution" in method_names:
        return True
    if "read" in method_names:
        for node in class_def.body:
            if isinstance(node, ast.FunctionDef) and node.name == "read":
                params = _method_param_names(node)
                if {"tenant_id", "execution_id"} <= params:
                    return True
    return False


def _is_configuration_provenance_class(class_def: ast.ClassDef, fields: frozenset[str]) -> bool:
    if _CONFIG_FINGERPRINT_PAIR <= fields:
        return True
    if _CONFIG_BINDING_FIELDS <= fields:
        return True
    if _CONFIG_REQUEST_FIELDS <= fields and "tenant_id" in fields:
        return True
    if "configured_binding" in fields and "request_id" in fields:
        return True
    if "current_revision" in fields and "configuration_fingerprint" in fields:
        return True
    return False


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _function_builds_boundary_policy_section(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name == "PolicyDecisionSection":
            return True
        for keyword in node.keywords:
            if keyword.arg == "policy" and isinstance(keyword.value, ast.Call):
                inner = _call_name(keyword.value)
                if inner == "PolicyDecisionSection":
                    return True
    ret = func.returns
    ret_name = _annotation_id(ret)
    if ret_name and "ExecutionBoundaryEvent" in ret_name:
        for node in ast.walk(func):
            if isinstance(node, ast.Call) and _call_name(node) == "ExecutionBoundaryEvent":
                return True
    return False


def _function_param_names(func: ast.FunctionDef | ast.AsyncFunctionDef) -> frozenset[str]:
    names = {arg.arg for arg in func.args.args}
    names.update(arg.arg for arg in func.args.kwonlyargs)
    return frozenset(names)


def _function_records_governance_policy_evidence(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    param_names = _function_param_names(func)
    if "evaluation_point" not in param_names or "decision" not in param_names:
        return False
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name and "governance_fact" in name.lower():
                return True
            if name == "build_governance_fact_from_policy_decision":
                return True
    return "recorder" in param_names


def _method_projects_governance_inspection(
    rel: str,
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    if func.name != "read_governance_decisions":
        return False
    if _annotation_id(func.returns) != "RuntimeInspectionGovernanceSection":
        return False
    return "runtime_inspection/adapters" in rel


def _function_pins_profile_revision(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    param_names = _function_param_names(func)
    if not {"revision", "execution_id"} <= param_names:
        return False
    return "pinning_store" in param_names


def _function_materializes_profile_revision(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    ret_name = _annotation_id(func.returns)
    if ret_name != "EffectiveProfileRevision":
        return False
    for node in ast.walk(func):
        if isinstance(node, ast.Call) and _call_name(node) == "EffectiveProfileRevision":
            return True
    return False


def _function_resolves_active_profile_revision(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    ret_name = _annotation_id(func.returns)
    if ret_name == "ActiveEffectiveProfileRevisionBinding":
        return True
    if ret_name != "EffectiveProfileRevision":
        return False
    for node in ast.walk(func):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "get_active":
                return True
    return False


def _function_validates_configuration_realization(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return func.name == "validate_realization_request_invariants"


def _function_realizes_configuration(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    if func.name != "realize_admitted":
        return False
    ret_name = _annotation_id(func.returns)
    return ret_name == "ExistingCapabilityConfigurationRealizationResult"


def _is_configuration_realization_service(class_def: ast.ClassDef) -> bool:
    for node in class_def.body:
        if isinstance(node, ast.FunctionDef) and node.name == "realize_admitted":
            if _function_realizes_configuration(node):
                return True
    return False


def _is_configuration_strategy_class(class_def: ast.ClassDef) -> bool:
    for node in class_def.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name != "realize":
            continue
        if _annotation_id(node.returns) == "ConfiguredCapabilityBinding":
            return True
    return False


def _is_integration_target_class(class_def: ast.ClassDef, fields: frozenset[str]) -> bool:
    return {"tenant_id", "current_revision", "provider_id"} <= fields


def _discover_in_tree(
    rel: str,
    tree: ast.Module,
    *,
    policy: bool,
    profile: bool,
    configuration: bool,
) -> set[tuple[str, str]]:
    discovered: set[tuple[str, str]] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            if policy and _function_builds_boundary_policy_section(node):
                discovered.add((rel, node.name))
            if policy and _function_records_governance_policy_evidence(node):
                discovered.add((rel, node.name))
            if profile and _function_pins_profile_revision(node):
                discovered.add((rel, node.name))
            if profile and _function_materializes_profile_revision(node):
                discovered.add((rel, node.name))
            if profile and _function_resolves_active_profile_revision(node):
                discovered.add((rel, node.name))
            if configuration and _function_validates_configuration_realization(node):
                discovered.add((rel, node.name))
            if configuration and _function_realizes_configuration(node):
                discovered.add((rel, node.name))
        if not isinstance(node, ast.ClassDef):
            continue
        fields = _class_field_names(node)
        if policy and _is_policy_provenance_class(node, fields):
            discovered.add((rel, node.name))
        if profile and (
            _is_profile_revision_class(node, fields)
            or _is_profile_revision_protocol(node)
            or _is_profile_revision_store_implementation(node)
        ):
            discovered.add((rel, node.name))
        if configuration and (
            _is_configuration_provenance_class(node, fields)
            or _is_configuration_realization_service(node)
            or _is_configuration_strategy_class(node)
            or _is_integration_target_class(node, fields)
        ):
            discovered.add((rel, node.name))
        for child in node.body:
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            qualified = f"{node.name}.{child.name}"
            if policy and _method_projects_governance_inspection(rel, child):
                discovered.add((rel, qualified))
            if profile and child.name == "admit_root_execution":
                if _is_protocol_stub_method(child):
                    continue
                if "execution" in _method_param_names(child) or "execution_id" in _method_param_names(
                    child,
                ):
                    discovered.add((rel, qualified))
            if configuration and _function_realizes_configuration(child):
                discovered.add((rel, qualified))
    return discovered


def _discover_domain_in_paths(
    py_paths: list[Path],
    *,
    policy: bool = False,
    profile: bool = False,
    configuration: bool = False,
) -> frozenset[tuple[str, str]]:
    discovered: set[tuple[str, str]] = set()
    for py_path in py_paths:
        rel = py_path.relative_to(_REPO_ROOT).as_posix()
        try:
            tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        except SyntaxError:
            continue
        discovered |= _discover_in_tree(
            rel,
            tree,
            policy=policy,
            profile=profile,
            configuration=configuration,
        )
    return frozenset(discovered)


def _discover_production_domain(
    *,
    policy: bool = False,
    profile: bool = False,
    configuration: bool = False,
) -> frozenset[tuple[str, str]]:
    return _discover_domain_in_paths(
        _iter_production_discovery_py_files(),
        policy=policy,
        profile=profile,
        configuration=configuration,
    )


def _discover_qualification_fixture_domain(
    *,
    policy: bool = False,
    profile: bool = False,
    configuration: bool = False,
) -> frozenset[tuple[str, str]]:
    return _discover_domain_in_paths(
        _iter_qualification_fixture_py_files(),
        policy=policy,
        profile=profile,
        configuration=configuration,
    )


def discover_policy_provenance_surfaces() -> frozenset[tuple[str, str]]:
    return _discover_production_domain(policy=True)


def discover_profile_revision_surfaces() -> frozenset[tuple[str, str]]:
    return _discover_production_domain(profile=True)


def discover_configuration_provenance_surfaces() -> frozenset[tuple[str, str]]:
    return _discover_production_domain(configuration=True)


def discover_qualification_policy_sentinel_surfaces() -> frozenset[tuple[str, str]]:
    return _discover_qualification_fixture_domain(policy=True)


def discover_qualification_profile_sentinel_surfaces() -> frozenset[tuple[str, str]]:
    return _discover_qualification_fixture_domain(profile=True)


def discover_qualification_configuration_sentinel_surfaces() -> frozenset[tuple[str, str]]:
    return _discover_qualification_fixture_domain(configuration=True)


def discover_policy_surfaces_in_source(rel_path: str, source: str) -> frozenset[tuple[str, str]]:
    tree = ast.parse(source, filename=rel_path)
    return frozenset(_discover_in_tree(rel_path, tree, policy=True, profile=False, configuration=False))


def discover_profile_surfaces_in_source(rel_path: str, source: str) -> frozenset[tuple[str, str]]:
    tree = ast.parse(source, filename=rel_path)
    return frozenset(_discover_in_tree(rel_path, tree, policy=False, profile=True, configuration=False))


def discover_configuration_surfaces_in_source(rel_path: str, source: str) -> frozenset[tuple[str, str]]:
    tree = ast.parse(source, filename=rel_path)
    return frozenset(_discover_in_tree(rel_path, tree, policy=False, profile=False, configuration=True))


__all__ = [
    "discover_configuration_provenance_surfaces",
    "discover_configuration_surfaces_in_source",
    "discover_qualification_configuration_sentinel_surfaces",
    "discover_qualification_policy_sentinel_surfaces",
    "discover_qualification_profile_sentinel_surfaces",
    "discover_policy_provenance_surfaces",
    "discover_policy_surfaces_in_source",
    "discover_profile_revision_surfaces",
    "discover_profile_surfaces_in_source",
    "repo_root",
]
