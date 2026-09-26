# © Artur Czarnecki. All rights reserved.

"""AST qualification for public contract module shape (EBH-2I)."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from testing_support.architecture.public_contract_boundary.discovery import (
    discover_public_contract_source_files,
    path_to_module_name,
)

_TYPE_IGNORE_RE = re.compile(r"#\s*type:\s*ignore\b")
_MUTABLE_REGISTRY_FIELD_NAMES = frozenset(
    {"_services", "_registry", "_cache", "_state"},
)
@dataclass(frozen=True, slots=True)
class ContractSurfacePurityDebtEntry:
    finding_id: str
    source_path: str
    line: int
    rule_id: str


# Pre-existing transport / decorator typing debt — removal tracked outside EBH-2I.
CONTRACT_SURFACE_PURITY_DEBT: tuple[ContractSurfacePurityDebtEntry, ...] = (
    ContractSurfacePurityDebtEntry(
        finding_id="D-CSP-01",
        source_path="intergrax/contracts/sandbox_network_egress.py",
        line=179,
        rule_id="architecture_masking_type_ignore",
    ),
    ContractSurfacePurityDebtEntry(
        finding_id="D-CSP-02",
        source_path="intergrax/contracts/delegated_correlation_query_index_backfill.py",
        line=42,
        rule_id="architecture_masking_type_ignore",
    ),
    ContractSurfacePurityDebtEntry(
        finding_id="D-CSP-03",
        source_path="intergrax/contracts/application_observability_attributes.py",
        line=68,
        rule_id="architecture_masking_cast",
    ),
    ContractSurfacePurityDebtEntry(
        finding_id="D-CSP-04",
        source_path="intergrax/contracts/application_observability_attributes.py",
        line=94,
        rule_id="architecture_masking_cast",
    ),
)


@dataclass(frozen=True, slots=True)
class ContractSurfacePurityViolation:
    source_path: str
    source_module: str
    rule_id: str
    line: int
    detail: str

    def as_message(self) -> str:
        return (
            f"{self.source_path}:{self.line}: {self.source_module} "
            f"[{self.rule_id}] {self.detail}"
        )


def _decorator_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    return None


def _is_frozen_dataclass(class_def: ast.ClassDef) -> bool:
    for decorator in class_def.decorator_list:
        name = _decorator_name(decorator)
        if name != "dataclass":
            continue
        if not isinstance(decorator, ast.Call):
            return False
        for keyword in decorator.keywords:
            if keyword.arg == "frozen" and isinstance(keyword.value, ast.Constant):
                return keyword.value.value is True
        return False
    return False


def _inherits_protocol(class_def: ast.ClassDef) -> bool:
    for base in class_def.bases:
        if isinstance(base, ast.Name) and base.id == "Protocol":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "Protocol":
            return True
    return False


def _inherits_exception(class_def: ast.ClassDef) -> bool:
    for base in class_def.bases:
        if isinstance(base, ast.Name) and base.id in {"Exception", "BaseException"}:
            return True
        if isinstance(base, ast.Attribute) and base.attr in {"Exception", "BaseException", "Error"}:
            return True
        if isinstance(base, ast.Name) and base.id.endswith("Error"):
            return True
    return False


def _inherits_enum(class_def: ast.ClassDef) -> bool:
    for base in class_def.bases:
        if isinstance(base, ast.Name) and base.id == "Enum":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "Enum":
            return True
    return False


def _class_has_abstract_method(class_def: ast.ClassDef) -> bool:
    for node in class_def.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            name = _decorator_name(decorator)
            if name == "abstractmethod":
                return True
    return False


def _mutable_state_field_targets(class_def: ast.ClassDef) -> list[str]:
    hits: list[str] = []
    for node in class_def.body:
        target_name: str | None = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_name = node.target.id
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    target_name = target.id
                    break
        if target_name in _MUTABLE_REGISTRY_FIELD_NAMES:
            hits.append(target_name)
    return hits


def _is_pydantic_model(class_def: ast.ClassDef) -> bool:
    for base in class_def.bases:
        name = None
        if isinstance(base, ast.Name):
            name = base.id
        elif isinstance(base, ast.Attribute):
            name = base.attr
        if name == "BaseModel":
            return True
    return False


def _concrete_implementation_class_violation(class_def: ast.ClassDef) -> str | None:
    if _inherits_protocol(class_def) or _inherits_exception(class_def) or _inherits_enum(class_def):
        return None
    if _class_has_abstract_method(class_def):
        return None
    if _is_pydantic_model(class_def):
        return None
    mutable_fields = _mutable_state_field_targets(class_def)
    if mutable_fields:
        return f"mutable registry-style fields on contract class: {', '.join(mutable_fields)}"
    if any(_decorator_name(dec) == "dataclass" for dec in class_def.decorator_list):
        if not _is_frozen_dataclass(class_def):
            method_names = {
                node.name
                for node in class_def.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if {"register", "seal", "close"} & method_names:
                return "non-frozen dataclass service registry implementation in contract module"
    return None


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _is_vendor_attribute_access_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr == "optional":
        root = func.value
        if isinstance(root, ast.Name) and root.id in {"attribute_access", "vendor_attribute_access"}:
            return True
    return False


def _second_arg_is_dynamic(node: ast.Call) -> bool:
    if len(node.args) < 2:
        return False
    attr = node.args[1]
    return not isinstance(attr, ast.Constant)


def _attr_name_suggests_enum_value(attr: ast.expr) -> bool:
    return isinstance(attr, ast.Attribute) and attr.attr == "value"


def _is_self_field_validation_getattr(node: ast.Call) -> bool:
    name = _call_name(node)
    if name not in {"getattr", "hasattr"} or len(node.args) < 2:
        return False
    obj = node.args[0]
    attr = node.args[1]
    if not isinstance(obj, ast.Name) or obj.id != "self":
        return False
    if isinstance(attr, ast.Constant):
        return True
    return not _attr_name_suggests_enum_value(attr)


def _is_dynamic_semantic_dispatch_call(node: ast.Call) -> bool:
    name = _call_name(node)
    if name in {"getattr", "hasattr"}:
        if _is_self_field_validation_getattr(node):
            return False
        return _second_arg_is_dynamic(node)
    if _is_vendor_attribute_access_call(node) and _second_arg_is_dynamic(node):
        return True
    return False


def _is_cast_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name) and func.id == "cast":
        return True
    if isinstance(func, ast.Attribute) and func.attr == "cast":
        return True
    return False


def _scan_type_ignore_lines(source: str) -> list[int]:
    lines: list[int] = []
    for index, line in enumerate(source.splitlines(), start=1):
        if _TYPE_IGNORE_RE.search(line):
            lines.append(index)
    return lines


def _debt_covers(
    violation: ContractSurfacePurityViolation,
    entry: ContractSurfacePurityDebtEntry,
) -> bool:
    return (
        violation.source_path == entry.source_path
        and violation.line == entry.line
        and violation.rule_id == entry.rule_id
    )


def _violations_for_source(
    *,
    source: str,
    rel_path: str,
    source_module: str,
) -> list[ContractSurfacePurityViolation]:
    violations: list[ContractSurfacePurityViolation] = []
    tree = ast.parse(source, filename=rel_path)

    for line in _scan_type_ignore_lines(source):
        violations.append(
            ContractSurfacePurityViolation(
                source_path=rel_path,
                source_module=source_module,
                rule_id="architecture_masking_type_ignore",
                line=line,
                detail="type ignore suppresses contract typing on public contract surface",
            ),
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            detail = _concrete_implementation_class_violation(node)
            if detail:
                violations.append(
                    ContractSurfacePurityViolation(
                        source_path=rel_path,
                        source_module=source_module,
                        rule_id="implementation_laundering",
                        line=node.lineno,
                        detail=detail,
                    ),
                )
        if isinstance(node, ast.Call):
            if _is_cast_call(node):
                violations.append(
                    ContractSurfacePurityViolation(
                        source_path=rel_path,
                        source_module=source_module,
                        rule_id="architecture_masking_cast",
                        line=node.lineno,
                        detail="typing.cast masks contract typing on public contract surface",
                    ),
                )
            if _is_dynamic_semantic_dispatch_call(node):
                violations.append(
                    ContractSurfacePurityViolation(
                        source_path=rel_path,
                        source_module=source_module,
                        rule_id="dynamic_semantic_dispatch",
                        line=node.lineno,
                        detail="dynamic attribute dispatch on public contract surface",
                    ),
                )
    return violations


def evaluate_contract_surface_purity(
    repo_root: Path,
    *,
    paths: tuple[Path, ...] | None = None,
    debt_entries: tuple[ContractSurfacePurityDebtEntry, ...] | None = None,
) -> tuple[ContractSurfacePurityViolation, ...]:
    intergrax_root = repo_root / "intergrax"
    target_paths = paths if paths is not None else discover_public_contract_source_files(repo_root)
    registry = debt_entries if debt_entries is not None else CONTRACT_SURFACE_PURITY_DEBT
    violations: list[ContractSurfacePurityViolation] = []

    for path in target_paths:
        if not path.is_file():
            continue
        rel_path = path.relative_to(repo_root).as_posix()
        source_module = path_to_module_name(path, intergrax_root=intergrax_root)
        try:
            source = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError):
            continue
        violations.extend(
            _violations_for_source(
                source=source,
                rel_path=rel_path,
                source_module=source_module,
            ),
        )

    unregistered = [
        violation
        for violation in violations
        if not any(_debt_covers(violation, entry) for entry in registry)
    ]
    unregistered.sort(key=lambda item: (item.source_path, item.line, item.rule_id))
    return tuple(unregistered)


def evaluate_contract_surface_purity_on_source(
    *,
    source: str,
    source_path: str = "intergrax/snippet/contracts/snippet.py",
    source_module: str = "intergrax.snippet.contracts.snippet",
    debt_entries: tuple[ContractSurfacePurityDebtEntry, ...] | None = (),
) -> tuple[ContractSurfacePurityViolation, ...]:
    violations = _violations_for_source(
        source=source,
        rel_path=source_path,
        source_module=source_module,
    )
    registry = debt_entries if debt_entries is not None else CONTRACT_SURFACE_PURITY_DEBT
    return tuple(
        violation
        for violation in violations
        if not any(_debt_covers(violation, entry) for entry in registry)
    )
