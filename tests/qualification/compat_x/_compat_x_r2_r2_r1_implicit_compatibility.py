# © Artur Czarnecki. All rights reserved.

"""Behavior-/structure-oriented implicit compatibility discovery (COMPAT-X-R2-R2-R1-R1)."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Final

from tests.qualification.compat_x._compat_x_ast_signals import parse_module
from tests.qualification.compat_x._compat_x_legacy_obligations import (
    legacy_persisted_compatibility_obligations,
)
from tests.qualification.compat_x._compat_x_persisted_inventory import PERSISTED_SCHEMA_FAMILIES
from tests.qualification.compat_x._compat_x_persisted_types import LegacyCompatibilityDisposition

_REPO_ROOT = Path(__file__).resolve().parents[3]

_CANONICAL_CURRENT_MODEL_MODULES: Final[dict[str, str]] = {
    "RuntimeEvent": "intergrax/contracts/runtime_event.py",
    "SuspendedExecutionOperationDescriptor": (
        "intergrax/contracts/execution/suspended_operation/descriptor.py"
    ),
    "AgentArtifactMetadata": "intergrax/agent_distribution/stores.py",
    "RuntimeMaterializationRecord": "intergrax/agent_distribution/runtime_materialization_record.py",
    "DeploymentInstanceRecord": "intergrax/agent_distribution/deployment.py",
    "ApplicationEnvironmentServingRecord": "intergrax/agent_distribution/stores.py",
    "TaskMemoryRecord": "intergrax/runtime/task_memory/models.py",
    "ConversationalMemoryPersistedPayload": "intergrax/memory/conversational_memory_persistence.py",
    "RegistrySnapshotPersistedPayload": "intergrax/applications/_shared/registry_snapshot_persistence.py",
}

_MIGRATION_BOUNDARY_MODULES: Final[frozenset[str]] = frozenset(
    {
        "intergrax/agent_distribution/sqlite_payload_legacy.py",
        "intergrax/runtime/task_memory/sqlite_row_legacy.py",
        "intergrax/memory/user_profile_serialization.py",
        "intergrax/memory/user_profile_legacy_wire.py",
        "intergrax/memory/conversational_memory_persistence.py",
        "intergrax/applications/_shared/registry_snapshot_persistence.py",
        "intergrax/contracts/execution/suspended_operation/descriptor_persisted.py",
        "intergrax/contracts/idempotency_store.py",
    }
)

_FORBIDDEN_SEMANTIC_ANNOTATION_FRAGMENTS: Final[tuple[str, ...]] = (
    "Any",
    "dict[str, Any]",
    "Dict[str, Any]",
    "list[dict[str, Any]]",
    "List[Dict[str, Any]]",
    "Callable[..., Any]",
)

_LEGACY_MODULE_MARKERS: Final[tuple[str, ...]] = (
    "_legacy.py",
    "descriptor_persisted.py",
    "LEGACY_UNVERSIONED_PRE_V1",
    "LEGACY_DESCRIPTOR_WITHOUT_AUTHORITY_SCOPE",
)


class ImplicitCompatibilityDisposition(StrEnum):
    EXPLICIT_SANCTIONED_LEGACY_READER = "EXPLICIT_SANCTIONED_LEGACY_READER"
    CURRENT_ONLY_NORMALIZATION = "CURRENT_ONLY_NORMALIZATION"
    NON_COMPAT_VALIDATION = "NON_COMPAT_VALIDATION"
    BLOCKER_IMPLICIT_COMPATIBILITY = "BLOCKER_IMPLICIT_COMPATIBILITY"


@dataclass(frozen=True, slots=True)
class ImplicitCompatibilityCandidate:
    module_path: str
    lineno: int
    signal: str
    detail: str
    enclosing_class: str | None = None
    enclosing_function: str | None = None


@dataclass(frozen=True, slots=True)
class ClassifiedImplicitCompatibility:
    candidate: ImplicitCompatibilityCandidate
    disposition: ImplicitCompatibilityDisposition


@dataclass(frozen=True, slots=True)
class ExplicitImplicitCompatibilityClassification:
    module_path: str
    class_name: str | None
    function_name: str
    signal: str
    disposition: ImplicitCompatibilityDisposition
    reason: str


@dataclass(frozen=True, slots=True)
class ImplicitCompatibilityReport:
    scanned_modules: tuple[str, ...]
    raw_candidates: tuple[ImplicitCompatibilityCandidate, ...]
    classified: tuple[ClassifiedImplicitCompatibility, ...]
    orphan_migration_modules: tuple[str, ...]
    strong_typing_violations: tuple[str, ...]
    module_wide_sanctioned_exemptions: int

    @property
    def blockers(self) -> tuple[ClassifiedImplicitCompatibility, ...]:
        return tuple(
            item
            for item in self.classified
            if item.disposition == ImplicitCompatibilityDisposition.BLOCKER_IMPLICIT_COMPATIBILITY
        )

    @property
    def sanctioned(self) -> tuple[ClassifiedImplicitCompatibility, ...]:
        return tuple(
            item
            for item in self.classified
            if item.disposition == ImplicitCompatibilityDisposition.EXPLICIT_SANCTIONED_LEGACY_READER
        )


_EXPLICIT_CLASSIFICATIONS: Final[tuple[ExplicitImplicitCompatibilityClassification, ...]] = (
    ExplicitImplicitCompatibilityClassification(
        "intergrax/contracts/runtime_event.py",
        "RuntimeEvent",
        "_validate_event_id_field",
        "pydantic_before_validator",
        ImplicitCompatibilityDisposition.CURRENT_ONLY_NORMALIZATION,
        "RuntimeEvent event_id normalization (current-only)",
    ),
    ExplicitImplicitCompatibilityClassification(
        "intergrax/contracts/runtime_event.py",
        "RuntimeEvent",
        "_validate_task_id_field",
        "pydantic_before_validator",
        ImplicitCompatibilityDisposition.CURRENT_ONLY_NORMALIZATION,
        "RuntimeEvent task_id normalization (current-only)",
    ),
    ExplicitImplicitCompatibilityClassification(
        "intergrax/contracts/runtime_event.py",
        "RuntimeEvent",
        "_validate_run_id_field",
        "pydantic_before_validator",
        ImplicitCompatibilityDisposition.CURRENT_ONLY_NORMALIZATION,
        "RuntimeEvent run_id normalization (current-only)",
    ),
    ExplicitImplicitCompatibilityClassification(
        "intergrax/contracts/runtime_event.py",
        "RuntimeEvent",
        "_validate_attempt_id_field",
        "pydantic_before_validator",
        ImplicitCompatibilityDisposition.CURRENT_ONLY_NORMALIZATION,
        "RuntimeEvent attempt_id normalization (current-only)",
    ),
    ExplicitImplicitCompatibilityClassification(
        "intergrax/contracts/runtime_event.py",
        "RuntimeEvent",
        "_validate_execution_id_field",
        "pydantic_before_validator",
        ImplicitCompatibilityDisposition.CURRENT_ONLY_NORMALIZATION,
        "RuntimeEvent execution_id normalization (current-only)",
    ),
    ExplicitImplicitCompatibilityClassification(
        "intergrax/contracts/runtime_event.py",
        "RuntimeEvent",
        "_validate_parent_event_id_field",
        "pydantic_before_validator",
        ImplicitCompatibilityDisposition.CURRENT_ONLY_NORMALIZATION,
        "RuntimeEvent parent_event_id normalization (current-only)",
    ),
)


def _sanctioned_boundary_symbols() -> frozenset[tuple[str, str]]:
    """(module_path, function_name) pairs registered via legacy obligations."""
    symbols: set[tuple[str, str]] = set()
    for obligation in legacy_persisted_compatibility_obligations():
        if obligation.migration_owner_path is None:
            continue
        if obligation.disposition not in (
            LegacyCompatibilityDisposition.EXPLICIT_LEGACY_READER,
            LegacyCompatibilityDisposition.EXPLICIT_MIGRATION,
            LegacyCompatibilityDisposition.CURRENT_ONLY_FAIL_CLOSED,
        ):
            continue
        module = obligation.migration_owner_path.replace("\\", "/")
        for symbol in obligation.sanctioned_reader_symbols:
            symbols.add((module, symbol))
    return frozenset(symbols)


def derive_compatibility_scan_universe() -> tuple[str, ...]:
    paths: set[str] = set()
    for family in PERSISTED_SCHEMA_FAMILIES:
        for reader in family.reader_paths:
            paths.add(reader.replace("\\", "/"))
        if family.migration_owner_path:
            paths.add(family.migration_owner_path.replace("\\", "/"))
        paths.add(family.owner_module_path.replace("\\", "/"))
    for obligation in legacy_persisted_compatibility_obligations():
        paths.add(obligation.reader_path.replace("\\", "/"))
        paths.add(obligation.semantic_owner_path.replace("\\", "/"))
        if obligation.migration_owner_path:
            paths.add(obligation.migration_owner_path.replace("\\", "/"))
    for module in _CANONICAL_CURRENT_MODEL_MODULES.values():
        paths.add(module)
    return tuple(sorted(paths))


def _read_module_source(module_rel: str) -> str | None:
    path = _REPO_ROOT / module_rel
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def _decorator_mode_before(dec: ast.expr) -> bool:
    if not isinstance(dec, ast.Call):
        return False
    for keyword in dec.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
            return keyword.value.value == "before"
    return False


def _is_before_validator(dec: ast.expr) -> bool:
    name = None
    if isinstance(dec, ast.Name):
        name = dec.id
    elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
        name = dec.func.id
    elif isinstance(dec, ast.Attribute):
        name = dec.attr
    elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
        name = dec.func.attr
    if name not in {"field_validator", "model_validator"}:
        return False
    return _decorator_mode_before(dec) if isinstance(dec, ast.Call) else False


def _enclosing_scope(
    tree: ast.Module, lineno: int
) -> tuple[str | None, str | None]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            end = getattr(node, "end_lineno", None) or node.lineno + 400
            if node.lineno <= lineno <= end:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                        item_end = getattr(item, "end_lineno", None) or item.lineno + 200
                        if item.lineno <= lineno <= item_end:
                            return node.name, item.name
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            item_end = getattr(node, "end_lineno", None) or node.lineno + 400
            if node.lineno <= lineno <= item_end:
                return None, node.name
    return None, None


def _function_spans(tree: ast.Module) -> dict[tuple[str | None, str], tuple[int, int]]:
    spans: dict[tuple[str | None, str], tuple[int, int]] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            end = getattr(node, "end_lineno", None) or node.lineno + 400
            spans[(None, node.name)] = (node.lineno, end)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                end = getattr(item, "end_lineno", None) or item.lineno + 200
                spans[(node.name, item.name)] = (item.lineno, end)
    return spans


def _lineno_in_function(
    module_rel: str, lineno: int, function_name: str, source_by_module: dict[str, str]
) -> bool:
    source = source_by_module.get(module_rel)
    if source is None:
        return False
    tree = parse_module(module_rel, source)
    for (_class_name, name), (start, end) in _function_spans(tree).items():
        if name == function_name and start <= lineno <= end:
            return True
    return False


def _candidate_in_sanctioned_boundary(
    candidate: ImplicitCompatibilityCandidate,
    source_by_module: dict[str, str],
) -> bool:
    module = candidate.module_path
    function_name = candidate.enclosing_function
    if function_name is None:
        return False
    if (module, function_name) not in _sanctioned_boundary_symbols():
        return False
    return _lineno_in_function(module, candidate.lineno, function_name, source_by_module)


def _discover_before_validators(module_rel: str, source: str) -> list[ImplicitCompatibilityCandidate]:
    tree = parse_module(module_rel, source)
    hits: list[ImplicitCompatibilityCandidate] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for dec in node.decorator_list:
            if not _is_before_validator(dec):
                continue
            class_name, _ = _enclosing_scope(tree, node.lineno)
            hits.append(
                ImplicitCompatibilityCandidate(
                    module_path=module_rel,
                    lineno=node.lineno,
                    signal="pydantic_before_validator",
                    detail=f"{class_name or '<module>'}.{node.name}",
                    enclosing_class=class_name,
                    enclosing_function=node.name,
                )
            )
    return hits


def _discover_schema_version_injection(module_rel: str, source: str) -> list[ImplicitCompatibilityCandidate]:
    hits: list[ImplicitCompatibilityCandidate] = []
    pattern = re.compile(
        r'if\s+["\']schema_version["\']\s+not\s+in\s+(\w+)\s*:'
        r"[\s\S]{0,400}?\1\[\s*[\"']schema_version[\"']\s*\]\s*=",
        re.MULTILINE,
    )
    tree = parse_module(module_rel, source)
    for match in pattern.finditer(source):
        lineno = source[: match.start()].count("\n") + 1
        class_name, function_name = _enclosing_scope(tree, lineno)
        hits.append(
            ImplicitCompatibilityCandidate(
                module_path=module_rel,
                lineno=lineno,
                signal="missing_schema_version_injection",
                detail=match.group(0).splitlines()[0][:120],
                enclosing_class=class_name,
                enclosing_function=function_name,
            )
        )
    return hits


def _discover_bare_model_validate(module_rel: str, source: str) -> list[ImplicitCompatibilityCandidate]:
    if ".model_validate(" not in source:
        return []
    hits: list[ImplicitCompatibilityCandidate] = []
    tree = parse_module(module_rel, source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        fn_source = ast.get_source_segment(source, node) or ""
        if ".model_validate(" not in fn_source:
            continue
        if "schema_version" in fn_source:
            continue
        class_name = None
        for parent in ast.walk(tree):
            if isinstance(parent, ast.ClassDef):
                for item in parent.body:
                    if item is node:
                        class_name = parent.name
        hits.append(
            ImplicitCompatibilityCandidate(
                module_path=module_rel,
                lineno=node.lineno,
                signal="bare_model_validate_without_schema_gate",
                detail=node.name,
                enclosing_class=class_name,
                enclosing_function=node.name,
            )
        )
    return hits


def _discover_exception_driven_dual_validate(
    module_rel: str, source: str
) -> list[ImplicitCompatibilityCandidate]:
    tree = parse_module(module_rel, source)
    hits: list[ImplicitCompatibilityCandidate] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        try_validate = any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "model_validate"
            for child in ast.walk(node)
        )
        if not try_validate:
            continue
        except_validate = False
        for handler in node.handlers:
            for child in ast.walk(handler):
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr == "model_validate"
                ):
                    except_validate = True
        if except_validate:
            class_name, function_name = _enclosing_scope(tree, node.lineno)
            hits.append(
                ImplicitCompatibilityCandidate(
                    module_path=module_rel,
                    lineno=node.lineno,
                    signal="exception_driven_dual_validate",
                    detail="try/except model_validate fallback",
                    enclosing_class=class_name,
                    enclosing_function=function_name,
                )
            )
    return hits


def discover_implicit_compatibility_candidates(
    *,
    extra_modules: tuple[tuple[str, str], ...] = (),
) -> tuple[ImplicitCompatibilityCandidate, ...]:
    candidates: list[ImplicitCompatibilityCandidate] = []
    modules = list(derive_compatibility_scan_universe())
    extra_paths = {rel for rel, _ in extra_modules}
    for rel, source in extra_modules:
        modules.append(rel)
        candidates.extend(_discover_before_validators(rel, source))
        candidates.extend(_discover_schema_version_injection(rel, source))
        candidates.extend(_discover_exception_driven_dual_validate(rel, source))
        candidates.extend(_discover_bare_model_validate(rel, source))
    scan_universe = frozenset(derive_compatibility_scan_universe()) | extra_paths
    migration_owner_modules = frozenset(
        obligation.migration_owner_path.replace("\\", "/")
        for obligation in legacy_persisted_compatibility_obligations()
        if obligation.migration_owner_path
    )
    canonical_modules = frozenset(_CANONICAL_CURRENT_MODEL_MODULES.values())
    for module_rel in modules:
        if module_rel in extra_paths:
            continue
        source = _read_module_source(module_rel)
        if source is None:
            continue
        if module_rel in scan_universe or module_rel in canonical_modules:
            candidates.extend(_discover_before_validators(module_rel, source))
            candidates.extend(_discover_schema_version_injection(module_rel, source))
            candidates.extend(_discover_exception_driven_dual_validate(module_rel, source))
        if module_rel in migration_owner_modules or module_rel in canonical_modules:
            candidates.extend(_discover_bare_model_validate(module_rel, source))
    return tuple(candidates)


def _explicit_classification(
    candidate: ImplicitCompatibilityCandidate,
) -> ClassifiedImplicitCompatibility | None:
    class_name = candidate.enclosing_class
    function_name = candidate.enclosing_function
    if function_name is None and "." in candidate.detail:
        parts = candidate.detail.split(".", 1)
        class_name = parts[0] if parts[0] != "<module>" else None
        function_name = parts[1]
    for entry in _EXPLICIT_CLASSIFICATIONS:
        if entry.module_path != candidate.module_path:
            continue
        if entry.signal != candidate.signal:
            continue
        if entry.class_name != class_name:
            continue
        if entry.function_name != function_name:
            continue
        return ClassifiedImplicitCompatibility(candidate, entry.disposition)
    return None


def _is_compatibility_relevant_persisted_model(class_name: str | None) -> bool:
    if class_name is None:
        return False
    return class_name in _CANONICAL_CURRENT_MODEL_MODULES


def _classify_candidate(
    candidate: ImplicitCompatibilityCandidate,
    source_by_module: dict[str, str],
) -> ClassifiedImplicitCompatibility:
    explicit = _explicit_classification(candidate)
    if explicit is not None:
        return explicit

    if _candidate_in_sanctioned_boundary(candidate, source_by_module):
        return ClassifiedImplicitCompatibility(
            candidate, ImplicitCompatibilityDisposition.EXPLICIT_SANCTIONED_LEGACY_READER
        )

    if candidate.signal == "pydantic_before_validator":
        class_name = candidate.enclosing_class
        if class_name is None and "." in candidate.detail:
            class_name = candidate.detail.split(".", 1)[0]
            if class_name == "<module>":
                class_name = None
        if _is_compatibility_relevant_persisted_model(class_name):
            return ClassifiedImplicitCompatibility(
                candidate, ImplicitCompatibilityDisposition.BLOCKER_IMPLICIT_COMPATIBILITY
            )
        return ClassifiedImplicitCompatibility(
            candidate, ImplicitCompatibilityDisposition.NON_COMPAT_VALIDATION
        )

    if candidate.signal in {
        "missing_schema_version_injection",
        "exception_driven_dual_validate",
        "bare_model_validate_without_schema_gate",
    }:
        return ClassifiedImplicitCompatibility(
            candidate, ImplicitCompatibilityDisposition.BLOCKER_IMPLICIT_COMPATIBILITY
        )

    return ClassifiedImplicitCompatibility(
        candidate, ImplicitCompatibilityDisposition.BLOCKER_IMPLICIT_COMPATIBILITY
    )


def _module_has_compatibility_signals(module_rel: str) -> bool:
    source = _read_module_source(module_rel)
    if source is None:
        return False
    if any(marker in source for marker in _LEGACY_MODULE_MARKERS):
        return True
    if discover_implicit_compatibility_candidates(extra_modules=((module_rel, source),)):
        return True
    return False


def discover_orphan_migration_modules() -> tuple[str, ...]:
    """Migration-like boundaries without a registered legacy obligation owner."""
    obligated_owners: set[str] = set()
    for obligation in legacy_persisted_compatibility_obligations():
        if obligation.migration_owner_path is None:
            continue
        if obligation.disposition in (
            LegacyCompatibilityDisposition.EXPLICIT_LEGACY_READER,
            LegacyCompatibilityDisposition.EXPLICIT_MIGRATION,
            LegacyCompatibilityDisposition.CURRENT_ONLY_FAIL_CLOSED,
        ):
            obligated_owners.add(obligation.migration_owner_path.replace("\\", "/"))

    orphans: list[str] = []
    candidate_modules = set(_MIGRATION_BOUNDARY_MODULES)
    for rel in derive_compatibility_scan_universe():
        if rel.endswith("_legacy.py") or rel.endswith("descriptor_persisted.py"):
            candidate_modules.add(rel)

    for module_rel in sorted(candidate_modules):
        if module_rel in obligated_owners:
            continue
        if module_rel == "intergrax/memory/user_profile_legacy_wire.py":
            continue
        if not (_REPO_ROOT / module_rel).is_file():
            continue
        if _module_has_compatibility_signals(module_rel):
            orphans.append(module_rel)
    return tuple(sorted(set(orphans)))


def _annotation_uses_forbidden_semantic_type(annotation: ast.expr) -> bool:
    text = ast.unparse(annotation)
    if any(fragment in text for fragment in _FORBIDDEN_SEMANTIC_ANNOTATION_FRAGMENTS):
        return True
    return isinstance(annotation, ast.Name) and annotation.id == "Any"


def _function_body_nodes(tree: ast.Module, function_name: str) -> list[ast.AST]:
    collected: list[ast.AST] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == function_name:
            collected.extend(ast.walk(node))
    for class_node in ast.walk(tree):
        if not isinstance(class_node, ast.ClassDef):
            continue
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef) and item.name == function_name:
                collected.extend(ast.walk(item))
    return collected


def _module_top_level_function_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            names.add(node.name)
    return names


def _callees_in_function(tree: ast.Module, function_name: str) -> set[str]:
    local_functions = _module_top_level_function_names(tree)
    callees: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if node.name != function_name:
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            callee_name: str | None = None
            if isinstance(child.func, ast.Name):
                callee_name = child.func.id
            elif isinstance(child.func, ast.Attribute):
                callee_name = child.func.attr
            if callee_name is not None and callee_name in local_functions:
                callees.add(callee_name)
    return callees


def _migration_boundary_functions_for_module(
    module_rel: str,
    source: str,
    reader_roots: set[str],
    helper_roots: set[str],
) -> set[str]:
    tree = parse_module(module_rel, source)
    local_functions = _module_top_level_function_names(tree)
    discovered: set[str] = {
        symbol for symbol in reader_roots if symbol in local_functions
    }
    for root in tuple(discovered):
        for callee in _callees_in_function(tree, root):
            if callee in local_functions and "legacy" in callee.lower():
                discovered.add(callee)
    for helper in helper_roots:
        if helper in local_functions:
            discovered.add(helper)
    queue = [name for name in discovered if "legacy" in name.lower()]
    while queue:
        current = queue.pop()
        for callee in _callees_in_function(tree, current):
            if callee not in local_functions or callee in discovered:
                continue
            discovered.add(callee)
            queue.append(callee)
    return discovered


def discover_strong_typing_violations(
    *,
    source_overrides: dict[str, str] | None = None,
) -> tuple[str, ...]:
    violations: list[str] = []
    reader_symbols_by_module: dict[str, set[str]] = {}
    helper_symbols_by_module: dict[str, set[str]] = {}
    for obligation in legacy_persisted_compatibility_obligations():
        if obligation.migration_owner_path is None:
            continue
        if obligation.disposition not in (
            LegacyCompatibilityDisposition.EXPLICIT_LEGACY_READER,
            LegacyCompatibilityDisposition.EXPLICIT_MIGRATION,
            LegacyCompatibilityDisposition.CURRENT_ONLY_FAIL_CLOSED,
        ):
            continue
        module = obligation.migration_owner_path.replace("\\", "/")
        for symbol in obligation.sanctioned_reader_symbols:
            reader_symbols_by_module.setdefault(module, set()).add(symbol)
        for symbol in obligation.migration_boundary_helper_symbols:
            helper_symbols_by_module.setdefault(module, set()).add(symbol)

    overrides = source_overrides or {}
    for module_rel in sorted(_MIGRATION_BOUNDARY_MODULES):
        source = overrides.get(module_rel) or _read_module_source(module_rel)
        if source is None:
            violations.append(f"missing migration boundary module {module_rel}")
            continue
        if re.search(r"\bmigrated:\s*Dict\[str,\s*Any\]", source):
            violations.append(f"{module_rel}: migrated_dict_semantic_contract")
        if re.search(r"\bmigrated:\s*dict\[str,\s*Any\]", source):
            violations.append(f"{module_rel}: migrated_dict_semantic_contract")
        tree = parse_module(module_rel, source)
        for class_node in ast.walk(tree):
            if not isinstance(class_node, ast.ClassDef):
                continue
            if not (class_node.name.startswith("Legacy") or class_node.name.startswith("_Legacy")):
                continue
            for item in class_node.body:
                if isinstance(item, ast.AnnAssign) and item.annotation is not None:
                    if _annotation_uses_forbidden_semantic_type(item.annotation):
                        violations.append(
                            f"{module_rel}:{item.lineno}: legacy_field "
                            f"{class_node.name} {ast.unparse(item.annotation)}"
                        )
        function_names = _migration_boundary_functions_for_module(
            module_rel,
            source,
            reader_symbols_by_module.get(module_rel, set()),
            helper_symbols_by_module.get(module_rel, set()),
        )
        for function_name in sorted(function_names):
            body_nodes = _function_body_nodes(tree, function_name)
            for node in body_nodes:
                if isinstance(node, ast.AnnAssign) and node.annotation is not None:
                    if _annotation_uses_forbidden_semantic_type(node.annotation):
                        violations.append(
                            f"{module_rel}:{node.lineno}: migration_boundary "
                            f"{function_name} {ast.unparse(node.annotation)}"
                        )
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    if node.returns is not None and _annotation_uses_forbidden_semantic_type(
                        node.returns
                    ):
                        violations.append(
                            f"{module_rel}:{node.lineno}: migration_boundary "
                            f"{node.name} return {ast.unparse(node.returns)}"
                        )
                    for arg in node.args.args + node.args.kwonlyargs:
                        if arg.annotation is not None and _annotation_uses_forbidden_semantic_type(
                            arg.annotation
                        ):
                            violations.append(
                                f"{module_rel}:{arg.lineno}: migration_boundary "
                                f"{node.name} param {arg.arg} {ast.unparse(arg.annotation)}"
                            )
    return tuple(violations)


def _source_map_for_candidates(
    candidates: tuple[ImplicitCompatibilityCandidate, ...],
    extra_modules: tuple[tuple[str, str], ...],
) -> dict[str, str]:
    source_by_module: dict[str, str] = dict(extra_modules)
    for candidate in candidates:
        if candidate.module_path in source_by_module:
            continue
        loaded = _read_module_source(candidate.module_path)
        if loaded is not None:
            source_by_module[candidate.module_path] = loaded
    return source_by_module


def build_implicit_compatibility_report(
    *,
    extra_modules: tuple[tuple[str, str], ...] = (),
) -> ImplicitCompatibilityReport:
    candidates = discover_implicit_compatibility_candidates(extra_modules=extra_modules)
    source_by_module = _source_map_for_candidates(candidates, extra_modules)
    classified = tuple(_classify_candidate(c, source_by_module) for c in candidates)
    source_overrides = dict(extra_modules)
    base_report = ImplicitCompatibilityReport(
        scanned_modules=derive_compatibility_scan_universe(),
        raw_candidates=candidates,
        classified=classified,
        orphan_migration_modules=discover_orphan_migration_modules(),
        strong_typing_violations=discover_strong_typing_violations(
            source_overrides=source_overrides or None
        ),
        module_wide_sanctioned_exemptions=0,
    )
    if extra_modules:
        return base_report
    return base_report


@lru_cache(maxsize=1)
def build_implicit_compatibility_report_cached() -> ImplicitCompatibilityReport:
    return build_implicit_compatibility_report()


def implicit_compatibility_blocker_count() -> int:
    return len(build_implicit_compatibility_report_cached().blockers)


def r2_r2_r1_implicit_gate_passes() -> bool:
    return r2_r2_r1_r1_implicit_gate_passes()


def r2_r2_r1_r1_implicit_gate_passes(
    *,
    extra_modules: tuple[tuple[str, str], ...] = (),
) -> bool:
    report = (
        build_implicit_compatibility_report(extra_modules=extra_modules)
        if extra_modules
        else build_implicit_compatibility_report_cached()
    )
    if report.blockers:
        return False
    if report.orphan_migration_modules:
        return False
    if report.strong_typing_violations:
        return False
    if report.module_wide_sanctioned_exemptions != 0:
        return False
    return True


def classify_implicit_compatibility_in_source(
    module_rel: str, source: str
) -> tuple[ClassifiedImplicitCompatibility, ...]:
    candidates = discover_implicit_compatibility_candidates(extra_modules=((module_rel, source),))
    module_candidates = [c for c in candidates if c.module_path == module_rel]
    source_by_module = {module_rel: source}
    return tuple(_classify_candidate(c, source_by_module) for c in module_candidates)


def synthetic_module_fails_implicit_gate(module_rel: str, source: str) -> bool:
    """Anti-tautology: synthetic violation must fail the qualification gate."""
    return not r2_r2_r1_r1_implicit_gate_passes(extra_modules=((module_rel, source),))
