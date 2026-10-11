# © Artur Czarnecki. All rights reserved.

"""Behavior-/structure-oriented implicit compatibility discovery (COMPAT-X-R2-R2-R1)."""

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


@dataclass(frozen=True, slots=True)
class ClassifiedImplicitCompatibility:
    candidate: ImplicitCompatibilityCandidate
    disposition: ImplicitCompatibilityDisposition


@dataclass(frozen=True, slots=True)
class ImplicitCompatibilityReport:
    scanned_modules: tuple[str, ...]
    raw_candidates: tuple[ImplicitCompatibilityCandidate, ...]
    classified: tuple[ClassifiedImplicitCompatibility, ...]
    orphan_migration_modules: tuple[str, ...]
    strong_typing_violations: tuple[str, ...]

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


def _sanctioned_migration_owner_paths() -> frozenset[str]:
    owners: set[str] = set()
    for obligation in legacy_persisted_compatibility_obligations():
        if obligation.migration_owner_path:
            owners.add(obligation.migration_owner_path.replace("\\", "/"))
    return frozenset(owners)


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


def _enclosing_class_name(tree: ast.Module, lineno: int) -> str | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            end = getattr(node, "end_lineno", None) or node.lineno + 400
            if node.lineno <= lineno <= end:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                        item_end = getattr(item, "end_lineno", None) or item.lineno + 200
                        if item.lineno <= lineno <= item_end:
                            return node.name
    return None


def _discover_before_validators(module_rel: str, source: str) -> list[ImplicitCompatibilityCandidate]:
    tree = parse_module(module_rel, source)
    hits: list[ImplicitCompatibilityCandidate] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for dec in node.decorator_list:
            if not _is_before_validator(dec):
                continue
            class_name = _enclosing_class_name(tree, node.lineno) or "<module>"
            hits.append(
                ImplicitCompatibilityCandidate(
                    module_path=module_rel,
                    lineno=node.lineno,
                    signal="pydantic_before_validator",
                    detail=f"{class_name}.{node.name}",
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
    for match in pattern.finditer(source):
        lineno = source[: match.start()].count("\n") + 1
        hits.append(
            ImplicitCompatibilityCandidate(
                module_path=module_rel,
                lineno=lineno,
                signal="missing_schema_version_injection",
                detail=match.group(0).splitlines()[0][:120],
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
        hits.append(
            ImplicitCompatibilityCandidate(
                module_path=module_rel,
                lineno=node.lineno,
                signal="bare_model_validate_without_schema_gate",
                detail=node.name,
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
            hits.append(
                ImplicitCompatibilityCandidate(
                    module_path=module_rel,
                    lineno=node.lineno,
                    signal="exception_driven_dual_validate",
                    detail="try/except model_validate fallback",
                )
            )
    return hits


def discover_implicit_compatibility_candidates(
    *,
    extra_modules: tuple[tuple[str, str], ...] = (),
) -> tuple[ImplicitCompatibilityCandidate, ...]:
    candidates: list[ImplicitCompatibilityCandidate] = []
    modules = list(derive_compatibility_scan_universe())
    for rel, source in extra_modules:
        modules.append(rel)
        candidates.extend(_discover_before_validators(rel, source))
        candidates.extend(_discover_schema_version_injection(rel, source))
        candidates.extend(_discover_exception_driven_dual_validate(rel, source))
        candidates.extend(_discover_bare_model_validate(rel, source))
    scan_universe = frozenset(derive_compatibility_scan_universe())
    owners = _sanctioned_migration_owner_paths()
    canonical_modules = frozenset(_CANONICAL_CURRENT_MODEL_MODULES.values())
    for module_rel in modules:
        if any(module_rel == rel for rel, _ in extra_modules):
            continue
        source = _read_module_source(module_rel)
        if source is None:
            continue
        if module_rel in scan_universe or module_rel in canonical_modules:
            candidates.extend(_discover_before_validators(module_rel, source))
            candidates.extend(_discover_schema_version_injection(module_rel, source))
            candidates.extend(_discover_exception_driven_dual_validate(module_rel, source))
        if module_rel in owners or module_rel in canonical_modules:
            candidates.extend(_discover_bare_model_validate(module_rel, source))
    return tuple(candidates)


def _classify_candidate(candidate: ImplicitCompatibilityCandidate) -> ClassifiedImplicitCompatibility:
    module = candidate.module_path
    owners = _sanctioned_migration_owner_paths()
    canonical_modules = frozenset(_CANONICAL_CURRENT_MODEL_MODULES.values())
    if module in owners:
        return ClassifiedImplicitCompatibility(
            candidate, ImplicitCompatibilityDisposition.EXPLICIT_SANCTIONED_LEGACY_READER
        )
    if candidate.signal == "pydantic_before_validator":
        class_name = candidate.detail.split(".", 1)[0]
        canonical_module = _CANONICAL_CURRENT_MODEL_MODULES.get(class_name)
        if canonical_module == module and class_name == "RuntimeEvent":
            return ClassifiedImplicitCompatibility(
                candidate, ImplicitCompatibilityDisposition.CURRENT_ONLY_NORMALIZATION
            )
        if canonical_module == module:
            return ClassifiedImplicitCompatibility(
                candidate, ImplicitCompatibilityDisposition.NON_COMPAT_VALIDATION
            )
        return ClassifiedImplicitCompatibility(
            candidate, ImplicitCompatibilityDisposition.NON_COMPAT_VALIDATION
        )
    if candidate.signal == "bare_model_validate_without_schema_gate":
        if module in owners:
            return ClassifiedImplicitCompatibility(
                candidate, ImplicitCompatibilityDisposition.EXPLICIT_SANCTIONED_LEGACY_READER
            )
        if module in canonical_modules:
            return ClassifiedImplicitCompatibility(
                candidate, ImplicitCompatibilityDisposition.BLOCKER_IMPLICIT_COMPATIBILITY
            )
        return ClassifiedImplicitCompatibility(
            candidate, ImplicitCompatibilityDisposition.NON_COMPAT_VALIDATION
        )
    if candidate.signal in {
        "missing_schema_version_injection",
        "exception_driven_dual_validate",
    }:
        if module in owners:
            return ClassifiedImplicitCompatibility(
                candidate, ImplicitCompatibilityDisposition.EXPLICIT_SANCTIONED_LEGACY_READER
            )
        return ClassifiedImplicitCompatibility(
            candidate, ImplicitCompatibilityDisposition.BLOCKER_IMPLICIT_COMPATIBILITY
        )
    return ClassifiedImplicitCompatibility(
        candidate, ImplicitCompatibilityDisposition.NON_COMPAT_VALIDATION
    )


def discover_orphan_migration_modules() -> tuple[str, ...]:
    """Migration modules in the persisted closed-world scan universe without an obligation owner."""
    owners = _sanctioned_migration_owner_paths()
    orphans: list[str] = []
    for rel in derive_compatibility_scan_universe():
        if not (rel.endswith("_legacy.py") or rel.endswith("descriptor_persisted.py")):
            continue
        if rel not in owners and (_REPO_ROOT / rel).is_file():
            orphans.append(rel)
    return tuple(sorted(set(orphans)))


def _annotation_uses_forbidden_semantic_type(annotation: ast.expr) -> bool:
    text = ast.unparse(annotation)
    return any(fragment in text for fragment in _FORBIDDEN_SEMANTIC_ANNOTATION_FRAGMENTS)


def discover_strong_typing_violations() -> tuple[str, ...]:
    violations: list[str] = []
    for module_rel in sorted(_MIGRATION_BOUNDARY_MODULES):
        source = _read_module_source(module_rel)
        if source is None:
            violations.append(f"missing migration boundary module {module_rel}")
            continue
        if re.search(r"\bmigrate:\s*Any\b", source):
            violations.append(f"{module_rel}: migrate_any")
        tree = parse_module(module_rel, source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and (
                node.name.startswith("Legacy") or node.name.startswith("_Legacy")
            ):
                for item in node.body:
                    if not isinstance(item, ast.AnnAssign) or item.annotation is None:
                        continue
                    if _annotation_uses_forbidden_semantic_type(item.annotation):
                        violations.append(
                            f"{module_rel}:{item.lineno}: legacy_field {node.name} {ast.unparse(item.annotation)}"
                        )
    return tuple(violations)


@lru_cache(maxsize=1)
def build_implicit_compatibility_report() -> ImplicitCompatibilityReport:
    candidates = discover_implicit_compatibility_candidates()
    classified = tuple(_classify_candidate(c) for c in candidates)
    return ImplicitCompatibilityReport(
        scanned_modules=derive_compatibility_scan_universe(),
        raw_candidates=candidates,
        classified=classified,
        orphan_migration_modules=discover_orphan_migration_modules(),
        strong_typing_violations=discover_strong_typing_violations(),
    )


def implicit_compatibility_blocker_count() -> int:
    return len(build_implicit_compatibility_report().blockers)


def r2_r2_r1_implicit_gate_passes() -> bool:
    report = build_implicit_compatibility_report()
    if report.blockers:
        return False
    if report.orphan_migration_modules:
        return False
    if report.strong_typing_violations:
        return False
    return True


def classify_implicit_compatibility_in_source(
    module_rel: str, source: str
) -> tuple[ClassifiedImplicitCompatibility, ...]:
    candidates = discover_implicit_compatibility_candidates(extra_modules=((module_rel, source),))
    module_candidates = [c for c in candidates if c.module_path == module_rel]
    return tuple(_classify_candidate(c) for c in module_candidates)
