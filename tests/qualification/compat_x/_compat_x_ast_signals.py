# © Artur Czarnecki. All rights reserved.

"""AST signal extraction for COMPAT-X mechanical discovery."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Final

from tests.qualification.compat_x._compat_x_types import CompatibilityCandidateContext

_VERSION_FIELD_NAMES: Final[frozenset[str]] = frozenset(
    {"schema_version", "contract_version", "payload_schema_version"}
)

_SCHEMA_CONST_RE = re.compile(r"^SCHEMA_[A-Z0-9_]+$")
_VERSION_CONST_SUFFIXES: Final[tuple[str, ...]] = (
    "_SCHEMA_VERSION",
    "_CONTRACT_VERSION",
)

_WIRE_CALL_NAMES: Final[frozenset[str]] = frozenset(
    {
        "model_dump",
        "model_validate",
        "model_validate_json",
        "model_dump_json",
    }
)

_PERSISTENCE_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "model_dump",
        "model_validate",
        "model_validate_json",
        "json.dumps",
        "encode",
        "persist",
        "append_event",
        "put_document",
        "save_checkpoint",
    }
)

_MIGRATION_NAME_FRAGMENTS: Final[tuple[str, ...]] = (
    "migrate",
    "migration",
    "upgrade",
    "legacy",
    "normalize",
    "decode_old",
    "from_legacy",
    "to_canonical",
)

_MIGRATION_PATH_FRAGMENTS: Final[tuple[str, ...]] = (
    "/migrations/",
    "_migration",
    "_legacy",
    "/legacy/",
)


@dataclass(frozen=True, slots=True)
class ClassVersionFieldSignal:
    module_path: str
    class_name: str
    field_name: str
    version_literal: str | None
    signal: str


@dataclass(frozen=True, slots=True)
class ModuleVersionConstantSignal:
    module_path: str
    const_name: str
    version_literal: str
    signal: str


@dataclass(frozen=True, slots=True)
class WirePersistenceSignal:
    module_path: str
    class_name: str | None
    call_name: str
    lineno: int
    signal: str


@dataclass(frozen=True, slots=True)
class MigrationSignal:
    module_path: str
    kind: str
    detail: str
    signal: str


@dataclass(frozen=True, slots=True)
class ShimSignal:
    module_path: str
    kind: str
    detail: str
    signal: str


@dataclass(frozen=True, slots=True)
class PublicExportSignal:
    module_path: str
    exported_name: str
    signal: str


def _literal_version(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int)):
        return str(node.value)
    if isinstance(node, ast.Subscript):
        return _literal_version(node.slice)
    return None


def _is_version_constant_name(name: str) -> bool:
    if name.startswith("UNSUPPORTED_"):
        return False
    return (
        name == "MANIFEST_SCHEMA_VERSION"
        or any(name.endswith(suffix) for suffix in _VERSION_CONST_SUFFIXES)
        or _SCHEMA_CONST_RE.match(name) is not None
    )


def extract_module_version_constants(
    module_path: str, tree: ast.Module
) -> list[ModuleVersionConstantSignal]:
    found: list[ModuleVersionConstantSignal] = []
    for node in tree.body:
        target: ast.expr | None = None
        value: ast.AST | None = None
        name: str | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            value = node.value
        if name is None or value is None or not _is_version_constant_name(name):
            continue
        version = _literal_version(value)
        if version is None:
            continue
        found.append(
            ModuleVersionConstantSignal(
                module_path=module_path,
                const_name=name,
                version_literal=version,
                signal=f"module_constant:{name}={version}",
            )
        )
    return found


def _class_field_from_annassign(
    module_path: str, class_name: str, node: ast.AnnAssign
) -> ClassVersionFieldSignal | None:
    if not isinstance(node.target, ast.Name):
        return None
    field_name = node.target.id
    if field_name not in _VERSION_FIELD_NAMES and not field_name.endswith("_version"):
        return None
    version = _literal_version(node.value)
    return ClassVersionFieldSignal(
        module_path=module_path,
        class_name=class_name,
        field_name=field_name,
        version_literal=version,
        signal=f"class_field:{class_name}.{field_name}",
    )


def _class_field_from_assign(
    module_path: str, class_name: str, node: ast.Assign
) -> ClassVersionFieldSignal | None:
    if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
        return None
    field_name = node.targets[0].id
    if field_name not in _VERSION_FIELD_NAMES:
        return None
    version = _literal_version(node.value)
    return ClassVersionFieldSignal(
        module_path=module_path,
        class_name=class_name,
        field_name=field_name,
        version_literal=version,
        signal=f"class_field:{class_name}.{field_name}",
    )


def extract_class_version_fields(module_path: str, tree: ast.Module) -> list[ClassVersionFieldSignal]:
    found: list[ClassVersionFieldSignal] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if isinstance(item, ast.AnnAssign):
                signal = _class_field_from_annassign(module_path, node.name, item)
                if signal is not None:
                    found.append(signal)
            elif isinstance(item, ast.Assign):
                signal = _class_field_from_assign(module_path, node.name, item)
                if signal is not None:
                    found.append(signal)
    return found


def _class_names_in_module(tree: ast.Module) -> frozenset[str]:
    return frozenset(n.name for n in tree.body if isinstance(n, ast.ClassDef))


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def extract_wire_persistence_signals(module_path: str, tree: ast.Module) -> list[WirePersistenceSignal]:
    classes = _class_names_in_module(tree)
    found: list[WirePersistenceSignal] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = _call_name(node)
        if call_name not in _WIRE_CALL_NAMES:
            continue
        related_class: str | None = None
        for arg in node.args:
            if isinstance(arg, ast.Name) and arg.id in classes:
                related_class = arg.id
                break
        found.append(
            WirePersistenceSignal(
                module_path=module_path,
                class_name=related_class,
                call_name=call_name or "call",
                lineno=getattr(node, "lineno", 0),
                signal=f"wire:{call_name}@{getattr(node, 'lineno', 0)}",
            )
        )
    return found


def class_has_schema_version_field(tree: ast.Module, class_name: str) -> bool:
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            target: str | None = None
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                target = item.target.id
            elif isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name):
                target = item.targets[0].id
            if target in _VERSION_FIELD_NAMES:
                return True
    return False


def extract_migration_signals(module_path: str, tree: ast.Module, source_text: str) -> list[MigrationSignal]:
    found: list[MigrationSignal] = []
    normalized = module_path.replace("\\", "/")
    for fragment in _MIGRATION_PATH_FRAGMENTS:
        if fragment in normalized:
            found.append(
                MigrationSignal(
                    module_path=module_path,
                    kind="path.dedicated",
                    detail=fragment,
                    signal=f"migration_path:{fragment}",
                )
            )
            break
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            lowered = node.name.lower()
            for frag in _MIGRATION_NAME_FRAGMENTS:
                if frag in lowered:
                    found.append(
                        MigrationSignal(
                            module_path=module_path,
                            kind="function.name",
                            detail=node.name,
                            signal=f"migration_fn:{node.name}",
                        )
                    )
                    break
        if isinstance(node, ast.Compare):
            for op in node.ops:
                if isinstance(op, ast.Eq):
                    for side in (node.left, *node.comparators):
                        if isinstance(side, ast.Attribute) and side.attr in _VERSION_FIELD_NAMES:
                            found.append(
                                MigrationSignal(
                                    module_path=module_path,
                                    kind="schema_version.branch",
                                    detail=f"line:{getattr(node, 'lineno', 0)}",
                                    signal=f"migration_branch:{getattr(node, 'lineno', 0)}",
                                )
                            )
                            break
    if "schema_version" in source_text and "legacy" in source_text.lower():
        if not any(s.kind == "path.dedicated" for s in found):
            found.append(
                MigrationSignal(
                    module_path=module_path,
                    kind="text.legacy_schema",
                    detail="legacy+schema_version",
                    signal="migration_text:legacy_schema",
                )
            )
    return found


def extract_shim_path_signal(module_path: str) -> ShimSignal | None:
    normalized = module_path.replace("\\", "/")
    if normalized.startswith("intergrax/compat/"):
        return ShimSignal(
            module_path=module_path,
            kind="path.compat_tree",
            detail="intergrax/compat",
            signal="shim_path:intergrax/compat",
        )
    return None


def module_defines_resolve_provider(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == "resolve_provider":
            return True
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef) and item.name == "resolve_provider":
                    return True
    return False


_AUTHORITY_CLASS_MARKERS: Final[tuple[str, ...]] = (
    "selector",
    "registry",
    "executor",
    "authorizer",
    "resolver",
    "dispatcher",
    "backend",
    "provider",
    "authority",
)

_AUTHORITY_METHOD_PREFIXES: Final[tuple[str, ...]] = (
    "resolve",
    "select",
    "dispatch",
    "execute",
    "invoke",
    "authorize",
    "admit",
    "register",
    "persist",
    "save",
)


_COMPAT_SOURCE_MARKERS: Final[tuple[str, ...]] = (
    "from_langchain",
    "to_langchain",
)

_COMPAT_SOURCE_SUBSTRINGS: Final[tuple[str, ...]] = (
    "adapter",
    "translate",
)

_COMPAT_PATH_MARKERS: Final[tuple[str, ...]] = (
    "/legacy/",
    "_legacy",
    "legacy_",
    "_adapter",
    "/compat/",
    "compat_",
)

_COMPAT_CLASS_NAME_PREFIXES: Final[tuple[str, ...]] = (
    "legacy",
    "compatibility",
    "compat",
)

def build_compatibility_candidate_context(
    module_path: str,
    source: str,
    tree: ast.Module,
) -> CompatibilityCandidateContext:
    """Mechanical signals only — must not call shim classifiers."""
    normalized = module_path.replace("\\", "/")
    kinds: list[str] = []
    path_signal = extract_shim_path_signal(module_path)
    if path_signal is not None:
        kinds.append(path_signal.kind)
    if normalized.startswith("synthetic/qualification/"):
        kinds.append("path.synthetic_qualification")
    lowered_path = normalized.lower()
    for marker in _COMPAT_PATH_MARKERS:
        if marker in lowered_path:
            kinds.append(f"path.marker:{marker}")
    lowered_source = source.lower()
    for marker in _COMPAT_SOURCE_MARKERS:
        if marker in lowered_source:
            kinds.append(f"source.{marker}")
    for substring in _COMPAT_SOURCE_SUBSTRINGS:
        if substring in lowered_source:
            kinds.append(f"source.contains:{substring}")
    _ = tree  # reserved for future AST-only shim discovery signals
    return CompatibilityCandidateContext(
        module_path=normalized,
        evidence_kinds=tuple(dict.fromkeys(kinds)),
    )


def is_compatibility_adapter_candidate(context: CompatibilityCandidateContext) -> bool:
    return bool(context.evidence_kinds)


def _class_name_suggests_compat_adapter_type(class_name: str) -> bool:
    lowered = class_name.lower()
    return any(lowered.startswith(prefix) for prefix in _COMPAT_CLASS_NAME_PREFIXES)


def _module_path_suggests_compat_adapter_seam(module_path: str) -> bool:
    lowered = module_path.replace("\\", "/").lower()
    if lowered.startswith("intergrax/compat/") or lowered.startswith("synthetic/qualification/"):
        return True
    return any(
        fragment in lowered
        for fragment in (
            "/legacy/",
            "_legacy",
            "legacy_",
            "/compat/",
        )
    )


_PARALLEL_AUTHORITY_PATH_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "/legacy/",
        "_legacy",
        "legacy_",
        "/compat/",
        "compat_",
    }
)


def parallel_authority_detection_applies(
    tree: ast.Module,
    context: CompatibilityCandidateContext,
) -> bool:
    """Parallel authority is bounded to compat seams — not every weak shim text signal."""
    if any(kind in ("path.compat_tree", "path.synthetic_qualification") for kind in context.evidence_kinds):
        return True
    for kind in context.evidence_kinds:
        if kind.startswith("path.marker:"):
            marker = kind.split(":", 1)[1]
            if marker in _PARALLEL_AUTHORITY_PATH_MARKERS:
                return True
    if any(kind.startswith("source.from_langchain") or kind.startswith("source.to_langchain") for kind in context.evidence_kinds):
        return True
    if _module_path_suggests_compat_adapter_seam(context.module_path):
        return True
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and _class_name_suggests_compat_adapter_type(node.name):
            return True
    return False


def compat_adapter_module_scope(module_path: str) -> bool:
    """Deprecated path gate — use ``is_compatibility_adapter_candidate`` with discovery context."""
    normalized = module_path.replace("\\", "/")
    if normalized.startswith("intergrax/compat/"):
        return True
    if normalized.startswith("synthetic/qualification/"):
        return True
    return False


def is_sanctioned_translation_callable(name: str) -> bool:
    lowered = name.lower()
    if lowered.startswith(("from_", "to_")):
        return True
    if lowered.startswith(("decode_", "normalize_")):
        return True
    if lowered in {"translate", "convert"} or lowered.startswith(("translate_", "convert_")):
        return True
    return False


_STRICT_AUTHORITY_METHOD_PREFIXES: Final[tuple[str, ...]] = (
    "select_",
    "resolve_",
    "dispatch_",
    "execute_",
    "invoke_",
    "authorize_",
    "admit_",
)

_TOP_LEVEL_PARALLEL_AUTHORITY_NAMES: Final[frozenset[str]] = frozenset({"resolve_provider"})


def _method_name_suggests_parallel_authority(name: str) -> bool:
    if is_sanctioned_translation_callable(name):
        return False
    lowered = name.lower()
    return any(lowered.startswith(prefix) for prefix in _AUTHORITY_METHOD_PREFIXES)


def _strict_parallel_authority_method_name(name: str) -> bool:
    if is_sanctioned_translation_callable(name):
        return False
    if name in _TOP_LEVEL_PARALLEL_AUTHORITY_NAMES:
        return True
    lowered = name.lower()
    return any(lowered.startswith(prefix) for prefix in _STRICT_AUTHORITY_METHOD_PREFIXES)


def _class_name_suggests_authority_carrier(class_name: str) -> bool:
    lowered = class_name.lower()
    return any(marker in lowered for marker in _AUTHORITY_CLASS_MARKERS)


def module_exhibits_parallel_authority(
    tree: ast.Module,
    context: CompatibilityCandidateContext,
) -> bool:
    """Bounded AST gate: parallel authority only for compatibility adapter candidates."""
    if not is_compatibility_adapter_candidate(context):
        return False
    if not parallel_authority_detection_applies(tree, context):
        return False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if _strict_parallel_authority_method_name(node.name):
                return True
        if not isinstance(node, ast.ClassDef):
            continue
        class_carrier = _class_name_suggests_authority_carrier(node.name)
        authority_methods: list[str] = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                if _method_name_suggests_parallel_authority(item.name):
                    authority_methods.append(item.name)
        if not authority_methods:
            continue
        if class_carrier:
            return True
        if any(_strict_parallel_authority_method_name(name) for name in authority_methods):
            return True
    if module_defines_resolve_provider(tree):
        return True
    return False


def module_is_translation_only_compat_adapter(
    tree: ast.Module,
    context: CompatibilityCandidateContext,
) -> bool:
    if not is_compatibility_adapter_candidate(context):
        return False
    callables: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            callables.append(node.name)
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    callables.append(f"{node.name}.{item.name}")
    if not callables:
        return False
    simple_names = [name.split(".")[-1] for name in callables]
    return all(is_sanctioned_translation_callable(name) for name in simple_names)


def extract_public_export_signals(module_path: str, tree: ast.Module) -> list[PublicExportSignal]:
    if not module_path.endswith("__init__.py"):
        return []
    found: list[PublicExportSignal] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                found.append(
                                    PublicExportSignal(
                                        module_path=module_path,
                                        exported_name=elt.value,
                                        signal=f"export:__all__:{elt.value}",
                                    )
                                )
    return found


def parse_module(module_path: str, source: str) -> ast.Module:
    return ast.parse(source, filename=module_path)
