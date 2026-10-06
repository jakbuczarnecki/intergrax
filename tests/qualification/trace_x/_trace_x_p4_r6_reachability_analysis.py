# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P4-R6 static production reachability analysis (P4-bounded, not a general call graph)."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from tests.qualification.trace_x._trace_x_p4_production_composition_types import (
    NonProductionReachabilityReason,
    ProductionCompositionSiteClassification,
    ProductionCompositionSiteKind,
    RegisteredNonProductionModelReachability,
    RegisteredProductionCompositionSite,
)
from tests.qualification.trace_x._trace_x_p4_registry_types import (
    ModelCallSurfaceClassification,
    RegisteredModelCallSurface,
)
from tests.qualification.trace_x._trace_x_p4_r6_reachability_types import (
    CompositionEdge,
    MechanicalReachabilityResult,
    ModelConsumerSurface,
    ReachabilityExpectationParityResult,
    ReachabilityReason,
    ReachabilityVerdict,
)
from tests.qualification.trace_x._trace_x_p4_support import (
    _PRODUCTION_EXCLUDE_DIR_NAMES,
    _PRODUCTION_SCAN_ROOTS,
    _REPO_ROOT,
    _ast_call_callee_root_name,
)

_INFERENCE_SURFACE = ModelConsumerSurface(
    path="intergrax/runtime/execution/inference.py",
    method="generate_structured",
)
_WRAPPER_MODULE = "intergrax/runtime/llm/model_call_runtime_evidence_adapter.py"
_STREAM_METHODS: Final[frozenset[str]] = frozenset({"stream_messages", "stream_with_tools"})


@dataclass(frozen=True, slots=True)
class ReachabilityEvaluationContext:
    production_composition_registry: tuple[RegisteredProductionCompositionSite, ...]
    extra_composition_edges: frozenset[CompositionEdge] = frozenset()


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


def _module_path_from_file(rel_posix: str) -> str:
    return rel_posix


def _parse_module(rel_posix: str) -> ast.Module | None:
    py_path = _REPO_ROOT / rel_posix
    if not py_path.is_file():
        return None
    try:
        return ast.parse(py_path.read_text(encoding="utf-8"))
    except SyntaxError:
        return None


def _imports_from_module(rel_posix: str) -> frozenset[str]:
    tree = _parse_module(rel_posix)
    if tree is None:
        return frozenset()
    current_pkg = rel_posix.replace("/", ".").removesuffix(".py")
    if current_pkg.endswith(".__init__"):
        current_pkg = current_pkg.removesuffix(".__init__")
    discovered: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                discovered.add(_resolve_import_to_path(alias.name, current_pkg))
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            base = node.module
            if node.level:
                pkg_parts = current_pkg.split(".")
                parent = ".".join(pkg_parts[: max(0, len(pkg_parts) - (node.level - 1))])
                base = f"{parent}.{node.module}" if parent else node.module
            discovered.add(_resolve_import_to_path(base, current_pkg))
            for alias in node.names:
                if alias.name == "*":
                    continue
                discovered.add(_resolve_import_to_path(f"{base}.{alias.name}", current_pkg))
    return frozenset(path for path in discovered if path)


def _resolve_import_to_path(dotted: str, _current_pkg: str) -> str:
    """Map import name to repo-relative module path (best-effort, P4 scan roots only)."""
    dotted = dotted.strip(".")
    if not dotted.startswith(("intergrax", "agents", "applications")):
        return ""
    candidate = dotted.replace(".", "/") + ".py"
    if (_REPO_ROOT / candidate).is_file():
        return candidate
    init_candidate = dotted.replace(".", "/") + "/__init__.py"
    if (_REPO_ROOT / init_candidate).is_file():
        return init_candidate
    return ""


def _sanctioned_production_module_paths(
    registry: tuple[RegisteredProductionCompositionSite, ...],
) -> frozenset[str]:
    return frozenset(
        row.path
        for row in registry
        if row.classification
        in (
            ProductionCompositionSiteClassification.SANCTIONED_PRODUCTION_ROOT,
            ProductionCompositionSiteClassification.SANCTIONED_PRODUCTION_ROUTER,
        )
    )


def _development_lab_module_paths(
    registry: tuple[RegisteredProductionCompositionSite, ...],
) -> frozenset[str]:
    return frozenset(
        row.path
        for row in registry
        if row.classification == ProductionCompositionSiteClassification.DEVELOPMENT_LAB_OR_NON_STRICT_RUNTIME
    )


def _import_closure(entry_modules: frozenset[str]) -> frozenset[str]:
    closure: set[str] = set()
    queue = [m for m in entry_modules if m]
    while queue:
        module = queue.pop()
        if module in closure or not module:
            continue
        closure.add(module)
        for imported in _imports_from_module(module):
            if imported and imported not in closure:
                queue.append(imported)
    return frozenset(closure)


_CLASS_NAMES_CACHE: dict[str, frozenset[str]] = {}
_IMPORTS_CACHE: dict[str, frozenset[str]] = {}


def _class_names_in_module(rel_posix: str) -> frozenset[str]:
    if rel_posix in _CLASS_NAMES_CACHE:
        return _CLASS_NAMES_CACHE[rel_posix]
    tree = _parse_module(rel_posix)
    if tree is None:
        result = frozenset()
    else:
        result = frozenset(
            node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        )
    _CLASS_NAMES_CACHE[rel_posix] = result
    return result


def _cached_imports(rel_posix: str) -> frozenset[str]:
    if rel_posix in _IMPORTS_CACHE:
        return _IMPORTS_CACHE[rel_posix]
    result = _imports_from_module(rel_posix)
    _IMPORTS_CACHE[rel_posix] = result
    return result


def _model_consumer_classes_by_module(
    surfaces: frozenset[ModelConsumerSurface],
) -> dict[str, frozenset[str]]:
    modules = frozenset(surface.path for surface in surfaces)
    return {
        module: _class_names_in_module(module)
        for module in modules
    }


def _resolve_callee_consumer_module(
    rel_posix: str,
    callee: str,
    consumer_modules: frozenset[str],
) -> str:
    if callee in _class_names_in_module(rel_posix) and rel_posix in consumer_modules:
        return rel_posix
    for imported in _cached_imports(rel_posix):
        if imported not in consumer_modules:
            continue
        if callee in _class_names_in_module(imported):
            return imported
    return ""


def _build_instantiation_adjacency(
    source_modules: frozenset[str],
    consumer_modules: frozenset[str],
) -> dict[str, frozenset[str]]:
    adjacency: dict[str, set[str]] = {module: set() for module in source_modules}
    for rel_posix in source_modules:
        tree = _parse_module(rel_posix)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            callee = _ast_call_callee_root_name(node)
            if callee is None:
                continue
            target = _resolve_callee_consumer_module(rel_posix, callee, consumer_modules)
            if target:
                adjacency.setdefault(rel_posix, set()).add(target)
    return {module: frozenset(targets) for module, targets in adjacency.items()}


def _wiring_closure(
    seed_modules: frozenset[str],
    consumer_modules: frozenset[str],
    import_closed_modules: frozenset[str],
    extra_edges: frozenset[CompositionEdge],
) -> frozenset[str]:
    adjacency = _build_instantiation_adjacency(import_closed_modules, consumer_modules)
    reachable: set[str] = set(seed_modules)
    queue = list(seed_modules)
    while queue:
        module = queue.pop()
        for target in adjacency.get(module, frozenset()):
            if target not in reachable:
                reachable.add(target)
                queue.append(target)
        for edge in extra_edges:
            if edge.source_module_path == module and edge.target_module_path not in reachable:
                reachable.add(edge.target_module_path)
                queue.append(edge.target_module_path)
    return frozenset(reachable)


def _sanctioned_modules_invoke_stream_methods(sanctioned_modules: frozenset[str]) -> bool:
    for module in sanctioned_modules:
        tree = _parse_module(module)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr not in _STREAM_METHODS:
                continue
            return True
    return False


def _inference_executor_mechanically_unreachable(
    registry: tuple[RegisteredProductionCompositionSite, ...],
    extra_edges: frozenset[CompositionEdge],
) -> bool:
    for row in registry:
        if row.site_kind == ProductionCompositionSiteKind.GOVERNED_INFERENCE_EXECUTOR_CALL:
            return False
        if (
            row.site_kind == ProductionCompositionSiteKind.STRATEGY_ROUTER_INFERENCE_EXECUTOR_KW
            and row.classification
            in (
                ProductionCompositionSiteClassification.SANCTIONED_PRODUCTION_ROOT,
                ProductionCompositionSiteClassification.SANCTIONED_PRODUCTION_ROUTER,
            )
        ):
            return False
    for edge in extra_edges:
        if edge.target_module_path == "intergrax/runtime/execution/inference.py":
            if edge.edge_kind.startswith("instantiate:") or edge.edge_kind == "synthetic:inference_executor":
                return False
        if edge.edge_kind == "synthetic:inference_executor":
            return False
    return True


def _expectation_verdict_for_reason(
    reason: NonProductionReachabilityReason,
) -> ReachabilityVerdict:
    if reason == NonProductionReachabilityReason.INTERNAL_INFERENCE_BACKEND_UNREACHABLE:
        return ReachabilityVerdict.NOT_REACHABLE_FROM_SANCTIONED_PRODUCTION_ROOT
    if reason == NonProductionReachabilityReason.TIER2_AGENT_NOT_SANCTIONED_COMPOSITION_ROOT:
        return ReachabilityVerdict.NOT_REACHABLE_FROM_SANCTIONED_PRODUCTION_ROOT
    if reason == NonProductionReachabilityReason.TIER3_APPLICATION_NOT_SANCTIONED_COMPOSITION_ROOT:
        return ReachabilityVerdict.NOT_REACHABLE_FROM_SANCTIONED_PRODUCTION_ROOT
    return ReachabilityVerdict.NOT_REACHABLE_FROM_SANCTIONED_PRODUCTION_ROOT


def evaluate_non_production_model_surface_reachability(
    surface: ModelConsumerSurface,
    *,
    model_registry: tuple[RegisteredModelCallSurface, ...],
    context: ReachabilityEvaluationContext,
    all_non_production_surfaces: frozenset[ModelConsumerSurface],
) -> MechanicalReachabilityResult:
    consumer_modules = frozenset(surface.path for surface in all_non_production_surfaces)
    sanctioned = _sanctioned_production_module_paths(context.production_composition_registry)
    lab_modules = _development_lab_module_paths(context.production_composition_registry)
    wired_modules = _wiring_closure(
        sanctioned,
        consumer_modules,
        sanctioned,
        context.extra_composition_edges,
    )

    if surface == _INFERENCE_SURFACE:
        if not _inference_executor_mechanically_unreachable(
            context.production_composition_registry,
            context.extra_composition_edges,
        ):
            return MechanicalReachabilityResult(
                surface=surface,
                verdict=ReachabilityVerdict.PRODUCTION_REACHABLE,
                reason=ReachabilityReason.INFERENCE_EXECUTOR_PRODUCTION_WIRED,
            )
        return MechanicalReachabilityResult(
            surface=surface,
            verdict=ReachabilityVerdict.NOT_REACHABLE_FROM_SANCTIONED_PRODUCTION_ROOT,
            reason=ReachabilityReason.INFERENCE_EXECUTOR_NO_PRODUCTION_CALLER,
        )

    if surface.path.startswith("agents/"):
        return MechanicalReachabilityResult(
            surface=surface,
            verdict=ReachabilityVerdict.NOT_REACHABLE_FROM_SANCTIONED_PRODUCTION_ROOT,
            reason=ReachabilityReason.TIER2_AGENT_MODULE,
        )

    if surface.path.startswith("applications/"):
        return MechanicalReachabilityResult(
            surface=surface,
            verdict=ReachabilityVerdict.NOT_REACHABLE_FROM_SANCTIONED_PRODUCTION_ROOT,
            reason=ReachabilityReason.TIER3_APPLICATION_MODULE,
        )

    if surface.path == _WRAPPER_MODULE and surface.method in _STREAM_METHODS:
        if _sanctioned_modules_invoke_stream_methods(sanctioned):
            return MechanicalReachabilityResult(
                surface=surface,
                verdict=ReachabilityVerdict.PRODUCTION_REACHABLE,
                reason=ReachabilityReason.SYNTHETIC_PRODUCTION_COMPOSITION_EDGE,
            )
        return MechanicalReachabilityResult(
            surface=surface,
            verdict=ReachabilityVerdict.NOT_REACHABLE_FROM_SANCTIONED_PRODUCTION_ROOT,
            reason=ReachabilityReason.WRAPPER_STREAM_NOT_INVOKED_FROM_SANCTIONED_COMPOSITION,
        )

    if surface.path in lab_modules and surface.path not in sanctioned:
        return MechanicalReachabilityResult(
            surface=surface,
            verdict=ReachabilityVerdict.DEVELOPMENT_LAB_ONLY,
            reason=ReachabilityReason.CONSUMER_NOT_INSTANTIATED_FROM_SANCTIONED_WIRING,
        )

    for edge in context.extra_composition_edges:
        if edge.target_module_path == surface.path:
            return MechanicalReachabilityResult(
                surface=surface,
                verdict=ReachabilityVerdict.PRODUCTION_REACHABLE,
                reason=ReachabilityReason.SYNTHETIC_PRODUCTION_COMPOSITION_EDGE,
            )

    if surface.path in wired_modules and surface.path not in sanctioned:
        return MechanicalReachabilityResult(
            surface=surface,
            verdict=ReachabilityVerdict.PRODUCTION_REACHABLE,
            reason=ReachabilityReason.SYNTHETIC_PRODUCTION_COMPOSITION_EDGE,
        )

    return MechanicalReachabilityResult(
        surface=surface,
        verdict=ReachabilityVerdict.NOT_REACHABLE_FROM_SANCTIONED_PRODUCTION_ROOT,
        reason=ReachabilityReason.CONSUMER_NOT_INSTANTIATED_FROM_SANCTIONED_WIRING,
    )


def evaluate_all_non_production_reachability(
    model_registry: tuple[RegisteredModelCallSurface, ...],
    context: ReachabilityEvaluationContext,
) -> tuple[MechanicalReachabilityResult, ...]:
    surfaces = frozenset(
        ModelConsumerSurface(path=row.path, method=row.method)
        for row in model_registry
        if row.classification == ModelCallSurfaceClassification.NON_PRODUCTION
    )
    return tuple(
        evaluate_non_production_model_surface_reachability(
            surface,
            model_registry=model_registry,
            context=context,
            all_non_production_surfaces=surfaces,
        )
        for surface in sorted(surfaces, key=lambda s: s.key)
    )


def compare_mechanical_reachability_to_expectations(
    mechanical: tuple[MechanicalReachabilityResult, ...],
    reachability_registry: tuple[RegisteredNonProductionModelReachability, ...],
) -> ReachabilityExpectationParityResult:
    registry_by_key = {row.key: row for row in reachability_registry}
    mechanical_by_key = {result.surface.key: result for result in mechanical}

    duplicate_registry_keys: set[tuple[str, str]] = set()
    seen: set[tuple[str, str]] = set()
    for row in reachability_registry:
        if row.key in seen:
            duplicate_registry_keys.add(row.key)
        seen.add(row.key)

    unknown = frozenset(key for key in mechanical_by_key if key not in registry_by_key)
    orphan = frozenset(key for key in registry_by_key if key not in mechanical_by_key)
    contradictions: set[tuple[str, str]] = set()
    unresolved: set[tuple[str, str]] = set()

    for key, result in mechanical_by_key.items():
        row = registry_by_key.get(key)
        if row is None:
            continue
        expected = _expectation_verdict_for_reason(row.reason)
        if result.verdict == ReachabilityVerdict.PRODUCTION_REACHABLE:
            contradictions.add(key)
        elif result.verdict != expected and result.verdict not in (
            ReachabilityVerdict.NOT_REACHABLE_FROM_SANCTIONED_PRODUCTION_ROOT,
            ReachabilityVerdict.DEVELOPMENT_LAB_ONLY,
            ReachabilityVerdict.TEST_OR_QUALIFICATION_ONLY,
        ):
            unresolved.add(key)

    return ReachabilityExpectationParityResult(
        duplicate_registry_keys=frozenset(duplicate_registry_keys),
        unknown=unknown,
        orphan=orphan,
        contradictions=contradictions,
        unresolved=unresolved,
    )


def build_synthetic_inference_executor_production_edge(
    composition_root_path: str,
) -> CompositionEdge:
    return CompositionEdge(
        source_module_path=composition_root_path,
        target_module_path="intergrax/runtime/execution/inference.py",
        edge_kind="synthetic:inference_executor",
    )


__all__ = [
    "ReachabilityEvaluationContext",
    "build_synthetic_inference_executor_production_edge",
    "compare_mechanical_reachability_to_expectations",
    "evaluate_all_non_production_reachability",
    "evaluate_non_production_model_surface_reachability",
]
