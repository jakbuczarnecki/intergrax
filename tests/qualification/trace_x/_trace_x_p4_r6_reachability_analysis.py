# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P4-R6/R7 static production reachability analysis (P4-bounded, not a general call graph)."""

from __future__ import annotations

import ast
import contextvars
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
    ProductionReachabilityGraphSnapshot,
    ReachabilityExpectationParityResult,
    ReachabilityReason,
    ReachabilityVerdict,
    UnresolvedCompositionReason,
    UnresolvedCompositionSite,
)
from tests.qualification.trace_x._trace_x_p4_support import (
    _PRODUCTION_EXCLUDE_DIR_NAMES,
    _REPO_ROOT,
    _ast_call_callee_root_name,
)

_INFERENCE_SURFACE = ModelConsumerSurface(
    path="intergrax/runtime/execution/inference.py",
    method="generate_structured",
)
_WRAPPER_MODULE = "intergrax/runtime/llm/model_call_runtime_evidence_adapter.py"
_STREAM_METHODS: Final[frozenset[str]] = frozenset({"stream_messages", "stream_with_tools"})
_UNRESOLVED_EDGE_PREFIX: Final[str] = "unresolved:"
_COMPOSITION_FACTORY_CALLEES: Final[frozenset[str]] = frozenset(
    {
        "StrategyExecutionRouter",
        "InferenceExecutor",
        "build_governed_inference_executor",
        "wrap_model_call_runtime_evidence",
        "ModelCallRuntimeEvidenceAdapter",
        "RuntimeConfig",
    },
)


@dataclass(frozen=True, slots=True)
class ReachabilityEvaluationContext:
    production_composition_registry: tuple[RegisteredProductionCompositionSite, ...]
    extra_composition_edges: frozenset[CompositionEdge] = frozenset()
    module_source_overrides: frozenset[tuple[str, str]] = frozenset()


@dataclass(frozen=True, slots=True)
class _ImportBinding:
    module_path: str
    symbol: str | None
    import_dotted: str | None = None
    in_scope_import_attempt: bool = False


_MODULE_SOURCE_OVERRIDES: contextvars.ContextVar[dict[str, str]] = contextvars.ContextVar(
    "_MODULE_SOURCE_OVERRIDES",
    default={},
)


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


def _parse_module(rel_posix: str) -> ast.Module | None:
    overrides = _MODULE_SOURCE_OVERRIDES.get()
    if rel_posix in overrides:
        try:
            return ast.parse(overrides[rel_posix])
        except SyntaxError:
            return None
    py_path = _REPO_ROOT / rel_posix
    if not py_path.is_file():
        return None
    try:
        return ast.parse(py_path.read_text(encoding="utf-8"))
    except SyntaxError:
        return None


def _module_dotted_from_posix(rel_posix: str) -> str:
    dotted = rel_posix.replace("/", ".").removesuffix(".py")
    if dotted.endswith(".__init__"):
        dotted = dotted.removesuffix(".__init__")
    return dotted


def _resolve_attr_chain_to_module(
    expr: ast.expr,
    bindings: dict[str, _ImportBinding],
) -> str | None:
    if isinstance(expr, ast.Name):
        binding = bindings.get(expr.id)
        if binding is None:
            return None
        if binding.import_dotted:
            return _resolve_import_to_path(binding.import_dotted, "") or None
        if binding.module_path:
            return binding.module_path
        return _resolve_import_to_path(expr.id, "") or None
    if isinstance(expr, ast.Attribute):
        dotted = _attribute_chain_to_dotted(expr)
        if dotted is None:
            return None
        return _resolve_import_to_path(dotted, "") or None
    return None


def _resolve_import_to_path(dotted: str, _current_pkg: str) -> str:
    dotted = dotted.strip(".")
    if not dotted.startswith(
        ("intergrax", "agents", "applications", "tests."),
    ):
        return ""
    candidate = dotted.replace(".", "/") + ".py"
    if (_REPO_ROOT / candidate).is_file():
        return candidate
    init_candidate = dotted.replace(".", "/") + "/__init__.py"
    if (_REPO_ROOT / init_candidate).is_file():
        return init_candidate
    return ""


def _attribute_chain_to_dotted(expr: ast.expr) -> str | None:
    segments: list[str] = []
    node: ast.expr = expr
    while isinstance(node, ast.Attribute):
        segments.insert(0, node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    dotted = node.id
    for segment in segments:
        dotted = f"{dotted}.{segment}"
    return dotted


def _current_package(rel_posix: str) -> str:
    current_pkg = rel_posix.replace("/", ".").removesuffix(".py")
    if current_pkg.endswith(".__init__"):
        current_pkg = current_pkg.removesuffix(".__init__")
    return current_pkg


def _imports_from_module(rel_posix: str) -> frozenset[str]:
    tree = _parse_module(rel_posix)
    if tree is None:
        return frozenset()
    current_pkg = _current_package(rel_posix)
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


def _import_bindings(rel_posix: str) -> dict[str, _ImportBinding]:
    tree = _parse_module(rel_posix)
    if tree is None:
        return {}
    current_pkg = _current_package(rel_posix)
    bindings: dict[str, _ImportBinding] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                dotted = alias.name
                if alias.asname:
                    local = alias.asname
                    path = _resolve_import_to_path(dotted, current_pkg)
                    bindings[local] = _ImportBinding(
                        module_path=path,
                        symbol=None,
                        import_dotted=dotted if "." in dotted else None,
                    )
                else:
                    root = dotted.split(".")[0]
                    path = _resolve_import_to_path(root, current_pkg)
                    bindings[root] = _ImportBinding(
                        module_path=path,
                        symbol=None,
                        import_dotted=None,
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            base = node.module
            if node.level:
                pkg_parts = current_pkg.split(".")
                parent = ".".join(pkg_parts[: max(0, len(pkg_parts) - (node.level - 1))])
                base = f"{parent}.{node.module}" if parent else node.module
            base_path = _resolve_import_to_path(base, current_pkg)
            in_scope = base.startswith(
                ("intergrax", "agents", "applications", "tests.qualification.trace_x.r8_fixtures"),
            )
            for alias in node.names:
                if alias.name == "*":
                    continue
                local = alias.asname or alias.name
                symbol_path = _resolve_import_to_path(f"{base}.{alias.name}", current_pkg)
                module_path = symbol_path or base_path
                bindings[local] = _ImportBinding(
                    module_path=module_path,
                    symbol=alias.name,
                    import_dotted=None,
                    in_scope_import_attempt=in_scope,
                )
    return bindings


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
_FUNCTION_NAMES_CACHE: dict[str, frozenset[str]] = {}
_IMPORTS_CACHE: dict[str, frozenset[str]] = {}
_BINDINGS_CACHE: dict[str, dict[str, _ImportBinding]] = {}


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


def _function_names_in_module(rel_posix: str) -> frozenset[str]:
    if rel_posix in _FUNCTION_NAMES_CACHE:
        return _FUNCTION_NAMES_CACHE[rel_posix]
    tree = _parse_module(rel_posix)
    if tree is None:
        result = frozenset()
    else:
        result = frozenset(
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
        )
    _FUNCTION_NAMES_CACHE[rel_posix] = result
    return result


def _cached_imports(rel_posix: str) -> frozenset[str]:
    if rel_posix in _IMPORTS_CACHE:
        return _IMPORTS_CACHE[rel_posix]
    result = _imports_from_module(rel_posix)
    _IMPORTS_CACHE[rel_posix] = result
    return result


def _cached_bindings(rel_posix: str) -> dict[str, _ImportBinding]:
    if rel_posix in _BINDINGS_CACHE:
        return _BINDINGS_CACHE[rel_posix]
    result = _import_bindings(rel_posix)
    _BINDINGS_CACHE[rel_posix] = result
    return result


def _resolve_class_in_module(module_path: str, class_name: str) -> str | None:
    if class_name not in _class_names_in_module(module_path):
        return ""
    return module_path


def _resolve_class_target_module(
    rel_posix: str,
    name: str,
    bindings: dict[str, _ImportBinding],
) -> str | None:
    if name in _class_names_in_module(rel_posix):
        return rel_posix
    if name not in bindings:
        return ""
    binding = bindings[name]
    if not binding.module_path:
        return None
    if binding.symbol is None:
        return ""
    return _resolve_class_in_module(binding.module_path, binding.symbol)


def _call_callee_symbol(call: ast.Call) -> str:
    callee = _ast_call_callee_root_name(call)
    return callee or ""


def _resolve_factory_target_module(
    rel_posix: str,
    name: str,
    bindings: dict[str, _ImportBinding],
) -> str | None:
    if name not in _COMPOSITION_FACTORY_CALLEES:
        return ""
    if name in bindings:
        binding = bindings[name]
        if not binding.module_path:
            return None
        return binding.module_path
    if name in _function_names_in_module(rel_posix):
        return rel_posix
    return ""


def _resolve_call_target_module(
    rel_posix: str,
    call: ast.Call,
) -> str | None:
    """Return target module path, empty if not a composition edge, None if ambiguous/unresolved."""
    bindings = _cached_bindings(rel_posix)
    func = call.func

    if isinstance(func, ast.Name):
        name = func.id
        factory_target = _resolve_factory_target_module(rel_posix, name, bindings)
        if factory_target is None:
            return None
        if factory_target:
            return factory_target
        if name in bindings:
            binding = bindings[name]
            if not binding.module_path:
                if binding.in_scope_import_attempt and binding.symbol is not None:
                    return None
                return ""
            if binding.symbol is None:
                return ""
            return _resolve_class_in_module(binding.module_path, binding.symbol)
        return _resolve_class_target_module(rel_posix, name, bindings)

    if isinstance(func, ast.Attribute):
        attr = func.attr
        factory_target = _resolve_factory_target_module(rel_posix, attr, bindings)
        if factory_target is None:
            return None
        if factory_target:
            return factory_target
        owner_module = _resolve_attr_chain_to_module(func.value, bindings)
        if owner_module is None:
            return None
        if not owner_module:
            return ""
        return _resolve_class_in_module(owner_module, attr)

    callee = _ast_call_callee_root_name(call)
    if callee is None:
        return ""
    factory_target = _resolve_factory_target_module(rel_posix, callee, bindings)
    if factory_target is None:
        return None
    if factory_target:
        return factory_target
    return _resolve_class_target_module(rel_posix, callee, bindings)


def _composition_call_is_resolution_relevant(
    rel_posix: str,
    call: ast.Call,
) -> bool:
    bindings = _cached_bindings(rel_posix)
    func = call.func
    if isinstance(func, ast.Name):
        name = func.id
        if name in _COMPOSITION_FACTORY_CALLEES:
            return True
        if name in _class_names_in_module(rel_posix):
            return True
        if name in bindings:
            binding = bindings[name]
            if binding.symbol is None:
                return False
            if binding.in_scope_import_attempt or binding.module_path:
                if binding.symbol in _COMPOSITION_FACTORY_CALLEES:
                    return True
                if binding.module_path and binding.symbol in _class_names_in_module(binding.module_path):
                    return True
                if binding.in_scope_import_attempt and not binding.module_path:
                    return True
        return False
    if isinstance(func, ast.Attribute):
        attr = func.attr
        if attr in _COMPOSITION_FACTORY_CALLEES:
            return True
        owner_module = _resolve_attr_chain_to_module(func.value, bindings)
        if owner_module and attr in _class_names_in_module(owner_module):
            return True
        return False
    callee = _call_callee_symbol(call)
    return callee in _COMPOSITION_FACTORY_CALLEES


def _discover_composition_edges(
    analysis_modules: frozenset[str],
) -> tuple[frozenset[CompositionEdge], tuple[UnresolvedCompositionSite, ...]]:
    edges: set[CompositionEdge] = set()
    unresolved_sites: list[UnresolvedCompositionSite] = []
    for rel_posix in analysis_modules:
        tree = _parse_module(rel_posix)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target = _resolve_call_target_module(rel_posix, node)
            if target is None:
                if _composition_call_is_resolution_relevant(rel_posix, node):
                    unresolved_sites.append(
                        UnresolvedCompositionSite(
                            source_module_path=rel_posix,
                            line_number=node.lineno,
                            callee_symbol=_call_callee_symbol(node),
                            reason=UnresolvedCompositionReason.COMPOSITION_RELEVANT_CALL_UNRESOLVED,
                        ),
                    )
                continue
            if target and target in analysis_modules:
                callee = _call_callee_symbol(node) or "call"
                edges.add(
                    CompositionEdge(
                        source_module_path=rel_posix,
                        target_module_path=target,
                        edge_kind=f"instantiate:{callee}",
                    ),
                )
    return frozenset(edges), tuple(unresolved_sites)


def _reachable_modules_bfs(
    seed_modules: frozenset[str],
    adjacency: dict[str, frozenset[str]],
) -> frozenset[str]:
    reachable: set[str] = set(seed_modules)
    queue = list(seed_modules)
    while queue:
        module = queue.pop()
        for target in adjacency.get(module, frozenset()):
            if target not in reachable:
                reachable.add(target)
                queue.append(target)
    return frozenset(reachable)


def _adjacency_from_edges(
    edges: frozenset[CompositionEdge],
    module_universe: frozenset[str],
) -> dict[str, frozenset[str]]:
    adjacency: dict[str, set[str]] = {module: set() for module in module_universe}
    for edge in edges:
        if edge.source_module_path in module_universe and edge.target_module_path in module_universe:
            adjacency.setdefault(edge.source_module_path, set()).add(edge.target_module_path)
    return {module: frozenset(targets) for module, targets in adjacency.items()}


def _shortest_path_edges(
    seeds: frozenset[str],
    target: str,
    adjacency: dict[str, frozenset[str]],
    all_edges: frozenset[CompositionEdge],
) -> tuple[CompositionEdge, ...]:
    if target not in seeds:
        parent: dict[str, str] = {}
        queue = list(seeds)
        visited = set(seeds)
        found = False
        while queue and not found:
            current = queue.pop(0)
            for nxt in adjacency.get(current, frozenset()):
                if nxt not in visited:
                    visited.add(nxt)
                    parent[nxt] = current
                    if nxt == target:
                        found = True
                        break
                    queue.append(nxt)
        if not found:
            return ()
        chain: list[str] = [target]
        while chain[-1] not in seeds:
            chain.append(parent[chain[-1]])
        chain.reverse()
    else:
        chain = [target]
    path_edges: list[CompositionEdge] = []
    for idx in range(len(chain) - 1):
        src, dst = chain[idx], chain[idx + 1]
        match = next(
            (
                edge
                for edge in all_edges
                if edge.source_module_path == src and edge.target_module_path == dst
            ),
            CompositionEdge(source_module_path=src, target_module_path=dst, edge_kind="path:hop"),
        )
        path_edges.append(match)
    return tuple(path_edges)


def _module_invokes_stream_methods(rel_posix: str) -> bool:
    tree = _parse_module(rel_posix)
    if tree is None:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in _STREAM_METHODS:
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


def build_production_reachability_graph(
    context: ReachabilityEvaluationContext,
) -> ProductionReachabilityGraphSnapshot:
    override_map = dict(context.module_source_overrides)
    override_token = _MODULE_SOURCE_OVERRIDES.set(override_map)
    try:
        sanctioned = _sanctioned_production_module_paths(context.production_composition_registry)
        lab_seeds = _development_lab_module_paths(context.production_composition_registry)
        import_closed = _import_closure(sanctioned)
        edge_modules = frozenset(
            edge.source_module_path for edge in context.extra_composition_edges
        ) | frozenset(edge.target_module_path for edge in context.extra_composition_edges)
        analysis_modules = import_closed | frozenset(override_map) | edge_modules
        discovered_edges, unresolved_from_parse = _discover_composition_edges(analysis_modules)
        all_edges = discovered_edges | context.extra_composition_edges
        unresolved_sites: list[UnresolvedCompositionSite] = list(unresolved_from_parse)
        for edge in context.extra_composition_edges:
            if edge.edge_kind.startswith(_UNRESOLVED_EDGE_PREFIX):
                unresolved_sites.append(
                    UnresolvedCompositionSite(
                        source_module_path=edge.source_module_path,
                        line_number=0,
                        callee_symbol=edge.edge_kind.removeprefix(_UNRESOLVED_EDGE_PREFIX),
                        reason=UnresolvedCompositionReason.COMPOSITION_RELEVANT_CALL_UNRESOLVED,
                    ),
                )
        module_universe = import_closed | frozenset(
            edge.source_module_path for edge in all_edges
        ) | frozenset(edge.target_module_path for edge in all_edges) | frozenset(override_map)
        adjacency = _adjacency_from_edges(all_edges, module_universe)
        production_reachable = _reachable_modules_bfs(sanctioned, adjacency)
        lab_reachable = _reachable_modules_bfs(lab_seeds, adjacency) if lab_seeds else frozenset()
        production_reachable_unresolved = tuple(
            site
            for site in unresolved_sites
            if site.source_module_path in production_reachable
        )
        unresolved_modules = frozenset(site.source_module_path for site in unresolved_sites)
        return ProductionReachabilityGraphSnapshot(
            sanctioned_seed_modules=sanctioned,
            import_closure_modules=import_closed,
            composition_edges=all_edges,
            production_reachable_modules=production_reachable,
            lab_reachable_modules=lab_reachable,
            unresolved_sites=tuple(unresolved_sites),
            production_reachable_unresolved_sites=production_reachable_unresolved,
            unresolved_modules=unresolved_modules,
        )
    finally:
        _MODULE_SOURCE_OVERRIDES.reset(override_token)


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
    graph: ProductionReachabilityGraphSnapshot,
) -> MechanicalReachabilityResult:
    sanctioned = graph.sanctioned_seed_modules
    lab_modules = _development_lab_module_paths(context.production_composition_registry)
    adjacency = _adjacency_from_edges(graph.composition_edges, graph.import_closure_modules)

    for edge in context.extra_composition_edges:
        if edge.edge_kind.startswith(_UNRESOLVED_EDGE_PREFIX) and edge.target_module_path == surface.path:
            return MechanicalReachabilityResult(
                surface=surface,
                verdict=ReachabilityVerdict.UNRESOLVED,
                reason=ReachabilityReason.AMBIGUOUS_COMPOSITION_EDGE,
            )

    if surface == _INFERENCE_SURFACE:
        inference_reachable = (
            "intergrax/runtime/execution/inference.py" in graph.production_reachable_modules
        )
        if inference_reachable or not _inference_executor_mechanically_unreachable(
            context.production_composition_registry,
            context.extra_composition_edges,
        ):
            path_edges = _shortest_path_edges(
                sanctioned,
                "intergrax/runtime/execution/inference.py",
                adjacency,
                graph.composition_edges,
            )
            return MechanicalReachabilityResult(
                surface=surface,
                verdict=ReachabilityVerdict.PRODUCTION_REACHABLE,
                reason=ReachabilityReason.INFERENCE_EXECUTOR_PRODUCTION_WIRED,
                reachable_from=sanctioned,
                evidence_edges=path_edges,
            )
        return MechanicalReachabilityResult(
            surface=surface,
            verdict=ReachabilityVerdict.NOT_REACHABLE_FROM_SANCTIONED_PRODUCTION_ROOT,
            reason=ReachabilityReason.INFERENCE_EXECUTOR_NO_PRODUCTION_CALLER,
        )

    if surface.path == _WRAPPER_MODULE and surface.method in _STREAM_METHODS:
        stream_modules = frozenset(
            module
            for module in graph.production_reachable_modules
            if module != _WRAPPER_MODULE and _module_invokes_stream_methods(module)
        )
        if stream_modules:
            return MechanicalReachabilityResult(
                surface=surface,
                verdict=ReachabilityVerdict.PRODUCTION_REACHABLE,
                reason=ReachabilityReason.SYNTHETIC_PRODUCTION_COMPOSITION_EDGE,
                reachable_from=sanctioned,
            )
        return MechanicalReachabilityResult(
            surface=surface,
            verdict=ReachabilityVerdict.NOT_REACHABLE_FROM_SANCTIONED_PRODUCTION_ROOT,
            reason=ReachabilityReason.WRAPPER_STREAM_NOT_INVOKED_FROM_SANCTIONED_COMPOSITION,
        )

    path_edges = _shortest_path_edges(
        sanctioned,
        surface.path,
        adjacency,
        graph.composition_edges,
    )
    reachable_from = sanctioned if surface.path in graph.production_reachable_modules else frozenset()

    if surface.path in graph.production_reachable_modules:
        return MechanicalReachabilityResult(
            surface=surface,
            verdict=ReachabilityVerdict.PRODUCTION_REACHABLE,
            reason=ReachabilityReason.SYNTHETIC_PRODUCTION_COMPOSITION_EDGE,
            reachable_from=reachable_from,
            evidence_edges=path_edges,
        )

    if (
        surface.path in graph.lab_reachable_modules
        and surface.path not in graph.production_reachable_modules
        and surface.path in lab_modules
    ):
        return MechanicalReachabilityResult(
            surface=surface,
            verdict=ReachabilityVerdict.DEVELOPMENT_LAB_ONLY,
            reason=ReachabilityReason.CONSUMER_NOT_INSTANTIATED_FROM_SANCTIONED_WIRING,
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
    graph = build_production_reachability_graph(context)
    return tuple(
        evaluate_non_production_model_surface_reachability(
            surface,
            model_registry=model_registry,
            context=context,
            all_non_production_surfaces=surfaces,
            graph=graph,
        )
        for surface in sorted(surfaces, key=lambda s: s.key)
    )


def compare_mechanical_reachability_to_expectations(
    mechanical: tuple[MechanicalReachabilityResult, ...],
    reachability_registry: tuple[RegisteredNonProductionModelReachability, ...],
    *,
    graph: ProductionReachabilityGraphSnapshot | None = None,
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
        if result.verdict == ReachabilityVerdict.UNRESOLVED:
            unresolved.add(key)
            continue
        if result.verdict == ReachabilityVerdict.PRODUCTION_REACHABLE:
            contradictions.add(key)
        elif result.verdict != expected and result.verdict not in (
            ReachabilityVerdict.NOT_REACHABLE_FROM_SANCTIONED_PRODUCTION_ROOT,
            ReachabilityVerdict.DEVELOPMENT_LAB_ONLY,
            ReachabilityVerdict.TEST_OR_QUALIFICATION_ONLY,
        ):
            unresolved.add(key)

    proof_incomplete = bool(
        graph is not None and graph.production_reachable_unresolved_sites,
    )
    return ReachabilityExpectationParityResult(
        duplicate_registry_keys=frozenset(duplicate_registry_keys),
        unknown=unknown,
        orphan=orphan,
        contradictions=contradictions,
        unresolved=unresolved,
        production_reachability_proof_incomplete=proof_incomplete,
    )


def qualify_discover_composition_edges(
    analysis_modules: frozenset[str],
    *,
    module_source_overrides: frozenset[tuple[str, str]] = frozenset(),
) -> tuple[frozenset[CompositionEdge], tuple[UnresolvedCompositionSite, ...]]:
    override_map = dict(module_source_overrides)
    token = _MODULE_SOURCE_OVERRIDES.set(override_map)
    try:
        return _discover_composition_edges(analysis_modules | frozenset(override_map))
    finally:
        _MODULE_SOURCE_OVERRIDES.reset(token)


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
    "build_production_reachability_graph",
    "build_synthetic_inference_executor_production_edge",
    "compare_mechanical_reachability_to_expectations",
    "evaluate_all_non_production_reachability",
    "evaluate_non_production_model_surface_reachability",
    "qualify_discover_composition_edges",
]
