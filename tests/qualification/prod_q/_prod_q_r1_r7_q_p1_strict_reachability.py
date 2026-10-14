# © Artur Czarnecki. All rights reserved.

"""Closed-world AST reachability for PROD-Q-R1-R7-Q-P1 strict production graph."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Closed-world production boundary (operator inventory — no repo-wide scan).
BOUNDARY_MODULE_RELPATHS: Final[tuple[str, ...]] = (
    "applications/governed_contractor_application/host/main.py",
    "applications/governed_contractor_application/host/strict_production_serving.py",
    "intergrax/applications/_shared/strict_production_serving_launcher.py",
    "intergrax/applications/_shared/production_host_composition.py",
    "intergrax/applications/_shared/strict_production_asgi_lifespan.py",
    "intergrax/applications/_shared/production_process_lifecycle.py",
    "intergrax/applications/_shared/production_lifecycle.py",
    "intergrax/applications/_shared/reference_production_lifecycle.py",
    "intergrax/applications/_shared/colocated_serving_runtime_composition.py",
    "intergrax/agent_distribution/deployment.py",
)

STRICT_PRODUCTION_ROOTS: Final[tuple[tuple[str, str], ...]] = (
    (
        "intergrax/applications/_shared/strict_production_serving_launcher.py",
        "open_strict_production_serving_session",
    ),
    (
        "applications/governed_contractor_application/host/strict_production_serving.py",
        "create_governed_contractor_strict_production_asgi_app",
    ),
    (
        "applications/governed_contractor_application/host/main.py",
        "run",
    ),
    (
        "applications/governed_contractor_application/host/main.py",
        "create_app",
    ),
    (
        "applications/governed_contractor_application/host/main.py",
        "create_governed_contractor_process_app",
    ),
)

# Execution / wiring violations (import-only or isinstance guards are not calls).
PROHIBITED_CALL_QUALIFIERS: Final[frozenset[str]] = frozenset(
    {
        "intergrax.agent_distribution.deployment.FakeInMemoryRuntimeDeploymentAdapter",
        "intergrax.applications._shared.reference_production_process_lifecycle.open_reference_production_process_lifecycle",
        "intergrax.applications._shared.reference_production_lifecycle.ReferenceProductionLifecycleLauncher",
        "intergrax.applications._shared.reference_production_lifecycle.wire_reference_production_lifecycle_services",
        "intergrax.applications._shared.reference_production_lifecycle.wire_durable_reference_production_lifecycle_services",
        "intergrax.applications._shared.production_process_composition.create_reference_production_process_composition",
    }
)

PROHIBITED_CALL_SUFFIXES: Final[frozenset[str]] = frozenset(
    {
        "FakeInMemoryRuntimeDeploymentAdapter",
        "open_reference_production_process_lifecycle",
        "ReferenceProductionLifecycleLauncher",
        "wire_reference_production_lifecycle_services",
        "wire_durable_reference_production_lifecycle_services",
        "create_reference_production_process_composition",
    }
)

GCA_STRICT_FACTORY_RELPATHS: Final[frozenset[str]] = frozenset(
    {
        "applications/governed_contractor_application/host/strict_production_serving.py",
        "applications/governed_contractor_application/host/main.py",
    }
)

DEPLOYMENT_ADAPTER_CONSTRUCTOR_SUFFIXES: Final[frozenset[str]] = frozenset(
    {
        "FakeInMemoryRuntimeDeploymentAdapter",
        "VenvBundleColocatedRuntimeDeploymentAdapter",
        "RuntimeDeploymentAdapter",
    }
)


@dataclass(frozen=True, slots=True)
class CallSiteRef:
    module_relpath: str
    qualname: str
    lineno: int
    resolved_target: str | None


@dataclass(frozen=True, slots=True)
class ProhibitedExecution:
    module_relpath: str
    qualname: str
    lineno: int
    kind: str
    detail: str


@dataclass
class ReachabilityReport:
    reachable_functions: set[tuple[str, str]] = field(default_factory=set)
    prohibited: list[ProhibitedExecution] = field(default_factory=list)
    unresolved_calls: list[CallSiteRef] = field(default_factory=list)
    gca_deployment_constructions: list[ProhibitedExecution] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _ModuleRecord:
    relpath: str
    path: Path
    tree: ast.Module
    imports: dict[str, str]
    functions: dict[str, ast.AST]
    methods: dict[tuple[str, str], ast.AST]


def repo_root() -> Path:
    return _REPO_ROOT


def boundary_module_paths() -> dict[str, Path]:
    return {rel: _REPO_ROOT / rel for rel in BOUNDARY_MODULE_RELPATHS}


def _parse_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _import_bindings(tree: ast.Module) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                local = alias.asname or alias.name
                bindings[local] = f"{module}.{alias.name}" if module else alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name
                bindings[local] = alias.name
    return bindings


def _collect_functions(tree: ast.Module) -> tuple[dict[str, ast.AST], dict[tuple[str, str], ast.AST]]:
    funcs: dict[str, ast.AST] = {}
    methods: dict[tuple[str, str], ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            funcs[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                    methods[(node.name, child.name)] = child
    return funcs, methods


def _build_index() -> dict[str, _ModuleRecord]:
    index: dict[str, _ModuleRecord] = {}
    for rel, path in boundary_module_paths().items():
        tree = _parse_module(path)
        funcs, methods = _collect_functions(tree)
        index[rel] = _ModuleRecord(
            relpath=rel,
            path=path,
            tree=tree,
            imports=_import_bindings(tree),
            functions=funcs,
            methods=methods,
        )
    return index


def _resolve_expr(
    node: ast.expr,
    imports: dict[str, str],
    locals_map: dict[str, str] | None = None,
) -> str | None:
    if isinstance(node, ast.Name):
        if locals_map and node.id in locals_map:
            return locals_map[node.id]
        return imports.get(node.id)
    if isinstance(node, ast.Attribute):
        base = _resolve_expr(node.value, imports, locals_map)
        if base is None:
            return None
        return f"{base}.{node.attr}"
    return None


def _call_target_qualifier(node: ast.Call, imports: dict[str, str]) -> str | None:
    return _resolve_expr(node.func, imports)


def _matches_prohibited(qual: str | None, local_name: str | None) -> str | None:
    if qual is not None:
        if qual in PROHIBITED_CALL_QUALIFIERS:
            return qual
        for suffix in PROHIBITED_CALL_SUFFIXES:
            if qual.endswith(f".{suffix}") or qual == suffix:
                return qual
    if local_name in PROHIBITED_CALL_SUFFIXES:
        return local_name
    return None


def _is_isinstance_guard(node: ast.Call, imports: dict[str, str] | None = None) -> bool:
    if not isinstance(node.func, ast.Name) or node.func.id != "isinstance":
        return False
    if len(node.args) < 2:
        return False
    second = node.args[1]
    if isinstance(second, ast.Name):
        if second.id == "FakeInMemoryRuntimeDeploymentAdapter":
            return True
        if imports:
            bound = imports.get(second.id, "")
            return bound.endswith("FakeInMemoryRuntimeDeploymentAdapter")
        return False
    if isinstance(second, ast.Attribute) and second.attr == "FakeInMemoryRuntimeDeploymentAdapter":
        return True
    return False


def _deployment_adapter_constructor_target(
    node: ast.Call,
    imports: dict[str, str],
) -> str | None:
    if _is_isinstance_guard(node, imports):
        return None
    qual = _call_target_qualifier(node, imports)
    candidates: list[str] = []
    if qual:
        candidates.append(qual)
    if isinstance(node.func, ast.Name):
        candidates.append(node.func.id)
        bound = imports.get(node.func.id)
        if bound:
            candidates.append(bound)
    for cand in candidates:
        for suffix in DEPLOYMENT_ADAPTER_CONSTRUCTOR_SUFFIXES:
            if cand == suffix or cand.endswith(f".{suffix}"):
                return cand
    return None


def _condition_is_not_ready_for_traffic(test: ast.expr) -> bool:
    inner = test
    negated = False
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        negated = True
        inner = test.operand
    if not negated:
        return False
    if not isinstance(inner, ast.Attribute) or inner.attr != "ready_for_traffic":
        return False
    readiness = inner.value
    return isinstance(readiness, ast.Attribute) and readiness.attr == "readiness"


def _call_resolves_to_create_governed_contractor_process_app(
    node: ast.Call,
    imports: dict[str, str],
) -> bool:
    qual = _call_target_qualifier(node, imports)
    if qual and qual.endswith("create_governed_contractor_process_app"):
        return True
    if isinstance(node.func, ast.Name):
        if node.func.id == "create_governed_contractor_process_app":
            return True
        bound = imports.get(node.func.id, "")
        return bound.endswith("create_governed_contractor_process_app")
    return False


def _walk_statement_list(stmts: list[ast.stmt]):
    for stmt in stmts:
        for node in ast.walk(stmt):
            yield node


def _last_lineno_in_statement_list(stmts: list[ast.stmt]) -> int:
    last = 0
    for node in _walk_statement_list(stmts):
        lineno = getattr(node, "lineno", None)
        if lineno:
            last = max(last, lineno)
    return last


def _guard_body_unconditionally_raises(body: list[ast.stmt], imports: dict[str, str]) -> bool:
    """True when the not-ready ``if`` body always raises (no nested if, no process app)."""
    if not body:
        return False
    if any(isinstance(stmt, ast.If) for stmt in body):
        return False
    for node in _walk_statement_list(body):
        if isinstance(node, ast.Call) and _call_resolves_to_create_governed_contractor_process_app(
            node,
            imports,
        ):
            return False
    return any(isinstance(stmt, ast.Raise) for stmt in body)


def _if_body_raises_without_process_app(if_node: ast.If, imports: dict[str, str]) -> bool:
    if not _condition_is_not_ready_for_traffic(if_node.test):
        return False
    return _guard_body_unconditionally_raises(if_node.body, imports)


def validate_strict_asgi_admission_structure(
    fnode: ast.AST,
    imports: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    guards = [
        node
        for node in ast.walk(fnode)
        if isinstance(node, ast.If) and _if_body_raises_without_process_app(node, imports)
    ]
    if not guards:
        errors.append(
            "P1-G05: strict ASGI factory must guard on session.readiness.ready_for_traffic "
            "and raise before create_governed_contractor_process_app",
        )
    process_calls = [
        node
        for node in ast.walk(fnode)
        if isinstance(node, ast.Call)
        and _call_resolves_to_create_governed_contractor_process_app(node, imports)
    ]
    if not process_calls:
        errors.append(
            "P1-G05: strict ASGI factory must call create_governed_contractor_process_app "
            "after admission",
        )
    if guards:
        primary_guard = min(guards, key=lambda node: node.lineno or 0)
        guard_floor = _last_lineno_in_statement_list(primary_guard.body)
        for call in process_calls:
            lineno = getattr(call, "lineno", None)
            if lineno is not None and lineno <= guard_floor:
                errors.append(
                    "P1-G05: create_governed_contractor_process_app must not run before "
                    "ready_for_traffic admission guard",
                )
                break
    for if_node in ast.walk(fnode):
        if not isinstance(if_node, ast.If):
            continue
        if not _condition_is_not_ready_for_traffic(if_node.test):
            continue
        for sub in _walk_statement_list(if_node.body):
            if isinstance(sub, ast.Call) and _call_resolves_to_create_governed_contractor_process_app(
                sub,
                imports,
            ):
                errors.append(
                    "P1-G05: create_governed_contractor_process_app must not run on "
                    "not-ready failure path",
                )
    return errors


def _function_qualname(module_relpath: str, name: str, class_name: str | None = None) -> str:
    if class_name:
        return f"{module_relpath}:{class_name}.{name}"
    return f"{module_relpath}:{name}"


def _locate_function(
    index: dict[str, _ModuleRecord],
    module_relpath: str,
    name: str,
) -> tuple[_ModuleRecord, ast.AST, str] | None:
    record = index.get(module_relpath)
    if record is None:
        return None
    if name in record.functions:
        return record, record.functions[name], _function_qualname(module_relpath, name)
    return None


def _module_relpath_for_python_module(module_qual: str) -> str | None:
    for rel in BOUNDARY_MODULE_RELPATHS:
        file_mod = rel.replace("/", ".").removesuffix(".py")
        if module_qual == file_mod:
            return rel
    if module_qual.startswith("governed_contractor_application."):
        candidate = "applications/" + module_qual.replace(".", "/") + ".py"
        if candidate in BOUNDARY_MODULE_RELPATHS:
            return candidate
    return None


def _resolve_callee_to_boundary_function(
    qual: str | None,
    index: dict[str, _ModuleRecord],
) -> tuple[str, str] | None:
    if qual is None:
        return None
    parts = qual.split(".")
    if len(parts) < 2:
        return None
    func_name = parts[-1]
    module_qual = ".".join(parts[:-1])
    rel = _module_relpath_for_python_module(module_qual)
    if rel is not None:
        record = index[rel]
        if func_name in record.functions:
            return rel, func_name
        for cls_name, method_name in record.methods:
            if func_name == method_name:
                return rel, f"{cls_name}.{method_name}"
    for rel, record in index.items():
        if record.imports.get(func_name) == qual:
            if func_name in record.functions:
                return rel, func_name
    return None


def analyze_module_snippet(
    source: str,
    *,
    module_label: str = "<snippet>",
) -> list[ProhibitedExecution]:
    tree = ast.parse(source, filename=module_label)
    imports = _import_bindings(tree)
    funcs, _ = _collect_functions(tree)
    violations: list[ProhibitedExecution] = []
    for fname, fnode in funcs.items():
        for node in ast.walk(fnode):
            if not isinstance(node, ast.Call):
                continue
            if _is_isinstance_guard(node, imports):
                continue
            qual = _call_target_qualifier(node, imports)
            local: str | None = None
            if isinstance(node.func, ast.Name):
                local = node.func.id
            matched = _matches_prohibited(qual, local)
            if matched:
                violations.append(
                    ProhibitedExecution(
                        module_relpath=module_label,
                        qualname=fname,
                        lineno=node.lineno,
                        kind="prohibited_call",
                        detail=matched,
                    ),
                )
    return violations


def analyze_strict_production_reachability() -> ReachabilityReport:
    index = _build_index()
    report = ReachabilityReport()
    queue: list[tuple[str, str]] = list(STRICT_PRODUCTION_ROOTS)
    seen: set[tuple[str, str]] = set()

    while queue:
        module_relpath, func_name = queue.pop(0)
        key = (module_relpath, func_name)
        if key in seen:
            continue
        seen.add(key)
        report.reachable_functions.add(key)

        located = _locate_function(index, module_relpath, func_name.split(".")[-1])
        if located is None:
            continue
        record, fnode, fq = located
        imports = record.imports

        for node in ast.walk(fnode):
            if not isinstance(node, ast.Call):
                continue
            qual = _call_target_qualifier(node, imports)
            local: str | None = None
            if isinstance(node.func, ast.Name):
                local = node.func.id

            if not _is_isinstance_guard(node, imports):
                matched = _matches_prohibited(qual, local)
                if matched:
                    report.prohibited.append(
                        ProhibitedExecution(
                            module_relpath=module_relpath,
                            qualname=fq,
                            lineno=node.lineno,
                            kind="prohibited_call",
                            detail=matched,
                        ),
                    )

            if module_relpath in GCA_STRICT_FACTORY_RELPATHS:
                ctor = _deployment_adapter_constructor_target(node, imports)
                if ctor:
                    report.gca_deployment_constructions.append(
                        ProhibitedExecution(
                            module_relpath=module_relpath,
                            qualname=fq,
                            lineno=node.lineno,
                            kind="gca_deployment_constructor",
                            detail=ctor,
                        ),
                    )

            callee: tuple[str, str] | None = None
            if isinstance(node.func, ast.Name) and node.func.id in record.functions:
                callee = (module_relpath, node.func.id)
            if callee is None:
                callee = _resolve_callee_to_boundary_function(qual, index)
            if callee is None and isinstance(node.func, ast.Name):
                imp = imports.get(node.func.id)
                callee = _resolve_callee_to_boundary_function(imp, index)
            if callee is not None and callee not in seen:
                queue.append(callee)
            elif qual is not None and qual.split(".")[-1] in {
                "open_production_process_lifecycle",
                "reject_reference_runtime_deployment_adapter",
                "wire_production_lifecycle_services",
                "bind_colocated_serving_runtime_after_activation",
                "create_governed_contractor_process_app",
                "attach_strict_production_serving_lifespan",
                "bootstrap_production_registry_projection",
            }:
                if callee is None:
                    report.unresolved_calls.append(
                        CallSiteRef(
                            module_relpath=module_relpath,
                            qualname=fq,
                            lineno=node.lineno,
                            resolved_target=qual,
                        ),
                    )

    return report


def assert_startup_chain_opener(index: dict[str, _ModuleRecord] | None = None) -> None:
    idx = index or _build_index()
    rel = "intergrax/applications/_shared/strict_production_serving_launcher.py"
    record = idx[rel]
    opener = record.functions["open_strict_production_serving_session"]
    found = False
    for node in ast.walk(opener):
        if isinstance(node, ast.Call):
            qual = _call_target_qualifier(node, record.imports)
            if qual and qual.endswith("open_production_process_lifecycle"):
                found = True
    if not found:
        raise AssertionError(
            "P1-G01: open_strict_production_serving_session must call "
            "open_production_process_lifecycle",
        )


def production_lifecycle_rejection_import_is_guard_only() -> None:
    rel = "intergrax/applications/_shared/production_lifecycle.py"
    path = _REPO_ROOT / rel
    tree = _parse_module(path)
    imports = _import_bindings(tree)
    assert "FakeInMemoryRuntimeDeploymentAdapter" in imports
    reject = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "reject_reference_runtime_deployment_adapter":
            reject = node
    assert reject is not None
    fake_calls = [
        n
        for n in ast.walk(reject)
        if isinstance(n, ast.Call)
        and not _is_isinstance_guard(n, imports)
        and (
            _call_target_qualifier(n, imports) or ""
        ).endswith("FakeInMemoryRuntimeDeploymentAdapter")
    ]
    assert not fake_calls, "reject helper must not instantiate FakeInMemoryRuntimeDeploymentAdapter"


def assert_strict_asgi_admission_guard(index: dict[str, _ModuleRecord] | None = None) -> None:
    idx = index or _build_index()
    rel = "applications/governed_contractor_application/host/strict_production_serving.py"
    record = idx[rel]
    factory = record.functions["create_governed_contractor_strict_production_asgi_app"]
    errors = validate_strict_asgi_admission_structure(factory, record.imports)
    if errors:
        raise AssertionError("\n".join(errors))


def analyze_asgi_admission_snippet(
    source: str,
    *,
    function_name: str = "create_governed_contractor_strict_production_asgi_app",
    module_label: str = "<snippet>",
) -> list[str]:
    tree = ast.parse(source, filename=module_label)
    imports = _import_bindings(tree)
    funcs, _ = _collect_functions(tree)
    fnode = funcs.get(function_name)
    if fnode is None:
        return [f"missing function {function_name}"]
    return validate_strict_asgi_admission_structure(fnode, imports)


def analyze_gca_deployment_snippet(
    source: str,
    *,
    module_relpath: str = "applications/governed_contractor_application/host/strict_production_serving.py",
    module_label: str = "<snippet>",
) -> list[ProhibitedExecution]:
    if module_relpath not in GCA_STRICT_FACTORY_RELPATHS:
        return []
    tree = ast.parse(source, filename=module_label)
    imports = _import_bindings(tree)
    funcs, _ = _collect_functions(tree)
    violations: list[ProhibitedExecution] = []
    for fname, fnode in funcs.items():
        for node in ast.walk(fnode):
            if not isinstance(node, ast.Call):
                continue
            ctor = _deployment_adapter_constructor_target(node, imports)
            if ctor:
                violations.append(
                    ProhibitedExecution(
                        module_relpath=module_relpath,
                        qualname=fname,
                        lineno=node.lineno,
                        kind="gca_deployment_constructor",
                        detail=ctor,
                    ),
                )
    return violations
