# © Artur Czarnecki. All rights reserved.

"""Bounded AST evidence for sandbox_host provider factory ownership (PROD-Q-P0)."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Final

_DECLARE_INTEGRATION_CONTRACT_Q: Final = (
    "intergrax.integrations.registry.contract_spec.declare_integration_contract"
)
_SANDBOX_HOST_CONTRACT_Q: Final = (
    "intergrax.runtime.integrations.categories.devops.SandboxHostIntegrationContract"
)
_REGISTER_FROM_MANIFEST_Q: Final = (
    "intergrax.integrations.registry.plugin_register.register_from_manifest"
)
_SANDBOX_HOST_BACKEND_Q: Final = (
    "intergrax.integrations.contracts.sandbox_host.SandboxHostBackend"
)

_UNRESOLVED_COMPOSITION_MARKERS = frozenset({"COMPOSITION_UNRESOLVED", ""})


@dataclass(frozen=True, slots=True)
class SandboxFactoryProviderSpec:
    provider_key: str
    integration_class: str
    typed_integration_factory: str
    legacy_catalog_factory: str
    register_function: str
    contract_spec_relpath: str
    integration_relpath: str
    register_relpath: str
    bundle_relpath: str


_SANDBOX_FACTORY_PROVIDER_SPECS: dict[str, SandboxFactoryProviderSpec] = {
    "e2b": SandboxFactoryProviderSpec(
        provider_key="e2b",
        integration_class="E2bSandboxHostIntegration",
        typed_integration_factory="create_e2b_sandbox_host_integration",
        legacy_catalog_factory="create_e2b_sandbox_host",
        register_function="register_e2b_integration",
        contract_spec_relpath="intergrax/integrations/providers/sandbox_host/e2b/contract_spec.py",
        integration_relpath="intergrax/integrations/providers/sandbox_host/e2b/integration.py",
        register_relpath="intergrax/integrations/providers/sandbox_host/e2b/register.py",
        bundle_relpath="intergrax/integrations/providers/sandbox_host/e2b/bundle.py",
    ),
    "modal": SandboxFactoryProviderSpec(
        provider_key="modal",
        integration_class="ModalSandboxHostIntegration",
        typed_integration_factory="create_modal_sandbox_host_integration",
        legacy_catalog_factory="create_modal_sandbox_host",
        register_function="register_modal_integration",
        contract_spec_relpath="intergrax/integrations/providers/sandbox_host/modal/contract_spec.py",
        integration_relpath="intergrax/integrations/providers/sandbox_host/modal/integration.py",
        register_relpath="intergrax/integrations/providers/sandbox_host/modal/register.py",
        bundle_relpath="intergrax/integrations/providers/sandbox_host/modal/bundle.py",
    ),
    "daytona": SandboxFactoryProviderSpec(
        provider_key="daytona",
        integration_class="DaytonaSandboxHostIntegration",
        typed_integration_factory="create_daytona_sandbox_host_integration",
        legacy_catalog_factory="create_daytona_sandbox_host",
        register_function="register_daytona_integration",
        contract_spec_relpath="intergrax/integrations/providers/sandbox_host/daytona/contract_spec.py",
        integration_relpath="intergrax/integrations/providers/sandbox_host/daytona/integration.py",
        register_relpath="intergrax/integrations/providers/sandbox_host/daytona/register.py",
        bundle_relpath="intergrax/integrations/providers/sandbox_host/daytona/bundle.py",
    ),
}


def sandbox_factory_provider_spec(provider_key: str) -> SandboxFactoryProviderSpec | None:
    return _SANDBOX_FACTORY_PROVIDER_SPECS.get(provider_key)


def _parse_module(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return None


def _import_map(tree: ast.Module) -> dict[str, str]:
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


def _local_qualified_names(tree: ast.Module, spec: SandboxFactoryProviderSpec) -> dict[str, str]:
    """Map module-local symbols to expected qualified names (narrow sandbox_host scope)."""
    mapping: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == spec.integration_class:
            mapping[node.name] = _integration_class_qname(spec)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if node.name == spec.typed_integration_factory:
                mapping[node.name] = _typed_factory_qname(spec)
            if node.name == spec.legacy_catalog_factory:
                mapping[node.name] = _legacy_factory_qname(spec)
    return mapping


def _resolve_name(
    name: str,
    imports: dict[str, str],
    locals_map: dict[str, str] | None = None,
) -> str | None:
    if locals_map and name in locals_map:
        return locals_map[name]
    if name not in imports:
        return None
    return imports[name]


def _resolve_expr(
    node: ast.expr,
    imports: dict[str, str],
    locals_map: dict[str, str] | None = None,
) -> str | None:
    if isinstance(node, ast.Name):
        return _resolve_name(node.id, imports, locals_map)
    if isinstance(node, ast.Attribute):
        base = _resolve_expr(node.value, imports, locals_map)
        if base is None:
            return None
        return f"{base}.{node.attr}"
    return None


def _top_level_assign(tree: ast.Module, name: str) -> ast.Assign | None:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node
    return None


def _kw_value(call: ast.Call, keyword: str) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == keyword:
            return kw.value
    return None


def _string_constant(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _integration_class_qname(spec: SandboxFactoryProviderSpec) -> str:
    return (
        f"intergrax.integrations.providers.sandbox_host.{spec.provider_key}.integration."
        f"{spec.integration_class}"
    )


def _typed_factory_qname(spec: SandboxFactoryProviderSpec) -> str:
    return (
        f"intergrax.integrations.providers.sandbox_host.{spec.provider_key}.bundle."
        f"{spec.typed_integration_factory}"
    )


def _legacy_factory_qname(spec: SandboxFactoryProviderSpec) -> str:
    return (
        f"intergrax.integrations.providers.sandbox_host.{spec.provider_key}.bundle."
        f"{spec.legacy_catalog_factory}"
    )


def _verify_contract_declaration(
    spec: SandboxFactoryProviderSpec,
    contract_tree: ast.Module,
    expected_typed_factory: str,
) -> tuple[bool, list[str]]:
    refs: list[str] = []
    imports = _import_map(contract_tree)
    assign = _top_level_assign(contract_tree, "CONTRACT_SPEC")
    if assign is None or not isinstance(assign.value, ast.Call):
        return False, refs
    call = assign.value
    callee = _resolve_expr(call.func, imports)
    if callee != _DECLARE_INTEGRATION_CONTRACT_Q:
        return False, refs
    refs.append(f"contract_spec:{spec.contract_spec_relpath}:CONTRACT_SPEC:declare_integration_contract")

    category = _kw_value(call, "category")
    if category is None or _string_constant(category) != "sandbox_host":
        return False, refs

    integration_class = _resolve_expr(
        _kw_value(call, "integration_class") or ast.Constant(value=None),
        imports,
    )
    if integration_class != _integration_class_qname(spec):
        return False, refs
    refs.append(f"ast:integration_class:{integration_class}")

    contract_class = _resolve_expr(_kw_value(call, "contract_class") or ast.Constant(value=None), imports)
    if contract_class != _SANDBOX_HOST_CONTRACT_Q:
        return False, refs
    refs.append(f"ast:contract_class:{contract_class}")

    factory_kw = _kw_value(call, "contract_factory")
    if factory_kw is None:
        return False, refs
    factory_ref = _resolve_expr(factory_kw, imports)
    expected_q = _typed_factory_qname(spec)
    if factory_ref != expected_q or expected_typed_factory != spec.typed_integration_factory:
        return False, refs
    refs.append(f"ast:contract_factory:{factory_ref}")

    specs_assign = _top_level_assign(contract_tree, "CONTRACT_SPECS")
    if specs_assign is None:
        return False, refs
    if not isinstance(specs_assign.value, ast.Tuple):
        return False, refs
    if not any(isinstance(elt, ast.Name) and elt.id == "CONTRACT_SPEC" for elt in specs_assign.value.elts):
        return False, refs
    refs.append(f"contract_specs_tuple:{spec.contract_spec_relpath}:CONTRACT_SPECS")
    return True, refs


def _verify_integration_class(spec: SandboxFactoryProviderSpec, integration_tree: ast.Module) -> tuple[bool, list[str]]:
    refs: list[str] = []
    imports = _import_map(integration_tree)
    locals_map = _local_qualified_names(integration_tree, spec)
    class_def: ast.ClassDef | None = None
    for node in integration_tree.body:
        if isinstance(node, ast.ClassDef) and node.name == spec.integration_class:
            class_def = node
            break
    if class_def is None:
        return False, refs
    if not class_def.bases:
        return False, refs
    base = _resolve_expr(class_def.bases[0], imports, locals_map)
    if base != _SANDBOX_HOST_CONTRACT_Q:
        return False, refs
    refs.append(f"integration_class:{spec.integration_relpath}:{spec.integration_class}")

    register_ok = False
    for node in integration_tree.body:
        if not isinstance(node, ast.Expr):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        call = node.value
        func = _resolve_expr(call.func, imports, locals_map)
        if func != f"{_SANDBOX_HOST_BACKEND_Q}.register":
            continue
        if not call.args:
            continue
        arg = _resolve_expr(call.args[0], imports, locals_map)
        if arg == _integration_class_qname(spec):
            register_ok = True
            refs.append(f"backend_register:{spec.integration_relpath}:SandboxHostBackend.register")
            break
    if not register_ok:
        return False, refs
    return True, refs


def _factory_returns_integration(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    spec: SandboxFactoryProviderSpec,
    imports: dict[str, str],
    locals_map: dict[str, str] | None = None,
) -> bool:
    integration_q = _integration_class_qname(spec)
    for node in ast.walk(func):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        value = node.value
        direct = _resolve_expr(value, imports, locals_map)
        if direct == integration_q:
            return True
        if isinstance(value, ast.Call):
            callee = _resolve_expr(value.func, imports, locals_map)
            if callee == integration_q:
                return True
            if callee and callee.startswith(f"{integration_q}."):
                return True
    return False


def _verify_typed_factory(
    spec: SandboxFactoryProviderSpec,
    bundle_tree: ast.Module,
    expected_symbol: str,
) -> tuple[bool, list[str]]:
    refs: list[str] = []
    imports = _import_map(bundle_tree)
    locals_map = _local_qualified_names(bundle_tree, spec)
    if expected_symbol != spec.typed_integration_factory:
        return False, refs
    func_def: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in bundle_tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == expected_symbol:
            func_def = node
            break
    if func_def is None:
        return False, refs
    if not _factory_returns_integration(func_def, spec, imports, locals_map):
        return False, refs
    refs.append(f"typed_factory:{spec.bundle_relpath}:{expected_symbol}")
    return True, refs


def _find_register_function(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _verify_registration(spec: SandboxFactoryProviderSpec, register_tree: ast.Module) -> tuple[bool, list[str]]:
    refs: list[str] = []
    imports = _import_map(register_tree)
    register_fn = _find_register_function(register_tree, spec.register_function)
    if register_fn is None:
        return False, refs

    manifest_q = (
        f"intergrax.integrations.providers.sandbox_host.{spec.provider_key}.manifest.MANIFEST"
    )
    contract_specs_q = (
        f"intergrax.integrations.providers.sandbox_host.{spec.provider_key}.contract_spec.CONTRACT_SPECS"
    )
    legacy_q = _legacy_factory_qname(spec)

    proven_call = False
    for node in ast.walk(register_fn):
        if not isinstance(node, ast.Call):
            continue
        callee = _resolve_expr(node.func, imports)
        if callee != _REGISTER_FROM_MANIFEST_Q:
            continue
        if len(node.args) < 2:
            continue
        manifest_arg = _resolve_expr(node.args[0], imports)
        legacy_arg = _resolve_expr(node.args[1], imports)
        contract_kw = _kw_value(node, "contract_specs")
        contract_arg = _resolve_expr(contract_kw, imports) if contract_kw is not None else None
        if manifest_arg != manifest_q:
            continue
        if legacy_arg != legacy_q:
            continue
        if contract_arg != contract_specs_q:
            continue
        proven_call = True
        refs.append(f"registration:{spec.register_relpath}:register_from_manifest")
        break

    if not proven_call:
        return False, refs
    return True, refs


@dataclass(frozen=True, slots=True)
class SandboxFactoryAstEvidence:
    proven: bool
    evidence_refs: tuple[str, ...]
    not_proven: tuple[str, ...]
    contract_declaration_owner: str
    integration_implementation_owner: str
    registration_composition_owner: str


def analyze_sandbox_factory_ast_evidence(
    repo: Path,
    provider_key: str,
    typed_factory_symbol: str,
    bundle_relpath: str,
) -> SandboxFactoryAstEvidence:
    """Structural factory graph proof for one sandbox_host provider (fail-closed)."""
    spec = sandbox_factory_provider_spec(provider_key)
    not_proven: list[str] = []
    if spec is None:
        return SandboxFactoryAstEvidence(
            proven=False,
            evidence_refs=(),
            not_proven=("provider_key:NOT_PROVEN",),
            contract_declaration_owner="",
            integration_implementation_owner="",
            registration_composition_owner="",
        )
    if bundle_relpath != spec.bundle_relpath:
        not_proven.append("bundle_path:NOT_PROVEN")
        return SandboxFactoryAstEvidence(
            proven=False,
            evidence_refs=(),
            not_proven=tuple(not_proven),
            contract_declaration_owner=spec.contract_spec_relpath,
            integration_implementation_owner=spec.integration_relpath,
            registration_composition_owner=spec.register_relpath,
        )

    paths = (
        spec.contract_spec_relpath,
        spec.integration_relpath,
        spec.register_relpath,
        spec.bundle_relpath,
    )
    for rel in paths:
        if not (repo / rel).is_file():
            not_proven.append(f"missing:{rel}")
    if not_proven:
        return SandboxFactoryAstEvidence(
            proven=False,
            evidence_refs=(),
            not_proven=tuple(not_proven),
            contract_declaration_owner=spec.contract_spec_relpath,
            integration_implementation_owner=spec.integration_relpath,
            registration_composition_owner=spec.register_relpath,
        )

    contract_tree = _parse_module(repo / spec.contract_spec_relpath)
    integration_tree = _parse_module(repo / spec.integration_relpath)
    register_tree = _parse_module(repo / spec.register_relpath)
    bundle_tree = _parse_module(repo / spec.bundle_relpath)
    if (
        contract_tree is None
        or integration_tree is None
        or register_tree is None
        or bundle_tree is None
    ):
        return SandboxFactoryAstEvidence(
            proven=False,
            evidence_refs=(),
            not_proven=("parse:NOT_PROVEN",),
            contract_declaration_owner=spec.contract_spec_relpath,
            integration_implementation_owner=spec.integration_relpath,
            registration_composition_owner=spec.register_relpath,
        )

    refs: list[str] = []
    ok_a, part = _verify_contract_declaration(spec, contract_tree, typed_factory_symbol)
    refs.extend(part)
    if not ok_a:
        not_proven.append("contract_declaration:NOT_PROVEN")

    ok_b, part = _verify_integration_class(spec, integration_tree)
    refs.extend(part)
    if not ok_b:
        not_proven.append("integration_class:NOT_PROVEN")

    ok_c, part = _verify_typed_factory(spec, bundle_tree, typed_factory_symbol)
    refs.extend(part)
    if not ok_c:
        not_proven.append("typed_factory:NOT_PROVEN")

    ok_d, part = _verify_registration(spec, register_tree)
    refs.extend(part)
    if not ok_d:
        not_proven.append("registration:NOT_PROVEN")

    proven = not not_proven
    return SandboxFactoryAstEvidence(
        proven=proven,
        evidence_refs=tuple(refs),
        not_proven=tuple(not_proven),
        contract_declaration_owner=spec.contract_spec_relpath,
        integration_implementation_owner=spec.integration_relpath,
        registration_composition_owner=spec.register_relpath,
    )
