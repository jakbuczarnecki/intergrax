# © Artur Czarnecki. All rights reserved.

"""ADR3-IMP-05 — enterprise qualification gates (regression protection for IMP-03/04)."""

from __future__ import annotations

import ast
import re
from pathlib import Path
import pytest

from intergrax.contracts.acp_metadata_keys import AcpMetadataKey
from intergrax.agents.persistence.tool_invoker_wiring import (
    resolve_declarative_tool_invoker_from_metadata,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]

_PRODUCTION_ROOTS = (
    _REPO / "intergrax" / "runtime",
    _REPO / "intergrax" / "agents",
    _REPO / "intergrax" / "applications",
)

_DECLARATIVE_FLOW_PATHS = (
    _REPO / "intergrax" / "agents" / "persistence" / "declarative_tool_executor.py",
    _REPO / "intergrax" / "runtime" / "nexus" / "agents" / "catalog_declarative_invoker.py",
    _REPO / "intergrax" / "runtime" / "nexus" / "agents" / "acp_uaep_shim.py",
)

_TOOL_INVOKER_WIRING = _REPO / "intergrax" / "agents" / "persistence" / "tool_invoker_wiring.py"
_RUN_BINDING = _REPO / "intergrax" / "agents" / "persistence" / "declarative_run_binding.py"
_L2_DECLARATIVE = _REPO / "intergrax" / "contracts" / "execution_bound_declarative_tool_invocation.py"
_L2_CATALOG = _REPO / "intergrax" / "contracts" / "execution_bound_catalog_tool_invocation.py"
_METADATA_KEYS = _REPO / "intergrax" / "contracts" / "acp_metadata_keys.py"

_CATALOG_CONCRETE_CONSTRUCTION_ALLOWLIST = frozenset(
    {
        _REPO / "intergrax" / "applications" / "_shared" / "declarative_tool_wiring.py",
    }
)

_CATALOG_GATEWAY_IMPORT_ALLOWLIST = frozenset(
    {
        _REPO / "intergrax" / "runtime" / "nexus" / "agents" / "catalog_declarative_invoker.py",
        _REPO / "intergrax" / "runtime" / "nexus" / "tools" / "catalog_dispatch.py",
        _REPO / "intergrax" / "runtime" / "nexus" / "tools" / "tool_gateway.py",
    }
)

_DIRECT_CONSUMER_MODULES = (
    _REPO / "intergrax" / "runtime" / "kernel" / "step_kernel.py",
    _REPO / "intergrax" / "runtime" / "nexus" / "nexus_loop.py",
    _REPO / "intergrax" / "runtime" / "nexus" / "execution" / "graph_executor.py",
    _REPO / "intergrax" / "runtime" / "execution" / "host_task.py",
    _REPO / "intergrax" / "applications" / "_shared" / "nexus_factory.py",
    _REPO / "intergrax" / "agents" / "authoring" / "acp_run.py",
    _REPO / "intergrax" / "agents" / "persistence" / "compensation_enqueue.py",
    _REPO / "applications" / "local_workspace_application" / "host" / "lkw_task_enricher.py",
)

ADR3_QUALIFICATION_TEST_PATHS: tuple[str, ...] = (
    "tests/unit/runtime/architecture/test_adr3_imp_01_catalog_l2_per_call_identity.py",
    "tests/unit/runtime/architecture/test_adr3_imp_02_declarative_l2_per_call_identity.py",
    "tests/unit/runtime/architecture/test_adr3_imp_03_l3_adapter_reconciliation.py",
    "tests/unit/runtime/architecture/test_adr3_imp_04_consumer_migration.py",
    "tests/unit/runtime/architecture/test_adr3_imp_05_qualification_gates.py",
    "tests/unit/runtime/architecture/test_ee_final_arch_zero_execution_bypass.py",
    "tests/unit/agents/persistence/test_catalog_declarative_invoker.py",
    "tests/unit/agents/persistence/test_tool_invoker_wiring.py",
    "tests/unit/applications/shared/test_declarative_tool_wiring.py",
)


def _production_py_files() -> list[Path]:
    paths: list[Path] = []
    for root in _PRODUCTION_ROOTS:
        paths.extend(root.rglob("*.py"))
    return paths


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _class_bases_include_protocol(node: ast.ClassDef) -> bool:
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id == "Protocol":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "Protocol":
            return True
    return False


def _invoke_identity_param_names(node: ast.AsyncFunctionDef) -> set[str]:
    return {arg.arg for arg in node.args.kwonlyargs}


def test_adr3_imp_05_qualification_suite_paths_exist() -> None:
    missing = [rel for rel in ADR3_QUALIFICATION_TEST_PATHS if not (_REPO / rel).is_file()]
    assert missing == [], f"missing qualification modules: {missing}"


def test_adr3_imp_05_no_narrow_declarative_tool_invoker_protocol_class() -> None:
    pattern = re.compile(r"class\s+DeclarativeToolInvoker\s*\(\s*Protocol")
    for path in _production_py_files():
        text = path.read_text(encoding="utf-8")
        stripped = text.replace("DeclarativeToolInvokerWithRunBinding", "")
        stripped = stripped.replace("ExecutionBoundDeclarativeToolInvoker", "")
        stripped = stripped.replace("CatalogDeclarativeToolInvoker", "")
        stripped = stripped.replace("CatalogHostDeclarativeToolInvoker", "")
        stripped = stripped.replace("CallableDeclarativeToolInvoker", "")
        assert not pattern.search(stripped), path.relative_to(_REPO)


def _returns_declarative_tool_invoke_result(node: ast.AsyncFunctionDef) -> bool:
    if node.returns is None:
        return False
    ret = ast.unparse(node.returns)
    return "DeclarativeToolInvokeResult" in ret


def test_adr3_imp_05_no_duplicate_declarative_invoke_protocol_owner_in_contracts() -> None:
    contracts_root = _REPO / "intergrax" / "contracts"
    owners: list[str] = []
    for path in contracts_root.rglob("*.py"):
        tree = _parse(path)
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if not _class_bases_include_protocol(node):
                continue
            for child in node.body:
                if not isinstance(child, ast.AsyncFunctionDef) or child.name != "invoke":
                    continue
                if not _returns_declarative_tool_invoke_result(child):
                    continue
                params = _invoke_identity_param_names(child)
                if {"tenant_id", "run_id", "task_id", "agent_id", "tool_id"} <= params:
                    owners.append(f"{path.relative_to(_REPO).as_posix()}:{node.name}")
    assert owners == [
        "intergrax/contracts/execution_bound_declarative_tool_invocation.py:ExecutionBoundDeclarativeToolInvoker",
    ]


def test_adr3_imp_05_declarative_run_binding_extends_canonical_without_invoke() -> None:
    tree = _parse(_RUN_BINDING)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "DeclarativeToolInvokerWithRunBinding":
            continue
        base_names = [
            base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
            for base in node.bases
        ]
        assert "ExecutionBoundDeclarativeToolInvoker" in base_names
        method_names = {
            child.name
            for child in node.body
            if isinstance(child, (ast.AsyncFunctionDef, ast.FunctionDef))
        }
        assert "invoke" not in method_names
        assert "bind_run" in method_names
        return
    raise AssertionError("DeclarativeToolInvokerWithRunBinding not found")


def test_adr3_imp_05_metadata_resolver_return_type_is_execution_bound() -> None:
    tree = _parse(_TOOL_INVOKER_WIRING)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "resolve_declarative_tool_invoker_from_metadata":
            assert node.returns is not None
            ret = ast.unparse(node.returns)
            assert "ExecutionBoundDeclarativeToolInvoker" in ret
            assert "CatalogDeclarativeToolInvoker" not in ret
            assert "Any" not in ret
            assert ret.strip() != "object"
            return
    raise AssertionError("resolve_declarative_tool_invoker_from_metadata not found")


def test_adr3_imp_05_single_metadata_resolver_definition() -> None:
    matches: list[str] = []
    for path in _production_py_files():
        tree = _parse(path)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "resolve_declarative_tool_invoker_from_metadata":
                matches.append(path.relative_to(_REPO).as_posix())
    assert matches == ["intergrax/agents/persistence/tool_invoker_wiring.py"]


def test_adr3_imp_05_metadata_key_not_duplicated_as_raw_literal() -> None:
    literal = AcpMetadataKey.DECLARATIVE_TOOL_INVOKER
    for path in _production_py_files():
        if path == _METADATA_KEYS:
            continue
        text = path.read_text(encoding="utf-8")
        assert literal not in text, f"raw metadata key literal in {path.relative_to(_REPO)}"


def test_adr3_imp_05_metadata_resolver_fail_closed_on_invalid_candidate() -> None:
    with pytest.raises(TypeError, match="ExecutionBoundDeclarativeToolInvoker"):
        resolve_declarative_tool_invoker_from_metadata(
            {AcpMetadataKey.DECLARATIVE_TOOL_INVOKER: object()},
        )


def test_adr3_imp_05_direct_consumers_avoid_catalog_concrete_import() -> None:
    for path in _DIRECT_CONSUMER_MODULES:
        assert path.is_file(), path
        text = path.read_text(encoding="utf-8")
        assert "CatalogDeclarativeToolInvoker" not in text, path.relative_to(_REPO)


def test_adr3_imp_05_catalog_concrete_construction_sanctioned_only() -> None:
    token = "CatalogDeclarativeToolInvoker("
    for path in _production_py_files():
        if token not in path.read_text(encoding="utf-8"):
            continue
        assert path in _CATALOG_CONCRETE_CONSTRUCTION_ALLOWLIST, path.relative_to(_REPO)


def test_adr3_imp_05_invoke_catalog_gateway_import_sanctioned_only() -> None:
    needle = "invoke_catalog_tool_request"
    for path in _production_py_files():
        text = path.read_text(encoding="utf-8")
        if needle not in text:
            continue
        assert path in _CATALOG_GATEWAY_IMPORT_ALLOWLIST, path.relative_to(_REPO)


def test_adr3_imp_05_declarative_flow_avoids_reflection_dispatch_helpers() -> None:
    forbidden = (
        "inspect.signature",
        "_declarative_invoker_requires_per_call_identity",
    )
    for path in _DECLARATIVE_FLOW_PATHS:
        source = path.read_text(encoding="utf-8")
        for fragment in forbidden:
            assert fragment not in source, f"{fragment} in {path.relative_to(_REPO)}"


def test_adr3_imp_05_declarative_flow_avoids_dynamic_capability_probing() -> None:
    forbidden = ("getattr(", "hasattr(", "setattr(")
    for path in _DECLARATIVE_FLOW_PATHS:
        source = path.read_text(encoding="utf-8")
        for fragment in forbidden:
            assert fragment not in source, f"{fragment} in {path.relative_to(_REPO)}"


def test_adr3_imp_05_executor_invoke_call_passes_explicit_identity() -> None:
    executor = _DECLARATIVE_FLOW_PATHS[0]
    tree = _parse(executor)
    required = frozenset(
        {
            "tenant_id",
            "run_id",
            "task_id",
            "agent_id",
            "tool_id",
            "args",
            "idempotency_key",
        }
    )
    for node in ast.walk(tree):
        if not isinstance(node, ast.Await):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        if not isinstance(func, ast.Attribute) or func.attr != "invoke":
            continue
        if not isinstance(func.value, ast.Name) or func.value.id != "invoker":
            continue
        keywords = {kw.arg for kw in call.keywords if kw.arg is not None}
        assert required <= keywords
        return
    raise AssertionError("invoker.invoke with explicit identity not found in executor")


def test_adr3_imp_05_uca_catalog_and_declarative_contracts_remain_separate() -> None:
    declarative_source = _L2_DECLARATIVE.read_text(encoding="utf-8")
    catalog_source = _L2_CATALOG.read_text(encoding="utf-8")
    adapter_source = _DECLARATIVE_FLOW_PATHS[1].read_text(encoding="utf-8")
    assert "ExecutionBoundCatalogToolInvoker" not in declarative_source
    assert "ExecutionBoundDeclarativeToolInvoker" not in catalog_source
    assert "ExecutionBoundCatalogToolInvoker" not in adapter_source


def test_adr3_imp_05_l2_declarative_contract_has_no_reverse_implementation_imports() -> None:
    tree = _parse(_L2_DECLARATIVE)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            module = node.module
            assert "runtime.nexus" not in module
            assert "agents.persistence" not in module
            assert "applications" not in module


def test_adr3_imp_05_direct_consumers_use_execution_bound_type_not_runtime_tool_invoker() -> None:
    for path in _DIRECT_CONSUMER_MODULES:
        text = path.read_text(encoding="utf-8")
        assert "RuntimeToolInvoker" not in text, path.relative_to(_REPO)


def test_adr3_imp_05_governance_fail_closed_test_present_in_qualification_suite() -> None:
    wiring_test = _REPO / "tests/unit/applications/shared/test_declarative_tool_wiring.py"
    source = wiring_test.read_text(encoding="utf-8")
    assert "test_build_declarative_invoker_fail_closed_without_governance_in_production_mode" in source
