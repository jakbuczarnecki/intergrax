# © Artur Czarnecki. All rights reserved.

"""ADR3-IMP-04 — consumer migration to execution-bound declarative contract (gates)."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from intergrax.agents.persistence.declarative_tool_executor import (
    CallableDeclarativeToolInvoker,
    execute_declarative_actions,
)
from intergrax.contracts.declarative_tool_invoke_result import DeclarativeToolInvokeResult
from intergrax.contracts.execution_bound_declarative_tool_invocation import (
    ExecutionBoundDeclarativeToolInvoker,
)
from intergrax.knowledge.contracts.validation import JsonObject
from intergrax.runtime.nexus.agents.catalog_declarative_invoker import (
    CatalogDeclarativeToolInvoker,
    CatalogHostDeclarativeToolInvoker,
)
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.tools.registry import ToolRegistry
from testing_support.catalog_declarative_invoker import (
    build_catalog_declarative_invoker_from_registry,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_EXECUTOR = _REPO / "intergrax" / "agents" / "persistence" / "declarative_tool_executor.py"
_UAEP_SHIM = _REPO / "intergrax" / "runtime" / "nexus" / "agents" / "acp_uaep_shim.py"
_CATALOG_DECLARATIVE = (
    _REPO / "intergrax" / "runtime" / "nexus" / "agents" / "catalog_declarative_invoker.py"
)
_PRODUCTION_ROOTS = (
    _REPO / "intergrax" / "runtime",
    _REPO / "intergrax" / "agents",
    _REPO / "intergrax" / "applications",
)


def _production_py_files() -> list[Path]:
    paths: list[Path] = []
    for root in _PRODUCTION_ROOTS:
        paths.extend(root.rglob("*.py"))
    return paths


def test_adr3_imp_04_executor_has_no_signature_reflection() -> None:
    source = _EXECUTOR.read_text(encoding="utf-8")
    assert "inspect" not in source
    assert "_declarative_invoker_requires_per_call_identity" not in source


def test_adr3_imp_04_narrow_declarative_protocol_removed_from_executor() -> None:
    tree = ast.parse(_EXECUTOR.read_text(encoding="utf-8"))
    class_names = {
        node.name for node in tree.body if isinstance(node, ast.ClassDef)
    }
    assert "DeclarativeToolInvoker" not in class_names


def test_adr3_imp_04_no_production_import_of_narrow_declarative_protocol() -> None:
    forbidden = "DeclarativeToolInvoker"
    allowlist = {
        _REPO / "intergrax" / "agents" / "persistence" / "declarative_run_binding.py",
    }
    for path in _production_py_files():
        if path in allowlist:
            continue
        text = path.read_text(encoding="utf-8")
        stripped = text.replace("ExecutionBoundDeclarativeToolInvoker", "")
        stripped = stripped.replace("DeclarativeToolInvokerWithRunBinding", "")
        stripped = stripped.replace("CallableDeclarativeToolInvoker", "")
        stripped = stripped.replace("CatalogDeclarativeToolInvoker", "")
        stripped = stripped.replace("CatalogHostDeclarativeToolInvoker", "")
        assert forbidden not in stripped, f"narrow protocol import in {path.relative_to(_REPO)}"


def test_adr3_imp_04_uaep_shim_does_not_couple_to_catalog_concrete() -> None:
    source = _UAEP_SHIM.read_text(encoding="utf-8")
    assert "CatalogDeclarativeToolInvoker" not in source
    assert "CatalogHostDeclarativeToolInvoker" in source


def test_adr3_imp_04_r1_catalog_host_inherits_execution_bound_contract() -> None:
    tree = ast.parse(_CATALOG_DECLARATIVE.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name != "CatalogHostDeclarativeToolInvoker":
            continue
        base_names: list[str] = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_names.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_names.append(base.attr)
        assert "ExecutionBoundDeclarativeToolInvoker" in base_names
        method_names = {
            child.name
            for child in node.body
            if isinstance(child, (ast.AsyncFunctionDef, ast.FunctionDef))
        }
        assert "invoke" not in method_names
        return
    raise AssertionError("CatalogHostDeclarativeToolInvoker not found")


def test_adr3_imp_04_r1_execution_bound_is_canonical_invoke_owner() -> None:
    canonical = _REPO / "intergrax" / "contracts" / "execution_bound_declarative_tool_invocation.py"
    tree = ast.parse(canonical.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ExecutionBoundDeclarativeToolInvoker":
            assert any(
                isinstance(child, ast.AsyncFunctionDef) and child.name == "invoke"
                for child in node.body
            )
            return
    raise AssertionError("ExecutionBoundDeclarativeToolInvoker.invoke not found")


@dataclass
class _StructuralCatalogHostInvoker:
    """Host-capable L2 invoker without catalog concrete subclass."""

    tool_invoker: RuntimeToolInvoker
    calls: list[str] = field(default_factory=list)

    async def invoke(
        self,
        *,
        tenant_id: str,
        run_id: str,
        task_id: str,
        agent_id: str,
        tool_id: str,
        args: JsonObject,
        idempotency_key: str | None,
    ) -> DeclarativeToolInvokeResult:
        self.calls.append(tool_id)
        _ = tenant_id, run_id, task_id, agent_id, args, idempotency_key
        return DeclarativeToolInvokeResult(status="success", output={"ok": True})


def test_adr3_imp_04_r1_structural_host_capability_without_catalog_subclass() -> None:
    registry = ToolRegistry()
    catalog = build_catalog_declarative_invoker_from_registry(registry)
    custom = _StructuralCatalogHostInvoker(tool_invoker=catalog.tool_invoker)
    assert isinstance(custom, ExecutionBoundDeclarativeToolInvoker)
    assert isinstance(custom, CatalogHostDeclarativeToolInvoker)
    assert not isinstance(custom, CatalogDeclarativeToolInvoker)
    assert isinstance(catalog, CatalogHostDeclarativeToolInvoker)


@dataclass
class _ExternalExecutionBoundInvoker:
    """Structural L2 invoker (no catalog concrete subclass)."""

    calls: list[tuple[str, str]] = field(default_factory=list)

    async def invoke(
        self,
        *,
        tenant_id: str,
        run_id: str,
        task_id: str,
        agent_id: str,
        tool_id: str,
        args: JsonObject,
        idempotency_key: str | None,
    ) -> DeclarativeToolInvokeResult:
        self.calls.append((run_id, tool_id))
        _ = tenant_id, task_id, agent_id, args, idempotency_key
        return DeclarativeToolInvokeResult(status="success", output={"ok": True})


@pytest.mark.asyncio
async def test_adr3_imp_04_external_invoker_reaches_executor_without_subclass() -> None:
    invoker = _ExternalExecutionBoundInvoker()
    assert isinstance(invoker, ExecutionBoundDeclarativeToolInvoker)
    result = await execute_declarative_actions(
        actions=[{"tool_id": "noop", "args": {}}],
        ledger=None,
        invoker=invoker,
        tenant_id="tenant-1",
        run_id="run-1",
        task_id="task-1",
        agent_id="agent-1",
    )
    assert result.results[0].status == "success"
    assert invoker.calls == [("run-1", "noop")]


@pytest.mark.asyncio
async def test_adr3_imp_04_callable_test_adapter_is_execution_bound() -> None:
    observed: list[str] = []

    async def _invoke(**kwargs: object) -> DeclarativeToolInvokeResult:
        observed.append(str(kwargs.get("run_id")))
        return DeclarativeToolInvokeResult(status="success")

    invoker = CallableDeclarativeToolInvoker(_invoke)
    await execute_declarative_actions(
        actions=[{"tool_id": "t", "args": {}}],
        ledger=None,
        invoker=invoker,
        run_id="run-x",
        task_id="task-x",
        agent_id="agent-x",
    )
    assert observed == ["run-x"]
