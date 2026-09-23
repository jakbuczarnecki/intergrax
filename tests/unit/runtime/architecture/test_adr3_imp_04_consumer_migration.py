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

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_EXECUTOR = _REPO / "intergrax" / "agents" / "persistence" / "declarative_tool_executor.py"
_UAEP_SHIM = _REPO / "intergrax" / "runtime" / "nexus" / "agents" / "acp_uaep_shim.py"
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
