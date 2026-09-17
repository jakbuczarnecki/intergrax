# © Artur Czarnecki. All rights reserved.

"""HOST-01 qualification gates (HOST-Q1..HOST-Q12)."""

from __future__ import annotations

import ast
import importlib
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import Request

from intergrax.applications._shared.harness_task_routes import HarnessAsyncRunRequest
from intergrax.applications._shared.mcp_nexus_server import execute_mcp_agent_task
from intergrax.contracts.execution_identity import mint_task_id
from intergrax.fastapi_core.errors.handlers import global_exception_handler
from intergrax.fastapi_core.errors.mapping import map_exception_to_api_error
from intergrax.runtime.execution.facade import Execution as ExecutionFacade
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.interactions.task_executor import HostTaskExecutionExecutor
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskResult, TaskState
from intergrax.runtime.task.task_result_authoritative_exposure_defaults import (
    terminal_task_result_exposure_no_decision_gate,
)
from governed_contractor_application.host.environment_profile import (
    build_governed_contractor_environment_profile,
)
from governed_contractor_application.host.execution_wiring import (
    build_governed_contractor_host_task_execution,
)
from governed_contractor_application.host.settings import GovernedContractorBackendSettings

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SHARED_HOST = _REPO_ROOT / "intergrax" / "applications" / "_shared"
_HOST_TASK = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "host_task.py"
_CONTRACTS_EXECUTION_REQUEST = _REPO_ROOT / "intergrax" / "contracts" / "execution_request.py"

_HOST_ADAPTER_FILES = (
    _SHARED_HOST / "mcp_nexus_server.py",
    _SHARED_HOST / "fastapi_mcp.py",
    _REPO_ROOT / "intergrax" / "runtime" / "task" / "host_task_execution_run_adapter.py",
    _REPO_ROOT / "intergrax" / "runtime" / "interactions" / "task_executor.py",
    _SHARED_HOST / "harness_task_routes.py",
)

_FORBIDDEN_HOST_ADAPTER_TOKENS = ("getattr(", "setattr(", "hasattr(", "GLOBAL_REGISTRY", "service_locator")

_FORBIDDEN_HOST_ADAPTER_IMPORT_PREFIXES = (
    "openai",
    "anthropic",
    "boto3",
    "sqlalchemy",
    "intergrax.integrations.providers",
    "intergrax.agents.uaep",
)

_TRANSPORT_IMPORT_MARKERS = ("fastapi", "fastmcp", "starlette", "uvicorn", "mcp.")


@dataclass(frozen=True, slots=True)
class CanonicalHostTaskIntent:
    tenant_id: str
    user_id: str
    message: str
    capability: str
    session_id: str | None
    intent: str | None


def _task_from_harness_async(body: HarnessAsyncRunRequest) -> CanonicalHostTaskIntent:
    return CanonicalHostTaskIntent(
        tenant_id=body.tenant_id,
        user_id=body.user_id,
        message=body.message,
        capability=body.capability,
        session_id=None,
        intent=None,
    )


def _task_from_mcp_intake(
    *,
    message: str,
    capability: str,
    tenant_id: str,
    user_id: str,
    session_id: str | None = None,
    intent: str | None = None,
) -> CanonicalHostTaskIntent:
    return CanonicalHostTaskIntent(
        tenant_id=tenant_id,
        user_id=user_id,
        message=message,
        capability=capability,
        session_id=session_id,
        intent=intent,
    )


def _build_mcp_task(intent: CanonicalHostTaskIntent) -> Task:
    context = (
        TaskContext(capability=intent.capability, intent=intent.intent)
        if intent.intent
        else TaskContext(capability=intent.capability)
    )
    return Task(
        tenant_id=intent.tenant_id,
        user_id=intent.user_id,
        session_id=intent.session_id,
        message=intent.message,
        context=context,
    )


def _build_harness_task(intent: CanonicalHostTaskIntent, metadata: dict[str, object] | None = None) -> Task:
    return Task(
        task_id=mint_task_id(),
        tenant_id=intent.tenant_id,
        user_id=intent.user_id,
        message=intent.message,
        context=TaskContext(capability=intent.capability),
        metadata=dict(metadata or {}),
    )


def _governed_host_execution(nexus_loop: NexusLoop):
    env = build_governed_contractor_environment_profile(GovernedContractorBackendSettings.from_env())
    return build_governed_contractor_host_task_execution(nexus_loop, env)


def _completed_task_result() -> TaskResult:
    from intergrax.contracts.execution_identity import mint_run_id

    return TaskResult(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        state=TaskState.COMPLETED,
        answer="ok",
        authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
    )


def test_host_q1_production_surfaces_use_host_task_execution_port() -> None:
    mcp_source = (_SHARED_HOST / "mcp_nexus_server.py").read_text(encoding="utf-8")
    run_adapter = (
        _REPO_ROOT / "intergrax" / "runtime" / "task" / "host_task_execution_run_adapter.py"
    ).read_text(encoding="utf-8")
    task_control = (_SHARED_HOST / "task_control_wiring.py").read_text(encoding="utf-8")
    assert "HostTaskExecutionPort" in mcp_source
    assert "host_execution.execute" in mcp_source
    assert "HostTaskExecutionPort" in run_adapter
    assert "HostTaskExecutionExecutor" in run_adapter
    assert "host_execution: HostTaskExecutionPort" in task_control


def test_host_q2_host_task_roots_on_root_execution_launcher() -> None:
    source = _HOST_TASK.read_text(encoding="utf-8")
    assert "DefaultRootExecutionLauncher" in source
    assert "root_authority_admission" in source
    assert "launcher.launch" in source


def test_host_q3_host_task_resolves_root_execution_context() -> None:
    source = _HOST_TASK.read_text(encoding="utf-8")
    assert "resolve_root_execution_context" in source
    assert "RootExecutionOptions" in source


def test_host_q4_host_task_uses_execution_runtime() -> None:
    source = _HOST_TASK.read_text(encoding="utf-8")
    assert "ExecutionRuntime" in source
    assert "class Execution" in source or "Execution(" in source
    assert "HostFacadeRootExecutionIntake" in source


def test_host_q5_fastapi_maps_internal_errors_without_secret_leak() -> None:
    secret = "HOST01-SECRET-TOKEN-xyz"
    error_type, status_code, message = map_exception_to_api_error(
        RuntimeError(f"boom {secret} traceback")
    )
    assert error_type.value == "internal_error"
    assert status_code == 500
    assert secret not in message
    assert "traceback" not in message.lower()


@pytest.mark.asyncio
async def test_host_q11_global_handler_hides_stack_and_secrets() -> None:
    secret = "HOST11-LEAK-SECRET"
    request = Request({"type": "http", "headers": [], "method": "GET", "path": "/"})
    response = await global_exception_handler(request, ValueError(f"fail {secret}"))
    body = response.body.decode("utf-8")
    assert secret not in body
    assert "Traceback" not in body
    assert "traceback" not in body.lower()


def test_host_q6_canonical_execution_request_has_no_transport_imports() -> None:
    text = _CONTRACTS_EXECUTION_REQUEST.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                assert not any(marker in mod for marker in _TRANSPORT_IMPORT_MARKERS), mod
        if isinstance(node, ast.ImportFrom) and node.module:
            mod = node.module
            assert not any(marker in mod for marker in _TRANSPORT_IMPORT_MARKERS), mod


def test_host_q8_host_adapter_modules_static_gate() -> None:
    violations: list[str] = []
    for path in _HOST_ADAPTER_FILES:
        source = path.read_text(encoding="utf-8")
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for token in _FORBIDDEN_HOST_ADAPTER_TOKENS:
            if token in source:
                violations.append(f"{rel}: forbidden token {token!r}")
    assert violations == [], "\n".join(violations)


def test_host_q12_host_adapter_import_layer_gate() -> None:
    violations: list[str] = []
    for path in _HOST_ADAPTER_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            mod = node.module
            for prefix in _FORBIDDEN_HOST_ADAPTER_IMPORT_PREFIXES:
                if mod == prefix or mod.startswith(prefix + "."):
                    violations.append(f"{rel}: forbidden import {mod!r}")
    assert violations == [], "\n".join(violations)


class _PluginHostAdapter:
    """Lightweight external host adapter — delegates only through HostTaskExecutionPort."""

    def __init__(self, inner: HostTaskExecutionPort) -> None:
        self._inner = inner

    async def execute(self, task: Task, **kwargs: object) -> TaskResult:
        return await self._inner.execute(task, **kwargs)


@pytest.mark.asyncio
async def test_host_q7_custom_adapter_delegates_without_core_change() -> None:
    inner = AsyncMock(spec=HostTaskExecutionPort)
    inner.execute = AsyncMock(return_value=_completed_task_result())
    adapter = _PluginHostAdapter(inner)
    task = Task(tenant_id="t", user_id="u", message="hi", context=TaskContext(capability="cap"))
    result = await adapter.execute(task)
    assert result.state is TaskState.COMPLETED
    inner.execute.assert_awaited_once()


def test_host_q10_mcp_and_http_harness_map_equivalent_task_semantics() -> None:
    harness_body = HarnessAsyncRunRequest(
        tenant_id="tenant-a",
        user_id="user-b",
        message="hello",
        capability="demo.cap",
        metadata={"source": "http"},
    )
    harness_intent = _task_from_harness_async(harness_body)
    mcp_intent = _task_from_mcp_intake(
        message="hello",
        capability="demo.cap",
        tenant_id="tenant-a",
        user_id="user-b",
    )
    assert harness_intent == mcp_intent
    harness_task = _build_harness_task(harness_intent, metadata={"source": "http"})
    mcp_task = _build_mcp_task(mcp_intent)
    assert harness_task.tenant_id == mcp_task.tenant_id
    assert harness_task.user_id == mcp_task.user_id
    assert harness_task.message == mcp_task.message
    assert harness_task.context.capability == mcp_task.context.capability


@pytest.mark.asyncio
async def test_host_q9_single_facade_invoke_per_adapter_path() -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    host_execution = _governed_host_execution(nexus_loop)
    facade_calls = 0
    original_execute = ExecutionFacade.execute

    async def _spy_execute(self, request, *, options, held_root_capacity_permit=None):
        nonlocal facade_calls
        facade_calls += 1
        return await original_execute(
            self,
            request,
            options=options,
            held_root_capacity_permit=held_root_capacity_permit,
        )

    with patch.object(ExecutionFacade, "execute", _spy_execute):
        with patch(
            "intergrax.runtime.execution.host_task.TaskBoundAgenticDelegate.execute",
            new_callable=AsyncMock,
            return_value=_completed_task_result(),
        ):
            await execute_mcp_agent_task(
                host_execution,
                message="once",
                capability="external_contractor.adapt",
                tenant_id="t",
                user_id="u",
            )
            assert facade_calls == 1

            facade_calls = 0
            executor = HostTaskExecutionExecutor(host_execution)
            task = Task(
                tenant_id="t",
                user_id="u",
                message="once",
                context=TaskContext(capability="external_contractor.adapt"),
            )
            await executor.execute(task)
            assert facade_calls == 1


def test_host_q_catalog_contract_module_importable() -> None:
    catalog = importlib.import_module("tests.qualification.host_01.catalog")
    assert len(catalog.HOST_01_Q_CATALOG) == 12
