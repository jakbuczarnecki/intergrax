# © Artur Czarnecki. All rights reserved.

"""HOST-01 qualification gates (HOST-Q1..HOST-Q12)."""

from __future__ import annotations

import ast
import importlib
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request

from intergrax.applications._shared.async_task_dispatch import (
    InMemoryAsyncTaskIndex,
    run_async_task_executor,
)
from intergrax.applications._shared.harness_task_routes import (
    HarnessAsyncRunRequest,
    task_from_harness_async_run_request,
)
from intergrax.applications._shared.mcp_nexus_server import (
    execute_mcp_agent_task,
    task_from_mcp_agent_intake,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.fastapi_core.errors.handlers import global_exception_handler
from intergrax.fastapi_core.errors.mapping import map_exception_to_api_error
from intergrax.fastapi_core.execution.models import ExecutionRequest as FastApiExecutionRequest
from intergrax.runtime.background_execution.bootstrap import BackgroundExecutionIdentity
from intergrax.runtime.execution.facade import Execution as ExecutionFacade
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.governance.default_root_execution_launcher import DefaultRootExecutionLauncher
from intergrax.runtime.interactions.task_executor import HostTaskExecutionExecutor
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.host_task_execution_run_adapter import HostTaskExecutionRunAdapter
from intergrax.runtime.task.nexus_worker_execution import NexusWorkerRuntime
from intergrax.runtime.task.task import Task, TaskContext, TaskResult, TaskState
from intergrax.runtime.task.task_result_authoritative_exposure_defaults import (
    terminal_task_result_exposure_no_decision_gate,
)
from intergrax.runtime.task.task_run_bridge import task_to_execution_payload
from intergrax.runtime.task.worker_payload import encode_execution_request
from governed_contractor_application.host.environment_profile import (
    build_governed_contractor_environment_profile,
)
from governed_contractor_application.host.execution_wiring import (
    build_governed_contractor_host_task_execution,
)
from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from tests.qualification.host_01.threaded_adapter_import_detector import (
    threaded_adapter_import_violations,
)

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

_HOST_ADAPTER_IMPORT_GATE_FILES = _HOST_ADAPTER_FILES + (
    _REPO_ROOT / "intergrax" / "runtime" / "task" / "queued_host_task_execution_adapter.py",
)

_PRODUCTION_COMPOSITION_ROOTS = (
    _REPO_ROOT / "intergrax" / "applications",
    _REPO_ROOT / "applications",
)

_FORBIDDEN_HOST_ADAPTER_TOKENS = ("getattr(", "setattr(", "hasattr(", "GLOBAL_REGISTRY", "service_locator")

_FORBIDDEN_HOST_ADAPTER_IMPORT_PREFIXES = (
    "openai",
    "anthropic",
    "boto3",
    "sqlalchemy",
    "intergrax.integrations.providers",
    "intergrax.runtime.nexus.uaep",
)

_TRANSPORT_IMPORT_MARKERS = ("fastapi", "fastmcp", "starlette", "uvicorn", "mcp.")


@dataclass(frozen=True, slots=True)
class _TaskSemanticCore:
    tenant_id: str
    user_id: str
    message: str
    capability: str
    session_id: str | None
    intent: str | None


def _task_semantic_core(task: Task) -> _TaskSemanticCore:
    return _TaskSemanticCore(
        tenant_id=task.tenant_id,
        user_id=task.user_id,
        message=task.message,
        capability=task.context.capability,
        session_id=task.session_id,
        intent=task.context.intent,
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


def _iter_production_composition_python_files() -> list[Path]:
    paths: list[Path] = []
    for root in _PRODUCTION_COMPOSITION_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            rel = path.relative_to(_REPO_ROOT).as_posix()
            # Materialized image build contexts — not live application composition roots.
            if "runtime-context" in rel:
                continue
            paths.append(path)
    return paths


def _collect_threaded_adapter_production_references() -> list[str]:
    violations: list[str] = []
    for path in _iter_production_composition_python_files():
        rel = path.relative_to(_REPO_ROOT).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except SyntaxError:
            continue
        violations.extend(threaded_adapter_import_violations(tree, rel))
    return violations


async def _await_async_index_task(index: InMemoryAsyncTaskIndex, task_id: str) -> None:
    task = index._tasks.get(task_id)
    assert task is not None
    await task


@pytest.mark.asyncio
async def test_host_q1_production_surfaces_use_host_task_execution_port() -> None:
    port = AsyncMock(spec=HostTaskExecutionPort)
    port.execute = AsyncMock(return_value=_completed_task_result())

    harness_body = HarnessAsyncRunRequest(
        tenant_id="tenant-q1",
        user_id="user-q1",
        message="http-path",
        capability="demo.cap",
        metadata={"channel": "http"},
    )
    harness_task = task_from_harness_async_run_request(harness_body)
    executor = HostTaskExecutionExecutor(port)
    index = InMemoryAsyncTaskIndex()
    await run_async_task_executor(executor, harness_task, index=index)
    await _await_async_index_task(index, harness_task.task_id)
    port.execute.assert_awaited_once()

    port.reset_mock()
    await execute_mcp_agent_task(
        port,
        message="mcp-path",
        capability="demo.cap",
        tenant_id="tenant-q1",
        user_id="user-q1",
    )
    port.execute.assert_awaited_once()

    port.reset_mock()
    run_service = MagicMock()
    adapter = HostTaskExecutionRunAdapter(port)
    adapter.bind_run_service(run_service)
    run_id = mint_run_id()
    core_request = FastApiExecutionRequest(
        run_id=run_id,
        tenant_id="tenant-q1",
        user_id="user-q1",
        input_payload={"message": "core-path", "capability": "demo.cap"},
    )
    await adapter.start_execution(core_request)
    port.execute.assert_awaited_once()
    run_service.mark_running.assert_called_once_with(run_id)
    run_service.mark_completed.assert_called_once()

    port.reset_mock()
    worker_runtime = NexusWorkerRuntime(port)
    task_id = mint_task_id()
    run_id = mint_run_id()
    identity = BackgroundExecutionIdentity(
        tenant_id="tenant-q1",
        task_id=TaskId(task_id),
        run_id=RunId(run_id),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    queue_task = Task(
        tenant_id="tenant-q1",
        user_id="user-q1",
        message="queue-worker-path",
        context=TaskContext(capability="demo.cap"),
    )
    encoded = encode_execution_request(
        FastApiExecutionRequest(
            run_id=str(identity.run_id),
            tenant_id="tenant-q1",
            user_id="user-q1",
            input_payload=task_to_execution_payload(queue_task),
        )
    )
    worker_runtime.execute_payload(
        encoded,
        tenant_id="tenant-q1",
        run_id=str(identity.run_id),
        execution_identity=identity,
    )
    port.execute.assert_awaited_once()

    scenario_source = (_SHARED_HOST / "scenario_runtime_baseline.py").read_text(encoding="utf-8")
    assert "host_execution.execute" in scenario_source

    mcp_source = (_SHARED_HOST / "mcp_nexus_server.py").read_text(encoding="utf-8")
    assert "HostTaskExecutionPort" in mcp_source


def test_host_threaded_execution_adapter_not_wired_in_production_composition() -> None:
    violations = _collect_threaded_adapter_production_references()
    assert violations == [], "\n".join(violations)


@pytest.mark.asyncio
async def test_host_q2_host_task_roots_on_root_execution_launcher() -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    host_execution = _governed_host_execution(nexus_loop)
    launch_calls = 0
    original_launch = DefaultRootExecutionLauncher.launch

    async def _spy_launch(self, request, *args, **kwargs):
        nonlocal launch_calls
        launch_calls += 1
        return await original_launch(self, request, *args, **kwargs)

    with patch.object(DefaultRootExecutionLauncher, "launch", _spy_launch):
        with patch(
            "intergrax.runtime.execution.host_task.TaskBoundAgenticDelegate.execute",
            new_callable=AsyncMock,
            return_value=_completed_task_result(),
        ):
            task = Task(
                tenant_id="t",
                user_id="u",
                message="governance",
                context=TaskContext(capability="external_contractor.adapt"),
            )
            await host_execution.execute(task)
            assert launch_calls == 1


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
    for path in _HOST_ADAPTER_IMPORT_GATE_FILES:
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
    harness_task = task_from_harness_async_run_request(harness_body)
    mcp_task = task_from_mcp_agent_intake(
        message="hello",
        capability="demo.cap",
        tenant_id="tenant-a",
        user_id="user-b",
    )
    assert _task_semantic_core(harness_task) == _task_semantic_core(mcp_task)


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
