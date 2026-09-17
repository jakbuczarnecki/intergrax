# © Artur Czarnecki. All rights reserved.

"""BG-01 qualification gates (BG-Q1..BG-Q15)."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import Callable, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

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
from intergrax.contracts.execution_terminal import ExecutionTerminalOutcome
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.fastapi_core.execution.models import ExecutionRequest as FastApiExecutionRequest
from intergrax.queueing.contracts.task_queue import (
    TaskHandle,
    TaskQueue,
    TaskRequest,
    TaskResult as QueueTaskResult,
    TaskStatus,
)
from intergrax.queueing.worker.execution import execute_logical_task
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.queueing.worker.result_codec import encode_logical_task_result
from intergrax.runtime.background_execution.bootstrap import BackgroundExecutionIdentity
from intergrax.runtime.background_execution.identity_persistence import (
    KvBackgroundExecutionIdentityPersistence,
)
from intergrax.runtime.background_execution.reentry_admission import (
    BackgroundExecutionReentryDisposition,
    admit_background_execution_reentry,
)
from intergrax.runtime.background_execution.transport_ref import BackgroundTransportExecutionRef
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.governance.default_root_execution_launcher import DefaultRootExecutionLauncher
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.nexus_worker_execution import NexusTaskWorkerOutput, NexusWorkerRuntime
from intergrax.runtime.task.queued_host_task_execution_adapter import QueuedHostTaskExecutionAdapter
from intergrax.runtime.task.task import Task, TaskContext, TaskResult, TaskState
from intergrax.runtime.task.task_result_authoritative_exposure_defaults import (
    terminal_task_result_exposure_no_decision_gate,
)
from intergrax.runtime.task.task_run_bridge import task_to_execution_payload
from intergrax.runtime.task.worker_payload import encode_execution_request
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.tools.execution_models import ToolExecutionResult
from governed_contractor_application.host.environment_profile import (
    build_governed_contractor_environment_profile,
)
from governed_contractor_application.host.execution_wiring import (
    build_governed_contractor_host_task_execution,
)
from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from tests.unit.runtime.background_execution.reentry_admission_doubles import (
    make_kv_admission_dependencies,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BG_INTAKE_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "background_execution"
_BG_WORKER_SURFACE = (
    _BG_INTAKE_ROOT,
    _REPO_ROOT / "intergrax" / "queueing" / "worker" / "execution.py",
    _REPO_ROOT / "intergrax" / "queueing" / "worker" / "dispatcher.py",
    _REPO_ROOT / "intergrax" / "queueing" / "worker" / "result_codec.py",
    _REPO_ROOT / "intergrax" / "runtime" / "task" / "nexus_worker_execution.py",
    _REPO_ROOT / "intergrax" / "runtime" / "task" / "queued_host_task_execution_adapter.py",
    _REPO_ROOT / "intergrax" / "runtime" / "task" / "worker_bootstrap.py",
)
_PRODUCTION_COMPOSITION_ROOTS = (
    _REPO_ROOT / "intergrax" / "applications",
    _REPO_ROOT / "applications",
)
_FORBIDDEN_BG_TOKENS = ("getattr(", "setattr(", "hasattr(", "GLOBAL_REGISTRY", "service_locator")
_FORBIDDEN_BG_IMPORT_PREFIXES = (
    "openai",
    "anthropic",
    "boto3",
    "sqlalchemy",
    "celery",
    "redis",
    "kafka",
    "intergrax.agents.uaep",
    "intergrax.integrations.providers",
)
_VENDOR_IMPORT_MARKERS = (
    "celery",
    "redis",
    "kafka",
    "boto3",
    "sqs",
    "rabbitmq",
    "azure",
)
_LEGACY_BACKGROUND_EXECUTION_TOKENS = (
    "NexusTaskExecutionAdapter",
    "QueuedNexusExecutionAdapter",
    "execute_root_task",
)
_BG_Q15_LAYER_FILES = (
    _REPO_ROOT / "intergrax" / "runtime" / "background_execution",
    _REPO_ROOT / "intergrax" / "queueing" / "worker" / "execution.py",
    _REPO_ROOT / "intergrax" / "runtime" / "task" / "nexus_worker_execution.py",
)


class _ExecKV(DistributedKVStore):
    def __init__(self) -> None:
        self._data: dict[tuple[str, str], bytes] = {}

    def get(self, tenant_id: str, key: str) -> bytes | None:
        return self._data.get((tenant_id, key))

    def set(
        self,
        tenant_id: str,
        key: str,
        value: bytes,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        self._data[(tenant_id, key)] = value

    def delete(self, tenant_id: str, key: str) -> None:
        self._data.pop((tenant_id, key), None)

    def compare_and_set(
        self,
        tenant_id: str,
        key: str,
        expected: bytes | None,
        new_value: bytes,
        *,
        ttl_seconds: int | None = None,
    ) -> bool:
        current = self.get(tenant_id, key)
        if expected is None and current is not None:
            return False
        if expected is not None and current != expected:
            return False
        self.set(tenant_id, key, new_value, ttl_seconds=ttl_seconds)
        return True


def _completed_task_result() -> TaskResult:
    return TaskResult(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        state=TaskState.COMPLETED,
        answer="ok",
        authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
    )


def _governed_host_execution(nexus_loop: NexusLoop):
    env = build_governed_contractor_environment_profile(GovernedContractorBackendSettings.from_env())
    return build_governed_contractor_host_task_execution(nexus_loop, env)


def _worker_identity(
    *,
    tenant_id: str = "tenant-bg",
    run_id: RunId | None = None,
    attempt_id: AttemptId | None = None,
    execution_id: ExecutionId | None = None,
) -> BackgroundExecutionIdentity:
    run = run_id or mint_run_id()
    return BackgroundExecutionIdentity(
        tenant_id=tenant_id,
        task_id=TaskId(mint_task_id()),
        run_id=run,
        attempt_id=attempt_id or mint_attempt_id(),
        execution_id=execution_id or mint_execution_id(),
    )


def _encoded_request(
    identity: BackgroundExecutionIdentity,
    *,
    tenant_id: str | None = None,
) -> bytes:
    scope = tenant_id or identity.tenant_id
    queue_task = Task(
        tenant_id=scope,
        user_id="user-bg",
        message="background",
        context=TaskContext(capability="external_contractor.adapt"),
    )
    return encode_execution_request(
        FastApiExecutionRequest(
            run_id=str(identity.run_id),
            tenant_id=scope,
            user_id="user-bg",
            input_payload=task_to_execution_payload(queue_task),
        )
    )


def _iter_python_files(paths: tuple[Path, ...]) -> list[Path]:
    files: list[Path] = []
    for root in paths:
        if root.is_file():
            files.append(root)
            continue
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            files.append(path)
    return files


def test_bg_q1_worker_invokes_host_task_execution_port() -> None:
    port = AsyncMock(spec=HostTaskExecutionPort)
    port.execute = AsyncMock(return_value=_completed_task_result())
    runtime = NexusWorkerRuntime(port)
    identity = _worker_identity()
    runtime.execute_payload(
        _encoded_request(identity),
        tenant_id=identity.tenant_id,
        run_id=str(identity.run_id),
        execution_identity=identity,
    )
    port.execute.assert_awaited_once()


def test_bg_q1_production_background_execution_surfaces_use_host_port() -> None:
    """Active production composition roots converge host-task workloads to HostTaskExecutionPort."""
    queue_wiring = (
        _REPO_ROOT / "intergrax" / "applications" / "_shared" / "queue_worker_wiring.py"
    ).read_text(encoding="utf-8-sig")
    lkw_factory = (
        _REPO_ROOT
        / "applications"
        / "local_workspace_application"
        / "host"
        / "background_worker_factory.py"
    ).read_text(encoding="utf-8-sig")
    worker_bootstrap = (
        _REPO_ROOT / "intergrax" / "runtime" / "task" / "worker_bootstrap.py"
    ).read_text(encoding="utf-8-sig")
    assert "QueuedHostTaskExecutionAdapter" in queue_wiring
    assert "create_nexus_celery_worker_app" in queue_wiring
    assert "HostTaskExecution" in lkw_factory
    assert "register_nexus_task_worker" in worker_bootstrap or "NexusWorkerRuntime" in worker_bootstrap


def test_bg_q2_identity_forwarded_to_host_execution() -> None:
    port = AsyncMock(spec=HostTaskExecutionPort)
    port.execute = AsyncMock(return_value=_completed_task_result())
    runtime = NexusWorkerRuntime(port)
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    identity = _worker_identity(run_id=run_id, attempt_id=attempt_id)
    runtime.execute_payload(
        _encoded_request(identity),
        tenant_id=identity.tenant_id,
        run_id=str(identity.run_id),
        execution_identity=identity,
    )
    _, kwargs = port.execute.await_args
    assert kwargs["run_id"] == run_id
    assert kwargs["attempt_id"] == attempt_id


def test_bg_q2_full_canonical_identity_on_task_and_host_kwargs() -> None:
    port = AsyncMock(spec=HostTaskExecutionPort)
    port.execute = AsyncMock(return_value=_completed_task_result())
    runtime = NexusWorkerRuntime(port)
    execution_id = mint_execution_id()
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    identity = BackgroundExecutionIdentity(
        tenant_id="tenant-bg-full",
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    runtime.execute_payload(
        _encoded_request(identity, tenant_id=identity.tenant_id),
        tenant_id=identity.tenant_id,
        run_id=str(identity.run_id),
        execution_identity=identity,
    )
    args, kwargs = port.execute.await_args
    task_arg = args[0]
    assert task_arg.tenant_id == identity.tenant_id
    assert task_arg.task_id == identity.task_id
    assert kwargs["run_id"] == run_id
    assert kwargs["attempt_id"] == attempt_id


def test_bg_q3_tenant_mismatch_blocks_execution() -> None:
    port = AsyncMock(spec=HostTaskExecutionPort)
    port.execute = AsyncMock(return_value=_completed_task_result())
    runtime = NexusWorkerRuntime(port)
    identity = _worker_identity(tenant_id="tenant-a")
    with pytest.raises(ValueError, match="tenant mismatch"):
        runtime.execute_payload(
            _encoded_request(identity, tenant_id="tenant-b"),
            tenant_id="tenant-a",
            run_id=str(identity.run_id),
            execution_identity=identity,
        )
    port.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_bg_q4_worker_host_stack_uses_root_launcher() -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    host_execution = _governed_host_execution(nexus_loop)
    runtime = NexusWorkerRuntime(host_execution)
    identity = _worker_identity()
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
            runtime.execute_payload(
                _encoded_request(identity),
                tenant_id=identity.tenant_id,
                run_id=str(identity.run_id),
                execution_identity=identity,
            )
            assert launch_calls == 1


def test_bg_q5_idempotent_logical_task_single_handler_invoke() -> None:
    calls = 0

    class _Out(BaseModel):
        value: str = "ok"

    def handler(
        *,
        tenant_id: str,
        run_id: str,
        payload: bytes,
        idempotency_key,
        execution_identity: BackgroundExecutionIdentity,
    ) -> ToolExecutionResult[_Out]:
        nonlocal calls
        _ = tenant_id, run_id, payload, idempotency_key, execution_identity
        calls += 1
        return ToolExecutionResult.ok(_Out())

    registry = TaskExecutionRegistry()
    registry.register("logical.demo", handler)
    store = InMemoryIdempotencyStore()
    identity = _worker_identity()
    kwargs = {
        "registry": registry,
        "logical_task_name": "logical.demo",
        "tenant_id": identity.tenant_id,
        "run_id": str(identity.run_id),
        "payload": b"payload",
        "idempotency_key": "idem-1",
        "idempotency_store": store,
        "execution_identity": identity,
    }
    execute_logical_task(**kwargs)
    execute_logical_task(**kwargs)
    assert calls == 1


def test_bg_q6_transport_retry_separate_from_attempt_lifecycle() -> None:
    dispatcher_source = (
        _REPO_ROOT / "intergrax" / "queueing" / "worker" / "dispatcher.py"
    ).read_text(encoding="utf-8")
    assert "RetryPolicy" in dispatcher_source
    assert "self.retry" in dispatcher_source
    assert "AttemptLifecycleService" in dispatcher_source
    assert "attempt_lifecycle.transition" not in dispatcher_source


def test_bg_q6_transport_redelivery_does_not_reconcile_new_attempt() -> None:
    deps = make_kv_admission_dependencies()
    transport = BackgroundTransportExecutionRef(
        tenant_id="tenant-a",
        provider="broker",
        transport_task_id="retry-attempt-stable",
    )
    first = admit_background_execution_reentry(
        transport_ref=transport,
        identity_persistence=deps.identity_persistence,
        attempt_lifecycle=deps.attempt_lifecycle,
        execution_terminal=deps.execution_terminal,
    )
    second = admit_background_execution_reentry(
        transport_ref=transport,
        identity_persistence=deps.identity_persistence,
        attempt_lifecycle=deps.attempt_lifecycle,
        execution_terminal=deps.execution_terminal,
    )
    assert second.identity.attempt_id == first.identity.attempt_id
    assert second.identity.execution_id == first.identity.execution_id


def test_bg_q7_terminal_redelivery_safe_disposition() -> None:
    deps = make_kv_admission_dependencies()
    transport = BackgroundTransportExecutionRef(
        tenant_id="tenant-a",
        provider="broker",
        transport_task_id="terminal-redelivery",
    )
    first = admit_background_execution_reentry(
        transport_ref=transport,
        identity_persistence=deps.identity_persistence,
        attempt_lifecycle=deps.attempt_lifecycle,
        execution_terminal=deps.execution_terminal,
    )
    deps.execution_terminal.commit_terminal_outcome(
        tenant_id=first.identity.tenant_id,
        task_id=str(first.identity.task_id),
        run_id=first.identity.run_id,
        outcome=ExecutionTerminalOutcome.COMPLETED,
    )
    redelivery = admit_background_execution_reentry(
        transport_ref=transport,
        identity_persistence=deps.identity_persistence,
        attempt_lifecycle=deps.attempt_lifecycle,
        execution_terminal=deps.execution_terminal,
    )
    assert redelivery.disposition is BackgroundExecutionReentryDisposition.TERMINAL_ALREADY_RECORDED


class _FakeTaskQueue(TaskQueue):
    """Plugin queue — synchronous dispatch for qualification only."""

    def __init__(self, worker: Callable[[TaskRequest], bytes]) -> None:
        self._worker = worker
        self._results: dict[str, QueueTaskResult] = {}

    def enqueue(self, request: TaskRequest) -> TaskHandle:
        handle = TaskHandle(task_id=request.run_id, provider="fake", tenant_id=request.tenant_id)
        try:
            output = self._worker(request)
            self._results[handle.task_id] = QueueTaskResult(
                status=TaskStatus.SUCCEEDED,
                output=output,
            )
        except Exception as exc:
            self._results[handle.task_id] = QueueTaskResult(
                status=TaskStatus.FAILED,
                error_message=str(exc),
            )
        return handle

    def get_status(self, handle: TaskHandle) -> TaskStatus:
        result = self._results.get(handle.task_id)
        if result is None:
            return TaskStatus.PENDING
        return result.status

    def get_result(self, handle: TaskHandle) -> Optional[QueueTaskResult]:
        return self._results.get(handle.task_id)


@pytest.mark.asyncio
async def test_bg_q8_custom_task_queue_plugin() -> None:
    port = AsyncMock(spec=HostTaskExecutionPort)
    port.execute = AsyncMock(return_value=_completed_task_result())
    runtime = NexusWorkerRuntime(port)
    identity_holder: list[BackgroundExecutionIdentity] = []

    def _worker(request: TaskRequest) -> bytes:
        identity = _worker_identity(tenant_id=request.tenant_id, run_id=RunId(request.run_id))
        identity_holder.append(identity)
        payload = runtime.execute_payload(
            request.payload,
            tenant_id=request.tenant_id,
            run_id=request.run_id,
            execution_identity=identity,
        )
        return encode_logical_task_result(
            ToolExecutionResult.ok(NexusTaskWorkerOutput(result_payload=payload))
        )

    queue = _FakeTaskQueue(_worker)
    run_service = MagicMock()
    adapter = QueuedHostTaskExecutionAdapter(queue, run_service, wait_for_result=True)
    run_id = mint_run_id()
    await adapter.start_execution(
        FastApiExecutionRequest(
            run_id=run_id,
            tenant_id="tenant-bg",
            user_id="user-bg",
            input_payload={"message": "plugin", "capability": "external_contractor.adapt"},
        )
    )
    port.execute.assert_awaited_once()
    assert identity_holder
    run_service.mark_completed.assert_called_once()
    run_service.mark_failed.assert_not_called()
    completed_args, completed_kwargs = run_service.mark_completed.call_args
    assert completed_args[0] == run_id
    result_payload = completed_kwargs.get("result_payload")
    assert isinstance(result_payload, dict)
    assert result_payload.get("answer") == "ok"
    assert result_payload.get("state") == TaskState.COMPLETED.value


def test_bg_q9_background_intake_import_layer_gate() -> None:
    violations: list[str] = []
    for path in _iter_python_files((_BG_INTAKE_ROOT,)):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            mod = node.module
            for prefix in ("sqlalchemy", "boto3", "celery", "redis", "kafka"):
                if mod == prefix or mod.startswith(prefix + "."):
                    violations.append(f"{rel}: forbidden import {mod!r}")
    assert violations == [], "\n".join(violations)


def test_bg_q10_resume_path_uses_host_execution() -> None:
    source = (
        _REPO_ROOT / "intergrax" / "runtime" / "task" / "nexus_worker_execution.py"
    ).read_text(encoding="utf-8")
    assert "_reconcile_resume_identity" in source
    assert "resume_checkpoint" in source
    assert "self._host_execution.execute" in source


@pytest.mark.asyncio
async def test_bg_q10_resume_checkpoint_forwarded_to_host_execute() -> None:
    from intergrax.runtime.long_running.models import TaskCheckpoint
    from intergrax.runtime.task.task_state import TaskState

    port = AsyncMock(spec=HostTaskExecutionPort)
    port.execute = AsyncMock(return_value=_completed_task_result())
    checkpoint = TaskCheckpoint(
        task_id=str(mint_task_id()),
        tenant_id="tenant-bg",
        resume_token="resume-token",
        task_state=TaskState.RUNNING,
        runtime=None,
    )
    runtime = NexusWorkerRuntime(port, checkpoint_store=MagicMock())
    identity = _worker_identity()
    with patch.object(
        NexusWorkerRuntime,
        "_reconcile_resume_identity",
        return_value=(identity, checkpoint),
    ):
        runtime.execute_payload(
            _encoded_request(identity),
            tenant_id=identity.tenant_id,
            run_id=str(identity.run_id),
            execution_identity=identity,
        )
    _, kwargs = port.execute.await_args
    assert kwargs["resume_checkpoint"] is checkpoint


def test_bg_q11_no_transport_lease_as_execution_timeout() -> None:
    worker_source = (
        _REPO_ROOT / "intergrax" / "runtime" / "task" / "nexus_worker_execution.py"
    ).read_text(encoding="utf-8")
    assert "visibility_timeout" not in worker_source
    execution_source = (
        _REPO_ROOT / "intergrax" / "queueing" / "worker" / "execution.py"
    ).read_text(encoding="utf-8")
    assert "lease_seconds" in execution_source
    assert "execution_deadline" not in execution_source


def _tier3_factory_paths() -> list[Path]:
    paths: list[Path] = []
    applications_root = _REPO_ROOT / "applications"
    for path in applications_root.rglob("host/factory.py"):
        if "__pycache__" in path.parts:
            continue
        if "runtime-context" in path.parts:
            continue
        paths.append(path)
    return paths


def test_bg_q12_no_legacy_alternate_execution_in_composition() -> None:
    violations: list[str] = []
    scoped_paths = _tier3_factory_paths() + [
        _REPO_ROOT / "intergrax" / "applications" / "_shared" / "queue_worker_wiring.py",
    ]
    for path in scoped_paths:
        if not path.is_file():
            continue
        source = path.read_text(encoding="utf-8-sig")
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for token in _LEGACY_BACKGROUND_EXECUTION_TOKENS:
            if token in source:
                violations.append(f"{rel}: forbidden token {token!r}")
    assert violations == [], "\n".join(violations)


def test_bg_q13_background_contracts_vendor_neutral() -> None:
    violations: list[str] = []
    for path in _iter_python_files((_BG_INTAKE_ROOT,)):
        text = path.read_text(encoding="utf-8").lower()
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for marker in _VENDOR_IMPORT_MARKERS:
            if f"import {marker}" in text or f"from {marker}" in text:
                violations.append(f"{rel}: vendor import marker {marker!r}")
    assert violations == [], "\n".join(violations)


def test_bg_q14_forbidden_integration_patterns_gate() -> None:
    violations: list[str] = []
    for path in _iter_python_files(_BG_WORKER_SURFACE):
        source = path.read_text(encoding="utf-8")
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for token in _FORBIDDEN_BG_TOKENS:
            if token in source:
                violations.append(f"{rel}: forbidden token {token!r}")
    assert violations == [], "\n".join(violations)


def test_bg_q15_background_worker_layer_gate() -> None:
    violations: list[str] = []
    for path in _iter_python_files(_BG_Q15_LAYER_FILES):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            mod = node.module
            for prefix in _FORBIDDEN_BG_IMPORT_PREFIXES:
                if mod == prefix or mod.startswith(prefix + "."):
                    violations.append(f"{rel}: forbidden import {mod!r}")
    assert violations == [], "\n".join(violations)


def test_bg_q_catalog_contract_module_importable() -> None:
    catalog = importlib.import_module("tests.qualification.bg_01.catalog")
    assert len(catalog.BG_01_Q_CATALOG) == 15
