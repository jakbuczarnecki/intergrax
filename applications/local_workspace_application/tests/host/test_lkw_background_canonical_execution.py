# © Artur Czarnecki. All rights reserved.

"""UE-11G-P2-B — LKW background worker canonical execution proofs."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from intergrax.applications._shared.host_queue_execution_wiring import HostQueueExecutionDependencies
from intergrax.contracts.execution_identity import (
    mint_execution_id,
    require_active_execution_id,
    require_active_execution_identity,
    validate_execution_id,
    validate_task_id,
)
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.runtime.background_execution.bootstrap import bootstrap_background_execution
from intergrax.runtime.background_execution.identity_persistence import (
    KvBackgroundExecutionIdentityPersistence,
)
from intergrax.runtime.background_execution.transport_ref import BackgroundTransportExecutionRef
from intergrax.runtime.execution.facade import Execution as ExecutionFacade
from intergrax.runtime.execution.request import ExecutionCapability, ExecutionRequest
from intergrax.runtime.execution.strategy import ExecutionStrategy, StrategyResolver
from intergrax.runtime.execution.strategy_router import StrategyExecutionRouter
from intergrax.runtime.execution.task_adapter import TaskExecutionInput
from intergrax.runtime.observability.causal_evidence_persistence import CausalEvidencePersistence
from intergrax.runtime.task.task import TaskResult, TaskState
from local_workspace_application.background_ingest.contracts import (
    LKW_BACKGROUND_INGEST_TASK_NAME,
    encode_background_ingest_job,
)
from local_workspace_application.background_ingest.handler import LKW_BACKGROUND_INGEST_CAPABILITY
from local_workspace_application.host.background_worker_main import (
    activate_local_workspace_reference_production_authority,
)
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.host.background_worker_factory import (
    build_local_workspace_background_worker_wiring,
)
from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST
from local_workspace_application.tests.background_ingest.test_background_ingest_handler import (
    _sample_job,
)
from local_workspace_application.tests.background_ingest.test_background_ingest_worker_handler import (
    _KV,
)

pytestmark = [pytest.mark.unit]


def _settings(monkeypatch: pytest.MonkeyPatch) -> LocalWorkspaceBackendSettings:
    monkeypatch.setenv("INTERGRAX_HARNESS_API_KEY", "test-lkw-background-canonical-key")
    monkeypatch.setenv(
        "INTERGRAX_DIAGNOSTIC_PROBLEM_LIST_CURSOR_SECRET",
        "unit-test-diagnostic-problem-list-cursor-secret",
    )
    monkeypatch.setenv("LOCAL_WORKSPACE_ENABLE_MESSAGE_BUS", "true")
    monkeypatch.setenv("LOCAL_WORKSPACE_ENABLE_KAFKA_MESSAGE_BUS", "true")
    monkeypatch.setenv("LOCAL_WORKSPACE_ENABLE_REDIS", "true")
    return LocalWorkspaceBackendSettings.from_env()


def _activated_projection(monkeypatch: pytest.MonkeyPatch):
    settings = _settings(monkeypatch)
    _, projection = activate_local_workspace_reference_production_authority(settings)
    return projection


def _bootstrap_identity(*, tenant_id: str, transport_task_id: str):
    return bootstrap_background_execution(
        transport_ref=BackgroundTransportExecutionRef(
            tenant_id=tenant_id,
            provider="test",
            transport_task_id=transport_task_id,
        ),
        identity_persistence=KvBackgroundExecutionIdentityPersistence(_KV()),
    )


def _build_worker_handler(monkeypatch: pytest.MonkeyPatch, projection: object):
    _settings(monkeypatch)
    with (
        patch(
            "local_workspace_application.host.background_worker_factory.resolve_host_queue_execution_dependencies",
            return_value=HostQueueExecutionDependencies(
                kv_store=MagicMock(spec=DistributedKVStore),
                causal_evidence_persistence=MagicMock(spec=CausalEvidencePersistence),
            ),
        ),
        patch(
            "local_workspace_application.host.background_worker_factory.create_kafka_worker",
            return_value=MagicMock(),
        ),
    ):
        wiring = build_local_workspace_background_worker_wiring(
            manifest=LOCAL_WORKSPACE_APPLICATION_MANIFEST,
            registry_projection=projection,
        )
    return wiring, wiring.registry.get_handler(LKW_BACKGROUND_INGEST_TASK_NAME)


@pytest.mark.asyncio
async def test_background_worker_handler_invokes_execution_facade_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = _activated_projection(monkeypatch)
    wiring, handler = _build_worker_handler(monkeypatch, projection)
    job = _sample_job()
    identity = _bootstrap_identity(tenant_id=job.tenant_id, transport_task_id="transport-facade-1")
    facade_calls = 0
    original_execute = ExecutionFacade.execute

    async def _spy_execute(self, request, *, options):
        nonlocal facade_calls
        facade_calls += 1
        return await original_execute(self, request, options=options)

    with patch.object(ExecutionFacade, "execute", _spy_execute):
        with patch(
            "intergrax.runtime.execution.host_task.TaskBoundAgenticDelegate.execute",
            new_callable=AsyncMock,
            return_value=TaskResult(
                task_id=identity.task_id,
                run_id=identity.run_id,
                state=TaskState.COMPLETED,
                answer="indexed",
                agent_id="local_indexer",
                metadata={"ingest_summary": {"used": True, "reason": "ingest_complete"}},
            ),
        ):
            result = handler(
                tenant_id=job.tenant_id,
                run_id="lkw.background_ingest.v1:broker-facade",
                payload=encode_background_ingest_job(job),
                idempotency_key=None,
                execution_identity=identity,
            )

    assert result.success is True
    assert facade_calls == 1
    assert wiring.host_execution is not None


@pytest.mark.asyncio
async def test_background_identity_propagates_to_strategy_router(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = _activated_projection(monkeypatch)
    _, handler = _build_worker_handler(monkeypatch, projection)
    job = _sample_job()
    identity = _bootstrap_identity(tenant_id=job.tenant_id, transport_task_id="transport-router-1")
    caller_execution_id = mint_execution_id()
    observed: dict[str, object] = {}

    async def _capture_execute(
        self: StrategyExecutionRouter[TaskExecutionInput, TaskResult, TaskResult],
        request: ExecutionRequest[TaskExecutionInput, TaskResult],
    ) -> TaskResult:
        active_run_id, active_attempt_id = require_active_execution_identity()
        observed["run_id"] = active_run_id
        observed["attempt_id"] = active_attempt_id
        observed["execution_id"] = require_active_execution_id()
        observed["strategy"] = StrategyResolver().resolve(request)
        observed["capabilities"] = request.capabilities
        return TaskResult(
            task_id=identity.task_id,
            run_id=identity.run_id,
            state=TaskState.COMPLETED,
            answer="indexed",
            agent_id="local_indexer",
            metadata={"ingest_summary": {"used": True, "reason": "ingest_complete"}},
        )

    with patch.object(StrategyExecutionRouter, "execute", _capture_execute):
        result = handler(
            tenant_id=job.tenant_id,
            run_id=str(identity.run_id),
            payload=encode_background_ingest_job(job),
            idempotency_key=None,
            execution_identity=identity,
        )

    assert result.success is True
    assert observed["run_id"] == identity.run_id
    assert observed["attempt_id"] == identity.attempt_id
    assert observed["strategy"] is ExecutionStrategy.AGENTIC
    assert observed["capabilities"] == frozenset({ExecutionCapability.AGENT})
    active_execution_id = observed["execution_id"]
    assert isinstance(active_execution_id, str)
    validate_execution_id(active_execution_id)
    assert active_execution_id != caller_execution_id


def test_runtime_task_preserves_background_task_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = _activated_projection(monkeypatch)
    wiring, handler = _build_worker_handler(monkeypatch, projection)
    captured_task_ids: list[str] = []
    job = _sample_job()
    identity = _bootstrap_identity(tenant_id=job.tenant_id, transport_task_id="transport-task-id-1")

    async def _capture_delegate_execute(self, request):
        captured_task_ids.append(self._task.task_id)
        return TaskResult(
            task_id=self._task.task_id,
            run_id=identity.run_id,
            state=TaskState.COMPLETED,
            answer="indexed",
            agent_id="local_indexer",
            metadata={"ingest_summary": {"used": True, "reason": "ingest_complete"}},
        )

    with patch(
        "intergrax.runtime.execution.host_task.TaskBoundAgenticDelegate.execute",
        new=_capture_delegate_execute,
    ):
        result = handler(
            tenant_id=job.tenant_id,
            run_id="lkw.background_ingest.v1:broker-task-id",
            payload=encode_background_ingest_job(job),
            idempotency_key=None,
            execution_identity=identity,
        )

    assert result.success is True
    assert len(captured_task_ids) == 1
    validate_task_id(captured_task_ids[0])
    assert captured_task_ids[0] == identity.task_id


def test_redelivery_preserves_task_and_run_with_new_attempt_and_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = _activated_projection(monkeypatch)
    _, handler = _build_worker_handler(monkeypatch, projection)
    job = _sample_job()
    transport_task_id = "transport-redelivery-p2b"
    persistence = KvBackgroundExecutionIdentityPersistence(_KV())
    transport_ref = BackgroundTransportExecutionRef(
        tenant_id=job.tenant_id,
        provider="test",
        transport_task_id=transport_task_id,
    )
    first_identity = bootstrap_background_execution(
        transport_ref=transport_ref,
        identity_persistence=persistence,
    )
    second_identity = bootstrap_background_execution(
        transport_ref=transport_ref,
        identity_persistence=persistence,
    )
    execution_ids: list[str] = []
    facade_calls = 0
    original_execute = ExecutionFacade.execute

    async def _spy_execute(self, request, *, options):
        nonlocal facade_calls
        facade_calls += 1
        return await original_execute(self, request, options=options)

    async def _stub_delegate_execute(self, request):
        execution_ids.append(require_active_execution_id())
        return TaskResult(
            task_id=self._task.task_id,
            run_id=first_identity.run_id,
            state=TaskState.COMPLETED,
            answer="indexed",
            agent_id="local_indexer",
            metadata={"ingest_summary": {"used": True, "reason": "ingest_complete"}},
        )

    with patch.object(ExecutionFacade, "execute", _spy_execute):
        with patch(
            "intergrax.runtime.execution.host_task.TaskBoundAgenticDelegate.execute",
            new=_stub_delegate_execute,
        ):
            first_result = handler(
                tenant_id=job.tenant_id,
                run_id="lkw.background_ingest.v1:redelivery-1",
                payload=encode_background_ingest_job(job),
                idempotency_key=None,
                execution_identity=first_identity,
            )
            second_result = handler(
                tenant_id=job.tenant_id,
                run_id="lkw.background_ingest.v1:redelivery-2",
                payload=encode_background_ingest_job(job),
                idempotency_key=None,
                execution_identity=second_identity,
            )

    assert first_result.success is True
    assert second_result.success is True
    assert facade_calls == 2
    assert first_identity.task_id == second_identity.task_id
    assert first_identity.run_id == second_identity.run_id
    assert first_identity.attempt_id == second_identity.attempt_id
    assert len(execution_ids) == 2
    validate_execution_id(execution_ids[0])
    validate_execution_id(execution_ids[1])
    assert execution_ids[0] != execution_ids[1]


def test_canonical_run_id_not_broker_run_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = _activated_projection(monkeypatch)
    _, handler = _build_worker_handler(monkeypatch, projection)
    job = _sample_job()
    identity = _bootstrap_identity(tenant_id=job.tenant_id, transport_task_id="transport-broker-leak")
    broker_run_id = "lkw.background_ingest.v1:definitely-not-canonical"
    assert broker_run_id != str(identity.run_id)
    observed_run_id: str | None = None

    async def _capture_router_execute(
        self: StrategyExecutionRouter[TaskExecutionInput, TaskResult, TaskResult],
        request: ExecutionRequest[TaskExecutionInput, TaskResult],
    ) -> TaskResult:
        nonlocal observed_run_id
        observed_run_id, _ = require_active_execution_identity()
        return TaskResult(
            task_id=identity.task_id,
            run_id=identity.run_id,
            state=TaskState.COMPLETED,
            answer="indexed",
            agent_id="local_indexer",
            metadata={"ingest_summary": {"used": True, "reason": "ingest_complete"}},
        )

    with patch.object(StrategyExecutionRouter, "execute", _capture_router_execute):
        result = handler(
            tenant_id=job.tenant_id,
            run_id=broker_run_id,
            payload=encode_background_ingest_job(job),
            idempotency_key=None,
            execution_identity=identity,
        )

    assert result.success is True
    assert observed_run_id == identity.run_id
    assert observed_run_id != broker_run_id


def test_tenant_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    projection = _activated_projection(monkeypatch)
    _, handler = _build_worker_handler(monkeypatch, projection)
    job = _sample_job()
    identity = _bootstrap_identity(tenant_id=job.tenant_id, transport_task_id="transport-tenant-mismatch")

    result = handler(
        tenant_id="tenant-other",
        run_id="lkw.background_ingest.v1:tenant-mismatch",
        payload=encode_background_ingest_job(job),
        idempotency_key=None,
        execution_identity=identity,
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.error_code == "background_ingest_tenant_mismatch"


@pytest.mark.asyncio
async def test_background_index_does_not_root_call_nexus_handle_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = _activated_projection(monkeypatch)
    wiring, handler = _build_worker_handler(monkeypatch, projection)
    wiring.runtime.nexus_loop.handle_task = AsyncMock()  # type: ignore[method-assign]
    job = _sample_job()
    identity = _bootstrap_identity(tenant_id=job.tenant_id, transport_task_id="transport-no-nexus")

    with patch(
        "intergrax.runtime.execution.host_task.TaskBoundAgenticDelegate.execute",
        new_callable=AsyncMock,
        return_value=TaskResult(
            task_id=identity.task_id,
            run_id=identity.run_id,
            state=TaskState.COMPLETED,
            answer="indexed",
            agent_id="local_indexer",
            metadata={"ingest_summary": {"used": True, "reason": "ingest_complete"}},
        ),
    ):
        result = handler(
            tenant_id=job.tenant_id,
            run_id="lkw.background_ingest.v1:no-nexus",
            payload=encode_background_ingest_job(job),
            idempotency_key=None,
            execution_identity=identity,
        )

    assert result.success is True
    wiring.runtime.nexus_loop.handle_task.assert_not_called()


def test_background_index_capability_is_agentic() -> None:
    assert LKW_BACKGROUND_INGEST_CAPABILITY == "local.workspace.index"
