# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from intergrax.contracts.execution_identity import (
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    validate_execution_id,
)
from intergrax.runtime.execution.facade import Execution as ExecutionFacade
from intergrax.runtime.execution.request import ExecutionCapability, ExecutionRequest
from intergrax.runtime.execution.strategy import ExecutionStrategy, StrategyResolver
from intergrax.runtime.execution.strategy_router import StrategyExecutionRouter
from intergrax.runtime.execution.task_adapter import TaskExecutionInput
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskResult, TaskState
from governed_contractor_application.host.environment_profile import (
    build_governed_contractor_environment_profile,
)
from governed_contractor_application.host.execution_wiring import (
    build_governed_contractor_host_task_execution,
)
from governed_contractor_application.host.factory import create_governed_contractor_backend_app
from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from governed_contractor_application.serving.fastapi_router import GovernedContractorRunService
from governed_contractor_application.tests.governed_contractor_ac3_projection import (
    build_governed_contractor_test_registry_projection,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from testing_support.builder import MeteringFakeLLMAdapter

pytestmark = [pytest.mark.unit]

_PREFIX = "/v1/governed_contractor"


def _build_service(nexus_loop: NexusLoop) -> GovernedContractorRunService:
    env = build_governed_contractor_environment_profile(GovernedContractorBackendSettings.from_env())
    host_execution = build_governed_contractor_host_task_execution(nexus_loop, env)
    return GovernedContractorRunService.from_host_execution(
        host_execution,
        default_agent_id="external_contractor_adapter",
    )


@pytest.fixture
def _stub_host_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = MeteringFakeLLMAdapter()

    def _resolve(
        env: object,
        agent_override: object | None = None,
        **_: object,
    ) -> object:
        del env
        if agent_override is not None:
            return agent_override
        return adapter

    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_llm_adapter",
        _resolve,
    )


@pytest.mark.asyncio
async def test_governed_contractor_root_uses_canonical_execution_facade() -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    service = _build_service(nexus_loop)
    task = Task(
        task_id=mint_task_id(),
        tenant_id="tenant-1",
        user_id="user-1",
        message="facade proof",
        agent_id="external_contractor_adapter",
        context=TaskContext(capability="external_contractor.adapt"),
    )
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
                task_id=task.task_id,
                run_id=mint_run_id(),
                state=TaskState.COMPLETED,
                answer="ok",
            ),
        ):
            from governed_contractor_application.serving.schemas import GovernedContractorRunRequestV1

            await service.run_task(
                GovernedContractorRunRequestV1(
                    message="facade proof",
                    capability="external_contractor.adapt",
                )
            )

    assert facade_calls == 1
    factory_source = (
        Path(__file__).resolve().parents[2] / "host" / "factory.py"
    ).read_text(encoding="utf-8")
    assert "build_governed_contractor_host_task_execution" in factory_source
    assert "UnifiedTaskRunner" not in factory_source


def test_governed_contractor_http_root_uses_canonical_execution_facade(
    _stub_host_llm: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "INTERGRAX_DIAGNOSTIC_PROBLEM_LIST_CURSOR_SECRET",
        "unit-test-diagnostic-problem-list-cursor-secret",
    )
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
                task_id=mint_task_id(),
                run_id=mint_run_id(),
                state=TaskState.COMPLETED,
                answer="ok",
            ),
        ):
            client = TestClient(
                create_governed_contractor_backend_app(
                    registry_projection=build_governed_contractor_test_registry_projection(),
                    document_store=InMemoryDocumentStore(),
                )
            )
            response = client.post(
                f"{_PREFIX}/run",
                json={"message": "hello", "capability": "external_contractor.adapt"},
            )

    assert response.status_code == 200
    assert facade_calls == 1


@pytest.mark.asyncio
async def test_governed_contractor_adapt_does_not_root_call_nexus_handle_task() -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    nexus_loop.handle_task = AsyncMock()  # type: ignore[method-assign]
    service = _build_service(nexus_loop)

    with patch(
        "intergrax.runtime.execution.host_task.TaskBoundAgenticDelegate.execute",
        new_callable=AsyncMock,
        return_value=TaskResult(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            state=TaskState.COMPLETED,
            answer="ok",
        ),
    ):
        from governed_contractor_application.serving.schemas import GovernedContractorRunRequestV1

        await service.run_task(
            GovernedContractorRunRequestV1(
                message="adapt proof",
                capability="external_contractor.adapt",
            )
        )

    nexus_loop.handle_task.assert_not_called()


@pytest.mark.asyncio
async def test_governed_contractor_adapt_reaches_strategy_router_with_agentic_capability() -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    service = _build_service(nexus_loop)
    captured: dict[str, object] = {}

    async def _capture_execute(
        self: StrategyExecutionRouter[TaskExecutionInput, TaskResult, TaskResult],
        request: ExecutionRequest[TaskExecutionInput, TaskResult],
    ) -> TaskResult:
        captured["strategy"] = StrategyResolver().resolve(request)
        captured["capabilities"] = request.capabilities
        return TaskResult(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            state=TaskState.COMPLETED,
            answer="routed",
        )

    with patch.object(StrategyExecutionRouter, "execute", _capture_execute):
        from governed_contractor_application.serving.schemas import GovernedContractorRunRequestV1

        await service.run_task(
            GovernedContractorRunRequestV1(
                message="strategy proof",
                capability="external_contractor.adapt",
            )
        )

    assert captured["strategy"] is ExecutionStrategy.AGENTIC
    assert captured["capabilities"] == frozenset({ExecutionCapability.AGENT})


@pytest.mark.asyncio
async def test_governed_contractor_root_execution_id_is_platform_owned() -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    service = _build_service(nexus_loop)
    caller_execution_id = mint_execution_id()
    observed: dict[str, object] = {}

    async def _capture_execute(
        self: StrategyExecutionRouter[TaskExecutionInput, TaskResult, TaskResult],
        request: ExecutionRequest[TaskExecutionInput, TaskResult],
    ) -> TaskResult:
        from intergrax.contracts.execution_identity import require_active_execution_id

        observed["execution_id"] = require_active_execution_id()
        return TaskResult(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            state=TaskState.COMPLETED,
            answer="identity",
        )

    with patch.object(StrategyExecutionRouter, "execute", _capture_execute):
        from governed_contractor_application.serving.schemas import GovernedContractorRunRequestV1

        await service.run_task(
            GovernedContractorRunRequestV1(
                message="identity proof",
                capability="external_contractor.adapt",
                metadata={"execution_id": caller_execution_id},
            )
        )

    active_execution_id = observed["execution_id"]
    assert isinstance(active_execution_id, str)
    validate_execution_id(active_execution_id)
    assert active_execution_id != caller_execution_id


@pytest.mark.asyncio
async def test_governed_contractor_request_produces_single_root_execution_invocation() -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    service = _build_service(nexus_loop)
    router_calls = 0
    facade_calls = 0
    original_facade_execute = ExecutionFacade.execute

    async def _count_facade_execute(self, request, *, options):
        nonlocal facade_calls
        facade_calls += 1
        return await original_facade_execute(self, request, options=options)

    async def _count_router_execute(
        self: StrategyExecutionRouter[TaskExecutionInput, TaskResult, TaskResult],
        request: ExecutionRequest[TaskExecutionInput, TaskResult],
    ) -> TaskResult:
        nonlocal router_calls
        router_calls += 1
        return TaskResult(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            state=TaskState.COMPLETED,
            answer="one",
        )

    with patch.object(ExecutionFacade, "execute", _count_facade_execute):
        with patch.object(StrategyExecutionRouter, "execute", _count_router_execute):
            from governed_contractor_application.serving.schemas import GovernedContractorRunRequestV1

            await service.run_task(
                GovernedContractorRunRequestV1(
                    message="single root",
                    capability="external_contractor.adapt",
                )
            )

    assert facade_calls == 1
    assert router_calls == 1
