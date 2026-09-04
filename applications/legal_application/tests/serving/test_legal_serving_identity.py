# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract
from legal_application.serving.fastapi_router import (
    DefaultLegalAgentService,
    LegalAgentServingConfig,
)
from legal_application.serving.schemas import LegalChatRequestV1
from intergrax.fastapi_core.context import RequestContext
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.applications._shared.host_task_execution_wiring import build_host_task_execution
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.task.task import TaskResult, TaskState
from testing_support.agent_registry_bootstrap import bootstrap_agent_registry_from_agents

pytestmark = pytest.mark.unit


class _DummyAgent(Agent):
    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="legal-test",
            name="Dummy",
            description="test",
            capabilities=["legal.contract_review"],
        )

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        raise RuntimeError("not used when HostTaskExecution.execute is mocked")


def _http_ctx(*, tenant: str, user: str) -> RequestContext:
    return RequestContext(
        request_id="req-id",
        path="/v1/legal/chat",
        method="POST",
        tenant_id=tenant,
        user_id=user,
        auth=None,
    )


def _service(*, identity_source: str) -> DefaultLegalAgentService:
    agent = _DummyAgent()
    registry = bootstrap_agent_registry_from_agents({"legal-test": agent})
    config = LegalAgentServingConfig(
        registry=registry,
        default_agent_id="legal-test",
        host_execution=build_host_task_execution(NexusLoop(registry), orchestration_triggers=frozenset()),
        identity_source=identity_source,  # type: ignore[arg-type]
    )
    return DefaultLegalAgentService(config=config)


@pytest.mark.asyncio
async def test_context_only_accepts_identity_from_request_context() -> None:
    svc = _service(identity_source="context_only")
    body = LegalChatRequestV1(message="ping", session_id="s1")
    task_result = TaskResult(
        task_id="r1",
        run_id="r1",
        state=TaskState.COMPLETED,
        answer="ok",
    )
    with patch(
        "intergrax.runtime.execution.host_task.HostTaskExecution.execute",
        new_callable=AsyncMock,
        return_value=task_result,
    ) as run_mock:
        out = await svc.run_legal_chat(body, _http_ctx(tenant="ten-a", user="user-a"))

    assert out.answer == "ok"
    run_mock.assert_awaited_once()
    executed_task = run_mock.await_args.args[0]
    assert executed_task.tenant_id == "ten-a"
    assert executed_task.user_id == "user-a"
    assert executed_task.context.capability == "legal.contract_review"


@pytest.mark.asyncio
async def test_context_only_401_when_context_missing_tenant() -> None:
    svc = _service(identity_source="context_only")
    body = LegalChatRequestV1(
        message="ping",
        session_id="s1",
        tenant_id="from-body",
        user_id="user-a",
    )
    ctx = RequestContext(
        request_id="req-id",
        path="/",
        method="POST",
        tenant_id=None,
        user_id="user-a",
        auth=None,
    )
    with pytest.raises(HTTPException) as ei:
        await svc.run_legal_chat(body, ctx)
    assert ei.value.status_code == 401


@pytest.mark.asyncio
async def test_context_only_400_on_tenant_body_mismatch() -> None:
    svc = _service(identity_source="context_only")
    body = LegalChatRequestV1(
        message="ping",
        session_id="s1",
        tenant_id="other-tenant",
        user_id="user-a",
    )
    with pytest.raises(HTTPException) as ei:
        await svc.run_legal_chat(body, _http_ctx(tenant="ten-a", user="user-a"))
    assert ei.value.status_code == 400
