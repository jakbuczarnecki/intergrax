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
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.task.task import TaskResult, TaskState

pytestmark = pytest.mark.unit


class _DummyAgent(Agent):
    def get_contract(self) -> AgentContract:
        return AgentContract(id="legal-test", name="Dummy", description="test")

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        raise RuntimeError("not used when UnifiedTaskRunner.run_runtime_request is mocked")


def _http_ctx(*, tenant: str, user: str) -> RequestContext:
    return RequestContext(
        request_id="req-id",
        path="/v1/legal/chat",
        method="POST",
        tenant_id=tenant,
        user_id=user,
        auth=None,
    )


@pytest.mark.asyncio
async def test_context_only_accepts_identity_from_request_context() -> None:
    config = LegalAgentServingConfig.from_agents(
        agents={"legal-test": _DummyAgent()},
        default_agent_id="legal-test",
        identity_source="context_only",
    )
    svc = DefaultLegalAgentService(config=config)
    body = LegalChatRequestV1(message="ping", session_id="s1")
    task_result = TaskResult(
        task_id="r1",
        run_id="r1",
        state=TaskState.COMPLETED,
        answer="ok",
    )
    with patch(
        "intergrax.runtime.task.unified_task_runner.UnifiedTaskRunner.run_runtime_request",
        new_callable=AsyncMock,
        return_value=task_result,
    ) as run_mock:
        out = await svc.run_legal_chat(body, _http_ctx(tenant="ten-a", user="user-a"))

    assert out.answer == "ok"
    run_mock.assert_awaited_once()
    assert run_mock.await_args.kwargs["tenant_id"] == "ten-a"
    assert run_mock.await_args.kwargs["user_id"] == "user-a"
    req = run_mock.await_args.args[0]
    assert req.tenant_id == "ten-a"
    assert req.user_id == "user-a"


@pytest.mark.asyncio
async def test_context_only_401_when_context_missing_tenant() -> None:
    config = LegalAgentServingConfig.from_agents(
        agents={"legal-test": _DummyAgent()},
        default_agent_id="legal-test",
        identity_source="context_only",
    )
    svc = DefaultLegalAgentService(config=config)
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
    config = LegalAgentServingConfig.from_agents(
        agents={"legal-test": _DummyAgent()},
        default_agent_id="legal-test",
        identity_source="context_only",
    )
    svc = DefaultLegalAgentService(config=config)
    body = LegalChatRequestV1(
        message="ping",
        session_id="s1",
        tenant_id="other-tenant",
        user_id="user-a",
    )
    with pytest.raises(HTTPException) as ei:
        await svc.run_legal_chat(body, _http_ctx(tenant="ten-a", user="user-a"))
    assert ei.value.status_code == 400
