# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from intergrax.agents.agent_contract import Agent
from legal_agent.serving.fastapi_router import (
    DefaultLegalAgentService,
    LegalAgentServingConfig,
)
from legal_agent.serving.schemas import LegalChatRequestV1
from intergrax.fastapi_core.context import RequestContext
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import (
    RouteInfo,
    RuntimeAnswer,
    RuntimeRequest,
    RuntimeStats,
    StopReason,
)

pytestmark = pytest.mark.unit


class _DummyAgent(Agent):
    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        raise RuntimeError("not used when AgentEngine.run is mocked")


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
    config = LegalAgentServingConfig(
        agents={"legal-test": _DummyAgent()},
        default_agent_id="legal-test",
        identity_source="context_only",
    )
    svc = DefaultLegalAgentService(config=config)
    body = LegalChatRequestV1(message="ping", session_id="s1")
    answer = RuntimeAnswer(
        answer="ok",
        stop_reason=StopReason.COMPLETED,
        run_id="r1",
        route=RouteInfo(strategy="legal_agent"),
        stats=RuntimeStats(),
    )
    with patch(
        "intergrax.agents.agent_engine.AgentEngine.run",
        new_callable=AsyncMock,
        return_value=answer,
    ) as run_mock:
        out = await svc.run_legal_chat(body, _http_ctx(tenant="ten-a", user="user-a"))

    assert out.answer == "ok"
    run_mock.assert_awaited_once()
    req = run_mock.await_args.args[0]
    assert req.tenant_id == "ten-a"
    assert req.user_id == "user-a"


@pytest.mark.asyncio
async def test_context_only_401_when_context_missing_tenant() -> None:
    config = LegalAgentServingConfig(
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
    config = LegalAgentServingConfig(
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
