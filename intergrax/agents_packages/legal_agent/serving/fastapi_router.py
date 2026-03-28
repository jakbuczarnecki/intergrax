# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""FastAPI integration aligned with :mod:`intergrax.fastapi_core.runs` (service Protocol + router + DI)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Protocol

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status

from intergrax.agents.agent_contract import Agent
from intergrax.agents.agent_engine import AgentEngine
from intergrax.agents_packages.legal_agent.serving.runtime_bridge import LegalApiV1RuntimeMapper
from intergrax.agents_packages.legal_agent.serving.schemas import (
    LegalChatRequestV1,
    LegalChatResponseV1,
)
from intergrax.fastapi_core.context import RequestContext, get_request_context


class LegalAgentService(Protocol):
    """HTTP-facing orchestration for legal chat (same role as :class:`intergrax.fastapi_core.runs.service.RunService`)."""

    async def run_legal_chat(
        self,
        body: LegalChatRequestV1,
        http_ctx: RequestContext,
    ) -> LegalChatResponseV1:
        ...


@dataclass(frozen=True)
class LegalAgentServingConfig:
    """Registry of Tier-2 agents exposed through this HTTP surface."""

    agents: Mapping[str, Agent]
    default_agent_id: str

    def __post_init__(self) -> None:
        if self.default_agent_id not in self.agents:
            raise ValueError(
                f"default_agent_id {self.default_agent_id!r} not in agents keys: {list(self.agents)!r}"
            )


@dataclass
class DefaultLegalAgentService:
    """
    Default :class:`LegalAgentService`: validate identity context, drive :class:`AgentEngine`, map responses.

    Inject a custom :class:`LegalApiV1RuntimeMapper` in tests or for extended API versions.
    """

    config: LegalAgentServingConfig
    mapper: LegalApiV1RuntimeMapper = field(default_factory=LegalApiV1RuntimeMapper)

    async def run_legal_chat(
        self,
        body: LegalChatRequestV1,
        http_ctx: RequestContext,
    ) -> LegalChatResponseV1:
        tenant = body.tenant_id or http_ctx.tenant_id
        user = body.user_id or http_ctx.user_id
        if not tenant:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="tenant_id is required (body.tenant_id or authenticated RequestContext).",
            )
        if not user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="user_id is required (body.user_id or authenticated RequestContext).",
            )

        engine = AgentEngine(agents=dict(self.config.agents))
        runtime_req = self.mapper.to_runtime_request(
            body,
            http_context=http_ctx,
            default_agent_id=self.config.default_agent_id,
            tenant_id=tenant,
            user_id=user,
        )
        try:
            answer = await engine.run(runtime_req)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"runtime_error: {exc.__class__.__name__}",
            ) from exc

        return self.mapper.to_legal_chat_response(
            answer,
            http_context=http_ctx,
            include_trace=body.include_trace,
        )


LegalAgentServingFacade = DefaultLegalAgentService

legal_agent_router = APIRouter(tags=["legal-agent"])


@legal_agent_router.post(
    "/chat",
    response_model=LegalChatResponseV1,
    status_code=status.HTTP_200_OK,
    summary="Run legal agent turn",
)
async def legal_chat(
    body: LegalChatRequestV1,
    http_ctx: RequestContext = Depends(get_request_context),
    service: LegalAgentService = Depends(),
) -> LegalChatResponseV1:
    return await service.run_legal_chat(body, http_ctx)


def create_legal_agent_router(
    *,
    service: LegalAgentService,
    prefix: str = "/v1/legal",
) -> APIRouter:
    """Router wired to a specific service instance (tests / composition without ``dependency_overrides``)."""

    router = APIRouter(prefix=prefix, tags=["legal-agent"])

    def _service_dep() -> LegalAgentService:
        return service

    @router.post(
        "/chat",
        response_model=LegalChatResponseV1,
        status_code=status.HTTP_200_OK,
        summary="Run legal agent turn",
    )
    async def _legal_chat(
        body: LegalChatRequestV1,
        http_ctx: RequestContext = Depends(get_request_context),
        svc: LegalAgentService = Depends(_service_dep),
    ) -> LegalChatResponseV1:
        return await svc.run_legal_chat(body, http_ctx)

    return router


def mount_legal_agent_routes(
    app: FastAPI,
    *,
    agents: Dict[str, Agent],
    default_agent_id: str,
    prefix: str = "/v1/legal",
    mapper: LegalApiV1RuntimeMapper | None = None,
) -> DefaultLegalAgentService:
    """
    Register legal routes on ``app`` via ``dependency_overrides`` (same pattern as ``RunService`` in ``create_app``).

    Returns the :class:`DefaultLegalAgentService` instance for tests or extra wiring.

    ``app`` must use ``RequestContextMiddleware`` (e.g. :func:`intergrax.fastapi_core.app_factory.create_app`).
    """
    config = LegalAgentServingConfig(agents=agents, default_agent_id=default_agent_id)
    svc = DefaultLegalAgentService(
        config=config,
        mapper=mapper or LegalApiV1RuntimeMapper(),
    )
    app.dependency_overrides[LegalAgentService] = lambda: svc
    app.include_router(legal_agent_router, prefix=prefix)
    return svc
