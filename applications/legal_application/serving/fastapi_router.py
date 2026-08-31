# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""FastAPI integration aligned with :mod:`intergrax.fastapi_core.runs`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Mapping, Optional, Protocol, Union

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status

LegalIdentitySource = Literal["body_or_context", "context_only"]

from intergrax.agents.agent_contract import Agent
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.nexus.tracing.persistence_models import RunTraceWriter
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task_run_bridge import mint_intake_execution_identity, task_from_runtime_request

from legal.legal_agent import LegalAgent
from legal_application.serving.runtime_bridge import LegalApiV1RuntimeMapper
from legal_application.serving.schemas import (
    LegalChatRequestV1,
    LegalChatResponseV1,
)
from intergrax.fastapi_core.context import RequestContext, get_request_context
from intergrax.runtime.nexus.policies.runtime_policies import DataCompliancePolicy
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RuntimeRequest


class LegalAgentService(Protocol):
    """HTTP-facing orchestration for legal chat."""

    async def run_legal_chat(
        self,
        body: LegalChatRequestV1,
        http_ctx: RequestContext,
    ) -> LegalChatResponseV1:
        ...


@dataclass(frozen=True)
class LegalAgentServingConfig:
    """Registry of Tier-2 agents exposed through this HTTP surface."""

    registry: AgentRegistry
    default_agent_id: str
    host_execution: HostTaskExecutionPort = field(repr=False)
    identity_source: LegalIdentitySource = "body_or_context"
    trace_store: Optional[RunTraceWriter] = None

    def __post_init__(self) -> None:
        if not self.registry.has(self.default_agent_id):
            raise ValueError(
                f"default_agent_id {self.default_agent_id!r} not in registry: "
                f"{self.registry.list_agent_ids()!r}"
            )

    @classmethod
    def from_agents(
        cls,
        agents: Mapping[str, Agent],
        *,
        default_agent_id: str,
        identity_source: LegalIdentitySource = "body_or_context",
        trace_store: Optional[RunTraceWriter] = None,
        host_execution: HostTaskExecutionPort,
    ) -> "LegalAgentServingConfig":
        registry = AgentRegistry.from_agents(dict(agents))
        return cls(
            registry=registry,
            default_agent_id=default_agent_id,
            identity_source=identity_source,
            trace_store=trace_store,
            host_execution=host_execution,
        )


@dataclass
class DefaultLegalAgentService:
    """Validate identity, drive canonical host execution, map responses."""

    config: LegalAgentServingConfig
    mapper: LegalApiV1RuntimeMapper = field(default_factory=LegalApiV1RuntimeMapper)

    def _data_compliance_for_request(self, runtime_req: RuntimeRequest) -> DataCompliancePolicy:
        agent = self.config.registry.get(runtime_req.agent_id or self.config.default_agent_id)
        if isinstance(agent, LegalAgent):
            return agent.data_compliance_policy
        return DataCompliancePolicy()

    async def run_legal_chat(
        self,
        body: LegalChatRequestV1,
        http_ctx: RequestContext,
    ) -> LegalChatResponseV1:
        tenant, user = self._resolve_identity(body, http_ctx)

        task_id, run_id = mint_intake_execution_identity()
        runtime_req = self.mapper.to_runtime_request(
            body,
            http_context=http_ctx,
            default_agent_id=self.config.default_agent_id,
            tenant_id=tenant,
            user_id=user,
            task_id=task_id,
            run_id=run_id,
        )
        task = task_from_runtime_request(
            runtime_req,
            tenant_id=tenant,
            user_id=user,
            capability="legal.contract_review",
        )

        try:
            result = await self.config.host_execution.execute(task)
            answer = RuntimeAnswer(
                run_id=result.run_id or run_id,
                answer=result.answer,
            )
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
            data_compliance=self._data_compliance_for_request(runtime_req),
        )

    def _resolve_identity(
        self,
        body: LegalChatRequestV1,
        http_ctx: RequestContext,
    ) -> tuple[str, str]:
        if self.config.identity_source == "context_only":
            tenant = http_ctx.tenant_id
            user = http_ctx.user_id
            if not tenant or not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=(
                        "tenant_id and user_id must be set on RequestContext "
                        "(configure AuthProvider / JWT or API key on the app)."
                    ),
                )
            if body.tenant_id is not None and body.tenant_id != tenant:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="tenant_id in request body conflicts with authenticated RequestContext.",
                )
            if body.user_id is not None and body.user_id != user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="user_id in request body conflicts with authenticated RequestContext.",
                )
        else:
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
        return tenant, user


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
    registry: Optional[AgentRegistry] = None,
    agents: Optional[Dict[str, Agent]] = None,
    default_agent_id: str,
    prefix: str = "/v1/legal",
    mapper: LegalApiV1RuntimeMapper | None = None,
    identity_source: LegalIdentitySource = "body_or_context",
    trace_store: Optional[RunTraceWriter] = None,
    host_execution: HostTaskExecutionPort,
) -> DefaultLegalAgentService:
    """
    Register legal routes on ``app`` via ``dependency_overrides``.

    Pass ``registry`` (preferred) or legacy ``agents`` dict.
    """
    if registry is None:
        if agents is None:
            raise ValueError("Either registry or agents must be provided.")
        config = LegalAgentServingConfig.from_agents(
            agents,
            default_agent_id=default_agent_id,
            identity_source=identity_source,
            trace_store=trace_store,
            host_execution=host_execution,
        )
    else:
        config = LegalAgentServingConfig(
            registry=registry,
            default_agent_id=default_agent_id,
            identity_source=identity_source,
            trace_store=trace_store,
            host_execution=host_execution,
        )

    svc = DefaultLegalAgentService(
        config=config,
        mapper=mapper or LegalApiV1RuntimeMapper(),
    )
    app.dependency_overrides[LegalAgentService] = lambda: svc
    app.include_router(legal_agent_router, prefix=prefix)
    return svc
