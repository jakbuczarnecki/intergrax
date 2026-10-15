# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status

from governed_contractor_application.host.settings import GovernedContractorIdentitySource
from governed_contractor_application.serving.schemas import (
    GovernedContractorRunRequestV1,
    GovernedContractorRunResponseV1,
)
from intergrax.applications._shared.strict_production_serving_execution_tenant_authority import (
    StrictProductionServingExecutionTenantBinding,
)
from intergrax.applications._shared.strict_production_serving_http_caller_authorization import (
    authorize_strict_production_serving_http_caller,
    request_identity_from_strict_production_http_auth,
)
from intergrax.applications._shared.strict_production_serving_http_execution_principal import (
    bind_strict_serving_http_caller_principal,
)
from intergrax.fastapi_core.auth.context import AuthContext
from intergrax.fastapi_core.auth.dependency import require_auth
from intergrax.fastapi_core.context import RequestContext, get_request_context
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead
from intergrax.runtime.task.task import Task, TaskContext


@dataclass(frozen=True)
class GovernedContractorServingConfig:
    host_execution: HostTaskExecutionPort
    default_agent_id: str
    identity_source: GovernedContractorIdentitySource = "body_or_context"
    strict_serving_tenant_binding: StrictProductionServingExecutionTenantBinding | None = None


@dataclass
class GovernedContractorRunService:
    config: GovernedContractorServingConfig

    @classmethod
    def from_config(cls, config: GovernedContractorServingConfig) -> GovernedContractorRunService:
        return cls(config=config)

    def _resolve_identity(
        self,
        body: GovernedContractorRunRequestV1,
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

    async def run_task(
        self,
        body: GovernedContractorRunRequestV1,
        http_ctx: RequestContext,
        *,
        auth: AuthContext | None = None,
    ) -> GovernedContractorRunResponseV1:
        binding = self.config.strict_serving_tenant_binding
        if binding is not None:
            if auth is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                )
            authorize_strict_production_serving_http_caller(auth, binding=binding)
            caller_principal = request_identity_from_strict_production_http_auth(auth)
        else:
            caller_principal = None

        tenant_id, user_id = self._resolve_identity(body, http_ctx)
        task = Task(
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=body.session_id,
            agent_id=self.config.default_agent_id,
            message=body.message,
            context=TaskContext(capability=body.capability or "external_contractor.adapt"),
            metadata=dict(body.metadata),
        )
        try:
            if caller_principal is not None:
                with bind_strict_serving_http_caller_principal(caller_principal):
                    result = await self.config.host_execution.execute(task)
            else:
                result = await self.config.host_execution.execute(task)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"run_error: {exc.__class__.__name__}",
            ) from exc
        return GovernedContractorRunResponseV1(
            task_id=result.task_id,
            run_id=result.run_id,
            state=result.state.value,
            answer=result.answer,
            agent_id=result.agent_id,
            metadata=dict(result.metadata),
        )


def mount_governed_contractor_routes(
    app: FastAPI,
    *,
    host_execution: HostTaskExecutionPort,
    registry: AgentRegistryRead,
    prefix: str = "/v1/governed_contractor",
    default_agent_id: str = "external_contractor_adapter",
    identity_source: GovernedContractorIdentitySource = "body_or_context",
    strict_serving_tenant_binding: StrictProductionServingExecutionTenantBinding | None = None,
) -> GovernedContractorRunService:
    config = GovernedContractorServingConfig(
        host_execution=host_execution,
        default_agent_id=default_agent_id,
        identity_source=identity_source,
        strict_serving_tenant_binding=strict_serving_tenant_binding,
    )
    service = GovernedContractorRunService.from_config(config)
    strict_ingress = strict_serving_tenant_binding is not None
    router = APIRouter(prefix=prefix, tags=["governed_contractor"])

    def _service_dep() -> GovernedContractorRunService:
        return service

    if strict_ingress:

        @router.post("/run", response_model=GovernedContractorRunResponseV1)
        async def run_agent(
            body: GovernedContractorRunRequestV1,
            http_ctx: RequestContext = Depends(get_request_context),
            auth: AuthContext = Depends(require_auth),
            svc: GovernedContractorRunService = Depends(_service_dep),
        ) -> GovernedContractorRunResponseV1:
            return await svc.run_task(body, http_ctx, auth=auth)

        @router.get("/agents")
        async def list_agents(
            _auth: AuthContext = Depends(require_auth),
        ) -> dict[str, list[dict[str, object]]]:
            agents: list[dict[str, object]] = []
            for agent_id in registry.list_agent_ids():
                contract = registry.get(agent_id).get_contract()
                agents.append(
                    {
                        "agent_id": contract.id,
                        "name": contract.name,
                        "capabilities": list(contract.capabilities),
                    }
                )
            return {"agents": agents}

    else:

        @router.post("/run", response_model=GovernedContractorRunResponseV1)
        async def run_agent(
            body: GovernedContractorRunRequestV1,
            http_ctx: RequestContext = Depends(get_request_context),
            svc: GovernedContractorRunService = Depends(_service_dep),
        ) -> GovernedContractorRunResponseV1:
            return await svc.run_task(body, http_ctx)

        @router.get("/agents")
        async def list_agents() -> dict[str, list[dict[str, object]]]:
            agents: list[dict[str, object]] = []
            for agent_id in registry.list_agent_ids():
                contract = registry.get(agent_id).get_contract()
                agents.append(
                    {
                        "agent_id": contract.id,
                        "name": contract.name,
                        "capabilities": list(contract.capabilities),
                    }
                )
            return {"agents": agents}

    app.include_router(router)
    return service
