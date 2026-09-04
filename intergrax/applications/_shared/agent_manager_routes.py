# © Artur Czarnecki. All rights reserved.

"""Read-oriented Agent Manager HTTP routes (Stage 14)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status

from intergrax.agent_distribution.agent_manager_models import (
    AgentManagerEntry,
    AgentManagerListFilters,
    AgentManagerListResult,
)
from intergrax.agent_distribution.agent_manager_query_service import (
    AgentManagerQueryService,
)
from intergrax.applications._shared.agent_platform_admin_routes import (
    _raise_admin_http,
    _require_agent_platform_admin_auth,
)


def mount_agent_manager_routes(
    app: FastAPI,
    *,
    query_service: AgentManagerQueryService,
    prefix: str = "/v1/agent-platform/manager",
) -> APIRouter:
    router = APIRouter(
        prefix=prefix,
        tags=["agent-platform-manager"],
        dependencies=[Depends(_require_agent_platform_admin_auth)],
    )
    env_prefix = "/applications/{application_id}/environments/{environment_id}"

    @router.get(f"{env_prefix}/agents", response_model=AgentManagerListResult)
    def list_agents(
        application_id: str,
        environment_id: str,
        catalog_source_id: str | None = None,
        provider_kind: str | None = None,
        category: str | None = None,
        publisher: str | None = None,
        installed: bool | None = None,
        bound: bool | None = None,
        enabled: bool | None = None,
        capability: str | None = None,
    ) -> AgentManagerListResult:
        try:
            from intergrax.agent_distribution.catalog import CatalogProviderKind

            parsed_kind = (
                CatalogProviderKind(provider_kind) if provider_kind is not None else None
            )
            filters = AgentManagerListFilters(
                catalog_source_id=catalog_source_id,
                provider_kind=parsed_kind,
                category=category,
                publisher=publisher,
                installed=installed,
                bound=bound,
                enabled=enabled,
                capability=capability,
            )
            return query_service.list_agents(
                application_id=application_id,
                application_environment_id=environment_id,
                filters=filters,
            )
        except Exception as exc:
            _raise_admin_http(exc)
            raise

    @router.get(
        f"{env_prefix}/agents/{{manager_entry_id}}",
        response_model=AgentManagerEntry,
    )
    def inspect_agent(
        application_id: str,
        environment_id: str,
        manager_entry_id: str,
    ) -> AgentManagerEntry:
        try:
            entry = query_service.inspect_agent(
                application_id=application_id,
                application_environment_id=environment_id,
                manager_entry_id=manager_entry_id,
            )
            if entry is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"agent manager entry {manager_entry_id} was not found",
                )
            return entry
        except HTTPException:
            raise
        except Exception as exc:
            _raise_admin_http(exc)
            raise

    app.include_router(router)
    return router
