# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Assemble FastAPI Core + Legal serving into a deployable application."""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from legal_application.serving.fastapi_router import mount_legal_agent_routes
from intergrax.fastapi_core.app_factory import create_app
from intergrax.fastapi_core.auth.api_key import ApiKeyConfig
from intergrax.fastapi_core.config import ApiConfig, ApiEnvironment
from intergrax.fastapi_core.runs.default_service import DefaultRunService
from intergrax.fastapi_core.runs.store_memory import InMemoryRunStore
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.tracing.in_memory_trace_store import InMemoryRunTraceStore
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.nexus_task_execution_adapter import NexusTaskExecutionAdapter

from legal_application.host.settings import LegalBackendSettings
from legal_application.host.wiring import build_legal_agent


def create_legal_backend_app(*, settings: Optional[LegalBackendSettings] = None) -> FastAPI:
    """
    Production host: Intergrax FastAPI Core (health, runs, middleware) + Legal Agent routes.

    Environment variables are read when ``settings`` is omitted (see :mod:`legal_application.host.settings`).

    Uvicorn::

        uvicorn legal_application.host.main:app --host 0.0.0.0 --port 8000
    """
    settings = settings or LegalBackendSettings.from_env()

    api_key_config = ApiKeyConfig(keys=settings.api_keys_map) if settings.api_keys_map else None

    agent = build_legal_agent(settings)
    registry = AgentRegistry()
    contract = agent.get_contract().model_copy(
        update={"id": settings.legal_default_agent_id},
    )
    registry.register(agent, contract=contract)

    trace_store = InMemoryRunTraceStore()
    nexus_loop = NexusLoop(registry, trace_store=trace_store)
    nexus_adapter = NexusTaskExecutionAdapter(nexus_loop)

    run_store = InMemoryRunStore()
    run_service = DefaultRunService(run_store, nexus_adapter)
    nexus_adapter.bind_run_service(run_service)

    api_cfg = ApiConfig(
        environment=settings.environment,
        api_prefix="/v1",
        cors_allow_origins=settings.cors_allow_origins,
        allowed_hosts=settings.allowed_hosts,
        api_key_config=api_key_config,
        run_store=run_store,
        run_service=run_service,
    )
    app = create_app(api_cfg)

    if settings.openapi_enabled_override is True:
        app.docs_url = "/docs"
        app.redoc_url = "/redoc"
        app.openapi_url = "/openapi.json"
    elif settings.openapi_enabled_override is False:
        app.docs_url = None
        app.redoc_url = None
        app.openapi_url = None

    if settings.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=sorted(settings.cors_allow_origins),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    mount_legal_agent_routes(
        app,
        registry=registry,
        default_agent_id=settings.legal_default_agent_id,
        prefix=settings.legal_route_prefix,
        identity_source=settings.identity_source,
        use_nexus_loop=settings.use_nexus_loop,
        trace_store=trace_store,
    )

    if settings.environment == ApiEnvironment.PROD:
        app.title = "Intergrax Legal API"
    else:
        app.title = "Intergrax Legal API (dev)"

    return app
