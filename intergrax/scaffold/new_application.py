# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tier-3 application scaffold CLI (Phase N.3–N.4) — lab and product profiles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from textwrap import dedent

from intergrax.applications._shared.build_deploy_doc import render_build_deploy_doc
from intergrax.applications._shared.docker_templates import write_application_docker
from intergrax.scaffold.agent_catalog import ScaffoldAgentSpec, resolve_agent_specs
from intergrax.scaffold.application_names import (
    ScaffoldApplicationNames,
    app_slug,
    env_prefix,
    pascal_case,
    short_id,
)

_PROFILES = ("lab", "product")

# Backward-compatible aliases for tests/tools.
_app_slug = app_slug
_short_id = short_id
_env_prefix = env_prefix
_pascal = pascal_case


def _write(path: Path, content: str, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"File already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _manifest_py(names: ScaffoldApplicationNames, specs: list[ScaffoldAgentSpec]) -> str:
    pkg = names.pkg
    short = names.short
    route_prefix = names.route_prefix
    env_prefix_value = names.env_prefix
    imports = "\n".join(f"from {s.module} import {s.class_name}" for s in specs)
    mounts = []
    for s in specs:
        caps = ", ".join(repr(c) for c in s.capabilities)
        cap_arg = f", capabilities=[{caps}]" if s.capabilities else ""
        mounts.append(f"        AgentBinding.mount({s.class_name}{cap_arg}),")
    mounts_block = "\n".join(mounts)
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        """Declarative agent roster for {pkg}."""

        from __future__ import annotations

        from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
        {imports}


        def build_{short}_manifest() -> ApplicationManifest:
            return ApplicationManifest.lab(
                app_id="{short}",
                name="{names.display} Lab Application",
                route_prefix="{route_prefix}",
                env_prefix="{env_prefix_value}",
                agents=[
        {mounts_block}
                ],
                description="Scaffolded Tier-3 lab environment (Phase N.3)",
            )


        APPLICATION_MANIFEST = build_{short}_manifest()
        '''
    )


def _agent_builders_py(names: ScaffoldApplicationNames, specs: list[ScaffoldAgentSpec]) -> str:
    builders_const = names.builders_const
    imports = "\n".join(f"from {s.module} import {s.class_name}" for s in specs)
    entries = "\n".join(
        f"    {s.class_name}: _zero_arg_factory({s.class_name})," for s in specs
    )
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        from __future__ import annotations

        from intergrax.agents.agent_contract import Agent
        from intergrax.applications.contracts.build_context import ApplicationBuildContext
        from intergrax.applications.contracts.factory import AgentFactory
        from intergrax.applications.contracts.manifest import AgentBinding
        {imports}


        def _zero_arg_factory(agent_cls: type[Agent]) -> AgentFactory:
            def _build(_ctx: ApplicationBuildContext, _binding: AgentBinding) -> Agent:
                return agent_cls()

            return _build


        {builders_const}: dict[type[Agent], AgentFactory] = {{
        {entries}
        }}
        '''
    )


def _settings_py(names: ScaffoldApplicationNames) -> str:
    pkg = names.pkg
    pascal = names.pascal
    env_prefix_value = names.env_prefix
    route_prefix = names.route_prefix
    port = names.port
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        from __future__ import annotations

        import os
        from dataclasses import dataclass

        from intergrax.fastapi_core.config import ApiEnvironment


        @dataclass(frozen=True)
        class {pascal}ApplicationSettings:
            """Environment for {pkg} (scaffolded lab profile)."""

            environment: ApiEnvironment = ApiEnvironment.DEV
            route_prefix: str = "{route_prefix}"
            backend_host: str = "127.0.0.1"
            backend_port: int = {port}
            include_interaction_routes: bool = True
            interaction_route_prefix: str = "/v1/interactions"
            include_scheduler: bool = True
            scheduler_poll_seconds: float | None = None
            interaction_surface: str = "auto"
            include_mcp: bool = True
            mcp_mount_path: str = "/mcp"

            @classmethod
            def from_env(cls) -> {pascal}ApplicationSettings:
                env_raw = (os.getenv("INTERGRAX_ENV") or "dev").strip().lower()
                environment = ApiEnvironment.PROD if env_raw == "prod" else ApiEnvironment.DEV
                prefix = (os.getenv("{env_prefix_value}ROUTE_PREFIX") or "{route_prefix}").strip() or "{route_prefix}"
                host = (os.getenv("{env_prefix_value}BACKEND_HOST") or "127.0.0.1").strip()
                port_raw = (os.getenv("{env_prefix_value}BACKEND_PORT") or "{port}").strip()
                include_interactions = (
                    os.getenv("{env_prefix_value}INCLUDE_INTERACTIONS") or "true"
                ).strip().lower() not in {{"0", "false", "no"}}
                include_scheduler = (
                    os.getenv("{env_prefix_value}INCLUDE_SCHEDULER") or "true"
                ).strip().lower() not in {{"0", "false", "no"}}
                interaction_prefix = (
                    os.getenv("{env_prefix_value}INTERACTION_ROUTE_PREFIX") or "/v1/interactions"
                ).strip() or "/v1/interactions"
                poll_raw = (os.getenv("INTERGRAX_SCHEDULER_POLL_SECONDS") or "").strip()
                scheduler_poll = float(poll_raw) if poll_raw else None
                interaction_surface = (
                    os.getenv("{env_prefix_value}INTERACTION_SURFACE") or "auto"
                ).strip().lower() or "auto"
                include_mcp = (os.getenv("{env_prefix_value}INCLUDE_MCP") or "true").strip().lower() not in {{
                    "0",
                    "false",
                    "no",
                }}
                mcp_mount = (os.getenv("{env_prefix_value}MCP_MOUNT_PATH") or "/mcp").strip() or "/mcp"
                return cls(
                    environment=environment,
                    route_prefix=prefix,
                    backend_host=host,
                    backend_port=int(port_raw),
                    include_interaction_routes=include_interactions,
                    interaction_route_prefix=interaction_prefix,
                    include_scheduler=include_scheduler,
                    scheduler_poll_seconds=scheduler_poll,
                    interaction_surface=interaction_surface,
                    include_mcp=include_mcp,
                    mcp_mount_path=mcp_mount,
                )
        '''
    )


def _wiring_py(names: ScaffoldApplicationNames) -> str:
    pkg = names.pkg
    short = names.short
    pascal = names.pascal
    builders_const = names.builders_const
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        from __future__ import annotations

        from intergrax.applications._shared.wiring import build_application_registry
        from intergrax.applications.contracts.build_context import ApplicationBuildContext
        from intergrax.runtime.registry.agent_registry import AgentRegistry
        from {pkg}.host.agent_builders import {builders_const}
        from {pkg}.host.settings import {pascal}ApplicationSettings
        from {pkg}.host.tool_wiring import wire_{short}_tools
        from {pkg}.manifest import build_{short}_manifest


        def build_{short}_registry(
            *,
            settings: {pascal}ApplicationSettings | None = None,
        ) -> AgentRegistry:
            settings = settings or {pascal}ApplicationSettings.from_env()
            manifest = build_{short}_manifest()
            tool_wiring = wire_{short}_tools(
                integration_profile=getattr(manifest, "integration_profile", None),
            )
            ctx = ApplicationBuildContext.for_manifest(
                manifest,
                settings=settings,
                tool_profile=tool_wiring.profile,
                tool_wiring_context=tool_wiring.wiring_context,
            )
            return build_application_registry(manifest, ctx, builders={builders_const})
        '''
    )


def _tool_wiring_py(names: ScaffoldApplicationNames) -> str:
    pkg = names.pkg
    short = names.short
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        """Tool catalog wiring for {pkg} (Phase O.8)."""

        from __future__ import annotations

        from intergrax.applications._shared.tool_wiring import ApplicationToolWiring, build_application_tool_wiring
        from intergrax.integrations.registry.profile import IntegrationProfile
        from intergrax.tools.registry.profile import ToolProfile


        def wire_{short}_tools(
            *,
            integration_profile: IntegrationProfile | None = None,
        ) -> ApplicationToolWiring:
            profile = ToolProfile(
                enabled=["rag.retrieve", "websearch.query", "sandbox.exec"],
            )
            return build_application_tool_wiring(
                profile,
                integration_profile=integration_profile,
            )
        '''
    )


def _integration_wiring_py(names: ScaffoldApplicationNames) -> str:
    pkg = names.pkg
    short = names.short
    pascal = names.pascal
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        """Integration composition for {pkg} (lab profile)."""

        from __future__ import annotations

        from dataclasses import dataclass
        from pathlib import Path
        from typing import Optional

        from intergrax.integrations.contracts.base import IntegrationCategory
        from intergrax.integrations.providers.relational_store.sqlite.bundle import (
            SQLiteIntegrationBundle,
            create_sqlite_integration,
        )
        from intergrax.integrations.registry.bootstrap import register_default_integrations
        from intergrax.integrations.registry.profile import IntegrationProfile
        from intergrax.integrations.registry.slugs import IntegrationSlug
        from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
        from intergrax.runtime.interactions.adapter_contract import InteractionAdapter
        from intergrax.runtime.interactions.factory import (
            InteractionSurface,
            create_interaction_adapter,
            resolve_interaction_settings,
        )
        from intergrax.runtime.long_running.persistence_contract import TaskCheckpointPersistence
        from intergrax.runtime.notifications.adapter_contract import NotificationAdapter
        from intergrax.runtime.nexus.tracing.in_memory_trace_store import InMemoryRunTraceStore
        from intergrax.runtime.nexus.tracing.persistence_models import RunTraceWriter
        from {pkg}.host.settings import {pascal}ApplicationSettings


        @dataclass(frozen=True)
        class {pascal}IntegrationWiring:
            profile: IntegrationProfile
            sqlite_bundle: SQLiteIntegrationBundle
            trace_store: RunTraceWriter
            runtime_event_store: RuntimeEventPersistence | None
            checkpoint_store: TaskCheckpointPersistence
            notification_adapter: NotificationAdapter
            interaction_adapter: InteractionAdapter
            trace_db_path: Path | None
            runtime_events_db_path: Path | None
            experiments_db_path: Path | None
            checkpoints_db_path: Path | None


        def _sqlite_config_overrides(
            *,
            db_path: Path | None,
            experiments_db_path: Path | None,
            runtime_events_db_path: Path | None,
            checkpoints_db_path: Path | None,
        ) -> dict[str, Path]:
            overrides: dict[str, Path] = {{}}
            if db_path is not None:
                overrides["trace_db"] = db_path
            if experiments_db_path is not None:
                overrides["experiments_db"] = experiments_db_path
            if runtime_events_db_path is not None:
                overrides["runtime_events_db"] = runtime_events_db_path
            if checkpoints_db_path is not None:
                overrides["task_checkpoints_db"] = checkpoints_db_path
            return overrides


        def create_{short}_interaction_adapter(
            settings: {pascal}ApplicationSettings,
        ) -> InteractionAdapter:
            surface = settings.interaction_surface.strip().lower()
            if surface in {{InteractionSurface.LAB.value, InteractionSurface.LAB_JSON.value}}:
                profile = IntegrationProfile.lab()
                return profile.resolve(IntegrationCategory.INTERACTION_SURFACE)
            return create_interaction_adapter(
                resolve_interaction_settings(surface=surface or InteractionSurface.AUTO.value)
            )


        def wire_{short}_integrations(
            *,
            settings: {pascal}ApplicationSettings,
            db_path: Path | None = None,
            experiments_db_path: Path | None = None,
            runtime_events_db_path: Path | None = None,
            checkpoints_db_path: Path | None = None,
        ) -> {pascal}IntegrationWiring:
            register_default_integrations()
            sqlite_overrides = _sqlite_config_overrides(
                db_path=db_path,
                experiments_db_path=experiments_db_path,
                runtime_events_db_path=runtime_events_db_path,
                checkpoints_db_path=checkpoints_db_path,
            )
            profile = IntegrationProfile.lab()
            if sqlite_overrides:
                profile = profile.model_copy(
                    update={{"options": {{IntegrationSlug.SQLITE: dict(sqlite_overrides)}}}}
                )
            sqlite_bundle = create_sqlite_integration(**sqlite_overrides)
            if db_path is None:
                trace_store: RunTraceWriter = InMemoryRunTraceStore()
                trace_db_path = None
            else:
                trace_store = sqlite_bundle.trace_store  # type: ignore[assignment]
                trace_db_path = db_path
            runtime_event_store = (
                sqlite_bundle.runtime_event_store if runtime_events_db_path is not None else None
            )
            notification_adapter = profile.resolve(IntegrationCategory.NOTIFICATION_CHANNEL)
            interaction_adapter = create_{short}_interaction_adapter(settings)
            return {pascal}IntegrationWiring(
                profile=profile,
                sqlite_bundle=sqlite_bundle,
                trace_store=trace_store,
                runtime_event_store=runtime_event_store,  # type: ignore[arg-type]
                checkpoint_store=sqlite_bundle.task_checkpoint_store,  # type: ignore[arg-type]
                notification_adapter=notification_adapter,  # type: ignore[arg-type]
                interaction_adapter=interaction_adapter,
                trace_db_path=trace_db_path,
                runtime_events_db_path=runtime_events_db_path,
                experiments_db_path=experiments_db_path or sqlite_bundle.paths.experiments,
                checkpoints_db_path=checkpoints_db_path or sqlite_bundle.paths.task_checkpoints,
            )
        '''
    )


def _factory_py(names: ScaffoldApplicationNames) -> str:
    pkg = names.pkg
    short = names.short
    pascal = names.pascal
    title = f"Intergrax {names.display} Lab Application"
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        """Assemble lab routes + debug API for {pkg}."""

        from __future__ import annotations

        from pathlib import Path
        from typing import Optional

        from fastapi import FastAPI

        from intergrax.debug.app import create_debug_app
        from intergrax.debug.hitl_service import DebugHitlResumeService
        from intergrax.debug.interaction_service import DebugInteractionIntakeService
        from intergrax.runtime.interactions.router import create_interaction_intake_router
        from intergrax.runtime.interactions.verification.factory import create_inbound_verifier
        from intergrax.runtime.long_running.wiring import wire_long_running_scheduler
        from intergrax.runtime.nexus.nexus_loop import NexusLoop
        from intergrax.runtime.registry.agent_registry import AgentRegistry
        from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner
        from {pkg}.host.integration_wiring import wire_{short}_integrations
        from {pkg}.host.settings import {pascal}ApplicationSettings
        from intergrax.applications._shared.fastapi_mcp import (
            couple_fastapi_with_mcp,
            make_scheduler_lifespan,
        )
        from {pkg}.host.wiring import build_{short}_registry
        from {pkg}.mcp.server import build_{short}_mcp_server
        from {pkg}.serving.fastapi_router import mount_{short}_routes


        def create_{short}_application(
            *,
            settings: Optional[{pascal}ApplicationSettings] = None,
            db_path: Path | None = None,
            experiments_db_path: Path | None = None,
            runtime_events_db_path: Path | None = None,
            checkpoints_db_path: Path | None = None,
            registry: Optional[AgentRegistry] = None,
        ) -> FastAPI:
            settings = settings or {pascal}ApplicationSettings.from_env()
            resolved_registry = registry or build_{short}_registry(settings=settings)
            integrations = wire_{short}_integrations(
                settings=settings,
                db_path=db_path,
                experiments_db_path=experiments_db_path,
                runtime_events_db_path=runtime_events_db_path,
                checkpoints_db_path=checkpoints_db_path,
            )
            nexus_loop = NexusLoop(
                resolved_registry,
                checkpoint_store=integrations.checkpoint_store,
                trace_store=integrations.trace_store,
                runtime_event_store=integrations.runtime_event_store,
                notification_adapter=integrations.notification_adapter,
            )
            task_runner = UnifiedTaskRunner(nexus_loop)
            scheduler_wiring = wire_long_running_scheduler(
                checkpoint_store=integrations.checkpoint_store,
                task_runner=task_runner,
                notification_adapter=integrations.notification_adapter,
                poll_interval_seconds=settings.scheduler_poll_seconds,
                enabled=settings.include_scheduler,
            )
            interaction_service = DebugInteractionIntakeService(
                nexus_loop=nexus_loop,
                adapter=integrations.interaction_adapter,
                verifier=create_inbound_verifier(),
            )
            hitl_service = DebugHitlResumeService(
                resolved_registry,
                checkpoint_store=integrations.checkpoint_store,
            )
            app = create_debug_app(
                db_path=integrations.trace_db_path,
                experiments_db_path=integrations.experiments_db_path,
                runtime_events_db_path=integrations.runtime_events_db_path,
                checkpoints_db_path=integrations.checkpoints_db_path,
                registry=resolved_registry,
                nexus_loop=nexus_loop,
                interaction_service=interaction_service,
                hitl_service=hitl_service,
                checkpoint_store=integrations.checkpoint_store,
                trace_store=integrations.trace_store,
                runtime_event_store=integrations.runtime_event_store,
            )
            app.title = "{title}"
            mount_{short}_routes(app, nexus_loop=nexus_loop, prefix=settings.route_prefix)
            if settings.include_interaction_routes:
                app.include_router(
                    create_interaction_intake_router(
                        interaction_service,
                        execute_default=True,
                    ),
                    prefix=settings.interaction_route_prefix,
                )
            scheduler = scheduler_wiring.scheduler if scheduler_wiring is not None else None
            if settings.include_mcp:
                mcp = build_{short}_mcp_server(
                    nexus_loop=nexus_loop,
                    route_prefix=settings.route_prefix,
                )
                extra_lifespans = [make_scheduler_lifespan(scheduler)] if scheduler else []
                app = couple_fastapi_with_mcp(
                    app,
                    mcp,
                    mount_path=settings.mcp_mount_path,
                    extra_lifespans=extra_lifespans,
                )
            elif scheduler is not None:

                @app.on_event("startup")
                async def _start_scheduler() -> None:
                    await scheduler.start()

                @app.on_event("shutdown")
                async def _stop_scheduler() -> None:
                    await scheduler.stop()
            return app
        '''
    )


def _main_py(names: ScaffoldApplicationNames) -> str:
    pkg = names.pkg
    short = names.short
    env_prefix_value = names.env_prefix
    port = names.port
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        import os

        from dotenv import load_dotenv

        from {pkg}.host.factory import create_{short}_application

        load_dotenv()

        app = create_{short}_application()


        def run() -> None:
            import uvicorn

            host = os.environ.get("{env_prefix_value}BACKEND_HOST", "127.0.0.1")
            port = int(os.environ.get("{env_prefix_value}BACKEND_PORT", "{port}"))
            uvicorn.run(
                "{pkg}.host.main:app",
                host=host,
                port=port,
                reload=os.environ.get("{env_prefix_value}BACKEND_RELOAD", "").lower()
                in {{"1", "true", "yes"}},
            )


        if __name__ == "__main__":
            run()
        '''
    )


def _serving_router_py(names: ScaffoldApplicationNames) -> str:
    short = names.short
    route_prefix = names.route_prefix
    pascal = names.pascal
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        from __future__ import annotations

        from dataclasses import dataclass
        from typing import Any, Optional

        from fastapi import APIRouter, FastAPI, HTTPException, status
        from pydantic import BaseModel, Field

        from intergrax.runtime.nexus.nexus_loop import NexusLoop
        from intergrax.runtime.task.task import Task, TaskContext
        from intergrax.runtime.task.task_run_bridge import new_run_id
        from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner


        class {pascal}RunRequestV1(BaseModel):
            tenant_id: str = "lab"
            user_id: str = "lab-user"
            session_id: Optional[str] = None
            message: str = Field(min_length=1)
            capability: str = Field(min_length=1)
            metadata: dict[str, Any] = Field(default_factory=dict)


        class {pascal}RunResponseV1(BaseModel):
            task_id: str
            run_id: Optional[str] = None
            state: str
            answer: str = ""
            agent_id: Optional[str] = None
            metadata: dict[str, Any] = Field(default_factory=dict)


        @dataclass
        class {pascal}RunService:
            task_runner: UnifiedTaskRunner

            @classmethod
            def from_nexus_loop(cls, nexus_loop: NexusLoop) -> {pascal}RunService:
                return cls(task_runner=UnifiedTaskRunner(nexus_loop))

            async def run_task(self, body: {pascal}RunRequestV1) -> {pascal}RunResponseV1:
                run_id = new_run_id()
                task = Task(
                    task_id=run_id,
                    tenant_id=body.tenant_id,
                    user_id=body.user_id,
                    session_id=body.session_id,
                    message=body.message,
                    context=TaskContext(capability=body.capability),
                    metadata=dict(body.metadata),
                )
                result = await self.task_runner.run_task(task)
                return {pascal}RunResponseV1(
                    task_id=result.task_id,
                    run_id=result.run_id,
                    state=result.state.value,
                    answer=result.answer,
                    agent_id=result.agent_id,
                    metadata=dict(result.metadata),
                )


        def mount_{short}_routes(
            app: FastAPI,
            *,
            nexus_loop: NexusLoop,
            prefix: str = "{route_prefix}",
        ) -> {pascal}RunService:
            service = {pascal}RunService.from_nexus_loop(nexus_loop)
            router = APIRouter(prefix=prefix, tags=["{short}"])

            @router.post("/run", response_model={pascal}RunResponseV1)
            async def run_agent(body: {pascal}RunRequestV1) -> {pascal}RunResponseV1:
                try:
                    return await service.run_task(body)
                except Exception as exc:
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail=f"run_error: {{exc.__class__.__name__}}",
                    ) from exc

            @router.get("/agents")
            async def list_agents() -> dict[str, list[dict[str, object]]]:
                agents: list[dict[str, object]] = []
                for agent_id in nexus_loop.registry.list_agent_ids():
                    contract = nexus_loop.registry.get(agent_id).get_contract()
                    agents.append(
                        {{
                            "agent_id": contract.id,
                            "name": contract.name,
                            "capabilities": list(contract.capabilities),
                        }}
                    )
                return {{"agents": agents}}

            app.include_router(router)
            return service
        '''
    )


def _mcp_server_py(names: ScaffoldApplicationNames, specs: list[ScaffoldAgentSpec]) -> str:
    pkg = names.pkg
    short = names.short
    display = names.display
    cap = specs[0].capabilities[0] if specs and specs[0].capabilities else "echo.basic"
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        """FastMCP server coupled to the {pkg} FastAPI host."""

        from __future__ import annotations

        from fastmcp import FastMCP

        from intergrax.applications._shared.mcp_nexus_server import build_nexus_mcp_server
        from intergrax.runtime.nexus.nexus_loop import NexusLoop


        def build_{short}_mcp_server(
            *,
            nexus_loop: NexusLoop,
            route_prefix: str,
        ) -> FastMCP:
            """MCP tools mirror the lab HTTP API (same NexusLoop / UnifiedTaskRunner)."""
            _ = route_prefix
            return build_nexus_mcp_server(
                name="{display} MCP",
                nexus_loop=nexus_loop,
                default_capability="{cap}",
            )
        '''
    )


def _env_example(env_prefix: str, route_prefix: str, port: int, specs: list[ScaffoldAgentSpec]) -> str:
    caps = specs[0].capabilities[0] if specs and specs[0].capabilities else "echo.basic"
    return dedent(
        f'''\
        # {env_prefix}* — copy to .env (gitignored) in this application directory.
        INTERGRAX_ENV=dev
        {env_prefix}BACKEND_HOST=127.0.0.1
        {env_prefix}BACKEND_PORT={port}
        {env_prefix}ROUTE_PREFIX={route_prefix}
        {env_prefix}INCLUDE_INTERACTIONS=true
        {env_prefix}INCLUDE_SCHEDULER=true
        {env_prefix}INTERACTION_SURFACE=auto
        {env_prefix}INCLUDE_MCP=true
        {env_prefix}MCP_MOUNT_PATH=/mcp
        # Example run capability for POST {route_prefix}/run
        # DEFAULT_CAPABILITY={caps}
        '''
    )


def _agent_dirs(specs: list[ScaffoldAgentSpec]) -> list[str]:
    return sorted({s.slug for s in specs})


def _readme(names: ScaffoldApplicationNames, specs: list[ScaffoldAgentSpec]) -> str:
    pkg = names.pkg
    short = names.short
    display = names.display
    env_prefix_value = names.env_prefix
    route_prefix = names.route_prefix
    port = names.port
    cap = specs[0].capabilities[0] if specs and specs[0].capabilities else "echo.basic"
    agents_list = ", ".join(s.class_name for s in specs)
    return dedent(
        f'''\
        # {display} Application (Tier-3)

        Scaffolded lab-profile application — debug API + ``POST {route_prefix}/run``.

        **Build & deploy:** [`BUILD_AND_DEPLOY.md`](BUILD_AND_DEPLOY.md)

        ## Three-command quickstart

        From **repository root**:

        ```bash
        uv run pytest applications/{pkg}/{pkg}_tests -q
        cp applications/{pkg}/.env.example applications/{pkg}/.env
        uv run uvicorn {pkg}.host.main:app --host 127.0.0.1 --port {port}
        applications/{pkg}/docker/build-docker.sh
        # Windows: applications\\{pkg}\\docker\\build-docker.bat
        ```

        ## Agents

        {agents_list}

        ## Start

        ```bash
        cp applications/{pkg}/.env.example applications/{pkg}/.env
        uv run uvicorn {pkg}.host.main:app --host 127.0.0.1 --port {port}
        ```

        ## Run

        ```bash
        curl -s http://127.0.0.1:{port}{route_prefix}/agents
        curl -s -X POST http://127.0.0.1:{port}{route_prefix}/run \\
          -H "Content-Type: application/json" \\
          -d '{{"message":"hello","capability":"{cap}"}}'
        ```

        ## MCP (FastMCP)

        FastMCP is mounted on the **same** uvicorn process as FastAPI (default ``/mcp``).
        Tools: ``list_agents``, ``run_agent`` — same Nexus loop as HTTP.

        Configure via ``{env_prefix_value}INCLUDE_MCP`` and ``{env_prefix_value}MCP_MOUNT_PATH``.

        ## Docs

        - Engine: `intergrax/applications/USAGE.md`
        - Layout: `applications/USAGE.md`
        '''
    )


def _smoke_test(names: ScaffoldApplicationNames, specs: list[ScaffoldAgentSpec]) -> str:
    pkg = names.pkg
    short = names.short
    route_prefix = names.route_prefix
    cap = specs[0].capabilities[0] if specs and specs[0].capabilities else "echo.basic"
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        from __future__ import annotations

        import pytest
        from fastapi.testclient import TestClient

        from {pkg}.host.factory import create_{short}_application

        pytestmark = [pytest.mark.unit]

        _PREFIX = "{route_prefix}"


        def test_{short}_application_lists_agents():
            client = TestClient(create_{short}_application())
            response = client.get(f"{{_PREFIX}}/agents")
            assert response.status_code == 200
            payload = response.json()
            assert "agents" in payload
            assert len(payload["agents"]) >= 1


        def test_{short}_application_run_echo():
            client = TestClient(create_{short}_application())
            response = client.post(
                f"{{_PREFIX}}/run",
                json={{"message": "hello", "capability": "{cap}"}},
            )
            assert response.status_code == 200
            body = response.json()
            assert body.get("state") == "completed"
        '''
    )


def _create_lab_application(
    *,
    names: ScaffoldApplicationNames,
    specs: list[ScaffoldAgentSpec],
    target: Path,
    profile: str,
    force: bool,
) -> None:
    agent_dirs = _agent_dirs(specs)
    cap = specs[0].capabilities[0] if specs and specs[0].capabilities else "echo.basic"
    health_path = f"{names.route_prefix}/agents"

    _write(target / "__init__.py", "", force=force)
    _write(target / "manifest.py", _manifest_py(names, specs), force=force)
    _write(target / "README.md", _readme(names, specs), force=force)
    _write(
        target / ".env.example",
        _env_example(names.env_prefix, names.route_prefix, names.port, specs),
        force=force,
    )

    _write(target / "host" / "__init__.py", "", force=force)
    _write(target / "host" / "settings.py", _settings_py(names), force=force)
    _write(target / "host" / "agent_builders.py", _agent_builders_py(names, specs), force=force)
    _write(target / "host" / "wiring.py", _wiring_py(names), force=force)
    _write(target / "host" / "integration_wiring.py", _integration_wiring_py(names), force=force)
    _write(target / "host" / "tool_wiring.py", _tool_wiring_py(names), force=force)
    _write(target / "host" / "factory.py", _factory_py(names), force=force)
    _write(target / "host" / "main.py", _main_py(names), force=force)

    _write(target / "serving" / "__init__.py", "", force=force)
    _write(target / "serving" / "fastapi_router.py", _serving_router_py(names), force=force)

    _write(target / "mcp" / "__init__.py", "", force=force)
    _write(target / "mcp" / "server.py", _mcp_server_py(names, specs), force=force)

    _write(target / names.tests_pkg / "__init__.py", "", force=force)
    _write(
        target / names.tests_pkg / "host" / f"test_{names.short}_host_smoke.py",
        _smoke_test(names, specs),
        force=force,
    )
    _write(target / names.tests_pkg / "host" / "__init__.py", "", force=force)

    write_application_docker(
        target,
        pkg=names.pkg,
        short=names.short,
        port=names.port,
        env_prefix=names.env_prefix,
        agent_dirs=agent_dirs,
        health_path=health_path,
        force=force,
    )

    _write(
        target / "BUILD_AND_DEPLOY.md",
        render_build_deploy_doc(
            pkg=names.pkg,
            short=names.short,
            port=names.port,
            env_prefix=names.env_prefix,
            route_prefix=names.route_prefix,
            profile=profile,
            agent_dirs=agent_dirs,
            example_capability=cap,
            health_path=health_path,
            tests_pkg=names.tests_pkg,
            display=names.display,
        ),
        force=force,
    )


def _create_product_application(
    *,
    names: ScaffoldApplicationNames,
    specs: list[ScaffoldAgentSpec],
    target: Path,
    profile: str,
    force: bool,
) -> None:
    from intergrax.scaffold import new_application_product as product_tpl

    agent_dirs = _agent_dirs(specs)
    cap = specs[0].capabilities[0] if specs and specs[0].capabilities else "echo.basic"
    health_path = "/health"

    _write(target / "__init__.py", "", force=force)
    _write(target / "manifest.py", product_tpl.manifest_py(names, specs), force=force)
    _write(target / "README.md", product_tpl.readme(names, specs), force=force)
    _write(
        target / ".env.example",
        product_tpl.env_example(names.env_prefix, names.route_prefix, names.port, specs),
        force=force,
    )

    _write(target / "host" / "__init__.py", "", force=force)
    _write(target / "host" / "settings.py", product_tpl.settings_py(names), force=force)
    _write(target / "host" / "agent_builders.py", product_tpl.agent_builders_py(names, specs), force=force)
    _write(target / "host" / "agent_factories.py", product_tpl.agent_factories_py(names, specs), force=force)
    _write(target / "host" / "wiring.py", product_tpl.wiring_py(names), force=force)
    _write(target / "host" / "integration_wiring.py", product_tpl.integration_wiring_py(names), force=force)
    _write(target / "host" / "tool_wiring.py", product_tpl.tool_wiring_py(names), force=force)
    _write(target / "host" / "factory.py", product_tpl.factory_py(names), force=force)
    _write(target / "host" / "main.py", product_tpl.main_py(names), force=force)

    _write(target / "serving" / "__init__.py", "", force=force)
    _write(target / "serving" / "schemas.py", product_tpl.schemas_py(names), force=force)
    _write(target / "serving" / "fastapi_router.py", product_tpl.serving_router_py(names, specs), force=force)

    _write(target / "mcp" / "__init__.py", "", force=force)
    _write(target / "mcp" / "server.py", product_tpl.mcp_server_py(names, specs), force=force)

    _write(target / names.tests_pkg / "__init__.py", "", force=force)
    _write(
        target / names.tests_pkg / "host" / f"test_{names.short}_host_smoke.py",
        product_tpl.smoke_test(names, specs),
        force=force,
    )
    _write(target / names.tests_pkg / "host" / "__init__.py", "", force=force)

    write_application_docker(
        target,
        pkg=names.pkg,
        short=names.short,
        port=names.port,
        env_prefix=names.env_prefix,
        agent_dirs=agent_dirs,
        health_path=health_path,
        force=force,
    )

    _write(
        target / "BUILD_AND_DEPLOY.md",
        render_build_deploy_doc(
            pkg=names.pkg,
            short=names.short,
            port=names.port,
            env_prefix=names.env_prefix,
            route_prefix=names.route_prefix,
            profile=profile,
            agent_dirs=agent_dirs,
            example_capability=cap,
            health_path=health_path,
            tests_pkg=names.tests_pkg,
            display=names.display,
        ),
        force=force,
    )


def create_application(
    *,
    name: str,
    agents: list[str],
    profile: str,
    root: Path,
    route_prefix: str | None = None,
    port: int = 8091,
    force: bool = False,
) -> Path:
    if profile not in _PROFILES:
        raise ValueError(f"Unsupported profile {profile!r}; choose: {', '.join(_PROFILES)}")

    names = ScaffoldApplicationNames.resolve(
        name,
        route_prefix=route_prefix,
        port=port,
    )
    specs = resolve_agent_specs(agents)

    target = root / "applications" / names.pkg
    if target.exists() and not force:
        raise FileExistsError(f"Application directory already exists: {target}")

    if profile == "lab":
        _create_lab_application(
            names=names,
            specs=specs,
            target=target,
            profile=profile,
            force=force,
        )
    else:
        _create_product_application(
            names=names,
            specs=specs,
            target=target,
            profile=profile,
            force=force,
        )

    return target


def register_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "new-application",
        help="Create applications/<name>_application/ (lab or product profile, Tier-3)",
    )
    parser.add_argument("name", help="Application name (e.g. my_lab → my_lab_application)")
    parser.add_argument(
        "--agents",
        default="echo",
        help="Comma-separated agent slugs (echo, research, signoff_probe, or your agent slug)",
    )
    parser.add_argument(
        "--profile",
        choices=_PROFILES,
        default="lab",
        help="Application profile: lab (debug API) or product (FastAPI Core host)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Default HTTP port (lab default 8091, product default 8000)",
    )
    parser.add_argument(
        "--prefix",
        dest="route_prefix",
        default=None,
        help="HTTP route prefix (default: /v1/<short_id>)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: cwd)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite if exists")


def _default_port(profile: str, port: int | None) -> int:
    if port is not None:
        return port
    return 8000 if profile == "product" else 8091


def run_new_application(args: argparse.Namespace) -> int:
    agent_list = [a.strip() for a in args.agents.split(",") if a.strip()]
    port = _default_port(args.profile, args.port)
    try:
        path = create_application(
            name=args.name,
            agents=agent_list,
            profile=args.profile,
            root=args.root.resolve(),
            route_prefix=args.route_prefix,
            port=port,
            force=args.force,
        )
    except (ValueError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    names = ScaffoldApplicationNames.resolve(
        args.name,
        route_prefix=args.route_prefix,
        port=port,
    )
    print(f"Created Tier-3 application at {path}  (profile={args.profile})")
    print(f"  Package: {names.pkg}  (app_id={names.short!r}, env={names.env_prefix})")
    print(f"  Start:  uv run uvicorn {names.pkg}.host.main:app --host 127.0.0.1 --port {names.port}")
    print(f"  Test:   uv run pytest {path / names.tests_pkg} -q")
    if args.profile == "product":
        print(f"  Health: GET http://127.0.0.1:{names.port}/health")
    print(f"  Agents: GET http://127.0.0.1:{names.port}{names.route_prefix}/agents")
    print(f"  Run:    POST http://127.0.0.1:{names.port}{names.route_prefix}/run")
    print(f"  MCP:    http://127.0.0.1:{names.port}/mcp  (FastMCP, coupled to FastAPI)")
    print(f"  Docker: applications/{names.pkg}/docker/build-docker.sh")
    print(f"          applications\\{names.pkg}\\docker\\build-docker.bat  (Windows)")
    print(f"  Deploy: applications/{names.pkg}/BUILD_AND_DEPLOY.md")
    print("  Docs:   intergrax/applications/USAGE.md · applications/USAGE.md")
    return 0
