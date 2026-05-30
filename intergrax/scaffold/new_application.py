# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tier-3 application scaffold CLI (Phase N.3) — lab profile with debug API."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from textwrap import dedent

from intergrax.scaffold.agent_catalog import ScaffoldAgentSpec, resolve_agent_specs

_PROFILES = ("lab",)


def _app_slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9_]+", "_", name.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug or slug[0].isdigit():
        raise ValueError(f"Invalid application name: {name!r}")
    if not slug.endswith("_application"):
        slug = f"{slug}_application"
    return slug


def _short_id(app_slug: str) -> str:
    if app_slug.endswith("_application"):
        return app_slug[: -len("_application")]
    return app_slug


def _env_prefix(short: str) -> str:
    return re.sub(r"[^A-Z0-9]", "_", short.upper()).strip("_") + "_"


def _pascal(short: str) -> str:
    return "".join(part.capitalize() for part in short.split("_"))


def _write(path: Path, content: str, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"File already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _manifest_py(pkg: str, specs: list[ScaffoldAgentSpec], route_prefix: str, env_prefix: str) -> str:
    imports = "\n".join(f"from {s.module} import {s.class_name}" for s in specs)
    mounts = []
    for s in specs:
        caps = ", ".join(repr(c) for c in s.capabilities)
        cap_arg = f", capabilities=[{caps}]" if s.capabilities else ""
        mounts.append(f"        AgentBinding.mount({s.class_name}{cap_arg}),")
    mounts_block = "\n".join(mounts)
    short = _short_id(pkg)
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
                name="{_pascal(short)} Lab Application",
                route_prefix="{route_prefix}",
                env_prefix="{env_prefix}",
                agents=[
        {mounts_block}
                ],
                description="Scaffolded Tier-3 lab environment (Phase N.3)",
            )


        APPLICATION_MANIFEST = build_{short}_manifest()
        '''
    )


def _agent_builders_py(specs: list[ScaffoldAgentSpec]) -> str:
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


        AGENT_BUILDERS: dict[type[Agent], AgentFactory] = {{
        {entries}
        }}
        '''
    )


def _settings_py(pkg: str, short: str, env_prefix: str, route_prefix: str, port: int) -> str:
    pascal = _pascal(short)
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

            @classmethod
            def from_env(cls) -> {pascal}ApplicationSettings:
                env_raw = (os.getenv("INTERGRAX_ENV") or "dev").strip().lower()
                environment = ApiEnvironment.PROD if env_raw == "prod" else ApiEnvironment.DEV
                prefix = (os.getenv("{env_prefix}ROUTE_PREFIX") or "{route_prefix}").strip() or "{route_prefix}"
                host = (os.getenv("{env_prefix}BACKEND_HOST") or "127.0.0.1").strip()
                port_raw = (os.getenv("{env_prefix}BACKEND_PORT") or "{port}").strip()
                include_interactions = (
                    os.getenv("{env_prefix}INCLUDE_INTERACTIONS") or "true"
                ).strip().lower() not in {{"0", "false", "no"}}
                include_scheduler = (
                    os.getenv("{env_prefix}INCLUDE_SCHEDULER") or "true"
                ).strip().lower() not in {{"0", "false", "no"}}
                interaction_prefix = (
                    os.getenv("{env_prefix}INTERACTION_ROUTE_PREFIX") or "/v1/interactions"
                ).strip() or "/v1/interactions"
                poll_raw = (os.getenv("INTERGRAX_SCHEDULER_POLL_SECONDS") or "").strip()
                scheduler_poll = float(poll_raw) if poll_raw else None
                interaction_surface = (
                    os.getenv("{env_prefix}INTERACTION_SURFACE") or "auto"
                ).strip().lower() or "auto"
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
                )
        '''
    )


def _wiring_py(pkg: str, short: str) -> str:
    pascal = _pascal(short)
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        from __future__ import annotations

        from intergrax.applications._shared.wiring import build_application_registry
        from intergrax.applications.contracts.build_context import ApplicationBuildContext
        from intergrax.runtime.registry.agent_registry import AgentRegistry
        from {pkg}.host.agent_builders import AGENT_BUILDERS
        from {pkg}.host.settings import {pascal}ApplicationSettings
        from {pkg}.manifest import build_{short}_manifest


        def build_{short}_registry(
            *,
            settings: {pascal}ApplicationSettings | None = None,
        ) -> AgentRegistry:
            settings = settings or {pascal}ApplicationSettings.from_env()
            manifest = build_{short}_manifest()
            ctx = ApplicationBuildContext.for_manifest(manifest, settings=settings)
            return build_application_registry(manifest, ctx, builders=AGENT_BUILDERS)
        '''
    )


def _integration_wiring_py(pkg: str, short: str) -> str:
    pascal = _pascal(short)
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        """Integration composition for {pkg} (lab profile)."""

        from __future__ import annotations

        from dataclasses import dataclass
        from pathlib import Path
        from typing import Optional

        from intergrax.integrations.contracts.base import IntegrationCategory
        from intergrax.integrations.providers.sqlite.bundle import (
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


def _factory_py(pkg: str, short: str) -> str:
    pascal = _pascal(short)
    title = f"Intergrax {_pascal(short)} Lab Application"
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
        from {pkg}.host.wiring import build_{short}_registry
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
            if scheduler_wiring is not None:
                scheduler = scheduler_wiring.scheduler

                @app.on_event("startup")
                async def _start_scheduler() -> None:
                    await scheduler.start()

                @app.on_event("shutdown")
                async def _stop_scheduler() -> None:
                    await scheduler.stop()
            return app
        '''
    )


def _main_py(pkg: str, short: str, env_prefix: str, port: int) -> str:
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

            host = os.environ.get("{env_prefix}BACKEND_HOST", "127.0.0.1")
            port = int(os.environ.get("{env_prefix}BACKEND_PORT", "{port}"))
            uvicorn.run(
                "{pkg}.host.main:app",
                host=host,
                port=port,
                reload=os.environ.get("{env_prefix}BACKEND_RELOAD", "").lower()
                in {{"1", "true", "yes"}},
            )


        if __name__ == "__main__":
            run()
        '''
    )


def _serving_router_py(pkg: str, short: str, route_prefix: str) -> str:
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


        class {_pascal(short)}RunRequestV1(BaseModel):
            tenant_id: str = "lab"
            user_id: str = "lab-user"
            session_id: Optional[str] = None
            message: str = Field(min_length=1)
            capability: str = Field(min_length=1)
            metadata: dict[str, Any] = Field(default_factory=dict)


        class {_pascal(short)}RunResponseV1(BaseModel):
            task_id: str
            run_id: Optional[str] = None
            state: str
            answer: str = ""
            agent_id: Optional[str] = None
            metadata: dict[str, Any] = Field(default_factory=dict)


        @dataclass
        class {_pascal(short)}RunService:
            task_runner: UnifiedTaskRunner

            @classmethod
            def from_nexus_loop(cls, nexus_loop: NexusLoop) -> {_pascal(short)}RunService:
                return cls(task_runner=UnifiedTaskRunner(nexus_loop))

            async def run_task(self, body: {_pascal(short)}RunRequestV1) -> {_pascal(short)}RunResponseV1:
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
                return {_pascal(short)}RunResponseV1(
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
        ) -> {_pascal(short)}RunService:
            service = {_pascal(short)}RunService.from_nexus_loop(nexus_loop)
            router = APIRouter(prefix=prefix, tags=["{short}"])

            @router.post("/run", response_model={_pascal(short)}RunResponseV1)
            async def run_agent(body: {_pascal(short)}RunRequestV1) -> {_pascal(short)}RunResponseV1:
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
        # Example run capability for POST {route_prefix}/run
        # DEFAULT_CAPABILITY={caps}
        '''
    )


def _readme(pkg: str, short: str, route_prefix: str, port: int, specs: list[ScaffoldAgentSpec]) -> str:
    cap = specs[0].capabilities[0] if specs and specs[0].capabilities else "echo.basic"
    agents_list = ", ".join(s.class_name for s in specs)
    return dedent(
        f'''\
        # {_pascal(short)} Application (Tier-3)

        Scaffolded lab-profile application — debug API + ``POST {route_prefix}/run``.

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

        ## Docs

        - Engine: `intergrax/applications/USAGE.md`
        - Layout: `applications/USAGE.md`
        '''
    )


def _smoke_test(pkg: str, short: str, route_prefix: str, specs: list[ScaffoldAgentSpec]) -> str:
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

    pkg = _app_slug(name)
    short = _short_id(pkg)
    env_prefix = _env_prefix(short)
    route_prefix = route_prefix or f"/v1/{short}"
    specs = resolve_agent_specs(agents)

    target = root / "applications" / pkg
    if target.exists() and not force:
        raise FileExistsError(f"Application directory already exists: {target}")

    tests_pkg = f"{pkg}_tests"

    _write(target / "__init__.py", "", force=force)
    _write(target / "manifest.py", _manifest_py(pkg, specs, route_prefix, env_prefix), force=force)
    _write(target / "README.md", _readme(pkg, short, route_prefix, port, specs), force=force)
    _write(target / ".env.example", _env_example(env_prefix, route_prefix, port, specs), force=force)

    _write(target / "host" / "__init__.py", "", force=force)
    _write(target / "host" / "settings.py", _settings_py(pkg, short, env_prefix, route_prefix, port), force=force)
    _write(target / "host" / "agent_builders.py", _agent_builders_py(specs), force=force)
    _write(target / "host" / "wiring.py", _wiring_py(pkg, short), force=force)
    _write(target / "host" / "integration_wiring.py", _integration_wiring_py(pkg, short), force=force)
    _write(target / "host" / "factory.py", _factory_py(pkg, short), force=force)
    _write(target / "host" / "main.py", _main_py(pkg, short, env_prefix, port), force=force)

    _write(target / "serving" / "__init__.py", "", force=force)
    _write(target / "serving" / "fastapi_router.py", _serving_router_py(pkg, short, route_prefix), force=force)

    _write(target / tests_pkg / "__init__.py", "", force=force)
    _write(
        target / tests_pkg / "host" / f"test_{short}_host_smoke.py",
        _smoke_test(pkg, short, route_prefix, specs),
        force=force,
    )
    _write(target / tests_pkg / "host" / "__init__.py", "", force=force)

    return target


def register_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "new-application",
        help="Create applications/<name>_application/ (lab profile, Tier-3)",
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
        help="Application profile (default: lab)",
    )
    parser.add_argument("--port", type=int, default=8091, help="Default HTTP port")
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


def run_new_application(args: argparse.Namespace) -> int:
    agent_list = [a.strip() for a in args.agents.split(",") if a.strip()]
    try:
        path = create_application(
            name=args.name,
            agents=agent_list,
            profile=args.profile,
            root=args.root.resolve(),
            route_prefix=args.route_prefix,
            port=args.port,
            force=args.force,
        )
    except (ValueError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    pkg = _app_slug(args.name)
    short = _short_id(pkg)
    port = args.port
    prefix = args.route_prefix or f"/v1/{short}"
    print(f"Created Tier-3 application at {path}")
    print(f"  Start:  uv run uvicorn {pkg}.host.main:app --host 127.0.0.1 --port {port}")
    print(f"  Test:   uv run pytest {path / f'{pkg}_tests'} -q")
    print(f"  Agents: GET http://127.0.0.1:{port}{prefix}/agents")
    print(f"  Run:    POST http://127.0.0.1:{port}{prefix}/run")
    print("  Docs:   intergrax/applications/USAGE.md · applications/USAGE.md")
    return 0
