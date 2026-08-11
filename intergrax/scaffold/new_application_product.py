# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tier-3 application scaffold templates — product profile (Phase N.4)."""

from __future__ import annotations

from textwrap import dedent

from intergrax.scaffold.agent_catalog import ScaffoldAgentSpec
from intergrax.scaffold.application_names import ScaffoldApplicationNames


def manifest_py(names: ScaffoldApplicationNames, specs: list[ScaffoldAgentSpec]) -> str:
    pkg = names.pkg
    short = names.short
    route_prefix = names.route_prefix
    env_prefix_value = names.env_prefix
    imports = "\n".join(f"from {s.module} import {s.class_name}" for s in specs)
    factory_imports = "\n".join(
        f"from {pkg}.host.agent_factories import build_{short}_{s.slug}_from_context"
        for s in specs
    )
    mounts = []
    for i, s in enumerate(specs):
        caps = ", ".join(repr(c) for c in s.capabilities)
        cap_arg = f", capabilities=[{caps}]" if s.capabilities else ""
        default_arg = ", default=True" if i == 0 else ""
        mounts.append(
            f"        AgentBinding.mount("
            f"{s.class_name}, "
            f"factory=build_{short}_{s.slug}_from_context"
            f"{cap_arg}{default_arg}),"
        )
    mounts_block = "\n".join(mounts)
    first_cap = specs[0].capabilities[0] if specs and specs[0].capabilities else "echo.basic"
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        """Declarative agent roster for {pkg} (product profile)."""

        from __future__ import annotations

        from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
        from intergrax.integrations.registry.profile import IntegrationProfile
        {imports}
        {factory_imports}


        def _resolve_integration_profile() -> IntegrationProfile:
            import json
            import os

            raw = os.environ.get("INTERGRAX_INTEGRATION_PROFILE_JSON", "").strip()
            if raw:
                return IntegrationProfile.model_validate_json(raw)
            return IntegrationProfile.legal_product()


        def build_{short}_manifest() -> ApplicationManifest:
            return ApplicationManifest.product(
                app_id="{short}",
                name="{names.display} API",
                route_prefix="{route_prefix}",
                env_prefix="{env_prefix_value}",
                default_capability="{first_cap}",
                integration_profile=_resolve_integration_profile(),
                agents=[
        {mounts_block}
                ],
                description="Scaffolded Tier-3 product host (Phase N.4)",
            )


        APPLICATION_MANIFEST = build_{short}_manifest()
        '''
    )


def settings_py(names: ScaffoldApplicationNames) -> str:
    pascal = names.pascal
    pkg = names.pkg
    env_prefix_value = names.env_prefix
    route_prefix = names.route_prefix
    port = names.port
    short = names.short
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        from __future__ import annotations

        import json
        import os
        from dataclasses import dataclass, field
        from typing import ClassVar, FrozenSet, Literal, Mapping, Optional

        from intergrax.applications.contracts.settings import EnvReader, IntergraxApplicationSettingsBase
        from intergrax.fastapi_core.auth.api_key import ApiKeyIdentity
        from intergrax.fastapi_core.config import ApiEnvironment

        {pascal}IdentitySource = Literal["body_or_context", "context_only"]


        def _parse_api_key_map(raw: Optional[str]) -> Mapping[str, ApiKeyIdentity]:
            if not raw or not raw.strip():
                return {{}}
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("{env_prefix_value}BACKEND_API_KEYS_JSON must be a JSON object.")
            out: dict[str, ApiKeyIdentity] = {{}}
            for key, val in data.items():
                if not isinstance(val, dict):
                    raise ValueError(f"Identity for key {{key!r}} must be an object.")
                tenant = val.get("tenant_id")
                if not tenant or not isinstance(tenant, str):
                    raise ValueError(f"tenant_id required for API key {{key!r}}.")
                user_id = val.get("user_id")
                scopes = val.get("scopes", ("*",))
                if isinstance(scopes, list):
                    scopes = tuple(str(s) for s in scopes)
                elif isinstance(scopes, tuple):
                    scopes = tuple(str(s) for s in scopes)
                else:
                    scopes = ("*",)
                out[str(key)] = ApiKeyIdentity(
                    tenant_id=str(tenant),
                    user_id=str(user_id) if user_id is not None else None,
                    scopes=scopes,
                )
            return out


        @dataclass(frozen=True, kw_only=True)
        class {pascal}BackendSettings(IntergraxApplicationSettingsBase):
            """Environment for {pkg} (scaffolded product profile)."""

            env_prefix: ClassVar[str] = "{env_prefix_value}"
            route_prefix: str = "{route_prefix}"
            backend_port: int = {port}
            include_scheduler: bool = False
            include_queue_worker: bool = False
            default_agent_id: str = "echo"
            identity_source: {pascal}IdentitySource = "body_or_context"
            cors_allow_origins: FrozenSet[str] = field(default_factory=frozenset)
            allowed_hosts: FrozenSet[str] = field(default_factory=frozenset)
            openapi_enabled_override: Optional[bool] = None
            api_keys_map: Mapping[str, ApiKeyIdentity] = field(default_factory=dict)
            interaction_execute_default: bool = True

            # ------------------------------------------------------------------
            # Application-specific settings
            # Add your own env-backed fields here.
            # ------------------------------------------------------------------

            @classmethod
            def _load_app_env(cls, env: EnvReader) -> dict[str, object]:
                env_raw = (
                    env.optional_str("BACKEND_ENV")
                    or (os.environ.get("INTERGRAX_ENV") or "dev").strip().lower()
                )
                environment = ApiEnvironment(env_raw)

                agent_id = env.str("DEFAULT_AGENT_ID", default="echo") or "echo"

                id_src_env = env.optional_str("IDENTITY_SOURCE")
                if id_src_env in {{"body_or_context", "context_only"}}:
                    identity_source = id_src_env
                else:
                    identity_source = (
                        "context_only" if environment == ApiEnvironment.PROD else "body_or_context"
                    )

                cors = env.csv_set("BACKEND_CORS_ORIGINS")
                hosts = env.csv_set("BACKEND_ALLOWED_HOSTS")

                openapi_override: Optional[bool] = None
                if env.raw("BACKEND_OPENAPI") is not None:
                    openapi_override = env.bool("BACKEND_OPENAPI")

                keys: Mapping[str, ApiKeyIdentity] = {{}}
                bootstrap_key = env.str("BACKEND_BOOTSTRAP_API_KEY", default="")
                if bootstrap_key:
                    tenant = env.str("BACKEND_BOOTSTRAP_TENANT_ID", default="")
                    user = env.str("BACKEND_BOOTSTRAP_USER_ID", default="")
                    if not tenant or not user:
                        raise ValueError(
                            "When {env_prefix_value}BACKEND_BOOTSTRAP_API_KEY is set, "
                            "{env_prefix_value}BACKEND_BOOTSTRAP_TENANT_ID and "
                            "{env_prefix_value}BACKEND_BOOTSTRAP_USER_ID are required."
                        )
                    keys = {{
                        bootstrap_key: ApiKeyIdentity(
                            tenant_id=tenant,
                            user_id=user,
                            scopes=("*",),
                        )
                    }}
                json_keys = env.str("BACKEND_API_KEYS_JSON", default="")
                if json_keys:
                    if keys:
                        raise ValueError(
                            "Use either {env_prefix_value}BACKEND_BOOTSTRAP_API_KEY or "
                            "{env_prefix_value}BACKEND_API_KEYS_JSON, not both."
                        )
                    keys = _parse_api_key_map(json_keys)

                if environment == ApiEnvironment.PROD and identity_source != "context_only":
                    raise ValueError(
                        "{env_prefix_value}BACKEND_ENV=prod requires "
                        "{env_prefix_value}IDENTITY_SOURCE=context_only (or omit to default)."
                    )
                if environment == ApiEnvironment.PROD and not env.bool(
                    "BACKEND_ALLOW_UNAUTHENTICATED", default=False
                ):
                    if not keys:
                        raise ValueError(
                            "Production {short} backend requires API keys: set "
                            "{env_prefix_value}BACKEND_BOOTSTRAP_API_KEY (+ tenant/user) or "
                            "{env_prefix_value}BACKEND_API_KEYS_JSON. "
                            "For local disaster debugging only, set "
                            "{env_prefix_value}BACKEND_ALLOW_UNAUTHENTICATED=true."
                        )

                return {{
                    "default_agent_id": agent_id,
                    "identity_source": identity_source,
                    "cors_allow_origins": cors,
                    "allowed_hosts": hosts,
                    "openapi_enabled_override": openapi_override,
                    "api_keys_map": keys,
                    "interaction_execute_default": env.bool(
                        "INTERACTION_EXECUTE_DEFAULT",
                        default=cls._field_default("interaction_execute_default"),  # type: ignore[arg-type]
                    ),
                }}
        '''
    )


def agent_factories_py(names: ScaffoldApplicationNames, specs: list[ScaffoldAgentSpec]) -> str:
    pkg = names.pkg
    short = names.short
    pascal = names.pascal
    imports = "\n".join(f"from {s.module} import {s.class_name}" for s in specs)
    builders_const = names.builders_const
    factories = []
    for s in specs:
        factories.append(
            dedent(
                f"""
                def build_{short}_{s.slug}_from_context(
                    ctx: ApplicationBuildContext,
                    binding: AgentBinding,
                ) -> {s.class_name}:
                    _ = ctx, binding
                    factory = {builders_const}.get({s.class_name})
                    if factory is None:
                        raise ValueError(f"No builder registered for {{binding.import_path!r}}")
                    return factory(ctx, binding)
                """
            ).strip()
        )
    factories_block = "\n\n\n".join(factories)
    return (
        dedent(
            f"""
            # © Artur Czarnecki. All rights reserved.

            from __future__ import annotations

            from intergrax.applications.contracts.build_context import ApplicationBuildContext
            from intergrax.applications.contracts.manifest import AgentBinding
            {imports}
            from {pkg}.host.agent_builders import {builders_const}


            """
        ).strip()
        + "\n\n\n"
        + factories_block
        + "\n"
    )


def agent_builders_py(names: ScaffoldApplicationNames, specs: list[ScaffoldAgentSpec]) -> str:
    from intergrax.scaffold.new_application import _agent_builders_py

    return _agent_builders_py(names, specs)


def wiring_py(names: ScaffoldApplicationNames) -> str:
    pkg = names.pkg
    short = names.short
    pascal = names.pascal
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        from __future__ import annotations

        from intergrax.applications._shared.environment_wiring import wire_application_environment
        from intergrax.applications._shared.wiring import build_application_registry
        from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
        from intergrax.applications.contracts.manifest import ApplicationManifest
        from intergrax.runtime.registry.agent_registry import AgentRegistry
        from {pkg}.host.agent_builders import {names.builders_const}
        from {pkg}.host.environment_profile import build_{short}_environment_profile
        from {pkg}.host.settings import {pascal}BackendSettings
        from {pkg}.manifest import APPLICATION_MANIFEST


        def build_{short}_manifest(settings: {pascal}BackendSettings) -> ApplicationManifest:
            binding = APPLICATION_MANIFEST.agents[0].model_copy(
                update={{"contract_id": settings.default_agent_id}}
            )
            agents = [binding, *APPLICATION_MANIFEST.agents[1:]]
            return APPLICATION_MANIFEST.model_copy(update={{"agents": agents}})


        def build_{short}_registry(
            settings: {pascal}BackendSettings | None = None,
        ) -> AgentRegistry:
            settings = settings or {pascal}BackendSettings.from_env()
            manifest = build_{short}_manifest(settings)
            env = manifest.environment or build_{short}_environment_profile(settings)
            if manifest.environment is None:
                manifest = manifest.model_copy(update={{"environment": env}})
            env_wiring = wire_application_environment(manifest, env, settings=settings)
            return build_application_registry(
                manifest,
                env_wiring.build_context,
                builders={names.builders_const},
            )
        '''
    )


def environment_profile_py(names: ScaffoldApplicationNames) -> str:
    pkg = names.pkg
    short = names.short
    pascal = names.pascal
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        """Tier-3 environment profile for {pkg} (Phase H-APP.5.5, DX-5.5)."""

        from __future__ import annotations

        from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
        from intergrax.fastapi_core.config import ApiEnvironment
        from intergrax.integrations.core.binding import IntegrationBinding
        from intergrax.integrations.registry.catalog_manifests import OTEL
        from {pkg}.host.settings import {pascal}BackendSettings


        def build_{short}_environment_profile(
            settings: {pascal}BackendSettings,
        ) -> ApplicationEnvironmentProfile:
            profile = ApplicationEnvironmentProfile.product_defaults(
                skill_bundles=["harness"],
                profile_id="{short}.product",
            )
            profile.observability_profile.otel_enabled = True
            profile.observability_profile.debug_surface_override = True
            otel_backend = IntegrationBinding.from_manifest(OTEL)
            profile.integration_profile = profile.integration_profile.model_copy(
                update={{
                    "observability_backend": otel_backend,
                    "options": {{**profile.integration_profile.options, OTEL.slug: {{}}}},
                }},
            )
            if settings.environment == ApiEnvironment.DEV:
                profile = profile.model_copy(
                    update={{
                        "reliability_profile": profile.reliability_profile.model_copy(
                            update={{"middleware_hook_timeout_seconds": 2.0}},
                        ),
                    }},
                )
            return profile.with_reference_host_platform_defaults()
        '''
    )


def tool_wiring_py(names: ScaffoldApplicationNames) -> str:
    from intergrax.scaffold.application_extension_templates import tool_wiring_local_extension_block

    pkg = names.pkg
    short = names.short
    local_block = tool_wiring_local_extension_block(names)
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        """Tool catalog wiring for {pkg} (Phase O.8)."""

        from __future__ import annotations

        from intergrax.applications._shared.tool_wiring import ApplicationToolWiring, build_application_tool_wiring
        from intergrax.integrations.registry.profile import IntegrationProfile
        from intergrax.tools.registry.profile import ToolProfile

        {local_block}

        def wire_{short}_tools(
            *,
            integration_profile: IntegrationProfile | None = None,
        ) -> ApplicationToolWiring:
            profile = ToolProfile(
                enabled=[
                    "rag.retrieve",
                    "rag.ingest_document",
                    "rag.list_collections",
                    "websearch.query",
                    "websearch.read_url",
                    "websearch.fetch_batch",
                    "local_prefix_echo.ping",
                ],
            )
            return build_application_tool_wiring(
                profile,
                integration_profile=integration_profile or IntegrationProfile(),
                extras={{"echo_prefix": "{short}"}},
            )
        '''
    )


def integration_wiring_py(names: ScaffoldApplicationNames) -> str:
    short = names.short
    pascal = names.pascal
    pkg = names.pkg
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        """Observability wiring for {pkg} (product profile)."""

        from __future__ import annotations

        from pathlib import Path

        from intergrax.applications._shared.integration_wiring import bootstrap_application_integration_catalog
        from intergrax.integrations.registry.profile import IntegrationProfile
        from intergrax.runtime.nexus.observability_wiring import NexusObservabilityStores, wire_nexus_observability


        def wire_{short}_integrations(
            *,
            trace_db_path: Path | None = None,
            runtime_events_db_path: Path | None = None,
            integration_profile: IntegrationProfile | None = None,
        ) -> NexusObservabilityStores:
            bootstrap_application_integration_catalog(integration_preset="full")
            return wire_nexus_observability(
                trace_db_path=trace_db_path,
                runtime_events_db_path=runtime_events_db_path,
                integration_profile=integration_profile or IntegrationProfile(),
            )
        '''
    )


def factory_py(names: ScaffoldApplicationNames) -> str:
    pkg = names.pkg
    short = names.short
    pascal = names.pascal
    title = f"Intergrax {names.display} API"
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        """Assemble FastAPI Core + product routes for {pkg}."""

        from __future__ import annotations

        from pathlib import Path
        from typing import Optional

        from fastapi import FastAPI
        from starlette.middleware.cors import CORSMiddleware

        from intergrax.applications._shared.workspace_cleanup_wiring import (
            apply_factory_lifespans,
            build_factory_lifespans,
        )
        from intergrax.applications._shared.interaction_wiring import wire_interaction_intake_service
        from intergrax.fastapi_core.app_factory import create_app
        from intergrax.fastapi_core.auth.api_key import ApiKeyConfig
        from intergrax.fastapi_core.config import ApiConfig
        from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
        from intergrax.applications._shared.platform_wiring import bootstrap_nexus_platform
        from intergrax.applications._shared.plugin_bootstrap import attach_plugin_shutdown
        from intergrax.runtime.interactions.router import create_interaction_intake_router
        from intergrax.applications._shared.harness_task_routes import mount_harness_task_routes
        from intergrax.applications._shared.task_control_wiring import (
            build_reliability_task_enricher,
            build_task_runner_with_enricher,
        )
        from intergrax.debug.store import open_default_task_checkpoint_persistence
        from intergrax.runtime.long_running.wiring import wire_long_running_scheduler
        from {pkg}.host.settings import {pascal}BackendSettings
        from {pkg}.host.environment_profile import build_{short}_environment_profile
        from {pkg}.manifest import build_{short}_manifest
        from {pkg}.serving.fastapi_router import mount_{short}_routes


        def create_{short}_backend_app(
            *,
            settings: Optional[{pascal}BackendSettings] = None,
            trace_db_path: Path | None = None,
            runtime_events_db_path: Path | None = None,
            checkpoints_db_path: Path | None = None,
        ) -> FastAPI:
            settings = settings or {pascal}BackendSettings.from_env()
            api_key_config = ApiKeyConfig(keys=settings.api_keys_map) if settings.api_keys_map else None

            manifest = build_{short}_manifest()
            env = manifest.environment or build_{short}_environment_profile(settings)
            runtime = build_harness_host_runtime(
                manifest,
                env,
                settings=settings,
                trace_db_path=trace_db_path,
                runtime_events_db_path=runtime_events_db_path,
                checkpoints_db_path=checkpoints_db_path,
                use_in_memory_trace=trace_db_path is None,
            )
            nexus_loop = runtime.nexus_loop
            registry = runtime.registry
            platform = bootstrap_nexus_platform(
                nexus_loop,
                trace_store=runtime.observability.trace_store,  # type: ignore[arg-type]
            )
            checkpoint_store = open_default_task_checkpoint_persistence(db_path=checkpoints_db_path)
            task_enricher = build_reliability_task_enricher(env)
            task_runner = build_task_runner_with_enricher(nexus_loop, task_enricher)
            scheduler_wiring = wire_long_running_scheduler(
                checkpoint_store=checkpoint_store,
                task_runner=task_runner,
                notification_adapter=None,
                poll_interval_seconds=settings.scheduler_poll_seconds,
                enabled=settings.include_scheduler,
            )
            interaction_service = wire_interaction_intake_service(
                nexus_loop,
                interaction_surface=settings.interaction_surface,
                task_enricher=task_enricher,
            )

            api_cfg = ApiConfig(
                environment=settings.environment,
                api_prefix="/v1",
                cors_allow_origins=settings.cors_allow_origins,
                allowed_hosts=settings.allowed_hosts,
                api_key_config=api_key_config,
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

            mount_{short}_routes(
                app,
                nexus_loop=nexus_loop,
                prefix=settings.route_prefix,
                default_agent_id=settings.default_agent_id,
            )

            if settings.include_task_control:
                mount_harness_task_routes(
                    app,
                    task_runner=task_runner,
                    checkpoint_store=checkpoint_store,
                    prefix=settings.task_control_route_prefix,
                    task_enricher=task_enricher,
                )

            if settings.include_interaction_routes:
                app.include_router(
                    create_interaction_intake_router(
                        interaction_service,
                        execute_default=settings.interaction_execute_default,
                    ),
                    prefix=settings.interaction_route_prefix,
                )

            app.title = "{title}" if settings.environment.value == "prod" else "{title} (dev)"

            scheduler = scheduler_wiring.scheduler if scheduler_wiring is not None else None
            if settings.include_mcp:
                from intergrax.applications._shared.mcp_import_guard import load_mcp_coupling

                couple_fastapi_with_mcp = load_mcp_coupling()
                from {pkg}.mcp.server import build_{short}_mcp_server

                mcp = build_{short}_mcp_server(
                    nexus_loop=nexus_loop,
                    route_prefix=settings.route_prefix,
                    tool_registry=runtime.env_wiring.tool_wiring.registry,
                )
                extra_lifespans = build_factory_lifespans(
                    runtime,
                    schedulers=[scheduler] if scheduler else None,
                )
                app = couple_fastapi_with_mcp(
                    app,
                    mcp,
                    mount_path=settings.mcp_mount_path,
                    extra_lifespans=extra_lifespans,
                )
            else:
                apply_factory_lifespans(app, runtime, schedulers=[scheduler] if scheduler else None)

            attach_plugin_shutdown(app, platform.shutdown_callbacks)
            return app
        '''
    )


def main_py(names: ScaffoldApplicationNames) -> str:
    pkg = names.pkg
    short = names.short
    env_prefix_value = names.env_prefix
    port = names.port
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        import os

        from dotenv import load_dotenv

        from {pkg}.host.factory import create_{short}_backend_app

        load_dotenv()

        app = create_{short}_backend_app()


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


def schemas_py(names: ScaffoldApplicationNames) -> str:
    pascal = names.pascal
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        from __future__ import annotations

        from typing import Any, Optional

        from pydantic import BaseModel, Field


        class {pascal}RunRequestV1(BaseModel):
            tenant_id: str = "default"
            user_id: str = "default-user"
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
        '''
    )


def serving_router_py(names: ScaffoldApplicationNames, specs: list[ScaffoldAgentSpec]) -> str:
    short = names.short
    pascal = names.pascal
    pkg = names.pkg
    route_prefix = names.route_prefix
    default_cap = specs[0].capabilities[0] if specs and specs[0].capabilities else "echo.basic"
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        from __future__ import annotations

        from dataclasses import dataclass

        from fastapi import APIRouter, FastAPI, HTTPException, status

        from intergrax.runtime.nexus.nexus_loop import NexusLoop
        from intergrax.runtime.task.task import Task, TaskContext
        from intergrax.runtime.task.task_run_bridge import new_run_id
        from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner
        from {pkg}.serving.schemas import {pascal}RunRequestV1, {pascal}RunResponseV1


        @dataclass
        class {pascal}RunService:
            task_runner: UnifiedTaskRunner
            default_agent_id: str

            @classmethod
            def from_nexus_loop(
                cls,
                nexus_loop: NexusLoop,
                *,
                default_agent_id: str,
            ) -> {pascal}RunService:
                return cls(
                    task_runner=UnifiedTaskRunner(nexus_loop),
                    default_agent_id=default_agent_id,
                )

            async def run_task(self, body: {pascal}RunRequestV1) -> {pascal}RunResponseV1:
                run_id = new_run_id()
                task = Task(
                    task_id=run_id,
                    tenant_id=body.tenant_id,
                    user_id=body.user_id,
                    session_id=body.session_id,
                    agent_id=self.default_agent_id,
                    message=body.message,
                    context=TaskContext(capability=body.capability or "{default_cap}"),
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
            default_agent_id: str = "echo",
        ) -> {pascal}RunService:
            service = {pascal}RunService.from_nexus_loop(
                nexus_loop,
                default_agent_id=default_agent_id,
            )
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


def mcp_server_py(names: ScaffoldApplicationNames, specs: list[ScaffoldAgentSpec]) -> str:
    from intergrax.scaffold.new_application import _mcp_server_py

    return _mcp_server_py(names, specs)


def env_example(
    env_prefix: str,
    route_prefix: str,
    port: int,
    specs: list[ScaffoldAgentSpec],
) -> str:
    caps = specs[0].capabilities[0] if specs and specs[0].capabilities else "echo.basic"
    return dedent(
        f'''\
        # {env_prefix}* — copy to .env (gitignored) in this application directory.
        INTERGRAX_ENV=dev
        {env_prefix}BACKEND_ENV=dev
        {env_prefix}BACKEND_HOST=127.0.0.1
        {env_prefix}BACKEND_PORT={port}
        {env_prefix}ROUTE_PREFIX={route_prefix}
        {env_prefix}DEFAULT_AGENT_ID=echo
        {env_prefix}IDENTITY_SOURCE=body_or_context
        {env_prefix}INCLUDE_MCP=false
        {env_prefix}MCP_MOUNT_PATH=/mcp
        {env_prefix}INCLUDE_INTERACTIONS=true
        {env_prefix}INTERACTION_ROUTE_PREFIX=/v1/interactions
        {env_prefix}INTERACTION_SURFACE=auto
        {env_prefix}INCLUDE_SCHEDULER=false
        {env_prefix}INCLUDE_TASK_CONTROL=true
        {env_prefix}INCLUDE_QUEUE_WORKER=true
        {env_prefix}TASK_CONTROL_ROUTE_PREFIX=/v1/tasks
        # Optional dev API key (prod requires keys or ALLOW_UNAUTHENTICATED=true):
        # {env_prefix}BACKEND_BOOTSTRAP_API_KEY=dev-key
        # {env_prefix}BACKEND_BOOTSTRAP_TENANT_ID=dev-tenant
        # {env_prefix}BACKEND_BOOTSTRAP_USER_ID=dev-user
        # Example capability for POST {route_prefix}/run
        # DEFAULT_CAPABILITY={caps}
        '''
    )


def readme(names: ScaffoldApplicationNames, specs: list[ScaffoldAgentSpec]) -> str:
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
        # {display} API (Tier-3)

        Scaffolded **product** profile — FastAPI Core (`/health`, `/v1/*`) + ``POST {route_prefix}/run``.

        **Architecture:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · **Plan:** [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md)

        **Build & deploy:** [`docs/BUILD_AND_DEPLOY.md`](docs/BUILD_AND_DEPLOY.md)

        ## Three-command quickstart

        From **repository root**:

        ```bash
        uv run pytest applications/{pkg}/tests -q
        cp applications/{pkg}/.env.example applications/{pkg}/.env
        uv run uvicorn {pkg}.host.main:app --host 127.0.0.1 --port {port}
        applications/{pkg}/docker/build-docker.sh
        ```

        ## Agents

        {agents_list}

        ## HTTP

        ```bash
        curl -s http://127.0.0.1:{port}/health
        curl -s http://127.0.0.1:{port}{route_prefix}/agents
        curl -s -X POST http://127.0.0.1:{port}{route_prefix}/run \\
          -H "Content-Type: application/json" \\
          -d '{{"message":"hello","capability":"{cap}"}}'
        ```

        ## MCP

        Default ``/mcp`` — ``list_agents``, ``run_agent``. Configure ``{env_prefix_value}INCLUDE_MCP``, ``{env_prefix_value}MCP_MOUNT_PATH``.

        ## Extending beyond the generic product skeleton

        This host uses ``POST {route_prefix}/run`` and ``/agents``. For chat-style routes,
        API-key auth, and domain-specific serving (like Legal), copy patterns from
        ``applications/legal_application/serving/`` after the scaffold — do not put agent logic here.

        ## Docs

        - Engine: `intergrax/applications/USAGE.md`
        - Layout: `applications/USAGE.md`
        - Full stack (agent + app): `python -m intergrax.scaffold new-stack <slug>`
        '''
    )


def smoke_test(names: ScaffoldApplicationNames, specs: list[ScaffoldAgentSpec]) -> str:
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

        from {pkg}.host.factory import create_{short}_backend_app

        pytestmark = [pytest.mark.unit]

        _PREFIX = "{route_prefix}"


        def test_{short}_backend_health():
            client = TestClient(create_{short}_backend_app())
            response = client.get("/health")
            assert response.status_code == 200


        def test_{short}_backend_lists_agents():
            client = TestClient(create_{short}_backend_app())
            response = client.get(f"{{_PREFIX}}/agents")
            assert response.status_code == 200
            assert "agents" in response.json()


        def test_{short}_backend_run():
            client = TestClient(create_{short}_backend_app())
            response = client.post(
                f"{{_PREFIX}}/run",
                json={{"message": "hello", "capability": "{cap}"}},
            )
            assert response.status_code == 200
            assert response.json().get("state") == "completed"
        '''
    )
