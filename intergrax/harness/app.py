# © Artur Czarnecki. All rights reserved.

"""HarnessApplication facade — LangGraph-style authoring entry (Phase DX-2.1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, Type

from fastapi import FastAPI

from intergrax.agents.agent_contract import Agent
from intergrax.applications._shared.harness_host_runtime import HarnessHostRuntime, build_harness_host_runtime
from intergrax.applications._shared.platform_wiring import bootstrap_nexus_platform
from intergrax.applications._shared.plugin_bootstrap import attach_plugin_shutdown
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.graph_builder import AgentGraph
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.harness.application_host import ApplicationHost
from intergrax.harness.yaml_loader import merge_manifest_with_files
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.runtime.registry.agent_registry import AgentRegistry


class HarnessApplication:
    """
    Fluent builder for a Tier-3 harness host.

    Example::

        app = (
            HarnessApplication("my_lab")
            .agents(EchoAgent)
            .integrations(IntegrationProfile.lab_stack())
            .mode("balanced")
            .build_fastapi()
        )
    """

    def __init__(
        self,
        app_id: str,
        *,
        name: str | None = None,
        route_prefix: str | None = None,
        env_prefix: str | None = None,
    ) -> None:
        slug = app_id.strip().lower()
        self._app_id = slug
        self._name = name or f"{slug.replace('_', ' ').title()} Application"
        self._route_prefix = route_prefix or f"/v1/{slug}"
        self._env_prefix = env_prefix or f"{slug.upper()}_"
        self._agent_types: list[type[Agent]] = []
        self._integration_profile: IntegrationProfile | None = None
        self._environment: ApplicationEnvironmentProfile | None = None
        self._graph: AgentGraph | None = None
        self._execution_mode: ExecutionMode = ExecutionMode.BALANCED
        self._llm_profile: LLMProfile | None = None
        self._host: ApplicationHost | None = None
        self._manifest: ApplicationManifest | None = None
        self._runtime: HarnessHostRuntime | None = None

    def agents(self, *agent_types: type[Agent]) -> HarnessApplication:
        self._agent_types.extend(agent_types)
        return self

    def integrations(self, profile: IntegrationProfile) -> HarnessApplication:
        self._integration_profile = profile
        return self

    def environment(self, profile: ApplicationEnvironmentProfile) -> HarnessApplication:
        self._environment = profile
        return self

    def graph(self, graph: AgentGraph) -> HarnessApplication:
        self._graph = graph
        return self

    def mode(self, mode: str | ExecutionMode) -> HarnessApplication:
        if isinstance(mode, str):
            self._execution_mode = ExecutionMode(mode)
        else:
            self._execution_mode = mode
        return self

    def llm(self, profile: LLMProfile) -> HarnessApplication:
        self._llm_profile = profile
        return self

    def hooks(self, host: ApplicationHost) -> HarnessApplication:
        self._host = host
        return self

    def manifest(self, manifest: ApplicationManifest) -> HarnessApplication:
        self._manifest = manifest
        return self

    def from_files(
        self,
        *,
        env_path: Path | None = None,
        agents_path: Path | None = None,
    ) -> HarnessApplication:
        base = self._build_manifest()
        self._manifest = merge_manifest_with_files(base, env_path=env_path, agents_path=agents_path)
        if self._manifest.environment is not None:
            self._environment = self._manifest.environment
        return self

    def _build_manifest(self) -> ApplicationManifest:
        if self._manifest is not None:
            return self._manifest
        bindings = [AgentBinding.mount(agent_type) for agent_type in self._agent_types]
        return ApplicationManifest.lab(
            app_id=self._app_id,
            name=self._name,
            route_prefix=self._route_prefix,
            env_prefix=self._env_prefix,
            agents=bindings,
            environment=self._resolve_environment(),
        )

    def _resolve_environment(self) -> ApplicationEnvironmentProfile:
        if self._environment is not None:
            env = self._environment
        else:
            env = ApplicationEnvironmentProfile.lab_defaults(profile_id=f"{self._app_id}.harness")
        if self._integration_profile is not None:
            env = env.model_copy(update={"integration_profile": self._integration_profile})
        if self._graph is not None:
            graph_spec = self._graph.build()
            env = env.model_copy(update={"graph_spec": graph_spec})
        if self._llm_profile is not None:
            env = env.model_copy(update={"llm_profile": self._llm_profile})
        env = env.model_copy(update={"execution_mode": self._execution_mode})
        return env

    def build_runtime(
        self,
        *,
        settings: Any = None,
        use_in_memory_trace: bool = True,
        trace_db_path: Path | None = None,
        runtime_events_db_path: Path | None = None,
    ) -> HarnessHostRuntime:
        manifest = self._build_manifest()
        environment = manifest.environment or self._resolve_environment()
        self._runtime = build_harness_host_runtime(
            manifest,
            environment,
            settings=settings,
            trace_db_path=trace_db_path,
            runtime_events_db_path=runtime_events_db_path,
            use_in_memory_trace=use_in_memory_trace,
        )
        _ = self._host
        return self._runtime

    def registry(self) -> AgentRegistry:
        if self._runtime is None:
            self.build_runtime()
        assert self._runtime is not None
        return self._runtime.registry

    def build_fastapi(
        self,
        *,
        settings: Any = None,
        mount_routes: bool = True,
    ) -> FastAPI:
        """Build a lab-style FastAPI app with run + agents routes."""
        runtime = self.build_runtime(settings=settings)
        from intergrax.harness.lab_fastapi import create_lab_fastapi_from_runtime

        return create_lab_fastapi_from_runtime(
            runtime,
            route_prefix=self._route_prefix,
            mount_routes=mount_routes,
        )

    def serve(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8090,
        settings: Any = None,
    ) -> None:
        """Run uvicorn on :meth:`build_fastapi` (local development)."""
        import uvicorn

        app = self.build_fastapi(settings=settings)
        uvicorn.run(app, host=host, port=port)
