# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import os
from dataclasses import dataclass

from intergrax.fastapi_core.config import ApiEnvironment


def _env_bool(name: str, *, default: bool) -> bool:
    raw = (os.getenv(name) or ("true" if default else "false")).strip().lower()
    return raw not in {"0", "false", "no"}


@dataclass(frozen=True)
class IntergraxAssistantApplicationSettings:
    """Environment for intergrax_assistant_application (harness chat lab)."""

    environment: ApiEnvironment = ApiEnvironment.DEV
    route_prefix: str = "/v1/intergrax_assistant"
    backend_host: str = "127.0.0.1"
    backend_port: int = 8096
    include_interaction_routes: bool = True
    interaction_route_prefix: str = "/v1/interactions"
    include_scheduler: bool = True
    include_task_control: bool = True
    task_control_route_prefix: str = "/v1/harness/tasks"
    scheduler_poll_seconds: float | None = None
    interaction_surface: str = "auto"
    include_mcp: bool = False
    mcp_mount_path: str = "/mcp"
    llm_env_prefix: str = "INTERGRAX_LLM"
    max_delegation_depth: int = 4
    discover_plugins: bool = False
    include_echo: bool = False
    include_legal: bool = False
    include_research: bool = False
    engine_planner: bool = False

    @classmethod
    def from_env(cls) -> IntergraxAssistantApplicationSettings:
        env_raw = (os.getenv("INTERGRAX_ENV") or "dev").strip().lower()
        environment = ApiEnvironment.PROD if env_raw == "prod" else ApiEnvironment.DEV
        prefix = (os.getenv("INTERGRAX_ASSISTANT_ROUTE_PREFIX") or "/v1/intergrax_assistant").strip() or "/v1/intergrax_assistant"
        host = (os.getenv("INTERGRAX_ASSISTANT_BACKEND_HOST") or "127.0.0.1").strip()
        port_raw = (os.getenv("INTERGRAX_ASSISTANT_BACKEND_PORT") or "8096").strip()
        include_interactions = _env_bool("INTERGRAX_ASSISTANT_INCLUDE_INTERACTIONS", default=True)
        include_scheduler = _env_bool("INTERGRAX_ASSISTANT_INCLUDE_SCHEDULER", default=True)
        include_task_control = _env_bool(
            "INTERGRAX_ASSISTANT_INCLUDE_TASK_CONTROL", default=True
        )
        task_control_prefix = (
            os.getenv("INTERGRAX_ASSISTANT_TASK_CONTROL_ROUTE_PREFIX")
            or "/v1/harness/tasks"
        ).strip() or "/v1/harness/tasks"
        interaction_prefix = (
            os.getenv("INTERGRAX_ASSISTANT_INTERACTION_ROUTE_PREFIX") or "/v1/interactions"
        ).strip() or "/v1/interactions"
        poll_raw = (os.getenv("INTERGRAX_SCHEDULER_POLL_SECONDS") or "").strip()
        scheduler_poll = float(poll_raw) if poll_raw else None
        interaction_surface = (
            os.getenv("INTERGRAX_ASSISTANT_INTERACTION_SURFACE") or "auto"
        ).strip().lower() or "auto"
        include_mcp = _env_bool("INTERGRAX_ASSISTANT_INCLUDE_MCP", default=False)
        mcp_mount = (os.getenv("INTERGRAX_ASSISTANT_MCP_MOUNT_PATH") or "/mcp").strip() or "/mcp"
        llm_prefix = (os.getenv("INTERGRAX_ASSISTANT_LLM_ENV_PREFIX") or "INTERGRAX_LLM").strip() or "INTERGRAX_LLM"
        depth_raw = (os.getenv("INTERGRAX_ASSISTANT_MAX_DELEGATION_DEPTH") or "4").strip()
        return cls(
            environment=environment,
            route_prefix=prefix,
            backend_host=host,
            backend_port=int(port_raw),
            include_interaction_routes=include_interactions,
            interaction_route_prefix=interaction_prefix,
            include_scheduler=include_scheduler,
            include_task_control=include_task_control,
            task_control_route_prefix=task_control_prefix,
            scheduler_poll_seconds=scheduler_poll,
            interaction_surface=interaction_surface,
            include_mcp=include_mcp,
            mcp_mount_path=mcp_mount,
            llm_env_prefix=llm_prefix,
            max_delegation_depth=max(1, min(32, int(depth_raw))),
            discover_plugins=_env_bool("INTERGRAX_DISCOVER_PLUGINS", default=False),
            include_echo=_env_bool("INTERGRAX_ASSISTANT_INCLUDE_ECHO", default=False),
            include_legal=_env_bool("INTERGRAX_ASSISTANT_INCLUDE_LEGAL", default=False),
            include_research=_env_bool("INTERGRAX_ASSISTANT_INCLUDE_RESEARCH", default=False),
            engine_planner=_env_bool("INTERGRAX_ASSISTANT_ENGINE_PLANNER", default=False),
        )
