# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import os
from dataclasses import dataclass

from intergrax.fastapi_core.config import ApiEnvironment


@dataclass(frozen=True)
class PocTemplateApplicationSettings:
    """Environment for poc_template_application (scaffolded lab profile)."""

    environment: ApiEnvironment = ApiEnvironment.DEV
    route_prefix: str = "/v1/poc_template"
    backend_host: str = "127.0.0.1"
    backend_port: int = 8095
    include_interaction_routes: bool = True
    interaction_route_prefix: str = "/v1/interactions"
    include_scheduler: bool = True
    scheduler_poll_seconds: float | None = None
    interaction_surface: str = "auto"
    include_mcp: bool = False
    mcp_mount_path: str = "/mcp"
    include_task_control: bool = True
    include_queue_worker: bool = False
    task_control_route_prefix: str = "/v1/tasks"

    @classmethod
    def from_env(cls) -> PocTemplateApplicationSettings:
        env_raw = (os.getenv("INTERGRAX_ENV") or "dev").strip().lower()
        environment = ApiEnvironment.PROD if env_raw == "prod" else ApiEnvironment.DEV
        prefix = (os.getenv("POC_TEMPLATE_ROUTE_PREFIX") or "/v1/poc_template").strip() or "/v1/poc_template"
        host = (os.getenv("POC_TEMPLATE_BACKEND_HOST") or "127.0.0.1").strip()
        port_raw = (os.getenv("POC_TEMPLATE_BACKEND_PORT") or "8095").strip()
        include_interactions = (
            os.getenv("POC_TEMPLATE_INCLUDE_INTERACTIONS") or "true"
        ).strip().lower() not in {"0", "false", "no"}
        include_scheduler = (
            os.getenv("POC_TEMPLATE_INCLUDE_SCHEDULER") or "true"
        ).strip().lower() not in {"0", "false", "no"}
        interaction_prefix = (
            os.getenv("POC_TEMPLATE_INTERACTION_ROUTE_PREFIX") or "/v1/interactions"
        ).strip() or "/v1/interactions"
        poll_raw = (os.getenv("INTERGRAX_SCHEDULER_POLL_SECONDS") or "").strip()
        scheduler_poll = float(poll_raw) if poll_raw else None
        interaction_surface = (
            os.getenv("POC_TEMPLATE_INTERACTION_SURFACE") or "auto"
        ).strip().lower() or "auto"
        include_mcp = (os.getenv("POC_TEMPLATE_INCLUDE_MCP") or "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        mcp_mount = (os.getenv("POC_TEMPLATE_MCP_MOUNT_PATH") or "/mcp").strip() or "/mcp"
        include_task_control = (
            os.getenv("POC_TEMPLATE_INCLUDE_TASK_CONTROL") or "true"
        ).strip().lower() not in {"0", "false", "no"}
        include_queue_worker = (
            os.getenv("POC_TEMPLATE_INCLUDE_QUEUE_WORKER") or "false"
        ).strip().lower() in {"1", "true", "yes", "on"}
        task_control_prefix = (
            os.getenv("POC_TEMPLATE_TASK_CONTROL_ROUTE_PREFIX") or "/v1/tasks"
        ).strip() or "/v1/tasks"
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
            include_task_control=include_task_control,
            include_queue_worker=include_queue_worker,
            task_control_route_prefix=task_control_prefix,
        )
