# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Declarative application host settings DTO for scaffolded Tier-3 applications."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.api_environment import ApiEnvironment


@dataclass(frozen=True, kw_only=True)
class IntergraxApplicationSettingsBase:
    """Platform-owned application host settings fields (env loading lives in host layer)."""

    environment: ApiEnvironment = ApiEnvironment.DEV
    route_prefix: str = "/v1/app"
    backend_host: str = "127.0.0.1"
    backend_port: int = 8091
    include_interaction_routes: bool = True
    interaction_route_prefix: str = "/v1/interactions"
    include_scheduler: bool = True
    scheduler_poll_seconds: float | None = None
    interaction_surface: str = "auto"
    include_mcp: bool = False
    mcp_mount_path: str = "/mcp"
    include_task_control: bool = True
    include_queue_worker: bool = True
    task_control_route_prefix: str = "/v1/tasks"


__all__ = ["IntergraxApplicationSettingsBase"]
