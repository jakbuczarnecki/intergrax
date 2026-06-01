# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from intergrax.fastapi_core.config import ApiEnvironment


@dataclass(frozen=True)
class LabApplicationSettings:
    """Environment for the universal lab application (Tier-3)."""

    environment: ApiEnvironment = ApiEnvironment.DEV
    route_prefix: str = "/v1/lab"
    include_mock_agents: bool = True
    include_echo: bool = True
    include_signoff_probe: bool = True
    include_research: bool = False
    include_interaction_routes: bool = True
    interaction_route_prefix: str = "/v1/interactions"
    include_scheduler: bool = True
    scheduler_poll_seconds: float | None = None
    interaction_surface: str = "auto"
    include_mcp: bool = True
    mcp_mount_path: str = "/mcp"
    harness: bool = False

    @classmethod
    def from_env(cls) -> LabApplicationSettings:
        env_raw = (os.getenv("INTERGRAX_ENV") or "dev").strip().lower()
        environment = ApiEnvironment.PROD if env_raw == "prod" else ApiEnvironment.DEV
        include_mocks = (os.getenv("LAB_INCLUDE_MOCK_AGENTS") or "true").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        include_echo = (os.getenv("LAB_INCLUDE_ECHO") or "true").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        include_signoff_probe = (
            os.getenv("LAB_INCLUDE_SIGNOFF_PROBE") or "true"
        ).strip().lower() not in {
            "0",
            "false",
            "no",
        }
        include_research = (os.getenv("LAB_INCLUDE_RESEARCH") or "false").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        include_interactions = (
            os.getenv("LAB_INCLUDE_INTERACTIONS") or "true"
        ).strip().lower() not in {
            "0",
            "false",
            "no",
        }
        prefix = (os.getenv("LAB_ROUTE_PREFIX") or "/v1/lab").strip() or "/v1/lab"
        interaction_prefix = (
            os.getenv("LAB_INTERACTION_ROUTE_PREFIX") or "/v1/interactions"
        ).strip() or "/v1/interactions"
        include_scheduler = (os.getenv("LAB_INCLUDE_SCHEDULER") or "true").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        poll_raw = (os.getenv("INTERGRAX_SCHEDULER_POLL_SECONDS") or "").strip()
        scheduler_poll = float(poll_raw) if poll_raw else None
        interaction_surface = (
            os.getenv("LAB_INTERACTION_SURFACE") or "auto"
        ).strip().lower() or "auto"
        include_mcp = (os.getenv("LAB_INCLUDE_MCP") or "true").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        mcp_mount = (os.getenv("LAB_MCP_MOUNT_PATH") or "/mcp").strip() or "/mcp"
        harness = (os.getenv("LAB_HARNESS") or "false").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        return cls(
            environment=environment,
            route_prefix=prefix,
            include_mock_agents=include_mocks,
            include_echo=include_echo,
            include_signoff_probe=include_signoff_probe,
            include_research=include_research,
            include_interaction_routes=include_interactions,
            interaction_route_prefix=interaction_prefix,
            include_scheduler=include_scheduler,
            scheduler_poll_seconds=scheduler_poll,
            interaction_surface=interaction_surface,
            include_mcp=include_mcp,
            mcp_mount_path=mcp_mount,
            harness=harness,
        )
