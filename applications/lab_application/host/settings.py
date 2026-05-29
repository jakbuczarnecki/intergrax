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
        return cls(
            environment=environment,
            route_prefix=prefix,
            include_mock_agents=include_mocks,
            include_echo=include_echo,
            include_signoff_probe=include_signoff_probe,
            include_research=include_research,
            include_interaction_routes=include_interactions,
            interaction_route_prefix=interaction_prefix,
        )
