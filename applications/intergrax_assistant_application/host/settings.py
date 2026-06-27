# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar

from intergrax.applications.contracts.settings import EnvReader, IntergraxApplicationSettingsBase


@dataclass(frozen=True, kw_only=True)
class IntergraxAssistantApplicationSettings(IntergraxApplicationSettingsBase):
    """Environment for intergrax_assistant_application (harness chat lab)."""

    env_prefix: ClassVar[str] = "INTERGRAX_ASSISTANT_"
    route_prefix: str = "/v1/intergrax_assistant"
    backend_port: int = 8096
    task_control_route_prefix: str = "/v1/harness/tasks"
    llm_env_prefix: str = "INTERGRAX_LLM"
    max_delegation_depth: int = 4
    discover_plugins: bool = False
    include_echo: bool = False
    include_legal: bool = False
    include_research: bool = False
    engine_planner: bool = False

    # ------------------------------------------------------------------
    # Application-specific settings
    # Add your own env-backed fields here.
    # ------------------------------------------------------------------

    @classmethod
    def _load_app_env(cls, env: EnvReader) -> dict[str, object]:
        depth_raw = env.str("MAX_DELEGATION_DEPTH", default="4")
        depth = max(1, min(32, int(depth_raw)))
        discover_raw = (os.environ.get("INTERGRAX_DISCOVER_PLUGINS") or "false").strip().lower()
        return {
            "llm_env_prefix": env.str("LLM_ENV_PREFIX", default="INTERGRAX_LLM"),
            "max_delegation_depth": depth,
            "discover_plugins": discover_raw not in {"0", "false", "no"},
            "include_echo": env.bool("INCLUDE_ECHO", default=False),
            "include_legal": env.bool("INCLUDE_LEGAL", default=False),
            "include_research": env.bool("INCLUDE_RESEARCH", default=False),
            "engine_planner": env.bool("ENGINE_PLANNER", default=False),
        }
