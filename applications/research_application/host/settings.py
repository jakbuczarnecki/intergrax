# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ResearchBackendSettings:
    host: str = "0.0.0.0"
    port: int = 8010
    route_prefix: str = "/v1/research"
    use_nexus_loop: bool = True

    @classmethod
    def from_env(cls) -> ResearchBackendSettings:
        use_legacy = _env_bool("RESEARCH_USE_LEGACY_AGENT_ENGINE", False)
        if os.environ.get("RESEARCH_USE_NEXUS_LOOP") is not None:
            use_nexus_loop = _env_bool("RESEARCH_USE_NEXUS_LOOP")
        else:
            use_nexus_loop = not use_legacy
        return cls(
            host=os.environ.get("RESEARCH_BACKEND_HOST", "0.0.0.0"),
            port=int(os.environ.get("RESEARCH_BACKEND_PORT", "8010")),
            route_prefix=os.environ.get("RESEARCH_ROUTE_PREFIX", "/v1/research"),
            use_nexus_loop=use_nexus_loop,
        )
