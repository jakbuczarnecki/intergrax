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

    @classmethod
    def from_env(cls) -> ResearchBackendSettings:
        return cls(
            host=os.environ.get("RESEARCH_BACKEND_HOST", "0.0.0.0"),
            port=int(os.environ.get("RESEARCH_BACKEND_PORT", "8010")),
            route_prefix=os.environ.get("RESEARCH_ROUTE_PREFIX", "/v1/research"),
        )
