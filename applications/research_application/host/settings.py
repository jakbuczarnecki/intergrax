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
    include_mcp: bool = True
    mcp_mount_path: str = "/mcp"
    enable_websearch: bool = True
    extra_enabled_tool_ids: tuple[str, ...] = ()
    websearch_executor: object | None = None

    @property
    def enabled_tool_ids(self) -> list[str]:
        ids: list[str] = list(self.extra_enabled_tool_ids)
        if self.enable_websearch and "websearch.query" not in ids:
            ids.append("websearch.query")
        return ids

    @classmethod
    def from_env(cls) -> ResearchBackendSettings:
        use_legacy = _env_bool("RESEARCH_USE_LEGACY_AGENT_ENGINE", False)
        if os.environ.get("RESEARCH_USE_NEXUS_LOOP") is not None:
            use_nexus_loop = _env_bool("RESEARCH_USE_NEXUS_LOOP")
        else:
            use_nexus_loop = not use_legacy
        include_mcp = _env_bool("RESEARCH_INCLUDE_MCP", default=True)
        mcp_mount = os.environ.get("RESEARCH_MCP_MOUNT_PATH", "/mcp").strip() or "/mcp"
        enable_websearch = _env_bool("RESEARCH_ENABLE_WEBSEARCH", default=True)
        extra_tools_raw = os.environ.get("RESEARCH_ENABLED_TOOLS", "").strip()
        extra_tools = tuple(x.strip() for x in extra_tools_raw.split(",") if x.strip())
        return cls(
            host=os.environ.get("RESEARCH_BACKEND_HOST", "0.0.0.0"),
            port=int(os.environ.get("RESEARCH_BACKEND_PORT", "8010")),
            route_prefix=os.environ.get("RESEARCH_ROUTE_PREFIX", "/v1/research"),
            use_nexus_loop=use_nexus_loop,
            include_mcp=include_mcp,
            mcp_mount_path=mcp_mount,
            enable_websearch=enable_websearch,
            extra_enabled_tool_ids=extra_tools,
        )
