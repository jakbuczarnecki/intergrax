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
    include_interaction_routes: bool = True
    interaction_route_prefix: str = "/v1/interactions"
    interaction_surface: str = "auto"
    interaction_execute_default: bool = True
    include_task_control: bool = True
    include_scheduler: bool = False
    include_queue_worker: bool = False
    task_control_route_prefix: str = "/v1/tasks"
    scheduler_poll_seconds: float | None = None
    enable_websearch: bool = True
    enable_rag: bool = False
    enable_rag_ingest: bool = False
    extra_enabled_tool_ids: tuple[str, ...] = ()
    websearch_executor: object | None = None

    @property
    def enabled_tool_ids(self) -> list[str]:
        ids: list[str] = list(self.extra_enabled_tool_ids)
        if self.enable_websearch and "websearch.query" not in ids:
            ids.append("websearch.query")
        if self.enable_rag and "rag.retrieve" not in ids:
            ids.append("rag.retrieve")
        if self.enable_rag_ingest and "rag.ingest_document" not in ids:
            ids.append("rag.ingest_document")
        return ids

    @classmethod
    def from_env(cls) -> ResearchBackendSettings:
        use_nexus_loop = _env_bool("RESEARCH_USE_NEXUS_LOOP", default=True)
        include_mcp = _env_bool("RESEARCH_INCLUDE_MCP", default=True)
        mcp_mount = os.environ.get("RESEARCH_MCP_MOUNT_PATH", "/mcp").strip() or "/mcp"
        enable_websearch = _env_bool("RESEARCH_ENABLE_WEBSEARCH", default=True)
        enable_rag = _env_bool("RESEARCH_ENABLE_RAG", default=False)
        enable_rag_ingest = _env_bool("RESEARCH_ENABLE_RAG_INGEST", default=False)
        extra_tools_raw = os.environ.get("RESEARCH_ENABLED_TOOLS", "").strip()
        extra_tools = tuple(x.strip() for x in extra_tools_raw.split(",") if x.strip())
        include_interactions = _env_bool("RESEARCH_INCLUDE_INTERACTIONS", default=True)
        interaction_prefix = (
            os.environ.get("RESEARCH_INTERACTION_ROUTE_PREFIX") or "/v1/interactions"
        ).strip() or "/v1/interactions"
        interaction_surface = (
            os.environ.get("RESEARCH_INTERACTION_SURFACE") or "auto"
        ).strip().lower() or "auto"
        interaction_execute = _env_bool("RESEARCH_INTERACTION_EXECUTE_DEFAULT", default=True)
        include_task_control = _env_bool("RESEARCH_INCLUDE_TASK_CONTROL", default=True)
        include_scheduler = _env_bool("RESEARCH_INCLUDE_SCHEDULER", default=False)
        include_queue_worker = _env_bool("RESEARCH_INCLUDE_QUEUE_WORKER", default=False)
        task_control_prefix = (
            os.environ.get("RESEARCH_TASK_CONTROL_ROUTE_PREFIX") or "/v1/tasks"
        ).strip() or "/v1/tasks"
        poll_raw = (os.environ.get("INTERGRAX_SCHEDULER_POLL_SECONDS") or "").strip()
        scheduler_poll = float(poll_raw) if poll_raw else None
        return cls(
            host=os.environ.get("RESEARCH_BACKEND_HOST", "0.0.0.0"),
            port=int(os.environ.get("RESEARCH_BACKEND_PORT", "8010")),
            route_prefix=os.environ.get("RESEARCH_ROUTE_PREFIX", "/v1/research"),
            use_nexus_loop=use_nexus_loop,
            include_mcp=include_mcp,
            mcp_mount_path=mcp_mount,
            include_interaction_routes=include_interactions,
            interaction_route_prefix=interaction_prefix,
            interaction_surface=interaction_surface,
            interaction_execute_default=interaction_execute,
            include_task_control=include_task_control,
            include_scheduler=include_scheduler,
            include_queue_worker=include_queue_worker,
            task_control_route_prefix=task_control_prefix,
            scheduler_poll_seconds=scheduler_poll,
            enable_websearch=enable_websearch,
            enable_rag=enable_rag,
            enable_rag_ingest=enable_rag_ingest,
            extra_enabled_tool_ids=extra_tools,
        )
