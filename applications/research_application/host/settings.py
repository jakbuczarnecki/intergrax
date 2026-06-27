# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from intergrax.applications.contracts.settings import EnvReader, IntergraxApplicationSettingsBase


@dataclass(frozen=True, kw_only=True)
class ResearchBackendSettings(IntergraxApplicationSettingsBase):
    """Environment for research_application (scaffolded lab profile)."""

    env_prefix: ClassVar[str] = "RESEARCH_"
    route_prefix: str = "/v1/research"
    backend_host: str = "0.0.0.0"
    backend_port: int = 8010
    include_queue_worker: bool = False
    use_nexus_loop: bool = True
    interaction_execute_default: bool = True
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

    # ------------------------------------------------------------------
    # Application-specific settings
    # Add your own env-backed fields here.
    # ------------------------------------------------------------------

    @classmethod
    def _load_app_env(cls, env: EnvReader) -> dict[str, object]:
        extra_tools_raw = env.optional_str("ENABLED_TOOLS")
        extra_tools = tuple(
            part.strip() for part in (extra_tools_raw or "").split(",") if part.strip()
        )
        return {
            "use_nexus_loop": env.bool("USE_NEXUS_LOOP", default=True),
            "interaction_execute_default": env.bool("INTERACTION_EXECUTE_DEFAULT", default=True),
            "enable_websearch": env.bool("ENABLE_WEBSEARCH", default=True),
            "enable_rag": env.bool("ENABLE_RAG", default=False),
            "enable_rag_ingest": env.bool("ENABLE_RAG_INGEST", default=False),
            "extra_enabled_tool_ids": extra_tools,
        }
