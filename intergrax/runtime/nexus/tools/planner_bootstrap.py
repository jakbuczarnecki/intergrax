# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog tool planner bootstrap on the runtime tool registry (TOOL-ENG-0)."""

from __future__ import annotations

from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.tools.catalog_tool_planner import CatalogToolPlanner
from intergrax.tools.registry.runtime import ToolRegistry


def wire_catalog_tool_planner_if_enabled(
    config: RuntimeConfig,
    registry: ToolRegistry,
    *,
    prompt_registry: YamlPromptRegistry | None = None,
) -> None:
    """
    Attach :class:`CatalogToolPlanner` to ``config.tool_planner`` when tools are enabled.

    Uses the same ``registry`` as ``RuntimeToolInvoker`` so planned ``tool_id`` values
    exist at invoke time. Skips when a planner is already set, ``tools_mode`` is ``off``,
    LLM is missing, or the registry has no tools.
    """
    if config.tool_planner is not None:
        return
    if config.tools_mode == "off":
        return
    if config.llm_adapter is None:
        return
    if not registry.tool_ids():
        return

    config.tool_planner = CatalogToolPlanner.from_registry(
        llm=config.llm_adapter,
        registry=registry,
        prompt_registry=prompt_registry,
        prompt_catalog_path=config.prompt_catalog_path,
        planner_prompt_id=config.tool_planner_prompt_id,
    )
    from intergrax.runtime.nexus.context.routing_snapshot_sync import (
        wire_secondary_llm_routing_surfaces,
    )

    wire_secondary_llm_routing_surfaces(config)
