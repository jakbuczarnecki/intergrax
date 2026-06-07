# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.knowledge.contracts import (
    KnowledgeGetPageInput,
    KnowledgePageOutput,
    KnowledgeSearchInput,
    KnowledgeSearchOutput,
)
from intergrax.tools.providers.knowledge.handlers import KnowledgeGetPageHandler, KnowledgeSearchHandler
from intergrax.tools.providers.knowledge.service import KNOWLEDGE_GET_PAGE_TOOL_ID, KNOWLEDGE_SEARCH_TOOL_ID
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

KNOWLEDGE_BUNDLE_ID = "knowledge"
KNOWLEDGE_TOOL_IDS: tuple[str, ...] = (KNOWLEDGE_GET_PAGE_TOOL_ID, KNOWLEDGE_SEARCH_TOOL_ID)


def register_knowledge_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=KNOWLEDGE_GET_PAGE_TOOL_ID,
            name=KNOWLEDGE_GET_PAGE_TOOL_ID,
            description="Fetch a wiki/knowledge-base page by provider id (provider-agnostic).",
            description_short="Get knowledge page.",
            input_schema=KnowledgeGetPageInput,
            output_schema=KnowledgePageOutput,
            error_mapping={},
            side_effects=False,
            injects_context=True,
            category="knowledge",
            risk_level=ToolRiskLevel.LOW,
            tags=("knowledge", "wiki"),
        ),
        KnowledgeGetPageHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=KNOWLEDGE_SEARCH_TOOL_ID,
            name=KNOWLEDGE_SEARCH_TOOL_ID,
            description="Search internal wiki/knowledge pages (provider-agnostic).",
            description_short="Search knowledge base.",
            input_schema=KnowledgeSearchInput,
            output_schema=KnowledgeSearchOutput,
            error_mapping={},
            side_effects=False,
            injects_context=True,
            category="knowledge",
            risk_level=ToolRiskLevel.LOW,
            tags=("knowledge", "wiki"),
        ),
        KnowledgeSearchHandler(ctx),
    )
