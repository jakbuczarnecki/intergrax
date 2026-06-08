# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.ltm.contracts import (
    LtmSearchInput,
    LtmSearchOutput,
    LtmWriteFactInput,
    LtmWriteFactOutput,
)
from intergrax.tools.providers.ltm.handlers import LtmSearchHandler, LtmWriteFactHandler
from intergrax.tools.providers.ltm.service import LTM_SEARCH_TOOL_ID, LTM_WRITE_FACT_TOOL_ID
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

LTM_BUNDLE_ID = "ltm"
LTM_TOOL_IDS: tuple[str, ...] = (LTM_SEARCH_TOOL_ID, LTM_WRITE_FACT_TOOL_ID)


def register_ltm_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=LTM_SEARCH_TOOL_ID,
            name=LTM_SEARCH_TOOL_ID,
            description="Search long-term user memory (vector retrieval or keyword fallback).",
            description_short="Search user LTM.",
            input_schema=LtmSearchInput,
            output_schema=LtmSearchOutput,
            error_mapping={},
            side_effects=False,
            category="ltm",
            risk_level=ToolRiskLevel.LOW,
            tags=("memory", "ltm", "retrieval"),
        ),
        LtmSearchHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=LTM_WRITE_FACT_TOOL_ID,
            name=LTM_WRITE_FACT_TOOL_ID,
            description="Persist a governed long-term user memory fact (MemoryKind tagged).",
            description_short="Write user LTM fact.",
            input_schema=LtmWriteFactInput,
            output_schema=LtmWriteFactOutput,
            error_mapping={},
            side_effects=True,
            category="ltm",
            risk_level=ToolRiskLevel.MEDIUM,
            tags=("memory", "ltm", "write"),
        ),
        LtmWriteFactHandler(ctx),
    )
