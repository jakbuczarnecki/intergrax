# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.memory.contracts import (
    MemoryDeleteKeyInput,
    MemoryDeleteKeyOutput,
    MemoryListKeysInput,
    MemoryListKeysOutput,
    MemoryReadInput,
    MemoryReadOutput,
    MemorySearchInput,
    MemorySearchOutput,
    MemoryWriteInput,
    MemoryWriteOutput,
)
from intergrax.tools.providers.memory.handlers import (
    MemoryDeleteKeyHandler,
    MemoryListKeysHandler,
    MemoryReadHandler,
    MemorySearchHandler,
    MemoryWriteHandler,
)
from intergrax.tools.providers.memory.service import (
    MEMORY_DELETE_KEY_TOOL_ID,
    MEMORY_LIST_KEYS_TOOL_ID,
    MEMORY_READ_TOOL_ID,
    MEMORY_SEARCH_TOOL_ID,
    MEMORY_WRITE_TOOL_ID,
)
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

MEMORY_BUNDLE_ID = "memory"
MEMORY_TOOL_IDS: tuple[str, ...] = (
    MEMORY_READ_TOOL_ID,
    MEMORY_WRITE_TOOL_ID,
    MEMORY_LIST_KEYS_TOOL_ID,
    MEMORY_DELETE_KEY_TOOL_ID,
    MEMORY_SEARCH_TOOL_ID,
)


def register_memory_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=MEMORY_READ_TOOL_ID,
            name=MEMORY_READ_TOOL_ID,
            description="Read a policy-scoped task memory record by namespace and key.",
            description_short="Read task memory.",
            input_schema=MemoryReadInput,
            output_schema=MemoryReadOutput,
            error_mapping={},
            side_effects=False,
            category="memory",
            risk_level=ToolRiskLevel.LOW,
            tags=("memory", "task"),
        ),
        MemoryReadHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=MEMORY_WRITE_TOOL_ID,
            name=MEMORY_WRITE_TOOL_ID,
            description="Write or merge a task memory record under policy guardrails.",
            description_short="Write task memory.",
            input_schema=MemoryWriteInput,
            output_schema=MemoryWriteOutput,
            error_mapping={},
            side_effects=True,
            category="memory",
            risk_level=ToolRiskLevel.MEDIUM,
            tags=("memory", "task"),
        ),
        MemoryWriteHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=MEMORY_LIST_KEYS_TOOL_ID,
            name=MEMORY_LIST_KEYS_TOOL_ID,
            description="List task memory keys in a namespace (optional prefix filter).",
            description_short="List task memory keys.",
            input_schema=MemoryListKeysInput,
            output_schema=MemoryListKeysOutput,
            error_mapping={},
            side_effects=False,
            category="memory",
            risk_level=ToolRiskLevel.LOW,
            tags=("memory", "task"),
        ),
        MemoryListKeysHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=MEMORY_DELETE_KEY_TOOL_ID,
            name=MEMORY_DELETE_KEY_TOOL_ID,
            description="Delete a policy-scoped task memory record by namespace and key.",
            description_short="Delete task memory key.",
            input_schema=MemoryDeleteKeyInput,
            output_schema=MemoryDeleteKeyOutput,
            error_mapping={},
            side_effects=True,
            category="memory",
            risk_level=ToolRiskLevel.MEDIUM,
            tags=("memory", "task"),
        ),
        MemoryDeleteKeyHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=MEMORY_SEARCH_TOOL_ID,
            name=MEMORY_SEARCH_TOOL_ID,
            description="Search task memory keys/values by substring within a namespace.",
            description_short="Search task memory.",
            input_schema=MemorySearchInput,
            output_schema=MemorySearchOutput,
            error_mapping={},
            side_effects=False,
            category="memory",
            risk_level=ToolRiskLevel.LOW,
            tags=("memory", "task", "search"),
        ),
        MemorySearchHandler(ctx),
    )
