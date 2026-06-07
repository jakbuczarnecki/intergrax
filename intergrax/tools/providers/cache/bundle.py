# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.cache.contracts import (
    CacheDeleteInput,
    CacheDeleteOutput,
    CacheGetInput,
    CacheGetOutput,
    CacheListKeysInput,
    CacheListKeysOutput,
    CacheSetInput,
    CacheSetOutput,
)
from intergrax.tools.providers.cache.handlers import (
    CacheDeleteHandler,
    CacheGetHandler,
    CacheListKeysHandler,
    CacheSetHandler,
)
from intergrax.tools.providers.cache.service import (
    CACHE_DELETE_TOOL_ID,
    CACHE_GET_TOOL_ID,
    CACHE_LIST_KEYS_TOOL_ID,
    CACHE_SET_TOOL_ID,
)
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

CACHE_BUNDLE_ID = "cache"
CACHE_TOOL_IDS: tuple[str, ...] = (
    CACHE_GET_TOOL_ID,
    CACHE_SET_TOOL_ID,
    CACHE_DELETE_TOOL_ID,
    CACHE_LIST_KEYS_TOOL_ID,
)


def register_cache_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=CACHE_GET_TOOL_ID,
            name=CACHE_GET_TOOL_ID,
            description="Read a tenant-scoped key from the configured key-value cache.",
            description_short="Get cache key.",
            input_schema=CacheGetInput,
            output_schema=CacheGetOutput,
            error_mapping={},
            side_effects=False,
            category="cache",
            risk_level=ToolRiskLevel.LOW,
            tags=("cache", "kv"),
        ),
        CacheGetHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=CACHE_SET_TOOL_ID,
            name=CACHE_SET_TOOL_ID,
            description="Store a tenant-scoped key in the key-value cache (optional TTL).",
            description_short="Set cache key.",
            input_schema=CacheSetInput,
            output_schema=CacheSetOutput,
            error_mapping={},
            side_effects=True,
            category="cache",
            risk_level=ToolRiskLevel.MEDIUM,
            tags=("cache", "kv"),
        ),
        CacheSetHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=CACHE_DELETE_TOOL_ID,
            name=CACHE_DELETE_TOOL_ID,
            description="Delete a tenant-scoped key from the configured key-value cache.",
            description_short="Delete cache key.",
            input_schema=CacheDeleteInput,
            output_schema=CacheDeleteOutput,
            error_mapping={},
            side_effects=True,
            category="cache",
            risk_level=ToolRiskLevel.MEDIUM,
            tags=("cache", "kv"),
        ),
        CacheDeleteHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=CACHE_LIST_KEYS_TOOL_ID,
            name=CACHE_LIST_KEYS_TOOL_ID,
            description="List cache keys for a tenant when the backend supports key listing.",
            description_short="List cache keys.",
            input_schema=CacheListKeysInput,
            output_schema=CacheListKeysOutput,
            error_mapping={},
            side_effects=False,
            category="cache",
            risk_level=ToolRiskLevel.LOW,
            tags=("cache", "kv"),
        ),
        CacheListKeysHandler(ctx),
    )
