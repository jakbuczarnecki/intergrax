# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.vector_store.contracts import (
    VectorStoreCountInput,
    VectorStoreCountOutput,
    VectorStoreDeleteInput,
    VectorStoreDeleteOutput,
    VectorStoreHealthInput,
    VectorStoreHealthOutput,
    VectorStoreListCollectionsInput,
    VectorStoreListCollectionsOutput,
)
from intergrax.tools.providers.vector_store.handlers import (
    VectorStoreCountHandler,
    VectorStoreDeleteHandler,
    VectorStoreHealthHandler,
    VectorStoreListCollectionsHandler,
)
from intergrax.tools.providers.vector_store.service import (
    VECTOR_STORE_COUNT_TOOL_ID,
    VECTOR_STORE_DELETE_TOOL_ID,
    VECTOR_STORE_HEALTH_TOOL_ID,
    VECTOR_STORE_LIST_COLLECTIONS_TOOL_ID,
)
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

VECTOR_STORE_BUNDLE_ID = "vector_store"
VECTOR_STORE_TOOL_IDS: tuple[str, ...] = (
    VECTOR_STORE_COUNT_TOOL_ID,
    VECTOR_STORE_DELETE_TOOL_ID,
    VECTOR_STORE_LIST_COLLECTIONS_TOOL_ID,
    VECTOR_STORE_HEALTH_TOOL_ID,
)


def register_vector_store_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=VECTOR_STORE_COUNT_TOOL_ID,
            name=VECTOR_STORE_COUNT_TOOL_ID,
            description="Return document count for the configured vector store backend.",
            description_short="Count vector store documents.",
            input_schema=VectorStoreCountInput,
            output_schema=VectorStoreCountOutput,
            error_mapping={},
            side_effects=False,
            category="vector_store",
            risk_level=ToolRiskLevel.LOW,
            tags=("vector_store", "retrieval", "read_only"),
        ),
        VectorStoreCountHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=VECTOR_STORE_DELETE_TOOL_ID,
            name=VECTOR_STORE_DELETE_TOOL_ID,
            description="Delete documents by id from the configured vector store backend.",
            description_short="Delete vector store documents.",
            input_schema=VectorStoreDeleteInput,
            output_schema=VectorStoreDeleteOutput,
            error_mapping={},
            side_effects=True,
            category="vector_store",
            risk_level=ToolRiskLevel.HIGH,
            tags=("vector_store", "retrieval", "write"),
        ),
        VectorStoreDeleteHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=VECTOR_STORE_LIST_COLLECTIONS_TOOL_ID,
            name=VECTOR_STORE_LIST_COLLECTIONS_TOOL_ID,
            description="List collection names exposed by the configured vector store backend.",
            description_short="List vector store collections.",
            input_schema=VectorStoreListCollectionsInput,
            output_schema=VectorStoreListCollectionsOutput,
            error_mapping={},
            side_effects=False,
            category="vector_store",
            risk_level=ToolRiskLevel.LOW,
            tags=("vector_store", "retrieval", "read_only"),
        ),
        VectorStoreListCollectionsHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=VECTOR_STORE_HEALTH_TOOL_ID,
            name=VECTOR_STORE_HEALTH_TOOL_ID,
            description="Probe vector store reachability via a lightweight count operation.",
            description_short="Probe vector store health.",
            input_schema=VectorStoreHealthInput,
            output_schema=VectorStoreHealthOutput,
            error_mapping={},
            side_effects=False,
            category="vector_store",
            risk_level=ToolRiskLevel.LOW,
            tags=("vector_store", "retrieval", "probe"),
        ),
        VectorStoreHealthHandler(ctx),
    )
