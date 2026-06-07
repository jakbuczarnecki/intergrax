# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.storage.contracts import (
    StorageDeleteInput,
    StorageDeleteOutput,
    StorageExistsInput,
    StorageExistsOutput,
    StorageGetInput,
    StorageGetOutput,
    StoragePresignedUrlInput,
    StoragePresignedUrlOutput,
    StoragePutInput,
    StoragePutOutput,
)
from intergrax.tools.providers.storage.handlers import (
    StorageDeleteHandler,
    StorageExistsHandler,
    StorageGetHandler,
    StoragePresignedUrlHandler,
    StoragePutHandler,
)
from intergrax.tools.providers.storage.service import (
    STORAGE_DELETE_TOOL_ID,
    STORAGE_EXISTS_TOOL_ID,
    STORAGE_GET_TOOL_ID,
    STORAGE_PRESIGNED_URL_TOOL_ID,
    STORAGE_PUT_TOOL_ID,
)
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

STORAGE_BUNDLE_ID = "storage"
STORAGE_TOOL_IDS: tuple[str, ...] = (
    STORAGE_GET_TOOL_ID,
    STORAGE_PUT_TOOL_ID,
    STORAGE_PRESIGNED_URL_TOOL_ID,
    STORAGE_DELETE_TOOL_ID,
    STORAGE_EXISTS_TOOL_ID,
)


def register_storage_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=STORAGE_GET_TOOL_ID,
            name=STORAGE_GET_TOOL_ID,
            description="Fetch an object from blob storage by key (base64 body in output).",
            description_short="Get storage object.",
            input_schema=StorageGetInput,
            output_schema=StorageGetOutput,
            error_mapping={},
            side_effects=False,
            category="storage",
            risk_level=ToolRiskLevel.LOW,
            tags=("storage", "object"),
        ),
        StorageGetHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=STORAGE_PUT_TOOL_ID,
            name=STORAGE_PUT_TOOL_ID,
            description="Upload or overwrite an object in blob storage (base64 body in input).",
            description_short="Put storage object.",
            input_schema=StoragePutInput,
            output_schema=StoragePutOutput,
            error_mapping={},
            side_effects=True,
            category="storage",
            risk_level=ToolRiskLevel.HIGH,
            tags=("storage", "object"),
        ),
        StoragePutHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=STORAGE_PRESIGNED_URL_TOOL_ID,
            name=STORAGE_PRESIGNED_URL_TOOL_ID,
            description="Generate a time-limited presigned URL for direct object access.",
            description_short="Presigned storage URL.",
            input_schema=StoragePresignedUrlInput,
            output_schema=StoragePresignedUrlOutput,
            error_mapping={},
            side_effects=False,
            category="storage",
            risk_level=ToolRiskLevel.MEDIUM,
            tags=("storage", "object"),
        ),
        StoragePresignedUrlHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=STORAGE_DELETE_TOOL_ID,
            name=STORAGE_DELETE_TOOL_ID,
            description="Delete an object from blob storage by key.",
            description_short="Delete storage object.",
            input_schema=StorageDeleteInput,
            output_schema=StorageDeleteOutput,
            error_mapping={},
            side_effects=True,
            category="storage",
            risk_level=ToolRiskLevel.HIGH,
            tags=("storage", "object"),
        ),
        StorageDeleteHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=STORAGE_EXISTS_TOOL_ID,
            name=STORAGE_EXISTS_TOOL_ID,
            description="Check whether an object exists in blob storage without returning body bytes.",
            description_short="Check storage object exists.",
            input_schema=StorageExistsInput,
            output_schema=StorageExistsOutput,
            error_mapping={},
            side_effects=False,
            category="storage",
            risk_level=ToolRiskLevel.LOW,
            tags=("storage", "object", "metadata"),
        ),
        StorageExistsHandler(ctx),
    )
