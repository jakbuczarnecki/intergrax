# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import base64

from intergrax.integrations.contracts.object_storage import ObjectStorage, PresignedUrlMethod
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
from intergrax.tools.registry.wiring import ToolWiringContext

STORAGE_GET_TOOL_ID = "storage.get"
STORAGE_PUT_TOOL_ID = "storage.put"
STORAGE_PRESIGNED_URL_TOOL_ID = "storage.presigned_url"
STORAGE_DELETE_TOOL_ID = "storage.delete"
STORAGE_EXISTS_TOOL_ID = "storage.exists"


def _require_storage(ctx: ToolWiringContext) -> ObjectStorage:
    storage = ctx.object_storage
    if storage is None:
        raise RuntimeError("object_storage_not_configured")
    return storage


def storage_get(ctx: ToolWiringContext, params: StorageGetInput) -> StorageGetOutput:
    stored = _require_storage(ctx).get(params.key.strip())
    if stored is None:
        return StorageGetOutput(key=params.key.strip(), found=False)
    return StorageGetOutput(
        key=stored.key,
        found=True,
        body_base64=base64.b64encode(stored.body).decode("ascii"),
        content_type=stored.content_type,
        size_bytes=stored.size_bytes or len(stored.body),
    )


def storage_put(ctx: ToolWiringContext, params: StoragePutInput) -> StoragePutOutput:
    body = base64.b64decode(params.body_base64)
    _require_storage(ctx).put(
        params.key.strip(),
        body,
        content_type=params.content_type,
    )
    return StoragePutOutput(key=params.key.strip(), stored=True, size_bytes=len(body))


def storage_presigned_url(ctx: ToolWiringContext, params: StoragePresignedUrlInput) -> StoragePresignedUrlOutput:
    method: PresignedUrlMethod = "PUT" if params.method == "PUT" else "GET"
    url = _require_storage(ctx).presigned_url(
        params.key.strip(),
        expires_in_seconds=params.expires_in_seconds,
        method=method,
    )
    return StoragePresignedUrlOutput(
        key=params.key.strip(),
        url=url,
        method=params.method,
        expires_in_seconds=params.expires_in_seconds,
    )


def storage_delete(ctx: ToolWiringContext, params: StorageDeleteInput) -> StorageDeleteOutput:
    _require_storage(ctx).delete(params.key.strip())
    return StorageDeleteOutput(key=params.key.strip(), deleted=True)


def storage_exists(ctx: ToolWiringContext, params: StorageExistsInput) -> StorageExistsOutput:
    stored = _require_storage(ctx).get(params.key.strip())
    if stored is None:
        return StorageExistsOutput(key=params.key.strip(), exists=False)
    return StorageExistsOutput(
        key=stored.key,
        exists=True,
        content_type=stored.content_type,
        size_bytes=stored.size_bytes or len(stored.body),
    )
