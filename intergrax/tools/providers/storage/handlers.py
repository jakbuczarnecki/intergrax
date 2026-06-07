# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
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
from intergrax.tools.providers.storage.service import (
    storage_delete,
    storage_exists,
    storage_get,
    storage_presigned_url,
    storage_put,
)


class StorageGetHandler(ServiceToolHandler[StorageGetInput, StorageGetOutput]):
    _service = storage_get


class StoragePutHandler(ServiceToolHandler[StoragePutInput, StoragePutOutput]):
    _service = storage_put


class StoragePresignedUrlHandler(ServiceToolHandler[StoragePresignedUrlInput, StoragePresignedUrlOutput]):
    _service = storage_presigned_url


class StorageDeleteHandler(ServiceToolHandler[StorageDeleteInput, StorageDeleteOutput]):
    _service = storage_delete


class StorageExistsHandler(ServiceToolHandler[StorageExistsInput, StorageExistsOutput]):
    _service = storage_exists
