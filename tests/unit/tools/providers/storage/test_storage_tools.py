# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import base64
from collections.abc import Mapping

import pytest
from intergrax.integrations.contracts.object_storage import StoredObject
from intergrax.tools.providers.storage.contracts import StorageGetInput, StoragePutInput
from intergrax.tools.providers.storage.service import storage_get, storage_put
from intergrax.tools.registry.wiring import ListStorageAccessAuditSink, ToolWiringContext

pytestmark = pytest.mark.unit

_TENANT = "tenant-tools"


def _storage_ctx(storage: InMemoryObjectStorage) -> ToolWiringContext:
    return ToolWiringContext(
        object_storage=storage,
        tenant_id=_TENANT,
        storage_audit_sink=ListStorageAccessAuditSink(),
    )


class InMemoryObjectStorage:
    def __init__(self) -> None:
        self._objects: dict[str, StoredObject] = {}

    def put(
        self,
        key: str,
        body: bytes,
        *,
        content_type: str = "application/octet-stream",
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        self._objects[key] = StoredObject(
            key=key,
            body=body,
            content_type=content_type,
            metadata=dict(metadata or {}),
            size_bytes=len(body),
        )

    def get(self, key: str) -> StoredObject | None:
        return self._objects.get(key)

    def delete(self, key: str) -> None:
        self._objects.pop(key, None)

    def presigned_url(self, key: str, *, expires_in_seconds: int = 3600, method: str = "GET") -> str:
        return f"https://example.test/{key}?method={method}&exp={expires_in_seconds}"

    def close(self) -> None:
        return None


def test_storage_put_and_get() -> None:
    storage = InMemoryObjectStorage()
    ctx = _storage_ctx(storage)
    body = b"hello blob"
    storage_put(ctx, StoragePutInput(key="docs/a.txt", body_base64=base64.b64encode(body).decode("ascii")))
    assert "tenants/tenant-tools/docs/a.txt" in storage._objects
    out = storage_get(ctx, StorageGetInput(key="docs/a.txt"))
    assert out.found is True
    assert base64.b64decode(out.body_base64) == body


def test_storage_not_configured() -> None:
    with pytest.raises(RuntimeError, match="object_storage_not_configured"):
        storage_get(
            ToolWiringContext(
                tenant_id=_TENANT,
                storage_audit_sink=ListStorageAccessAuditSink(),
            ),
            StorageGetInput(key="missing"),
        )
