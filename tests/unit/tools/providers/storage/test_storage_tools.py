# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import base64
from typing import Mapping, Optional

import pytest

from intergrax.integrations.contracts.object_storage import ObjectStorage, StoredObject
from intergrax.tools.providers.storage.contracts import StorageGetInput, StoragePutInput
from intergrax.tools.providers.storage.service import storage_get, storage_put
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


class InMemoryObjectStorage:
    def __init__(self) -> None:
        self._objects: dict[str, StoredObject] = {}

    def put(
        self,
        key: str,
        body: bytes,
        *,
        content_type: str = "application/octet-stream",
        metadata: Optional[Mapping[str, str]] = None,
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
    ctx = ToolWiringContext(object_storage=storage)
    body = b"hello blob"
    storage_put(ctx, StoragePutInput(key="docs/a.txt", body_base64=base64.b64encode(body).decode("ascii")))
    out = storage_get(ctx, StorageGetInput(key="docs/a.txt"))
    assert out.found is True
    assert base64.b64decode(out.body_base64) == body


def test_storage_not_configured() -> None:
    with pytest.raises(RuntimeError, match="object_storage_not_configured"):
        storage_get(ToolWiringContext(), StorageGetInput(key="missing"))
