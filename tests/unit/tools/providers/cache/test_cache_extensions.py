# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import base64

import pytest

from intergrax.tools.providers.cache.contracts import CacheDeleteInput, CacheListKeysInput, CacheSetInput
from intergrax.tools.providers.cache.service import cache_delete, cache_list_keys, cache_set
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


class InMemoryCache:
    def __init__(self) -> None:
        self.store: dict[tuple[str, str], bytes] = {}

    def get(self, tenant_id: str, key: str) -> bytes | None:
        return self.store.get((tenant_id, key))

    def set(self, tenant_id: str, key: str, value: bytes, *, ttl_seconds: int | None = None) -> None:
        self.store[(tenant_id, key)] = value

    def delete(self, tenant_id: str, key: str) -> None:
        self.store.pop((tenant_id, key), None)

    def set_if_absent(self, tenant_id: str, key: str, value: bytes, *, ttl_seconds: int | None = None) -> bool:
        if (tenant_id, key) in self.store:
            return False
        self.set(tenant_id, key, value, ttl_seconds=ttl_seconds)
        return True

    def list_keys(self, tenant_id: str, *, prefix: str = "", limit: int = 100) -> list[str]:
        keys = [key for (tid, key) in self.store if tid == tenant_id and key.startswith(prefix)]
        return keys[:limit]


def test_cache_delete_and_list_keys() -> None:
    cache = InMemoryCache()
    ctx = ToolWiringContext(key_value_cache=cache)
    cache_set(
        ctx,
        CacheSetInput(tenant_id="t1", key="alpha", value_base64=base64.b64encode(b"x").decode("ascii")),
    )
    listed = cache_list_keys(ctx, CacheListKeysInput(tenant_id="t1"))
    assert listed.total == 1
    cache_delete(ctx, CacheDeleteInput(tenant_id="t1", key="alpha"))
    listed_after = cache_list_keys(ctx, CacheListKeysInput(tenant_id="t1"))
    assert listed_after.total == 0
