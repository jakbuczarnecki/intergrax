# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import base64

from intergrax.integrations.contracts.key_value_cache import KeyValueCache
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
from intergrax.tools.registry.runtime_bindings import KeyValueCacheListerBinding
from intergrax.tools.registry.wiring import ToolWiringContext

CACHE_GET_TOOL_ID = "cache.get"
CACHE_SET_TOOL_ID = "cache.set"
CACHE_DELETE_TOOL_ID = "cache.delete"
CACHE_LIST_KEYS_TOOL_ID = "cache.list_keys"


def _require_cache(ctx: ToolWiringContext) -> KeyValueCache:
    cache = ctx.key_value_cache
    if cache is None:
        raise RuntimeError("key_value_cache_not_configured")
    return cache


def cache_get(ctx: ToolWiringContext, params: CacheGetInput) -> CacheGetOutput:
    value = _require_cache(ctx).get(params.tenant_id.strip(), params.key.strip())
    if value is None:
        return CacheGetOutput(
            tenant_id=params.tenant_id.strip(),
            key=params.key.strip(),
            found=False,
        )
    return CacheGetOutput(
        tenant_id=params.tenant_id.strip(),
        key=params.key.strip(),
        found=True,
        value_base64=base64.b64encode(value).decode("ascii"),
    )


def cache_set(ctx: ToolWiringContext, params: CacheSetInput) -> CacheSetOutput:
    body = base64.b64decode(params.value_base64)
    _require_cache(ctx).set(
        params.tenant_id.strip(),
        params.key.strip(),
        body,
        ttl_seconds=params.ttl_seconds,
    )
    return CacheSetOutput(
        tenant_id=params.tenant_id.strip(),
        key=params.key.strip(),
        stored=True,
    )


def cache_delete(ctx: ToolWiringContext, params: CacheDeleteInput) -> CacheDeleteOutput:
    _require_cache(ctx).delete(params.tenant_id.strip(), params.key.strip())
    return CacheDeleteOutput(
        tenant_id=params.tenant_id.strip(),
        key=params.key.strip(),
    )


def cache_list_keys(ctx: ToolWiringContext, params: CacheListKeysInput) -> CacheListKeysOutput:
    cache = _require_cache(ctx)
    if not isinstance(cache, KeyValueCacheListerBinding):
        raise RuntimeError("cache_list_keys_not_supported")
    keys = list(
        cache.list_keys(
            params.tenant_id.strip(),
            prefix=params.prefix,
            limit=params.limit,
        )
    )
    return CacheListKeysOutput(tenant_id=params.tenant_id.strip(), keys=keys, total=len(keys))
