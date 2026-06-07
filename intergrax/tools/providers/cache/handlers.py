# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
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
from intergrax.tools.providers.cache.service import cache_delete, cache_get, cache_list_keys, cache_set


class CacheGetHandler(ServiceToolHandler[CacheGetInput, CacheGetOutput]):
    _service = cache_get


class CacheSetHandler(ServiceToolHandler[CacheSetInput, CacheSetOutput]):
    _service = cache_set


class CacheDeleteHandler(ServiceToolHandler[CacheDeleteInput, CacheDeleteOutput]):
    _service = cache_delete


class CacheListKeysHandler(ServiceToolHandler[CacheListKeysInput, CacheListKeysOutput]):
    _service = cache_list_keys
