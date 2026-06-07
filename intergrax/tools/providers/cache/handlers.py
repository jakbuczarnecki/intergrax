# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.cache.contracts import CacheGetInput, CacheGetOutput, CacheSetInput, CacheSetOutput
from intergrax.tools.providers.cache.service import cache_get, cache_set


class CacheGetHandler(ServiceToolHandler[CacheGetInput, CacheGetOutput]):
    _service = cache_get


class CacheSetHandler(ServiceToolHandler[CacheSetInput, CacheSetOutput]):
    _service = cache_set
