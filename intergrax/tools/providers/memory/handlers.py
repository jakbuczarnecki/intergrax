# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.memory.contracts import (
    MemoryDeleteKeyInput,
    MemoryDeleteKeyOutput,
    MemoryListKeysInput,
    MemoryListKeysOutput,
    MemoryReadInput,
    MemoryReadOutput,
    MemorySearchInput,
    MemorySearchOutput,
    MemoryWriteInput,
    MemoryWriteOutput,
)
from intergrax.tools.providers.memory.service import (
    memory_delete_key,
    memory_list_keys,
    memory_read,
    memory_search,
    memory_write,
)


class MemoryReadHandler(ServiceToolHandler[MemoryReadInput, MemoryReadOutput]):
    _service = memory_read


class MemoryWriteHandler(ServiceToolHandler[MemoryWriteInput, MemoryWriteOutput]):
    _service = memory_write


class MemoryListKeysHandler(ServiceToolHandler[MemoryListKeysInput, MemoryListKeysOutput]):
    _service = memory_list_keys


class MemoryDeleteKeyHandler(ServiceToolHandler[MemoryDeleteKeyInput, MemoryDeleteKeyOutput]):
    _service = memory_delete_key


class MemorySearchHandler(ServiceToolHandler[MemorySearchInput, MemorySearchOutput]):
    _service = memory_search
