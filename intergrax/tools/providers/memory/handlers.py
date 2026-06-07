# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.memory.contracts import (
    MemoryListKeysInput,
    MemoryListKeysOutput,
    MemoryReadInput,
    MemoryReadOutput,
    MemoryWriteInput,
    MemoryWriteOutput,
)
from intergrax.tools.providers.memory.service import memory_list_keys, memory_read, memory_write


class MemoryReadHandler(ServiceToolHandler[MemoryReadInput, MemoryReadOutput]):
    _service = memory_read


class MemoryWriteHandler(ServiceToolHandler[MemoryWriteInput, MemoryWriteOutput]):
    _service = memory_write


class MemoryListKeysHandler(ServiceToolHandler[MemoryListKeysInput, MemoryListKeysOutput]):
    _service = memory_list_keys
