# © Artur Czarnecki. All rights reserved.

"""Provider instance factories for isolated qualification runs (MEM-ENT-13)."""

from __future__ import annotations

from typing import Protocol, TypeVar

T = TypeVar("T", contravariant=False)


class MemoryProviderInstanceFactory(Protocol[T]):
    """Creates and disposes ephemeral provider instances for qualification."""

    async def create(self) -> T: ...

    async def dispose(self, instance: T) -> None: ...
