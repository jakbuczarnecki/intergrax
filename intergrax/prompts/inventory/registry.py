# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Set

from .models import (
    PromptInstructionInventory,
    PromptInstructionLocation,
    PromptInstructionKind,
)


class PromptInventoryBuilder:
    """
    Builder used to gradually collect instruction locations
    """

    def __init__(self) -> None:
        self._items: Set[PromptInstructionLocation] = set()

    def add(
        self,
        kind: PromptInstructionKind,
        module: str,
        symbol: str,
        description: str,
    ) -> None:
        self._items.add(
            PromptInstructionLocation(
                kind=kind,
                module=module,
                symbol=symbol,
                description=description,
            )
        )

    def build(self) -> PromptInstructionInventory:
        return PromptInstructionInventory(items=frozenset(self._items))
