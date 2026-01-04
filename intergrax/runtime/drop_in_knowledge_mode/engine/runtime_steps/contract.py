# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from typing import Protocol
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_context import RuntimeContext
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_state import RuntimeState

class RuntimeStep(Protocol):
    async def run(self, state: RuntimeState, ctx: RuntimeContext) -> None:
        ...
