# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Optional, Protocol

from pydantic import BaseModel

from intergrax.tools.execution_models import ToolExecutionResult


class IdempotencyStore(Protocol):
    """
    Idempotency port for tool invocations.

    Contract:
    - check(key) returns a previously saved ToolExecutionResult or None
    - save(key, result) persists the ToolExecutionResult for future deduplication
    """

    def check(self, key: str) -> Optional[ToolExecutionResult[BaseModel]]:
        ...

    def save(self, key: str, result: ToolExecutionResult[BaseModel]) -> None:
        ...
