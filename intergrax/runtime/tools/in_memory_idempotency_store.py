# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel

from intergrax.runtime.tools.idempotency_store import IdempotencyStore
from intergrax.tools.execution_models import ToolExecutionResult


class InMemoryIdempotencyStore(IdempotencyStore):
    """
    In-memory idempotency store.

    Intended for:
    - unit tests
    - local runtime
    - deterministic retry verification

    Not production-safe (no persistence, no concurrency guarantees).
    """

    def __init__(self) -> None:
        self._store: Dict[str, ToolExecutionResult[BaseModel]] = {}

    def check(self, key: str) -> Optional[ToolExecutionResult[BaseModel]]:
        return self._store.get(key)

    def save(self, key: str, result: ToolExecutionResult[BaseModel]) -> None:
        self._store[key] = result
