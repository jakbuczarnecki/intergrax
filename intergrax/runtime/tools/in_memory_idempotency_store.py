# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from typing import Dict, Optional, Tuple

from pydantic import BaseModel

from intergrax.contracts.idempotency_store import (
    IdempotencyStore,
    InvocationStatus,
)
from intergrax.contracts.persistence_topology import PersistenceTopology
from intergrax.tools.execution_models import ToolExecutionResult


class InMemoryIdempotencyStore(IdempotencyStore):
    """Process-local ledger — state is lost on process restart."""

    @property
    def persistence_topology(self) -> PersistenceTopology:
        return PersistenceTopology.PROCESS_LOCAL

    def __init__(self) -> None:
        self._store: Dict[
            Tuple[str, str],
            Tuple[InvocationStatus, Optional[ToolExecutionResult[BaseModel]]],
        ] = {}

    def get_status(
        self,
        tenant_id: str,
        key: str,
    ) -> Optional[InvocationStatus]:
        entry = self._store.get((tenant_id, key))
        if entry is None:
            return None
        return entry[0]

    def record_started(
        self,
        tenant_id: str,
        key: str,
        lease_seconds: Optional[int] = None,
    ) -> None:
        composite_key = (tenant_id, key)

        if composite_key in self._store:
            status, _ = self._store[composite_key]
            if status == InvocationStatus.STARTED:
                raise RuntimeError(
                    f"Invocation already started for key={key}"
                )
            if status == InvocationStatus.COMPLETED:
                raise RuntimeError(
                    f"Invocation already completed for key={key}"
                )

        self._store[composite_key] = (InvocationStatus.STARTED, None)

    def record_completed(
        self,
        tenant_id: str,
        key: str,
        result: ToolExecutionResult[BaseModel],
        completed_ttl_seconds: Optional[int] = None,
    ) -> None:
        composite_key = (tenant_id, key)

        current = self._store.get(composite_key)
        if current is None:
            raise RuntimeError(
                "Cannot mark completed without STARTED state."
            )

        status, _ = current
        if status != InvocationStatus.STARTED:
            raise RuntimeError(
                "Cannot mark completed: invalid state transition."
            )

        self._store[composite_key] = (
            InvocationStatus.COMPLETED,
            result,
        )

    def get_completed_result(
        self,
        tenant_id: str,
        key: str,
    ) -> Optional[ToolExecutionResult[BaseModel]]:
        entry = self._store.get((tenant_id, key))
        if entry is None:
            return None

        status, result = entry
        if status == InvocationStatus.COMPLETED:
            return result

        return None