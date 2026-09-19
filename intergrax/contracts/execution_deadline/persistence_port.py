# © Artur Czarnecki. All rights reserved.

"""Durable execution deadline authority persistence port."""

from __future__ import annotations

from typing import Protocol

from intergrax.contracts.execution_identity import RunId


class ExecutionDeadlinePersistenceError(RuntimeError):
    """Raised when durable deadline authority cannot be loaded or stored safely."""


class ExecutionDeadlineAuthorityPersistencePort(Protocol):
    def load(self, *, tenant_id: str, run_id: RunId) -> bytes | None:
        """Return encoded snapshot bytes or ``None`` when absent."""

    def compare_and_create(
        self,
        *,
        tenant_id: str,
        run_id: RunId,
        expected: bytes | None,
        encoded_snapshot: bytes,
    ) -> bool:
        """Atomically store snapshot when ``expected`` matches current bytes."""
