# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Relational store integration contract (§7.1.2, Phase M.2)."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@runtime_checkable
class RelationalStore(Protocol):
    """
    Backend-agnostic SQL persistence facade.

    Transport only — ``connect`` / ``execute`` / ``fetch_all`` / ``close`` do not
    certify multi-host safety, CAS, claim/lease, exactly-once, or arbitrary domain
    port transaction semantics. Domain stores (IdempotencyStore, CheckpointStore, …)
    declare ``persistence_topology`` and own correctness guarantees.

    Implementations: sqlite, postgresql, mysql, …
    """

    def connect(self) -> None:
        """Open or validate a connection pool / file handle."""

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        """Run a statement that does not return rows."""

    def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Mapping[str, Any]]:
        """Run a query and return row mappings."""

    def close(self) -> None:
        """Release resources."""
