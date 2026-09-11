# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider-neutral vector index metadata read contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.integrations.contracts.vector_index_administration import VectorIndexIdentity


@dataclass(frozen=True, slots=True)
class VectorIndexPointPayload:
    """Typed payload for one logical vector index point."""

    payload: dict[str, str | int]


@runtime_checkable
class VectorIndexMetadataReader(Protocol):
    """Read durable index metadata and bounded fallback points by logical identity."""

    def retrieve_point_by_logical_id(
        self,
        identity: VectorIndexIdentity,
        logical_point_id: str,
    ) -> VectorIndexPointPayload | None: ...

    def retrieve_first_point_payload(
        self,
        identity: VectorIndexIdentity,
        *,
        limit: int = 1,
    ) -> VectorIndexPointPayload | None: ...

    def close(self) -> None: ...


__all__ = [
    "VectorIndexMetadataReader",
    "VectorIndexPointPayload",
]
