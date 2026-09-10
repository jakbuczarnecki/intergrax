# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Qdrant vector index metadata reader — vendor SDK ownership boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.integrations.contracts.base import IntegrationDependencyError
from intergrax.integrations.contracts.vector_index_administration import VectorIndexIdentity
from intergrax.integrations.contracts.vector_index_metadata import (
    VectorIndexMetadataReader,
    VectorIndexPointPayload,
)
from intergrax.integrations.providers.vector_store.qdrant.config import QdrantIntegrationConfig
from intergrax.integrations.providers.vector_store.qdrant.point_ids import (
    normalize_qdrant_logical_point_id,
)


type QdrantProviderPayloadInput = Mapping[str, str | int] | None


@runtime_checkable
class QdrantMetadataReadClient(Protocol):
    def retrieve(
        self,
        collection_name: str,
        ids: Sequence[str | int],
        *,
        with_payload: bool,
        with_vectors: bool,
    ) -> Sequence[QdrantMetadataPoint]: ...

    def scroll(
        self,
        collection_name: str,
        *,
        limit: int,
        with_payload: bool,
        with_vectors: bool,
    ) -> tuple[Sequence[QdrantMetadataPoint], str | int | None]: ...

    def close(self) -> None: ...


@runtime_checkable
class QdrantMetadataPoint(Protocol):
    payload: QdrantProviderPayloadInput


def _physical_index_name(identity: VectorIndexIdentity) -> str:
    return f"{identity.logical_name}__tenant__{identity.tenant_id}"


def _typed_payload(raw_payload: QdrantProviderPayloadInput) -> dict[str, str | int] | None:
    if raw_payload is None:
        return None
    converted: dict[str, str | int] = {}
    for key, value in raw_payload.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, str) or isinstance(value, int):
            converted[key] = value
    if not converted:
        return None
    return converted


@dataclass(slots=True)
class QdrantVectorIndexMetadataReader:
    """Qdrant implementation of ``VectorIndexMetadataReader``."""

    _client: QdrantMetadataReadClient
    _config: QdrantIntegrationConfig

    def retrieve_point_by_logical_id(
        self,
        identity: VectorIndexIdentity,
        logical_point_id: str,
    ) -> VectorIndexPointPayload | None:
        point_id = normalize_qdrant_logical_point_id(logical_point_id)
        try:
            points = self._client.retrieve(
                _physical_index_name(identity),
                (point_id,),
                with_payload=True,
                with_vectors=False,
            )
        except (OSError, ConnectionError, TimeoutError, IntegrationDependencyError):
            return None
        if not points:
            return None
        typed = _typed_payload(points[0].payload)
        if typed is None:
            return None
        return VectorIndexPointPayload(payload=typed)

    def retrieve_first_point_payload(
        self,
        identity: VectorIndexIdentity,
        *,
        limit: int = 1,
    ) -> VectorIndexPointPayload | None:
        if limit <= 0:
            return None
        try:
            points, _offset = self._client.scroll(
                _physical_index_name(identity),
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        except (OSError, ConnectionError, TimeoutError, IntegrationDependencyError):
            return None
        if not points:
            return None
        typed = _typed_payload(points[0].payload)
        if typed is None:
            return None
        return VectorIndexPointPayload(payload=typed)

    def close(self) -> None:
        self._client.close()


__all__ = [
    "QdrantMetadataReadClient",
    "QdrantVectorIndexMetadataReader",
]
