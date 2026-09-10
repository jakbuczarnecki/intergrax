# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Public Qdrant vector data-plane client opener."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from intergrax.integrations.providers.vector_store.qdrant.config import QdrantIntegrationConfig
from intergrax.integrations.providers.vector_store.qdrant.client_factory import (
    build_qdrant_control_plane_client,
)

type QdrantProviderDistance = str
type QdrantProviderPayloadInput = Mapping[str, str | int] | None
type QdrantProviderVectorInput = list[float] | dict[str, list[float]] | None


class QdrantDataPlanePointView(Protocol):
    id: str | int
    payload: QdrantProviderPayloadInput
    vector: QdrantProviderVectorInput


class QdrantVectorParamsView(Protocol):
    size: int
    distance: QdrantProviderDistance


class QdrantCollectionParamsView(Protocol):
    vectors: QdrantVectorParamsView | dict[str, QdrantVectorParamsView] | None


class QdrantCollectionConfigView(Protocol):
    params: QdrantCollectionParamsView


class QdrantCollectionInfoView(Protocol):
    config: QdrantCollectionConfigView


class QdrantUpsertPointView(Protocol):
    id: str | int
    vector: list[float] | dict[str, list[float]]
    payload: dict[str, str | int]


@runtime_checkable
class QdrantVectorDataPlaneClient(Protocol):
    """Narrow Qdrant client surface for vector upsert and retrieval."""

    def retrieve(
        self,
        collection_name: str,
        ids: Sequence[str | int],
        *,
        with_payload: bool,
        with_vectors: bool,
    ) -> Sequence[QdrantDataPlanePointView]: ...

    def upsert(
        self,
        collection_name: str,
        points: Sequence[QdrantUpsertPointView],
    ) -> None: ...

    def get_collection(self, collection_name: str) -> QdrantCollectionInfoView: ...

    def close(self) -> None: ...


def open_qdrant_vector_data_plane_client(
    config: QdrantIntegrationConfig,
) -> QdrantVectorDataPlaneClient:
    """Open a typed Qdrant data-plane client without exposing the raw SDK type."""
    return build_qdrant_control_plane_client(config)


__all__ = [
    "QdrantVectorDataPlaneClient",
    "open_qdrant_vector_data_plane_client",
]
