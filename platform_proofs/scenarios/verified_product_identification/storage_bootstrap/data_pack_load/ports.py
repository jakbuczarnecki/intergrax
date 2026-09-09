"""Provider-neutral ports for Data Pack storage bootstrap."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.embedding import (
    EmbeddingDataPackRecord,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.manifest import (
    DataPackManifest,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.relational import (
    RelationalDataPackRecord,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    BootstrapProgress,
    BootstrapRequest,
    BootstrapResult,
    RelationalBatch,
    StorageLoadBatchResult,
    VectorBatch,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.progress import (
    BootstrapProgressSinkPort,
)


@dataclass(frozen=True, slots=True)
class PairedDataPackRecord:
    relational: RelationalDataPackRecord
    embedding: EmbeddingDataPackRecord


class DataPackBootstrapReaderPort(Protocol):
    """Scenario-owned streaming reader — bounded memory per shard/batch."""

    def read_manifest(self) -> DataPackManifest: ...

    def iter_paired_records(self) -> Iterator[PairedDataPackRecord]: ...

    def close(self) -> None: ...


class RelationalStorageLoadPort(Protocol):
    """Relational backend adapter — upsert or insert-if-absent semantics."""

    def write_batch(self, batch: RelationalBatch) -> StorageLoadBatchResult: ...

    def verify_batch(self, batch: RelationalBatch) -> StorageLoadBatchResult: ...


class VectorStorageLoadPort(Protocol):
    """Vector backend adapter — consumes pre-materialized dense embeddings only."""

    def write_batch(self, batch: VectorBatch) -> StorageLoadBatchResult: ...

    def verify_batch(self, batch: VectorBatch) -> StorageLoadBatchResult: ...


class StorageBootstrapServicePort(Protocol):
    def run(
        self,
        request: BootstrapRequest,
        *,
        progress_sink: BootstrapProgressSinkPort | None = None,
    ) -> BootstrapResult: ...
