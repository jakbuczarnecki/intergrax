"""Provider-neutral data pack ports."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
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
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.results import (
    ValidationReport,
)


class DataPackReaderPort(Protocol):
    def read_manifest(self) -> DataPackManifest: ...

    def read_relational_records(self) -> tuple[RelationalDataPackRecord, ...]: ...

    def read_embedding_records(self) -> tuple[EmbeddingDataPackRecord, ...]: ...

    def validate_integrity(self) -> ValidationReport: ...

    def close(self) -> None: ...


class DataPackWriterPort(Protocol):
    def write_relational_shard(
        self,
        shard_ordinal: int,
        records: Sequence[RelationalDataPackRecord],
    ) -> Path: ...

    def write_embedding_shard(
        self,
        shard_ordinal: int,
        records: Sequence[EmbeddingDataPackRecord],
    ) -> Path: ...

    def write_manifest(self, manifest: DataPackManifest) -> None: ...

    def close(self) -> None: ...
