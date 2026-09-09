"""Filesystem Data Pack reader for bounded-memory storage bootstrap."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    VpiDataPackCompatibilityError,
    VpiDataPackFormatError,
    VpiDataPackIntegrityError,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.manifest import (
    DataPackManifest,
    read_manifest_file,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.paths import (
    DataPackPaths,
    resolve_data_pack_paths,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.shard_index import (
    ShardDescriptor,
    ShardIndex,
    read_shard_index_file,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.status import (
    DataPackStatus,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.errors import (
    StorageBootstrapIdentityError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.mapping import (
    assert_paired_identity,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.ports import (
    DataPackBootstrapReaderPort,
    PairedDataPackRecord,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.reader.errors import (
    DataPackReaderError,
    DataPackReaderIntegrityError,
    DataPackReaderOrderingError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.reader.parquet_streaming import (
    iter_embedding_records,
    iter_relational_records,
)

_DEFAULT_PARQUET_BATCH_SIZE = 256


class FilesystemDataPackBootstrapReader(DataPackBootstrapReaderPort):
    """Streaming reader over a finalized canonical Data Pack on local filesystem."""

    def __init__(
        self,
        artifact_root: Path,
        *,
        parquet_batch_size: int = _DEFAULT_PARQUET_BATCH_SIZE,
    ) -> None:
        if parquet_batch_size <= 0:
            raise ValueError("parquet_batch_size must be > 0")
        self._paths = resolve_data_pack_paths(artifact_root)
        self._parquet_batch_size = parquet_batch_size
        self._manifest: DataPackManifest | None = None
        self._shard_index: ShardIndex | None = None
        self._closed = False

    @property
    def paths(self) -> DataPackPaths:
        return self._paths

    def read_manifest(self) -> DataPackManifest:
        if self._closed:
            raise DataPackReaderIntegrityError("reader is closed")
        if self._manifest is None:
            self._manifest = self._load_manifest()
        if self._manifest.status is not DataPackStatus.READY:
            raise DataPackReaderIntegrityError(
                f"data pack status must be READY (got {self._manifest.status.value})"
            )
        return self._manifest

    def iter_paired_records(self) -> Iterator[PairedDataPackRecord]:
        manifest = self.read_manifest()
        shard_index = self._load_shard_index(manifest)
        self._validate_shard_index(shard_index, manifest)

        expected_global_row_index = 0
        records_yielded = 0

        for relational_descriptor, embedding_descriptor in zip(
            shard_index.relational_shards,
            shard_index.embedding_shards,
            strict=True,
        ):
            self._validate_shard_pair(
                relational_descriptor,
                embedding_descriptor,
                expected_start_global_row_index=expected_global_row_index,
            )
            relational_path = self._paths.root / relational_descriptor.relative_path
            embedding_path = self._paths.root / embedding_descriptor.relative_path
            if not relational_path.is_file():
                raise DataPackReaderIntegrityError(
                    f"missing relational shard: {relational_descriptor.relative_path}"
                )
            if not embedding_path.is_file():
                raise DataPackReaderIntegrityError(
                    f"missing embedding shard: {embedding_descriptor.relative_path}"
                )

            relational_iter = iter_relational_records(
                relational_path,
                batch_size=self._parquet_batch_size,
            )
            embedding_iter = iter_embedding_records(
                embedding_path,
                expected_dimension=manifest.embedding_identity.dimension,
                batch_size=self._parquet_batch_size,
            )

            shard_rows = 0
            try:
                for relational, embedding in zip(relational_iter, embedding_iter, strict=True):
                    if relational.global_row_index != expected_global_row_index:
                        raise DataPackReaderOrderingError(
                            "DATA_PACK_ORDERING_VIOLATION: "
                            f"expected global_row_index {expected_global_row_index}, "
                            f"found {relational.global_row_index}"
                        )

                    pair = PairedDataPackRecord(relational=relational, embedding=embedding)
                    try:
                        assert_paired_identity(pair)
                    except StorageBootstrapIdentityError as exc:
                        raise DataPackReaderIntegrityError(str(exc)) from exc
                    yield pair

                    expected_global_row_index += 1
                    records_yielded += 1
                    shard_rows += 1
            except ValueError as exc:
                raise DataPackReaderIntegrityError(
                    f"relational/embedding row count mismatch in shard "
                    f"{relational_descriptor.ordinal}"
                ) from exc

            if shard_rows != relational_descriptor.record_count:
                raise DataPackReaderIntegrityError(
                    f"shard {relational_descriptor.ordinal} record_count mismatch: "
                    f"expected {relational_descriptor.record_count}, streamed {shard_rows}"
                )

        if records_yielded != manifest.record_count:
            raise DataPackReaderIntegrityError(
                "DATA_PACK_RECORD_COUNT_MISMATCH: "
                f"manifest declares {manifest.record_count}, streamed {records_yielded}"
            )

    def close(self) -> None:
        self._closed = True

    def _load_manifest(self) -> DataPackManifest:
        try:
            return read_manifest_file(self._paths.manifest_file)
        except json.JSONDecodeError as exc:
            raise DataPackReaderIntegrityError(
                f"malformed manifest JSON: {self._paths.manifest_file}"
            ) from exc
        except (
            VpiDataPackIntegrityError,
            VpiDataPackFormatError,
            VpiDataPackCompatibilityError,
        ) as exc:
            raise DataPackReaderIntegrityError(str(exc)) from exc

    def _load_shard_index(self, manifest: DataPackManifest) -> ShardIndex:
        if self._shard_index is not None:
            return self._shard_index
        shard_index_path = self._paths.root / manifest.shards_index_path
        if not shard_index_path.is_file():
            raise DataPackReaderIntegrityError(f"shard index not found: {shard_index_path}")
        try:
            self._shard_index = read_shard_index_file(shard_index_path)
        except json.JSONDecodeError as exc:
            raise DataPackReaderIntegrityError(
                f"malformed shard index JSON: {shard_index_path}"
            ) from exc
        except (VpiDataPackIntegrityError, VpiDataPackFormatError) as exc:
            raise DataPackReaderIntegrityError(str(exc)) from exc
        return self._shard_index

    @staticmethod
    def _validate_shard_index(shard_index: ShardIndex, manifest: DataPackManifest) -> None:
        if shard_index.shard_count != manifest.shard_count:
            raise DataPackReaderIntegrityError(
                f"shard_count mismatch: manifest={manifest.shard_count}, "
                f"index={shard_index.shard_count}"
            )
        relational_ordinals = [descriptor.ordinal for descriptor in shard_index.relational_shards]
        embedding_ordinals = [descriptor.ordinal for descriptor in shard_index.embedding_shards]
        expected_ordinals = list(range(1, shard_index.shard_count + 1))
        if relational_ordinals != expected_ordinals:
            raise DataPackReaderOrderingError(
                f"DATA_PACK_ORDERING_VIOLATION: relational shard ordinals {relational_ordinals}"
            )
        if embedding_ordinals != expected_ordinals:
            raise DataPackReaderOrderingError(
                f"DATA_PACK_ORDERING_VIOLATION: embedding shard ordinals {embedding_ordinals}"
            )

    @staticmethod
    def _validate_shard_pair(
        relational_descriptor: ShardDescriptor,
        embedding_descriptor: ShardDescriptor,
        *,
        expected_start_global_row_index: int,
    ) -> None:
        if relational_descriptor.ordinal != embedding_descriptor.ordinal:
            raise DataPackReaderIntegrityError(
                f"shard ordinal mismatch: relational={relational_descriptor.ordinal} "
                f"embedding={embedding_descriptor.ordinal}"
            )
        if relational_descriptor.record_count != embedding_descriptor.record_count:
            raise DataPackReaderIntegrityError(
                f"shard {relational_descriptor.ordinal} record_count mismatch: "
                f"relational={relational_descriptor.record_count} "
                f"embedding={embedding_descriptor.record_count}"
            )
        expected_end = (
            expected_start_global_row_index + relational_descriptor.record_count - 1
        )
        if relational_descriptor.record_count > 0 and expected_end < expected_start_global_row_index:
            raise DataPackReaderIntegrityError(
                f"invalid shard row range for ordinal {relational_descriptor.ordinal}"
            )
