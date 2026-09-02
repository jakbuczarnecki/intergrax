"""Provider-neutral bootstrap ports — orchestrator depends on contracts only."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from platform_proofs.scenarios.verified_product_identification.application.domain.search_representation import (
    DerivedOfferSearchRepresentation,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.wdc_source_offer import (
    WdcSourceOffer,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.results import (
    EmbeddingProbeResult,
    ValidationReport,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.model import (
    VpiBootstrapManifest,
)


@dataclass(frozen=True, slots=True)
class CatalogIngestRecord:
    global_row_index: int
    record_json: str
    source_offer: WdcSourceOffer
    representation: DerivedOfferSearchRepresentation


@dataclass(frozen=True, slots=True)
class CatalogIngestBatch:
    batch_ordinal: int
    records: tuple[CatalogIngestRecord, ...]


@dataclass(frozen=True, slots=True)
class CatalogIngestBatchResult:
    """Authoritative catalog totals after the batch commit."""

    source_offer_count: int
    identifier_count: int
    structured_attribute_count: int


@dataclass(frozen=True, slots=True)
class SearchIndexIngestRecord:
    logical_point_id: str
    dense_embedding: tuple[float, ...]
    lexical_text: str
    source_ref: SourceRecordRef
    derivation_version: str
    dataset_checksum: str
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int


@dataclass(frozen=True, slots=True)
class SearchIndexIngestBatch:
    batch_ordinal: int
    records: tuple[SearchIndexIngestRecord, ...]


@dataclass(frozen=True, slots=True)
class SearchIndexIngestBatchResult:
    """Authoritative search-index point total after the batch upsert."""

    point_count: int


class CatalogBootstrapPort(Protocol):
    def probe_readiness(self) -> ValidationReport: ...

    def prepare(self, manifest: VpiBootstrapManifest) -> ValidationReport: ...

    def ingest_batch(self, batch: CatalogIngestBatch) -> CatalogIngestBatchResult: ...

    def validate(self, manifest: VpiBootstrapManifest) -> ValidationReport: ...

    def read_manifest(self) -> VpiBootstrapManifest | None: ...

    def write_manifest(self, manifest: VpiBootstrapManifest) -> None: ...

    def close(self) -> None: ...


class SearchIndexBootstrapPort(Protocol):
    def probe_readiness(self) -> ValidationReport: ...

    def prepare(self, manifest: VpiBootstrapManifest) -> ValidationReport: ...

    def ingest_batch(self, batch: SearchIndexIngestBatch) -> SearchIndexIngestBatchResult: ...

    def validate(self, manifest: VpiBootstrapManifest) -> ValidationReport: ...

    def count_points(self) -> int: ...

    def close(self) -> None: ...


class EmbeddingExecutionPort(Protocol):
    def probe(self) -> EmbeddingProbeResult: ...

    def embed_batch(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]: ...

    def close(self) -> None: ...
