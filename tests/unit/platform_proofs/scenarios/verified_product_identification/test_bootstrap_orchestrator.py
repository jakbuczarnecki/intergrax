"""Unit tests for provider-neutral VPI bootstrap orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from platform_proofs.scenarios.verified_product_identification.application.config.embedding_configuration import (
    EMBEDDING_CONFIGURATION_VERSION,
    VpiEmbeddingConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.search_representation import (
    SEARCH_REPRESENTATION_DERIVATION_VERSION,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.config import (
    DatasetVerificationMode,
    VpiBootstrapConfig,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.errors import (
    VpiBootstrapCompatibilityError,
    VpiBootstrapProviderError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.ports import (
    CatalogIngestBatch,
    CatalogIngestBatchResult,
    CatalogIngestRecord,
    SearchIndexIngestBatch,
    SearchIndexIngestBatchResult,
    SearchIndexIngestRecord,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.results import (
    ValidationCheck,
    ValidationReport,
    ValidationStatus,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.compatibility import (
    BootstrapCompatibilityIdentity,
    assert_manifest_compatible,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.model import (
    BOOTSTRAP_IMPLEMENTATION_VERSION,
    CATALOG_SCHEMA_VERSION,
    SEARCH_INDEX_SCHEMA_VERSION,
    BootstrapState,
    VpiBootstrapManifest,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.orchestration.orchestrator import (
    VpiBootstrapDependencies,
    VpiBootstrapOrchestrator,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.validation.embedding_gate import (
    RegistryEmbeddingReadinessProbe,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.validation.ready_gate import (
    evaluate_ready_gate,
)
from platform_proofs.scenarios.verified_product_identification.ingest.source_reader.parquet_dataset import (
    iter_dataset_rows,
)
from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider
from intergrax.rag.embedding.registry.embedding_provider_registry import EmbeddingProviderRegistry
from intergrax.rag.embedding.registry.profile import EmbeddingProfile

pytestmark = pytest.mark.unit


def _sample_manifest(**overrides: object) -> VpiBootstrapManifest:
    base = VpiBootstrapManifest(
        state=BootstrapState.INITIALIZING,
        dataset_path="/data/selected_offers.parquet",
        dataset_checksum="abc123",
        dataset_record_count=100,
        search_representation_derivation_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        embedding_configuration_version=EMBEDDING_CONFIGURATION_VERSION,
        embedding_provider="hf",
        embedding_model="fake-model",
        embedding_dimension=8,
        catalog_schema_version=CATALOG_SCHEMA_VERSION,
        search_index_schema_version=SEARCH_INDEX_SCHEMA_VERSION,
        bootstrap_implementation_version=BOOTSTRAP_IMPLEMENTATION_VERSION,
        catalog_id="wdc-v2-selected",
        source_revision=None,
        checkpoint_batch_ordinal=None,
        checkpoint_rows_processed=0,
        target_max_records=2,
        catalog_source_offer_count=0,
        catalog_identifier_count=0,
        catalog_structured_attribute_count=0,
        search_point_count=0,
    )
    if not overrides:
        return base
    fields = {key: getattr(base, key) for key in base.__dataclass_fields__}
    fields.update(overrides)
    return VpiBootstrapManifest(**fields)


@dataclass
class FakeCatalogPort:
    manifest: VpiBootstrapManifest | None = None
    batches: list[CatalogIngestBatch] = field(default_factory=list)
    fail_on_batch: int | None = None
    prepare_calls: int = 0

    def probe_readiness(self) -> ValidationReport:
        return ValidationReport.from_checks(
            (ValidationCheck("postgresql_reachable", ValidationStatus.PASS, "ok"),)
        )

    def prepare(self, manifest: VpiBootstrapManifest) -> ValidationReport:
        self.prepare_calls += 1
        return ValidationReport.from_checks(
            (ValidationCheck("postgresql_schema_ready", ValidationStatus.PASS, "ok"),)
        )

    def ingest_batch(self, batch: CatalogIngestBatch) -> CatalogIngestBatchResult:
        if self.fail_on_batch == batch.batch_ordinal:
            raise VpiBootstrapProviderError("catalog ingest failed")
        self.batches.append(batch)
        identifiers = sum(len(record.representation.exact.terms) for record in batch.records)
        structured = sum(
            len(record.representation.structured.attributes) for record in batch.records
        )
        return CatalogIngestBatchResult(
            source_offers_ingested=len(batch.records),
            identifiers_ingested=identifiers,
            structured_attributes_ingested=structured,
        )

    def validate(self, manifest: VpiBootstrapManifest) -> ValidationReport:
        return ValidationReport.from_checks(
            (
                ValidationCheck(
                    "source_offer_count",
                    ValidationStatus.PASS,
                    f"count={manifest.catalog_source_offer_count}",
                ),
            )
        )

    def read_manifest(self) -> VpiBootstrapManifest | None:
        return self.manifest

    def write_manifest(self, manifest: VpiBootstrapManifest) -> None:
        self.manifest = manifest

    def close(self) -> None:
        return None


@dataclass
class FakeSearchPort:
    batches: list[SearchIndexIngestBatch] = field(default_factory=list)
    fail_on_batch: int | None = None
    point_count: int = 0

    def probe_readiness(self) -> ValidationReport:
        return ValidationReport.from_checks(
            (ValidationCheck("qdrant_reachable", ValidationStatus.PASS, "ok"),)
        )

    def prepare(self, manifest: VpiBootstrapManifest) -> ValidationReport:
        _ = manifest
        return ValidationReport.from_checks(
            (ValidationCheck("qdrant_collection_created", ValidationStatus.PASS, "ok"),)
        )

    def ingest_batch(self, batch: SearchIndexIngestBatch) -> SearchIndexIngestBatchResult:
        if self.fail_on_batch == batch.batch_ordinal:
            raise VpiBootstrapProviderError("search ingest failed")
        self.batches.append(batch)
        self.point_count += len(batch.records)
        return SearchIndexIngestBatchResult(points_ingested=len(batch.records))

    def validate(self, manifest: VpiBootstrapManifest) -> ValidationReport:
        return ValidationReport.from_checks(
            (
                ValidationCheck(
                    "qdrant_point_count",
                    ValidationStatus.PASS if self.point_count >= manifest.checkpoint_rows_processed else ValidationStatus.FAIL,
                    f"points={self.point_count}",
                ),
            )
        )

    def count_points(self) -> int:
        return self.point_count

    def close(self) -> None:
        return None


class FakeEmbeddingProbe:
    def __init__(self, *, should_pass: bool = True) -> None:
        self.should_pass = should_pass
        self.calls = 0

    def probe(self) -> ValidationReport:
        self.calls += 1
        status = ValidationStatus.PASS if self.should_pass else ValidationStatus.FAIL
        return ValidationReport.from_checks(
            (ValidationCheck("embedding_gate0", status, "probe"),)
        )


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self, *, dimension: int = 8) -> None:
        self._dimension = dimension

    def provider_name(self) -> str:
        return "hf"

    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            seed = sum(ord(char) for char in text) % 7 + 1
            rows.append(np.full(self._dimension, seed, dtype=np.float32))
        return np.stack(rows, axis=0)


def _orchestrator(
    tmp_path: Path,
    *,
    catalog: FakeCatalogPort | None = None,
    search: FakeSearchPort | None = None,
    embedding_probe: FakeEmbeddingProbe | None = None,
    max_records: int = 2,
    monkeypatch: pytest.MonkeyPatch,
) -> VpiBootstrapOrchestrator:
    dataset_path = tmp_path / "selected_offers.parquet"
    _write_tiny_parquet(dataset_path, row_count=5)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        '{"output_sha256":"testchecksum","selected_record_count":5}',
        encoding="utf-8",
    )

    config = VpiBootstrapConfig(
        dataset_path=dataset_path,
        dataset_manifest_path=manifest_path,
        dataset_verification_mode=DatasetVerificationMode.FAST,
        catalog_id="wdc-v2-selected",
        source_revision=None,
        max_records=max_records,
        source_batch_size=2,
        vector_batch_size=2,
        catalog_schema_version=CATALOG_SCHEMA_VERSION,
        search_index_schema_version=SEARCH_INDEX_SCHEMA_VERSION,
        bootstrap_implementation_version=BOOTSTRAP_IMPLEMENTATION_VERSION,
        postgresql_schema="vpi",
        qdrant_collection_name="vpi_offers",
        embedding_configuration=VpiEmbeddingConfiguration(
            profile=EmbeddingProfile(provider="hf", model="fake-model"),
            expected_dimension=8,
        ),
    )

    registry = EmbeddingProviderRegistry([FakeEmbeddingProvider(dimension=8)])

    def _fake_registry(*_args: object, **_kwargs: object) -> EmbeddingProviderRegistry:
        return registry

    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.storage_bootstrap.orchestration.orchestrator.create_default_registry",
        _fake_registry,
    )

    return VpiBootstrapOrchestrator(
        config=config,
        dependencies=VpiBootstrapDependencies(
            catalog=catalog or FakeCatalogPort(),
            search=search or FakeSearchPort(),
            embedding_probe=embedding_probe or FakeEmbeddingProbe(),
        ),
    )


def _write_tiny_parquet(path: Path, *, row_count: int) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    records = []
    for index in range(row_count):
        records.append(
            '{"id":"%s","title":"relay module","identifiers":[{"gtin":"%s"}],"keyValuePairs":{"voltage":"24V"}}'
            % (1000 + index, 1000000000000 + index)
        )
    table = pa.table({"record_json": records})
    pq.write_table(table, path)


def test_stage_ordering_and_ready(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    catalog = FakeCatalogPort()
    search = FakeSearchPort()
    orchestrator = _orchestrator(tmp_path, catalog=catalog, search=search, monkeypatch=monkeypatch)

    report = orchestrator.run()

    assert report.final_state is BootstrapState.READY
    assert catalog.prepare_calls == 1
    assert len(catalog.batches) == 1
    assert len(search.batches) == 1
    assert catalog.manifest is not None
    assert catalog.manifest.checkpoint_rows_processed == 2


def test_embedding_gate_blocks_before_ingest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    catalog = FakeCatalogPort()
    search = FakeSearchPort()
    orchestrator = _orchestrator(
        tmp_path,
        catalog=catalog,
        search=search,
        embedding_probe=FakeEmbeddingProbe(should_pass=False),
        monkeypatch=monkeypatch,
    )

    report = orchestrator.run()

    assert report.final_state is BootstrapState.FAILED
    assert catalog.batches == []
    assert search.batches == []


def test_catalog_failure_not_ready(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    catalog = FakeCatalogPort(fail_on_batch=0)
    orchestrator = _orchestrator(tmp_path, catalog=catalog, monkeypatch=monkeypatch)

    report = orchestrator.run()

    assert report.final_state is BootstrapState.FAILED
    assert catalog.manifest is not None
    assert catalog.manifest.state is BootstrapState.FAILED


def test_search_failure_not_ready(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    search = FakeSearchPort(fail_on_batch=0)
    orchestrator = _orchestrator(tmp_path, search=search, monkeypatch=monkeypatch)

    report = orchestrator.run()

    assert report.final_state is BootstrapState.FAILED


def test_retry_same_batch_idempotent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    catalog = FakeCatalogPort()
    search = FakeSearchPort()
    orchestrator = _orchestrator(tmp_path, catalog=catalog, search=search, monkeypatch=monkeypatch)

    first = orchestrator.run()
    second = orchestrator.run()

    assert first.final_state is BootstrapState.READY
    assert second.final_state is BootstrapState.READY
    assert len(catalog.batches) == 1


def test_incompatible_manifest_fails_closed() -> None:
    existing = _sample_manifest(embedding_model="other-model")
    expected = BootstrapCompatibilityIdentity(
        dataset_checksum="abc123",
        dataset_record_count=100,
        search_representation_derivation_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        embedding_configuration_version=EMBEDDING_CONFIGURATION_VERSION,
        embedding_provider="hf",
        embedding_model="fake-model",
        embedding_dimension=8,
        catalog_schema_version=CATALOG_SCHEMA_VERSION,
        search_index_schema_version=SEARCH_INDEX_SCHEMA_VERSION,
        bootstrap_implementation_version=BOOTSTRAP_IMPLEMENTATION_VERSION,
        catalog_id="wdc-v2-selected",
    )
    with pytest.raises(VpiBootstrapCompatibilityError, match="embedding_model"):
        assert_manifest_compatible(existing=existing, expected=expected)


def test_derivation_version_mismatch_fails_closed() -> None:
    existing = _sample_manifest(search_representation_derivation_version="v1")
    expected = BootstrapCompatibilityIdentity(
        dataset_checksum="abc123",
        dataset_record_count=100,
        search_representation_derivation_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        embedding_configuration_version=EMBEDDING_CONFIGURATION_VERSION,
        embedding_provider="hf",
        embedding_model="fake-model",
        embedding_dimension=8,
        catalog_schema_version=CATALOG_SCHEMA_VERSION,
        search_index_schema_version=SEARCH_INDEX_SCHEMA_VERSION,
        bootstrap_implementation_version=BOOTSTRAP_IMPLEMENTATION_VERSION,
        catalog_id="wdc-v2-selected",
    )
    with pytest.raises(VpiBootstrapCompatibilityError, match="derivation"):
        assert_manifest_compatible(existing=existing, expected=expected)


def test_dataset_checksum_mismatch_fails_closed() -> None:
    existing = _sample_manifest(dataset_checksum="other")
    expected = BootstrapCompatibilityIdentity(
        dataset_checksum="abc123",
        dataset_record_count=100,
        search_representation_derivation_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        embedding_configuration_version=EMBEDDING_CONFIGURATION_VERSION,
        embedding_provider="hf",
        embedding_model="fake-model",
        embedding_dimension=8,
        catalog_schema_version=CATALOG_SCHEMA_VERSION,
        search_index_schema_version=SEARCH_INDEX_SCHEMA_VERSION,
        bootstrap_implementation_version=BOOTSTRAP_IMPLEMENTATION_VERSION,
        catalog_id="wdc-v2-selected",
    )
    with pytest.raises(VpiBootstrapCompatibilityError, match="dataset_checksum"):
        assert_manifest_compatible(existing=existing, expected=expected)


def test_deterministic_batching(tmp_path: Path) -> None:
    dataset_path = tmp_path / "rows.parquet"
    _write_tiny_parquet(dataset_path, row_count=5)
    batches = list(
        iter_dataset_rows(dataset_path, batch_size=2, start_row_index=0, max_records=4)
    )
    assert len(batches) == 2
    assert batches[0][1][0].global_row_index == 0
    assert batches[1][1][-1].global_row_index == 3


def test_checkpoint_advances_only_after_successful_batch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    catalog = FakeCatalogPort()
    orchestrator = _orchestrator(tmp_path, catalog=catalog, monkeypatch=monkeypatch)
    orchestrator.run()
    assert catalog.manifest is not None
    assert catalog.manifest.checkpoint_batch_ordinal == 0
    assert catalog.manifest.checkpoint_rows_processed == 2


def test_alternate_fake_ports_without_orchestrator_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class AltCatalog(FakeCatalogPort):
        def prepare(self, manifest: VpiBootstrapManifest) -> ValidationReport:
            return ValidationReport.from_checks(
                (ValidationCheck("alt_catalog", ValidationStatus.PASS, "alt"),)
            )

    class AltSearch(FakeSearchPort):
        def prepare(self, manifest: VpiBootstrapManifest) -> ValidationReport:
            return ValidationReport.from_checks(
                (ValidationCheck("alt_search", ValidationStatus.PASS, "alt"),)
            )

    orchestrator = _orchestrator(
        tmp_path,
        catalog=AltCatalog(),
        search=AltSearch(),
        monkeypatch=monkeypatch,
    )
    report = orchestrator.run()
    assert report.final_state is BootstrapState.READY


def test_ready_gate_requires_all_checks() -> None:
    manifest = _sample_manifest(
        state=BootstrapState.VALIDATING,
        checkpoint_rows_processed=2,
        catalog_source_offer_count=2,
        search_point_count=2,
    )
    report = evaluate_ready_gate(
        manifest=manifest,
        embedding_report=ValidationReport.from_checks(
            (ValidationCheck("embedding_gate0", ValidationStatus.PASS, "ok"),)
        ),
        catalog_report=ValidationReport.from_checks(
            (ValidationCheck("source_offer_count", ValidationStatus.PASS, "ok"),)
        ),
        search_report=ValidationReport.from_checks(
            (ValidationCheck("qdrant_point_count", ValidationStatus.FAIL, "missing"),)
        ),
        checkpoint_complete=True,
    )
    assert report.status is ValidationStatus.FAIL


def test_registry_embedding_gate_with_fake_provider() -> None:
    registry = EmbeddingProviderRegistry([FakeEmbeddingProvider(dimension=8)])
    configuration = VpiEmbeddingConfiguration(
        profile=EmbeddingProfile(provider="hf", model="fake-model"),
        expected_dimension=8,
    )
    probe = RegistryEmbeddingReadinessProbe(configuration, registry=registry)
    report = probe.probe()
    assert report.status is ValidationStatus.PASS
