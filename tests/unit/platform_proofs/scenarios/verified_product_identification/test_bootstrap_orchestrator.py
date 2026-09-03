"""Unit tests for provider-neutral VPI bootstrap orchestration."""

from __future__ import annotations

import ast
from dataclasses import dataclass, fields, field
from pathlib import Path
from types import SimpleNamespace

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
    EmbeddingProbeResult,
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
from platform_proofs.scenarios.verified_product_identification.integrations.search_store.platform_bootstrap_adapter import (
    PlatformSearchIndexBootstrapAdapter,
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
from intergrax.integrations.contracts.base import HealthStatus
from intergrax.integrations.contracts.vector_index_administration import (
    VectorIndexDescription,
    VectorIndexIdentity,
    VectorSearchCapability,
)
from intergrax.integrations.contracts.vector_store import VectorStoreScope
from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider
from intergrax.rag.embedding.registry.embedding_provider_registry import EmbeddingProviderRegistry
from intergrax.rag.embedding.registry.profile import EmbeddingProfile

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[5]
VPI_BOOTSTRAP_ROOT = (
    REPO_ROOT / "platform_proofs" / "scenarios" / "verified_product_identification" / "storage_bootstrap"
)
VPI_INTEGRATIONS_ROOT = (
    REPO_ROOT / "platform_proofs" / "scenarios" / "verified_product_identification" / "integrations"
)
ORCHESTRATOR_PATH = (
    REPO_ROOT
    / "platform_proofs"
    / "scenarios"
    / "verified_product_identification"
    / "storage_bootstrap"
    / "orchestration"
    / "orchestrator.py"
)


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
    values = {item.name: getattr(base, item.name) for item in fields(base)}
    values.update(overrides)
    return VpiBootstrapManifest(**values)


@dataclass
class FakeCatalogPort:
    manifest: VpiBootstrapManifest | None = None
    batches: list[CatalogIngestBatch] = field(default_factory=list)
    fail_on_batch: int | None = None
    prepare_calls: int = 0
    source_offer_count: int = 0
    identifier_count: int = 0
    structured_attribute_count: int = 0
    _ingested_offer_ids: set[str] = field(default_factory=set)

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
        for record in batch.records:
            offer_id = record.representation.source_ref.offer_id.value
            if offer_id in self._ingested_offer_ids:
                continue
            self._ingested_offer_ids.add(offer_id)
            self.source_offer_count += 1
            self.identifier_count += len(record.representation.exact.terms)
            self.structured_attribute_count += len(record.representation.structured.attributes)
        return CatalogIngestBatchResult(
            source_offer_count=self.source_offer_count,
            identifier_count=self.identifier_count,
            structured_attribute_count=self.structured_attribute_count,
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
            (ValidationCheck("search_index_reachable", ValidationStatus.PASS, "ok"),)
        )

    def prepare(self, manifest: VpiBootstrapManifest) -> ValidationReport:
        _ = manifest
        return ValidationReport.from_checks(
            (ValidationCheck("search_index_created", ValidationStatus.PASS, "ok"),)
        )

    def ingest_batch(self, batch: SearchIndexIngestBatch) -> SearchIndexIngestBatchResult:
        if self.fail_on_batch == batch.batch_ordinal:
            raise VpiBootstrapProviderError("search ingest failed")
        self.batches.append(batch)
        self.point_count += len(batch.records)
        return SearchIndexIngestBatchResult(point_count=self.point_count)

    def validate(self, manifest: VpiBootstrapManifest) -> ValidationReport:
        return ValidationReport.from_checks(
            (
                ValidationCheck(
                    "search_index_point_count",
                    ValidationStatus.PASS
                    if self.point_count >= manifest.checkpoint_rows_processed
                    else ValidationStatus.FAIL,
                    f"points={self.point_count}",
                ),
            )
        )

    def count_points(self) -> int:
        return self.point_count

    def close(self) -> None:
        return None


class FakeEmbeddingPort:
    def __init__(self, *, should_pass: bool = True, dimension: int = 8) -> None:
        self.should_pass = should_pass
        self.dimension = dimension
        self.probe_calls = 0
        self.embed_calls = 0
        self.embed_instance_id = id(self)

    def probe(self) -> EmbeddingProbeResult:
        self.probe_calls += 1
        status = ValidationStatus.PASS if self.should_pass else ValidationStatus.FAIL
        return EmbeddingProbeResult(
            status=status,
            provider="hf",
            model="fake-model",
            resolved_dimension=self.dimension,
            probe_vector_count=3,
            detail="probe",
        )

    def embed_batch(self, texts) -> tuple[tuple[float, ...], ...]:
        self.embed_calls += 1
        return tuple(
            tuple(float(index + 1) for _ in range(self.dimension)) for index, _ in enumerate(texts)
        )

    def close(self) -> None:
        return None


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


def _bootstrap_config(
    tmp_path: Path,
    *,
    max_records: int | None = 2,
    row_count: int = 5,
) -> VpiBootstrapConfig:
    dataset_path = tmp_path / "selected_offers.parquet"
    _write_tiny_parquet(dataset_path, row_count=row_count)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        f'{{"output_sha256":"testchecksum","selected_record_count":{row_count}}}',
        encoding="utf-8",
    )
    return VpiBootstrapConfig(
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


def _orchestrator(
    tmp_path: Path,
    *,
    catalog: FakeCatalogPort | None = None,
    search: FakeSearchPort | None = None,
    embedding: FakeEmbeddingPort | None = None,
    max_records: int | None = 2,
    row_count: int = 5,
) -> VpiBootstrapOrchestrator:
    return VpiBootstrapOrchestrator(
        config=_bootstrap_config(tmp_path, max_records=max_records, row_count=row_count),
        dependencies=VpiBootstrapDependencies(
            catalog=catalog or FakeCatalogPort(),
            search=search or FakeSearchPort(),
            embedding=embedding or FakeEmbeddingPort(),
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


def test_stage_ordering_and_ready(tmp_path: Path) -> None:
    catalog = FakeCatalogPort()
    search = FakeSearchPort()
    embedding = FakeEmbeddingPort()
    orchestrator = _orchestrator(tmp_path, catalog=catalog, search=search, embedding=embedding)

    report = orchestrator.run()

    assert report.final_state is BootstrapState.READY
    assert catalog.prepare_calls == 1
    assert len(catalog.batches) == 1
    assert len(search.batches) == 1
    assert catalog.manifest is not None
    assert catalog.manifest.checkpoint_rows_processed == 2
    assert report.embedding_probe is not None
    assert report.embedding_probe.status is ValidationStatus.PASS
    assert embedding.probe_calls >= 1
    assert embedding.embed_calls == 1


def test_embedding_gate_blocks_before_ingest(tmp_path: Path) -> None:
    catalog = FakeCatalogPort()
    search = FakeSearchPort()
    orchestrator = _orchestrator(
        tmp_path,
        catalog=catalog,
        search=search,
        embedding=FakeEmbeddingPort(should_pass=False),
    )

    report = orchestrator.run()

    assert report.final_state is BootstrapState.FAILED
    assert catalog.batches == []
    assert search.batches == []


def test_catalog_failure_not_ready(tmp_path: Path) -> None:
    catalog = FakeCatalogPort(fail_on_batch=0)
    orchestrator = _orchestrator(tmp_path, catalog=catalog)

    report = orchestrator.run()

    assert report.final_state is BootstrapState.FAILED
    assert catalog.manifest is not None
    assert catalog.manifest.state is BootstrapState.FAILED


def test_search_failure_not_ready(tmp_path: Path) -> None:
    search = FakeSearchPort(fail_on_batch=0)
    orchestrator = _orchestrator(tmp_path, search=search)

    report = orchestrator.run()

    assert report.final_state is BootstrapState.FAILED


def test_retry_same_batch_idempotent(tmp_path: Path) -> None:
    catalog = FakeCatalogPort()
    search = FakeSearchPort()
    orchestrator = _orchestrator(tmp_path, catalog=catalog, search=search)

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


def test_partial_final_batch_when_max_records_not_batch_aligned(tmp_path: Path) -> None:
    dataset_path = tmp_path / "rows.parquet"
    _write_tiny_parquet(dataset_path, row_count=10)
    batches = list(
        iter_dataset_rows(dataset_path, batch_size=4, start_row_index=0, max_records=5)
    )
    assert len(batches) == 2
    assert len(batches[0][1]) == 4
    assert len(batches[1][1]) == 1
    assert batches[1][1][0].global_row_index == 4


def test_resume_emits_remaining_rows_when_final_batch_is_partial(tmp_path: Path) -> None:
    dataset_path = tmp_path / "rows.parquet"
    _write_tiny_parquet(dataset_path, row_count=10)
    batches = list(
        iter_dataset_rows(
            dataset_path,
            batch_size=4,
            start_row_index=4,
            start_batch_ordinal=1,
            max_records=3,
        )
    )
    assert len(batches) == 1
    assert len(batches[0][1]) == 3
    assert batches[0][1][0].global_row_index == 4
    assert batches[0][1][-1].global_row_index == 6


def test_checkpoint_advances_only_after_successful_batch(tmp_path: Path) -> None:
    catalog = FakeCatalogPort()
    orchestrator = _orchestrator(tmp_path, catalog=catalog)
    orchestrator.run()
    assert catalog.manifest is not None
    assert catalog.manifest.checkpoint_batch_ordinal == 0
    assert catalog.manifest.checkpoint_rows_processed == 2


def test_alternate_fake_ports_without_orchestrator_changes(tmp_path: Path) -> None:
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


def test_orchestrator_has_no_create_default_registry_import() -> None:
    source = ORCHESTRATOR_PATH.read_text(encoding="utf-8")
    assert "create_default_registry" not in source


def test_orchestrator_has_no_concrete_embedding_provider_import() -> None:
    source = ORCHESTRATOR_PATH.read_text(encoding="utf-8")
    assert "EmbeddingProvider" not in source
    assert "EmbeddingProviderRegistry" not in source


def test_alternate_embedding_execution_port(tmp_path: Path) -> None:
    class AltEmbedding(FakeEmbeddingPort):
        def probe(self) -> EmbeddingProbeResult:
            result = super().probe()
            return EmbeddingProbeResult(
                status=result.status,
                provider="alt",
                model="alt-model",
                resolved_dimension=result.resolved_dimension,
                probe_vector_count=result.probe_vector_count,
                detail="alt-probe",
            )

    embedding = AltEmbedding()
    orchestrator = _orchestrator(tmp_path, embedding=embedding)
    report = orchestrator.run()
    assert report.final_state is BootstrapState.READY
    assert report.embedding_probe is not None
    assert report.embedding_probe.provider == "alt"


def test_same_embedding_instance_for_gate0_and_batches(tmp_path: Path) -> None:
    embedding = FakeEmbeddingPort()
    orchestrator = _orchestrator(tmp_path, embedding=embedding)
    orchestrator.run()
    assert embedding.probe_calls >= 1
    assert embedding.embed_calls == 1


def test_verify_ready_then_full_continues(tmp_path: Path) -> None:
    catalog = FakeCatalogPort()
    search = FakeSearchPort()
    embedding = FakeEmbeddingPort()
    verify = _orchestrator(
        tmp_path,
        catalog=catalog,
        search=search,
        embedding=embedding,
        max_records=2,
        row_count=5,
    )
    first = verify.run()
    assert first.final_state is BootstrapState.READY
    assert catalog.manifest is not None
    assert catalog.manifest.checkpoint_rows_processed == 2

    full = _orchestrator(
        tmp_path,
        catalog=catalog,
        search=search,
        embedding=embedding,
        max_records=None,
        row_count=5,
    )
    second = full.run()
    assert second.final_state is BootstrapState.READY
    assert catalog.manifest.checkpoint_rows_processed == 5
    assert len(catalog.batches) == 3


def test_verify_ready_then_verify_no_reingest(tmp_path: Path) -> None:
    catalog = FakeCatalogPort()
    search = FakeSearchPort()
    embedding = FakeEmbeddingPort()
    orchestrator = _orchestrator(
        tmp_path,
        catalog=catalog,
        search=search,
        embedding=embedding,
        max_records=2,
        row_count=5,
    )
    first = orchestrator.run()
    assert first.final_state is BootstrapState.READY
    second = orchestrator.run()
    assert second.final_state is BootstrapState.READY
    assert len(catalog.batches) == 1


def test_partial_resume_to_target(tmp_path: Path) -> None:
    catalog = FakeCatalogPort()
    search = FakeSearchPort(point_count=2)
    embedding = FakeEmbeddingPort()
    catalog.manifest = _sample_manifest(
        state=BootstrapState.INGESTING,
        dataset_record_count=5,
        target_max_records=2,
        checkpoint_batch_ordinal=0,
        checkpoint_rows_processed=2,
        catalog_source_offer_count=2,
        catalog_identifier_count=2,
        catalog_structured_attribute_count=2,
        search_point_count=2,
        dataset_checksum="testchecksum",
    )
    catalog.source_offer_count = 2
    catalog.identifier_count = 2
    catalog.structured_attribute_count = 2
    catalog._ingested_offer_ids = {"1000", "1001"}
    orchestrator = _orchestrator(
        tmp_path,
        catalog=catalog,
        search=search,
        embedding=embedding,
        max_records=4,
        row_count=5,
    )
    report = orchestrator.run()
    assert report.final_state is BootstrapState.READY
    assert catalog.manifest is not None
    assert catalog.manifest.checkpoint_rows_processed == 4


def test_pg_success_qdrant_fail_retry_does_not_overcount(tmp_path: Path) -> None:
    catalog = FakeCatalogPort()
    search = FakeSearchPort(fail_on_batch=0)
    embedding = FakeEmbeddingPort()
    orchestrator = _orchestrator(tmp_path, catalog=catalog, search=search, embedding=embedding)
    first = orchestrator.run()
    assert first.final_state is BootstrapState.FAILED
    assert catalog.source_offer_count == 2

    search.fail_on_batch = None
    second = orchestrator.run()
    assert second.final_state is BootstrapState.READY
    assert catalog.manifest is not None
    assert catalog.manifest.catalog_source_offer_count == 2
    assert catalog.manifest.search_point_count == 2


def test_ready_fast_path_executes_real_gate0(tmp_path: Path) -> None:
    catalog = FakeCatalogPort()
    search = FakeSearchPort(point_count=2)
    embedding = FakeEmbeddingPort()
    catalog.manifest = _sample_manifest(
        state=BootstrapState.READY,
        dataset_record_count=5,
        target_max_records=2,
        checkpoint_batch_ordinal=0,
        checkpoint_rows_processed=2,
        catalog_source_offer_count=2,
        catalog_identifier_count=2,
        catalog_structured_attribute_count=2,
        search_point_count=2,
        dataset_checksum="testchecksum",
    )
    orchestrator = _orchestrator(
        tmp_path,
        catalog=catalog,
        search=search,
        embedding=embedding,
        max_records=2,
        row_count=5,
    )
    report = orchestrator.run()
    assert report.final_state is BootstrapState.READY
    assert embedding.probe_calls == 1
    assert embedding.embed_calls == 0
    assert report.validation is not None
    assert any(check.name == "embedding.embedding_gate0" for check in report.validation.checks)


class _RestartIndexAdmin:
    def __init__(self, description: VectorIndexDescription) -> None:
        self._description = description

    def probe(self) -> HealthStatus:
        return HealthStatus(slug="qdrant", healthy=True)

    def describe_index(self, identity: VectorIndexIdentity) -> VectorIndexDescription:
        return self._description

    def prepare_index(self, spec) -> object:
        raise AssertionError("READY fast path must not prepare collection")

    def close(self) -> None:
        return None


class _RestartVectorStore:
    def add_records(self, records, *, scope: VectorStoreScope):
        raise AssertionError("READY fast path must not ingest")

    def query(self, *args, **kwargs):
        raise NotImplementedError

    def delete(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def count(self, *, scope: VectorStoreScope) -> int:
        return 2


def test_ready_fast_path_validates_persisted_qdrant_without_prepare(tmp_path: Path) -> None:
    catalog = FakeCatalogPort()
    embedding = FakeEmbeddingPort()
    identity = VectorIndexIdentity(logical_name="vpi_offers", tenant_id="default")
    description = VectorIndexDescription(
        identity=identity,
        exists=True,
        reachable=True,
        point_count=2,
        dense_dimension=8,
        present_capabilities=frozenset(
            {VectorSearchCapability.DENSE, VectorSearchCapability.SPARSE_LEXICAL}
        ),
        dense_channel_name="dense",
        sparse_lexical_channel_name="sparse",
    )
    search = PlatformSearchIndexBootstrapAdapter(
        _index_admin=_RestartIndexAdmin(description),
        _vector_store=_RestartVectorStore(),
        _index_identity=identity,
        _dense_channel_name="dense",
        _sparse_channel_name="sparse",
        _sparse_required=True,
    )
    assert search._dimension is None
    catalog.manifest = _sample_manifest(
        state=BootstrapState.READY,
        dataset_record_count=5,
        target_max_records=2,
        checkpoint_batch_ordinal=0,
        checkpoint_rows_processed=2,
        catalog_source_offer_count=2,
        catalog_identifier_count=2,
        catalog_structured_attribute_count=2,
        search_point_count=2,
        dataset_checksum="testchecksum",
    )
    orchestrator = _orchestrator(
        tmp_path,
        catalog=catalog,
        search=search,
        embedding=embedding,
        max_records=2,
        row_count=5,
    )
    report = orchestrator.run()
    assert report.final_state is BootstrapState.READY
    assert embedding.probe_calls == 1
    assert embedding.embed_calls == 0
    assert search._dimension is None
    assert report.validation is not None
    assert any(
        check.name == "search.search_index_dense_dimension" and check.status is ValidationStatus.PASS
        for check in report.validation.checks
    )


def test_requested_target_below_checkpoint_fails_closed(tmp_path: Path) -> None:
    catalog = FakeCatalogPort()
    catalog.manifest = _sample_manifest(
        state=BootstrapState.READY,
        dataset_record_count=5,
        target_max_records=4,
        checkpoint_batch_ordinal=1,
        checkpoint_rows_processed=4,
        catalog_source_offer_count=4,
        catalog_identifier_count=4,
        catalog_structured_attribute_count=4,
        search_point_count=4,
        dataset_checksum="testchecksum",
    )
    orchestrator = _orchestrator(tmp_path, catalog=catalog, max_records=2, row_count=5)
    report = orchestrator.run()
    assert report.final_state is BootstrapState.FAILED
    assert "below existing checkpoint" in (report.failure_detail or "")


def _iter_production_python_files(*roots: Path):
    for root in roots:
        for path in root.rglob("*.py"):
            if path.name.startswith("test_"):
                continue
            yield path


def test_no_vendor_sdk_import_in_platform_search_adapter() -> None:
    adapter_source = (
        VPI_INTEGRATIONS_ROOT / "search_store" / "platform_bootstrap_adapter.py"
    ).read_text(encoding="utf-8")
    assert "qdrant_client" not in adapter_source
    assert "QdrantClient" not in adapter_source


def test_no_reflection_in_vpi_bootstrap_production_code() -> None:
    forbidden_names = {"getattr", "setattr", "hasattr", "inspect"}
    for path in _iter_production_python_files(VPI_BOOTSTRAP_ROOT, VPI_INTEGRATIONS_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        assert forbidden_names.isdisjoint(names), f"forbidden reflection in {path}"


def test_no_any_or_object_contracts_in_vpi_bootstrap_production_code() -> None:
    for path in _iter_production_python_files(VPI_BOOTSTRAP_ROOT, VPI_INTEGRATIONS_ROOT):
        source = path.read_text(encoding="utf-8")
        assert "dict[str, Any]" not in source, path
        assert ": Any" not in source, path
        assert "_client: object" not in source, path
        assert "_sparse_encoder: object" not in source, path
