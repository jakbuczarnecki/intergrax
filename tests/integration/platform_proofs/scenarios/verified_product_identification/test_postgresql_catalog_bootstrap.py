"""PostgreSQL catalog bootstrap integration — skipped when provider unavailable."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.model import (
    BOOTSTRAP_IMPLEMENTATION_VERSION,
    BootstrapState,
    VpiBootstrapManifest,
)
from platform_proofs.scenarios.verified_product_identification.ingest.pipeline.derive_batch import (
    build_catalog_ingest_batch,
)
from platform_proofs.scenarios.verified_product_identification.ingest.source_reader.parquet_dataset import (
    DatasetRow,
)
from platform_proofs.scenarios.verified_product_identification.integrations.catalog_store.postgresql.adapter import (
    PostgreSQLCatalogBootstrapAdapter,
)

pytestmark = [pytest.mark.integration]


def _postgres_available() -> bool:
    dsn = os.getenv("INTERGRAX_POSTGRESQL_DSN", "").strip()
    host = os.getenv("INTERGRAX_POSTGRESQL_HOST", "").strip()
    return bool(dsn or host)


@pytest.mark.skipif(not _postgres_available(), reason="PostgreSQL not configured locally")
def test_postgresql_catalog_bootstrap_idempotent(tmp_path: Path) -> None:
    schema_name = f"vpi_test_{uuid.uuid4().hex[:8]}"
    adapter = PostgreSQLCatalogBootstrapAdapter.from_env(
        schema_name=schema_name,
        ingestion_batch_label=BOOTSTRAP_IMPLEMENTATION_VERSION,
    )
    manifest = VpiBootstrapManifest(
        state=BootstrapState.INITIALIZING,
        dataset_path=str(tmp_path / "selected_offers.parquet"),
        dataset_checksum="integration-checksum",
        dataset_record_count=2,
        search_representation_derivation_version="v2",
        embedding_configuration_version="v1",
        embedding_provider="fake",
        embedding_model="fake-model",
        embedding_dimension=8,
        catalog_schema_version="v1",
        search_index_schema_version="v1",
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

    first_prepare = adapter.prepare(manifest)
    second_prepare = adapter.prepare(manifest)
    assert first_prepare.status.value == "PASS"
    assert second_prepare.status.value == "PASS"

    dataset_path = tmp_path / "selected_offers.parquet"
    _write_sample_parquet(dataset_path)
    batch = build_catalog_ingest_batch(
        batch_ordinal=0,
        rows=(
            DatasetRow(global_row_index=0, record_json=_sample_record(1)),
            DatasetRow(global_row_index=1, record_json=_sample_record(2)),
        ),
        catalog_id="wdc-v2-selected",
        source_revision=None,
    )
    adapter.ingest_batch(batch)
    adapter.ingest_batch(batch)

    adapter.write_manifest(
        manifest.with_checkpoint(
            batch_ordinal=0,
            rows_processed=2,
            catalog_source_offer_count=2,
            catalog_identifier_count=2,
            catalog_structured_attribute_count=2,
            search_point_count=0,
        )
    )
    stored = adapter.read_manifest()
    assert stored is not None
    assert stored.dataset_checksum == "integration-checksum"


def _sample_record(index: int) -> str:
    return (
        '{"id":"%s","title":"relay module","identifiers":[{"gtin":"%s"}],'
        '"keyValuePairs":{"voltage":"24V"}}'
    ) % (1000 + index, 1000000000000 + index)


def _write_sample_parquet(path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "record_json": [
                _sample_record(1),
                _sample_record(2),
            ]
        }
    )
    pq.write_table(table, path)
