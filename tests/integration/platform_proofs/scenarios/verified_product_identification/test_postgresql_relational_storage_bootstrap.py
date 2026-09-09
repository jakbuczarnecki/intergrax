"""Bounded PostgreSQL qualification for relational Data Pack storage bootstrap."""

from __future__ import annotations

import json
import os
import uuid

import pytest

from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductOfferId,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.adapter import (
    PostgreSqlRelationalStorageAdapter,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    RelationalBatch,
    RelationalLoadRecord,
    RelationalTargetId,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.errors import (
    StorageBootstrapWriteError,
)

pytestmark = [pytest.mark.integration]


def _postgres_available() -> bool:
    dsn = os.getenv("INTERGRAX_POSTGRESQL_DSN", "").strip()
    host = os.getenv("INTERGRAX_POSTGRESQL_HOST", "").strip()
    return bool(dsn or host)


def _record(index: int, *, offer_suffix: str, semantic_hash: str = "hash-a") -> RelationalLoadRecord:
    return RelationalLoadRecord(
        source_ref=SourceRecordRef(
            offer_id=ProductOfferId(f"offer-{offer_suffix}"),
            catalog_id="wdc-v2-selected",
            source_revision=None,
        ),
        global_row_index=index,
        record_json=json.dumps({"id": f"offer-{offer_suffix}", "title": "relay"}),
        semantic_text=f"semantic-{offer_suffix}",
        semantic_text_hash=semantic_hash,
        derivation_version="v1",
    )


def _adapter() -> PostgreSqlRelationalStorageAdapter:
    schema_name = f"vpi_rel_bootstrap_{uuid.uuid4().hex[:10]}"
    adapter = PostgreSqlRelationalStorageAdapter.from_env(schema_name=schema_name)
    adapter.prepare_target(RelationalTargetId("vpi-products"))
    return adapter


@pytest.mark.skipif(not _postgres_available(), reason="PostgreSQL not configured locally")
def test_postgresql_relational_schema_preparation() -> None:
    adapter = _adapter()
    adapter.prepare_target(RelationalTargetId("vpi-products"))


@pytest.mark.skipif(not _postgres_available(), reason="PostgreSQL not configured locally")
def test_postgresql_relational_insert_and_readback() -> None:
    adapter = _adapter()
    batch = RelationalBatch(
        batch_number=0,
        target=RelationalTargetId("vpi-products"),
        records=tuple(_record(index, offer_suffix=str(index)) for index in range(5)),
    )
    write = adapter.write_batch(batch)
    assert write.written_count == 5
    verify = adapter.verify_batch(batch)
    assert verify.failed_count == 0


@pytest.mark.skipif(not _postgres_available(), reason="PostgreSQL not configured locally")
def test_postgresql_relational_identical_retry() -> None:
    adapter = _adapter()
    batch = RelationalBatch(
        batch_number=0,
        target=RelationalTargetId("vpi-products"),
        records=(_record(0, offer_suffix="0"),),
    )
    first = adapter.write_batch(batch)
    second = adapter.write_batch(batch)
    assert first.written_count == 1
    assert second.skipped_count == 1


@pytest.mark.skipif(not _postgres_available(), reason="PostgreSQL not configured locally")
def test_postgresql_relational_content_conflict_rejected() -> None:
    adapter = _adapter()
    adapter.write_batch(
        RelationalBatch(
            batch_number=0,
            target=RelationalTargetId("vpi-products"),
            records=(_record(0, offer_suffix="0", semantic_hash="hash-a"),),
        )
    )
    with pytest.raises(StorageBootstrapWriteError):
        adapter.write_batch(
            RelationalBatch(
                batch_number=1,
                target=RelationalTargetId("vpi-products"),
                records=(_record(0, offer_suffix="0", semantic_hash="hash-b"),),
            )
        )


@pytest.mark.skipif(not _postgres_available(), reason="PostgreSQL not configured locally")
def test_postgresql_relational_transaction_rollback() -> None:
    adapter = _adapter()
    good = _record(0, offer_suffix="0")
    bad = _record(0, offer_suffix="0", semantic_hash="hash-b")
    with pytest.raises(StorageBootstrapWriteError):
        adapter.write_batch(
            RelationalBatch(
                batch_number=0,
                target=RelationalTargetId("vpi-products"),
                records=(good, bad),
            )
        )
    verify = adapter.verify_batch(
        RelationalBatch(
            batch_number=0,
            target=RelationalTargetId("vpi-products"),
            records=(good,),
        )
    )
    assert verify.failed_count == 1


@pytest.mark.skipif(not _postgres_available(), reason="PostgreSQL not configured locally")
def test_postgresql_relational_unique_identity_enforced() -> None:
    adapter = _adapter()
    adapter.write_batch(
        RelationalBatch(
            batch_number=0,
            target=RelationalTargetId("vpi-products"),
            records=(_record(10, offer_suffix="a"),),
        )
    )
    with pytest.raises(StorageBootstrapWriteError):
        adapter.write_batch(
            RelationalBatch(
                batch_number=1,
                target=RelationalTargetId("vpi-products"),
                records=(_record(10, offer_suffix="b"),),
            )
        )
