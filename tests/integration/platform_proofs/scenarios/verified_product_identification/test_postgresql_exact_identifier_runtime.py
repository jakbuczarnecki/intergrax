"""Bounded PostgreSQL runtime qualification for exact identifier lookup."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from platform_proofs.scenarios.verified_product_identification.application.contracts.queries import (
    ExactIdentifierQuery,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductIdentifier,
    ProductIdentifierType,
    ProductOfferId,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)
from platform_proofs.scenarios.verified_product_identification.qualification.integration.storage_environment import (
    postgres_environment_available,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.adapter import (
    PostgreSqlRelationalStorageAdapter,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.exact_identifier_lookup import (
    PostgreSqlExactIdentifierLookupAdapter,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    RelationalBatch,
    RelationalLoadRecord,
    RelationalTargetId,
)
from tests.integration.platform_proofs.scenarios.verified_product_identification.conftest import (
    drop_postgresql_schema,
    runtime_qualification_target_name,
)

pytestmark = [pytest.mark.integration]

_CATALOG_ID = "vpi-runtime-catalog"


def _record_json(offer_id: str) -> str:
    return json.dumps(
        {
            "id": offer_id,
            "identifiers": [
                {"/gtin13": "[8806095123456]"},
                {"/mpn": "[MZ-V9P2T0BW]"},
                {"/sku": "[SKU-NEUTRAL-01]"},
                {"/productID": "[PROD-NEUTRAL-01]"},
            ],
            "title": f"Runtime offer {offer_id}",
        },
        ensure_ascii=False,
    )


def _relational_record(index: int, offer_id: str) -> RelationalLoadRecord:
    return RelationalLoadRecord(
        source_ref=SourceRecordRef(
            offer_id=ProductOfferId(offer_id),
            catalog_id=_CATALOG_ID,
            source_revision="rev-runtime",
        ),
        global_row_index=index,
        record_json=_record_json(offer_id),
        semantic_text=f"semantic-{offer_id}",
        semantic_text_hash=f"hash-{offer_id}",
        derivation_version="v2",
    )


def _query(identifier_type: ProductIdentifierType, value: str) -> ExactIdentifierQuery:
    return ExactIdentifierQuery(
        identifier=ProductIdentifier(identifier_type=identifier_type, value=value),
        limit=5,
    )


@pytest.mark.skipif(not postgres_environment_available(), reason="PostgreSQL not configured")
def test_postgresql_exact_identifier_runtime_qualification(tmp_path: Path) -> None:
    _ = tmp_path
    schema_name = runtime_qualification_target_name()
    storage = PostgreSqlRelationalStorageAdapter.from_env(schema_name=schema_name)
    lookup = PostgreSqlExactIdentifierLookupAdapter.from_env(schema_name=schema_name)
    try:
        storage.prepare_target(RelationalTargetId("vpi-products"))
        batch = RelationalBatch(
            batch_number=0,
            target=RelationalTargetId("vpi-products"),
            records=(
                _relational_record(0, "offer-runtime-a"),
                _relational_record(1, "offer-runtime-b"),
            ),
        )
        write_result = storage.write_batch(batch)
        assert write_result.failed_count == 0
        assert write_result.written_count == 2

        gtin = lookup.lookup(_query(ProductIdentifierType.GTIN, "[8806095123456]"))
        assert gtin.failure is None
        assert len(gtin.candidates) == 2

        mpn = lookup.lookup(_query(ProductIdentifierType.MPN, "[MZ-V9P2T0BW]"))
        assert mpn.failure is None
        assert len(mpn.candidates) == 2

        sku = lookup.lookup(_query(ProductIdentifierType.SKU, "[SKU-NEUTRAL-01]"))
        assert sku.failure is None
        assert len(sku.candidates) == 2

        product_id = lookup.lookup(_query(ProductIdentifierType.PRODUCT_ID, "[PROD-NEUTRAL-01]"))
        assert product_id.failure is None
        assert len(product_id.candidates) == 2

        zero = lookup.lookup(_query(ProductIdentifierType.GTIN, "0000000000000"))
        assert zero.failure is None
        assert zero.candidates == ()

        limited = lookup.lookup(
            ExactIdentifierQuery(
                identifier=ProductIdentifier(
                    identifier_type=ProductIdentifierType.GTIN,
                    value="8806095123456",
                ),
                limit=1,
            )
        )
        assert len(limited.candidates) == 1

        with storage._provider.connection() as session:
            explain = session.execute(
                """
                EXPLAIN
                SELECT catalog_id, offer_id
                FROM vpi_product_identifiers
                WHERE identifier_type = %s AND normalized_value = %s
                LIMIT 1
                """,
                ("gtin", "8806095123456"),
            ).fetchall()
        explain_text = "\n".join(str(row) for row in explain).lower()
        assert "vpi_product_identifiers_lookup_idx" in explain_text or "index" in explain_text
    finally:
        drop_postgresql_schema(storage)
