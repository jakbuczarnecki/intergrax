"""Unit tests for WDC identifier projection into indexed storage rows."""

from __future__ import annotations

import json

import pytest

from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductIdentifierType,
    ProductOfferId,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.identifier_projection import (
    project_identifiers_from_load_record,
    project_identifiers_from_source_ref,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    RelationalLoadRecord,
)

pytestmark = pytest.mark.unit

_CATALOG_ID = "catalog-neutral"


def _source_ref(offer_id: str) -> SourceRecordRef:
    return SourceRecordRef(
        offer_id=ProductOfferId(offer_id),
        catalog_id=_CATALOG_ID,
        source_revision="rev-1",
    )


def _electronics_record_json() -> str:
    return json.dumps(
        {
            "id": "offer-electronics",
            "identifiers": [
                {"/gtin13": "[8806095123456]"},
                {"/mpn": "[MZ-V9P2T0BW]"},
                {"/sku": "[SKU-NEUTRAL-01]"},
                {"/productID": "[PROD-NEUTRAL-01]"},
            ],
            "title": "Neutral NVMe Storage Device 2TB",
        },
        ensure_ascii=False,
    )


def _load_record(record_json: str, offer_id: str = "offer-electronics") -> RelationalLoadRecord:
    return RelationalLoadRecord(
        source_ref=_source_ref(offer_id),
        global_row_index=1,
        record_json=record_json,
        semantic_text="semantic",
        semantic_text_hash="hash-a",
        derivation_version="v2",
    )


def test_projection_maps_all_supported_identifier_families() -> None:
    rows = project_identifiers_from_load_record(_load_record(_electronics_record_json()))
    by_type = {row.identifier_type: row for row in rows}
    assert set(by_type) == {
        ProductIdentifierType.GTIN,
        ProductIdentifierType.MPN,
        ProductIdentifierType.SKU,
        ProductIdentifierType.PRODUCT_ID,
    }
    assert by_type[ProductIdentifierType.GTIN].normalized_value == "8806095123456"
    assert by_type[ProductIdentifierType.GTIN].source_field == "/gtin13"
    assert by_type[ProductIdentifierType.MPN].normalized_value == "MZ-V9P2T0BW"
    assert by_type[ProductIdentifierType.SKU].source_value == "[SKU-NEUTRAL-01]"
    assert by_type[ProductIdentifierType.PRODUCT_ID].source_field == "/productID"


def test_projection_ignores_unsupported_identifier_keys() -> None:
    record_json = json.dumps(
        {
            "id": "offer-unknown",
            "identifiers": [{"/ean": "[1234567890123]"}, {"/gtin13": "[8806095123456]"}],
        }
    )
    rows = project_identifiers_from_load_record(_load_record(record_json, "offer-unknown"))
    assert len(rows) == 1
    assert rows[0].identifier_type is ProductIdentifierType.GTIN


def test_projection_skips_invalid_gtin_values() -> None:
    record_json = json.dumps(
        {
            "id": "offer-short-gtin",
            "identifiers": [{"/gtin13": "[12345]"}],
        }
    )
    rows = project_identifiers_from_load_record(_load_record(record_json, "offer-short-gtin"))
    assert rows == ()


def test_projection_rejects_offer_id_mismatch() -> None:
    with pytest.raises(ValueError, match="offer id must match"):
        project_identifiers_from_load_record(
            _load_record(_electronics_record_json(), offer_id="other-offer")
        )


def test_projection_from_source_ref_helper() -> None:
    rows = project_identifiers_from_source_ref(
        record_json=_electronics_record_json(),
        source_ref=_source_ref("offer-electronics"),
    )
    assert len(rows) == 4
