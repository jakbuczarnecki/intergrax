"""SourceRecordRef ordering semantics — domain-level canonical helper."""

from __future__ import annotations

import pytest

from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductOfferId,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
    source_ref_sort_key,
)

pytestmark = pytest.mark.unit


def _source_ref(
    *,
    catalog_id: str,
    offer_id: str,
    source_revision: str | None,
) -> SourceRecordRef:
    return SourceRecordRef(
        offer_id=ProductOfferId(offer_id),
        catalog_id=catalog_id,
        source_revision=source_revision,
    )


def test_source_ref_sort_key_orders_catalog_then_offer_then_revision() -> None:
    refs = (
        _source_ref(catalog_id="catalog-b", offer_id="offer-a", source_revision=None),
        _source_ref(catalog_id="catalog-a", offer_id="offer-z", source_revision="rev-2"),
        _source_ref(catalog_id="catalog-a", offer_id="offer-a", source_revision=None),
        _source_ref(catalog_id="catalog-a", offer_id="offer-a", source_revision="rev-1"),
        _source_ref(catalog_id="catalog-a", offer_id="offer-a", source_revision="rev-2"),
    )
    ordered = sorted(refs, key=source_ref_sort_key)
    assert [source_ref_sort_key(ref) for ref in ordered] == [
        ("catalog-a", "offer-a", ""),
        ("catalog-a", "offer-a", "rev-1"),
        ("catalog-a", "offer-a", "rev-2"),
        ("catalog-a", "offer-z", "rev-2"),
        ("catalog-b", "offer-a", ""),
    ]


def test_source_ref_sort_key_normalizes_none_revision_to_empty_string() -> None:
    ref = _source_ref(catalog_id="catalog-a", offer_id="offer-a", source_revision=None)
    assert source_ref_sort_key(ref) == ("catalog-a", "offer-a", "")


def test_source_ref_sort_key_preserves_explicit_empty_revision() -> None:
    none_ref = _source_ref(catalog_id="catalog-a", offer_id="offer-a", source_revision=None)
    empty_ref = _source_ref(catalog_id="catalog-a", offer_id="offer-a", source_revision="")
    assert source_ref_sort_key(none_ref) == source_ref_sort_key(empty_ref)
