"""Scenario-owned lexical projection for indexed BM25 persistence."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.application.catalog.derive_search_representation import (
    derive_search_representation,
    flatten_lexical_text,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.wdc_source_offer import (
    parse_wdc_source_offer_json,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.bm25_engine import (
    build_term_frequencies,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.lexical_projection import (
    LexicalIndexProjectionRecord,
    lexical_document_hash,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    RelationalLoadRecord,
)


def _source_revision_norm(source_revision: str | None) -> str:
    return source_revision or ""


@dataclass(frozen=True, slots=True)
class ProjectedLexicalPosting:
    catalog_id: str
    offer_id: str
    source_revision_norm: str
    term: str
    term_frequency: int


def project_lexical_from_load_record(
    record: RelationalLoadRecord,
) -> LexicalIndexProjectionRecord | None:
    """Project one lexical document from a relational load record."""
    source_offer = parse_wdc_source_offer_json(record.record_json)
    if source_offer.offer_id != record.source_ref.offer_id.value:
        msg = "record_json offer id must match source_ref.offer_id"
        raise ValueError(msg)

    representation = derive_search_representation(
        source_offer,
        source_ref=record.source_ref,
        derivation_version=record.derivation_version,
    )
    lexical_document = flatten_lexical_text(representation.lexical)
    if not lexical_document.strip():
        return None

    tokens, term_frequencies = build_term_frequencies(lexical_document)
    if not tokens or not term_frequencies:
        return None

    document_hash = lexical_document_hash(
        lexical_document=lexical_document,
        derivation_version=record.derivation_version,
    )
    return LexicalIndexProjectionRecord(
        source_ref=record.source_ref,
        lexical_document=lexical_document,
        document_hash=document_hash,
        document_length=len(tokens),
        derivation_version=record.derivation_version,
    )


def project_lexical_postings(
    projection: LexicalIndexProjectionRecord,
) -> tuple[ProjectedLexicalPosting, ...]:
    _, term_frequencies = build_term_frequencies(projection.lexical_document)
    revision_norm = _source_revision_norm(projection.source_ref.source_revision)
    return tuple(
        ProjectedLexicalPosting(
            catalog_id=projection.source_ref.catalog_id,
            offer_id=projection.source_ref.offer_id.value,
            source_revision_norm=revision_norm,
            term=term,
            term_frequency=term_frequency,
        )
        for term, term_frequency in sorted(term_frequencies.items())
    )
