"""Deterministic query set derived from proof-50 relational records."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.application.catalog.identifier_normalization import (
    classify_wdc_identifier_type,
    normalize_exact_lookup_value,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.queries import (
    ExactIdentifierQuery,
    LexicalSearchQuery,
    StructuredAttributeConstraint,
    StructuredConstraintOperator,
    StructuredSearchQuery,
    VectorSearchQuery,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductIdentifier,
    ProductIdentifierType,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)
from platform_proofs.scenarios.verified_product_identification.application.catalog.derive_search_representation import (
    derive_search_representation,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.wdc_source_offer import (
    parse_wdc_source_offer_json,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.relational import (
    RelationalDataPackRecord,
)


@dataclass(frozen=True, slots=True)
class ProofQueryCase:
    query_id: str
    channel: str
    expected_offer_id: str
    negative: bool
    exact_query: ExactIdentifierQuery | None = None
    lexical_query: LexicalSearchQuery | None = None
    structured_query: StructuredSearchQuery | None = None
    vector_query: VectorSearchQuery | None = None


def _first_identifier(record: RelationalDataPackRecord) -> ProductIdentifier | None:
    source_offer = parse_wdc_source_offer_json(record.record_json)
    for entry in source_offer.identifiers:
        identifier_type = classify_wdc_identifier_type(entry.source_key)
        if identifier_type is None:
            continue
        lookup_value = normalize_exact_lookup_value(identifier_type, entry.source_value)
        if lookup_value:
            return ProductIdentifier(
                identifier_type=identifier_type,
                value=entry.source_value,
                source_field=entry.source_key,
            )
    return None


def _first_structured_attribute(record: RelationalDataPackRecord) -> tuple[str, str] | None:
    source_offer = parse_wdc_source_offer_json(record.record_json)
    representation = derive_search_representation(
        source_offer,
        source_ref=record.source_ref,
        derivation_version=record.derivation_version,
    )
    for attribute in representation.structured.attributes:
        lookup_name = attribute.canonical_key or attribute.source_key
        lookup_value = attribute.normalized_text_value or attribute.source_value
        if lookup_name.strip() and lookup_value.strip():
            return lookup_name, lookup_value
    return None


def build_proof_query_set(
    records: tuple[RelationalDataPackRecord, ...],
) -> tuple[ProofQueryCase, ...]:
    if not records:
        return ()
    ordered = sorted(records, key=lambda record: record.source_ref.offer_id.value)
    cases: list[ProofQueryCase] = []

    exact_record = next((record for record in ordered if record.has_identifiers), ordered[0])
    exact_identifier = _first_identifier(exact_record)
    if exact_identifier is not None:
        cases.append(
            ProofQueryCase(
                query_id="exact-identifier-1",
                channel="exact",
                expected_offer_id=exact_record.source_ref.offer_id.value,
                negative=False,
                exact_query=ExactIdentifierQuery(identifier=exact_identifier, limit=5),
            )
        )

    lexical_record = next(
        (record for record in ordered if record.title),
        ordered[0],
    )
    title = lexical_record.title or lexical_record.source_ref.offer_id.value
    cases.append(
        ProofQueryCase(
            query_id="lexical-title-brand-1",
            channel="lexical",
            expected_offer_id=lexical_record.source_ref.offer_id.value,
            negative=False,
            lexical_query=LexicalSearchQuery(
                query_text=title[: min(len(title), 24)],
                limit=10,
            ),
        )
    )

    structured_record = next(
        (record for record in ordered if record.has_structured_attributes),
        ordered[0],
    )
    structured_attribute = _first_structured_attribute(structured_record)
    if structured_attribute is not None:
        cases.append(
            ProofQueryCase(
                query_id="structured-attribute-1",
                channel="structured",
                expected_offer_id=structured_record.source_ref.offer_id.value,
                negative=False,
                structured_query=StructuredSearchQuery(
                    constraints=(
                        StructuredAttributeConstraint(
                            attribute_name=structured_attribute[0],
                            operator=StructuredConstraintOperator.CONTAINS,
                            value=structured_attribute[1][: min(len(structured_attribute[1]), 32)],
                        ),
                    ),
                    limit=10,
                ),
            )
        )

    vector_record = next(
        (record for record in ordered if record.description),
        ordered[0],
    )
    vector_text = vector_record.description or vector_record.semantic_text[:120]
    cases.append(
        ProofQueryCase(
            query_id="vector-semantic-1",
            channel="vector",
            expected_offer_id=vector_record.source_ref.offer_id.value,
            negative=False,
            vector_query=VectorSearchQuery(query_text=vector_text[:120], limit=10),
        )
    )

    negative_record = ordered[-1]
    cases.append(
        ProofQueryCase(
            query_id="negative-unrelated-token",
            channel="lexical",
            expected_offer_id=negative_record.source_ref.offer_id.value,
            negative=True,
            lexical_query=LexicalSearchQuery(
                query_text="__vpi_proof_negative_token_5c4d1__",
                limit=5,
            ),
        )
    )
    return tuple(cases)
