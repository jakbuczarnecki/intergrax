"""Scenario-owned product identification domain models."""

from platform_proofs.scenarios.verified_product_identification.application.domain.candidates import (
    ChannelCandidateBatch,
    ChannelScore,
    ExactChannelScore,
    LexicalChannelScore,
    MultiChannelCandidateCollection,
    ProductCandidate,
    RetrievalChannel,
    StructuredChannelScore,
    VectorChannelScore,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductIdentifier,
    ProductIdentifierType,
    ProductOfferId,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.search_representation import (
    SEARCH_REPRESENTATION_DERIVATION_VERSION,
    DerivedOfferSearchRepresentation,
    ExactIdentifierTerm,
    ExactSearchRepresentation,
    LexicalSearchRepresentation,
    SemanticContributingField,
    SemanticSearchRepresentation,
    StructuredAttribute,
    StructuredSearchRepresentation,
    StructuredTextFragment,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    ProductSourceProvenance,
    ProductSourceRecord,
    SourceRecordRef,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.wdc_source_offer import (
    WdcIdentifierEntry,
    WdcKeyValuePair,
    WdcSourceOffer,
    parse_wdc_record_json,
    parse_wdc_source_offer,
    parse_wdc_source_offer_json,
)

__all__ = (
    "ChannelCandidateBatch",
    "ChannelScore",
    "DerivedOfferSearchRepresentation",
    "ExactChannelScore",
    "ExactIdentifierTerm",
    "ExactSearchRepresentation",
    "LexicalChannelScore",
    "LexicalSearchRepresentation",
    "MultiChannelCandidateCollection",
    "ProductCandidate",
    "ProductIdentifier",
    "ProductIdentifierType",
    "ProductOfferId",
    "ProductSourceProvenance",
    "ProductSourceRecord",
    "RetrievalChannel",
    "SEARCH_REPRESENTATION_DERIVATION_VERSION",
    "SemanticContributingField",
    "SemanticSearchRepresentation",
    "SourceRecordRef",
    "StructuredAttribute",
    "StructuredChannelScore",
    "StructuredSearchRepresentation",
    "StructuredTextFragment",
    "VectorChannelScore",
    "WdcIdentifierEntry",
    "WdcKeyValuePair",
    "WdcSourceOffer",
    "parse_wdc_record_json",
    "parse_wdc_source_offer",
    "parse_wdc_source_offer_json",
)
