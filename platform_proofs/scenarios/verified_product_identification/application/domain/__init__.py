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
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    ProductSourceProvenance,
    ProductSourceRecord,
    SourceRecordRef,
)

__all__ = (
    "ChannelCandidateBatch",
    "ChannelScore",
    "ExactChannelScore",
    "LexicalChannelScore",
    "MultiChannelCandidateCollection",
    "ProductCandidate",
    "ProductIdentifier",
    "ProductIdentifierType",
    "ProductOfferId",
    "ProductSourceProvenance",
    "ProductSourceRecord",
    "RetrievalChannel",
    "SourceRecordRef",
    "StructuredChannelScore",
    "VectorChannelScore",
)
