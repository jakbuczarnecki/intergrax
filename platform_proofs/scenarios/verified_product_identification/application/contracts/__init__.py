"""Scenario-owned catalog/search contracts."""

from platform_proofs.scenarios.verified_product_identification.application.contracts.failures import (
    CatalogSearchFailure,
    CatalogSearchFailureKind,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.identification_context import (
    HypothesisRejectionEvidence,
    MissingDistinguishingRequirement,
    MissingRequirementOrigin,
    NegativeAttributeConstraint,
    ProductIdentificationQueryContext,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.queries import (
    ExactIdentifierQuery,
    LexicalSearchQuery,
    StructuredAttributeConstraint,
    StructuredConstraintOperator,
    StructuredSearchQuery,
    VectorSearchQuery,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.results import (
    ExactIdentifierLookupResult,
    LexicalSearchResult,
    SourceRecordFetchResult,
    StructuredSearchResult,
    VectorSearchResult,
)

__all__ = (
    "HypothesisRejectionEvidence",
    "MissingDistinguishingRequirement",
    "MissingRequirementOrigin",
    "NegativeAttributeConstraint",
    "ProductIdentificationQueryContext",
    "CatalogSearchFailure",
    "CatalogSearchFailureKind",
    "ExactIdentifierLookupResult",
    "ExactIdentifierQuery",
    "LexicalSearchQuery",
    "LexicalSearchResult",
    "SourceRecordFetchResult",
    "StructuredAttributeConstraint",
    "StructuredConstraintOperator",
    "StructuredSearchQuery",
    "StructuredSearchResult",
    "VectorSearchQuery",
    "VectorSearchResult",
)
