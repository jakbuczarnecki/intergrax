"""Product identity hypothesis layer — propositions with evidence, not verified truth."""

from platform_proofs.scenarios.verified_product_identification.application.identity.composition import (
    build_product_identity_hypothesis_service,
)
from platform_proofs.scenarios.verified_product_identification.application.identity.contracts import (
    IdentityContradiction,
    IdentityContradictionType,
    IdentityEvidence,
    IdentityEvidenceProvenance,
    IdentityEvidenceStrengthClass,
    IdentityEvidenceType,
    IdentityHypothesisConfiguration,
    IdentityHypothesisMember,
    ProductIdentityHypothesis,
    ProductIdentityHypothesisCollection,
    ProductIdentityHypothesisRequest,
)
from platform_proofs.scenarios.verified_product_identification.application.identity.errors import (
    IdentityEvidenceUnavailableError,
    IdentityHypothesisError,
)
from platform_proofs.scenarios.verified_product_identification.application.identity.service import (
    ProductIdentityHypothesisService,
)
from platform_proofs.scenarios.verified_product_identification.application.identity.strategy import (
    DeterministicEvidenceIdentityHypothesisStrategy,
    ProductIdentityHypothesisStrategy,
)

__all__ = (
    "DeterministicEvidenceIdentityHypothesisStrategy",
    "IdentityContradiction",
    "IdentityContradictionType",
    "IdentityEvidence",
    "IdentityEvidenceProvenance",
    "IdentityEvidenceStrengthClass",
    "IdentityEvidenceType",
    "IdentityEvidenceUnavailableError",
    "IdentityHypothesisConfiguration",
    "IdentityHypothesisError",
    "IdentityHypothesisMember",
    "ProductIdentityHypothesis",
    "ProductIdentityHypothesisCollection",
    "ProductIdentityHypothesisRequest",
    "ProductIdentityHypothesisService",
    "ProductIdentityHypothesisStrategy",
    "build_product_identity_hypothesis_service",
)
