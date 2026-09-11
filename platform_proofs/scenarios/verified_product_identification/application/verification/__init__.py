"""Terminal product identification verification and abstention (5C10)."""

from platform_proofs.scenarios.verified_product_identification.application.verification.composition import (
    build_product_identification_verification_service,
)
from platform_proofs.scenarios.verified_product_identification.application.verification.contracts import (
    ConstraintRequirementStatus,
    HypothesisVerificationCollection,
    HypothesisVerificationState,
    IdentityHypothesisVerification,
    MissingRequirement,
    ProductIdentificationDecision,
    ProductIdentificationDecisionReasonCode,
    ProductIdentificationOutcome,
    ProductIdentificationVerificationOutcome,
    ProductIdentificationVerificationRequest,
)
from platform_proofs.scenarios.verified_product_identification.application.verification.decision_policy import (
    DeterministicProductIdentificationDecisionPolicy,
    ProductIdentificationDecisionPolicy,
)
from platform_proofs.scenarios.verified_product_identification.application.verification.errors import (
    ProductIdentificationVerificationError,
)
from platform_proofs.scenarios.verified_product_identification.application.verification.service import (
    IdentityVerificationService,
    ProductIdentificationDecisionService,
    ProductIdentificationVerificationService,
)
from platform_proofs.scenarios.verified_product_identification.application.verification.verification_policy import (
    DeterministicIdentityVerificationPolicy,
    IdentityVerificationPolicy,
)

__all__ = (
    "ConstraintRequirementStatus",
    "DeterministicIdentityVerificationPolicy",
    "DeterministicProductIdentificationDecisionPolicy",
    "HypothesisVerificationCollection",
    "HypothesisVerificationState",
    "IdentityHypothesisVerification",
    "IdentityVerificationPolicy",
    "IdentityVerificationService",
    "MissingRequirement",
    "ProductIdentificationDecision",
    "ProductIdentificationDecisionPolicy",
    "ProductIdentificationDecisionReasonCode",
    "ProductIdentificationDecisionService",
    "ProductIdentificationOutcome",
    "ProductIdentificationVerificationError",
    "ProductIdentificationVerificationOutcome",
    "ProductIdentificationVerificationRequest",
    "ProductIdentificationVerificationService",
    "build_product_identification_verification_service",
)
