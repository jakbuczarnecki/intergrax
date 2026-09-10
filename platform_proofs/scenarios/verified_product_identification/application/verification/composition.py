"""Constructor injection for verification layer."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.application.verification.decision_policy import (
    DeterministicProductIdentificationDecisionPolicy,
    ProductIdentificationDecisionPolicy,
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


def build_product_identification_verification_service(
    *,
    verification_policy: IdentityVerificationPolicy | None = None,
    decision_policy: ProductIdentificationDecisionPolicy | None = None,
) -> ProductIdentificationVerificationService:
    resolved_verification = (
        verification_policy
        if verification_policy is not None
        else DeterministicIdentityVerificationPolicy()
    )
    resolved_decision = (
        decision_policy
        if decision_policy is not None
        else DeterministicProductIdentificationDecisionPolicy()
    )
    return ProductIdentificationVerificationService(
        identity_verification_service=IdentityVerificationService(
            verification_policy=resolved_verification,
        ),
        decision_service=ProductIdentificationDecisionService(
            decision_policy=resolved_decision,
        ),
    )
