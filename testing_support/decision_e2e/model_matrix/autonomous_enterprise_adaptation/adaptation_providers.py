# © Artur Czarnecki. All rights reserved.

"""Enterprise adaptation provider plugins (DS-E2E-15J-L13)."""

from __future__ import annotations

from dataclasses import dataclass

from testing_support.decision_e2e.model_matrix.autonomous_enterprise_adaptation.contracts import (
    AdaptationExecutionStatus,
    ApprovedAdaptationRequest,
)
from testing_support.decision_e2e.model_matrix.autonomous_enterprise_adaptation.protocol import (
    AdaptationApplyOutcome,
)

_DEFAULT_PROVIDER_ID = "default_enterprise_adaptation"
_DEFAULT_PROVIDER_VERSION = "1"


@dataclass(frozen=True, slots=True)
class DefaultEnterpriseAdaptationProvider:
    """Default controlled adaptation — records strategy activation, no runtime mutation."""

    @property
    def provider_id(self) -> str:
        return _DEFAULT_PROVIDER_ID

    @property
    def provider_version(self) -> str:
        return _DEFAULT_PROVIDER_VERSION

    def apply_adaptation(
        self, approved_change: ApprovedAdaptationRequest
    ) -> AdaptationApplyOutcome:
        reference = (
            f"adapted:{approved_change.adaptation_id}:"
            f"v{approved_change.version}:"
            f"{approved_change.source_reference.proposal_id}"
        )
        return AdaptationApplyOutcome(
            status=AdaptationExecutionStatus.APPLIED,
            applied_change_reference=reference,
            summary=(
                f"Activated approved adaptation {approved_change.adaptation_id} "
                f"for scope {approved_change.scope.scope_id}."
            ),
        )


__all__ = [
    "DefaultEnterpriseAdaptationProvider",
]
