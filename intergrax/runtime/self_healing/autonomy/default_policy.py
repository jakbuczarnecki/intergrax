# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Safe default autonomy policy — recommend-only posture (SELF-HEALING R6.1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.self_healing.autonomy.level import AutonomyLevel
from intergrax.contracts.self_healing.autonomy.policy import (
    AutonomyConstraintDescriptor,
    AutonomyPolicyOutcome,
)
from intergrax.contracts.self_healing.autonomy.request import AutonomyControlRequest


@dataclass(frozen=True, slots=True)
class DefaultAutonomyPolicy:
    _policy_id: str = "platform.default_autonomy"
    _policy_version: str = "1.0.0"

    @property
    def policy_id(self) -> str:
        return self._policy_id

    def evaluate(self, request: AutonomyControlRequest) -> AutonomyPolicyOutcome:
        _ = request
        return AutonomyPolicyOutcome(
            policy_id=self.policy_id,
            policy_version=self._policy_version,
            suggested_level=AutonomyLevel.RECOMMEND_ONLY,
            constraint_descriptors=(
                AutonomyConstraintDescriptor(
                    code="enterprise.safe_default",
                    description="Advisory recommendations only; no automated execution path.",
                ),
            ),
            rationale="Platform default restricts autonomy to recommendation consumption.",
        )


__all__ = ["DefaultAutonomyPolicy"]
