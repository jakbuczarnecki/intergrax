# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

from intergrax.logging import IntergraxLogging
from intergrax.runtime.replay.policy import PolicyDecision, PolicyDecisionType


# Base class ensures strong semantic contract
class PolicyActionHandler(ABC):

    @abstractmethod
    def handle(self, run_id: str, decision: PolicyDecision) -> None:
        ...


@dataclass(slots=True)
class LoggingPolicyAction(PolicyActionHandler):
    """
    Emits structured governance logs.
    """

    def __post_init__(self) -> None:
        self._logger = IntergraxLogging.get_logger(__name__, component="governance")

    def handle(self, run_id: str, decision: PolicyDecision) -> None:
        self._logger.warning(
            "Policy decision",
            extra={
                "run_id": run_id,
                "decision": decision.decision.value,
                "reasons": decision.reasons,
            },
        )


@dataclass(slots=True)
class BlockingPolicyAction(PolicyActionHandler):
    """
    Stops execution when decision is BLOCK.
    """

    def handle(self, run_id: str, decision: PolicyDecision) -> None:
        if decision.decision == PolicyDecisionType.BLOCK:
            raise RuntimeError(f"Run {run_id} blocked by policy: {decision.reasons}")


@dataclass(slots=True)
class AlertingPolicyAction(PolicyActionHandler):
    """
    Emits alerts for non-ALLOW decisions.
    """

    def __post_init__(self) -> None:
        self._logger = IntergraxLogging.get_logger(__name__, component="governance")

    def handle(self, run_id: str, decision: PolicyDecision) -> None:
        if decision.decision != PolicyDecisionType.ALLOW:
            self._logger.error(
                "Policy alert",
                extra={
                    "run_id": run_id,
                    "decision": decision.decision.value,
                    "reasons": decision.reasons,
                },
            )
