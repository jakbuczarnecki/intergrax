# © Artur Czarnecki. All rights reserved.

"""HITL approval gate for policy-learning apply (Phase W-ADAPT-4.6)."""

from __future__ import annotations

from typing import Protocol

from intergrax.runtime.adaptive.adaptation_models import AdaptationProposalPackage
from intergrax.runtime.architecture.adaptive_governance import AdaptiveLoopKind


class PolicyLearningApprovalStore(Protocol):
    """Tracks human approval for policy-learning proposals."""

    def is_approved(self, proposal_id: str) -> bool: ...

    def record_approval(self, proposal_id: str, *, approver_id: str) -> None: ...

    def clear(self) -> None: ...


class InMemoryPolicyLearningApprovalStore:
    """In-process approval store for unit tests."""

    def __init__(self) -> None:
        self._approved: dict[str, str] = {}

    def is_approved(self, proposal_id: str) -> bool:
        return proposal_id in self._approved

    def record_approval(self, proposal_id: str, *, approver_id: str) -> None:
        self._approved[proposal_id] = approver_id

    def clear(self) -> None:
        self._approved.clear()


class PolicyLearningApprovalRequiredError(ValueError):
    """Raised when apply is attempted without required human approval."""


def require_policy_learning_approval(
    package: AdaptationProposalPackage,
    *,
    approval_store: PolicyLearningApprovalStore,
) -> None:
    """Block apply for policy-learning proposals until human approval is recorded."""
    kind = package.candidate.proposal.envelope.kind
    if kind != AdaptiveLoopKind.POLICY_LEARNING:
        return
    if approval_store.is_approved(package.proposal_id):
        return
    raise PolicyLearningApprovalRequiredError(
        f"Policy learning proposal '{package.proposal_id}' requires human approval before apply"
    )
