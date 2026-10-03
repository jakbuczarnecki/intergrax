# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Neutral Decision flow gate contracts (host wiring / Graph / UAEP seam)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, Protocol, TypeVar, runtime_checkable

from intergrax.contracts.decision_authorization import (
    DecisionAuthorizationEvaluator,
    DecisionExecutionAction,
    DecisionExecutionAuthorization,
    DecisionGovernancePolicyContext,
)
from intergrax.contracts.decision_finalization import (
    DecisionFinalizeDisposition,
    DecisionFinalizeGuardState,
)
from intergrax.contracts.decision_human_review import (
    DecisionHumanReviewPending,
    DecisionHumanReviewPort,
)
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionId,
    DecisionScope,
)
from intergrax.contracts.decision_lifecycle import DecisionLifecycleState
from intergrax.contracts.decision_record import (
    AuthoritativeAcceptedDecision,
    CandidateDecision,
    DecisionArtifactKind,
)
from intergrax.contracts.decision_resolution import AuthoritativeResolutionRecord
from intergrax.contracts.decision_revision import (
    DecisionRevisionDecision,
    DecisionRevisionPolicy,
)
from intergrax.contracts.decision_verification import VerificationResult

T = TypeVar("T")


class DecisionFlowScope(str, Enum):
    """Host invocation scopes supported by one configured gate."""

    GRAPH_FINAL = "graph_final"
    UAEP_STEP = "uaep_step"


class DecisionFlowHostAction(str, Enum):
    """Execution-facing instruction for the hosting Graph or UAEP surface."""

    CONTINUE = "continue"
    BLOCK = "block"
    PENDING_HUMAN = "pending_human"


class DecisionCriticAuthorityConflictError(ValueError):
    """Raised when Decision and legacy Critic both claim production authority."""


@runtime_checkable
class DecisionVerificationPipelinePort(Protocol[T]):
    """Verification orchestration consumed by one Decision flow gate."""

    async def verify(self, candidate: CandidateDecision[T]) -> VerificationResult: ...


@dataclass(frozen=True, slots=True)
class DecisionFlowIdentitySeed:
    """Neutral identity inputs without Graph, UAEP, or Nexus types."""

    scope: DecisionScope
    tenant_id: str
    execution: DecisionExecutionLineage
    decision_id: DecisionId | None = None


@dataclass(frozen=True, slots=True)
class DecisionFlowGovernanceSpec(Generic[T]):
    """Optional governance evaluation for one governed execution action."""

    action: DecisionExecutionAction
    policy_context: DecisionGovernancePolicyContext
    evaluator: DecisionAuthorizationEvaluator


@dataclass(frozen=True, slots=True)
class DecisionFlowGateCapabilities(Generic[T]):
    """Immutable capability bundle composed by the hosting application."""

    verification_pipeline: DecisionVerificationPipelinePort[T]
    revision_policy: DecisionRevisionPolicy
    scopes: frozenset[DecisionFlowScope]
    human_review_port: DecisionHumanReviewPort | None = None
    governance_spec: DecisionFlowGovernanceSpec[T] | None = None
    request_human_on_revision_exhausted: bool = True


@dataclass(frozen=True, slots=True)
class DecisionFlowRequest(Generic[T]):
    """One immutable decision-flow evaluation request."""

    identity_seed: DecisionFlowIdentitySeed
    artifact_kind: DecisionArtifactKind
    payload: T
    flow_scope: DecisionFlowScope
    finalize_guard_state: DecisionFinalizeGuardState[T] | None = None


@dataclass(frozen=True, slots=True)
class DecisionFlowResult(Generic[T]):
    """Typed semantic outcome for one decision-flow evaluation."""

    host_action: DecisionFlowHostAction
    flow_scope: DecisionFlowScope
    candidate: CandidateDecision[T]
    verification_result: VerificationResult
    lifecycle_state: DecisionLifecycleState
    accepted_decision: AuthoritativeAcceptedDecision[T] | None = None
    resolution_record: AuthoritativeResolutionRecord | None = None
    human_review_pending: DecisionHumanReviewPending | None = None
    authorization: DecisionExecutionAuthorization | None = None
    revision_decision: DecisionRevisionDecision | None = None
    finalize_disposition: DecisionFinalizeDisposition | None = None
    authority_reason: str | None = None


class DecisionFlowGate(Protocol[T]):
    """Neutral reusable authority seam for Graph, UAEP, and orchestration hosts."""

    @property
    def capabilities(self) -> DecisionFlowGateCapabilities[T]:
        """Return immutable configured capabilities."""
        ...

    def supports_scope(self, flow_scope: DecisionFlowScope) -> bool:
        """Return whether this gate is configured for one host scope."""
        ...

    async def evaluate(
        self,
        request: DecisionFlowRequest[T],
    ) -> DecisionFlowResult[T]:
        """Run canonical decision lifecycle composition for one candidate."""
        ...


__all__ = [
    "DecisionCriticAuthorityConflictError",
    "DecisionFlowGate",
    "DecisionFlowGateCapabilities",
    "DecisionFlowGovernanceSpec",
    "DecisionFlowHostAction",
    "DecisionFlowIdentitySeed",
    "DecisionFlowRequest",
    "DecisionFlowResult",
    "DecisionFlowScope",
    "DecisionVerificationPipelinePort",
]
