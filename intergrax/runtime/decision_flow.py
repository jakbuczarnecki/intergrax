# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Reusable production Decision flow gate (DS-MIG-01).

Thin composition around canonical Decision lifecycle, verification, revision,
human review, governance, and finalization — not a second runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, Protocol, TypeVar

from intergrax.contracts.decision_authorization import (
    DecisionAuthorizationEvaluator,
    DecisionExecutionAction,
    DecisionExecutionAuthorization,
    DecisionGovernanceDisposition,
    DecisionGovernanceEvaluationInput,
    DecisionGovernancePolicyContext,
    authoritative_decision_ref,
    evaluate_decision_governance_with,
)
from intergrax.contracts.decision_finalization import (
    DecisionFinalizeDisposition,
    DecisionFinalizeGuardResult,
    DecisionFinalizeGuardState,
    DecisionFinalizationKey,
    guard_decision_finalization,
    initial_decision_finalize_guard,
)
from intergrax.contracts.decision_human_review import (
    DecisionHumanReviewPending,
    DecisionHumanReviewPort,
    decision_human_review_request,
    governance_requires_human_review_reason,
    revision_exhausted_human_review_reason,
)
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionId,
    DecisionIdentity,
    DecisionScope,
    initial_decision_version,
    mint_decision_id,
)
from intergrax.contracts.decision_lifecycle import (
    DecisionLifecycleStage,
    DecisionLifecycleState,
)
from intergrax.contracts.decision_record import (
    AuthoritativeAcceptedDecision,
    CandidateDecision,
    DecisionArtifactKind,
    DecisionProposalRef,
    candidate_decision,
    candidate_decision_ref,
)
from intergrax.contracts.decision_resolution import (
    AuthoritativeResolutionRecord,
    DecisionResolution,
)
from intergrax.contracts.decision_revision import (
    DecisionRevisionDecision,
    DecisionRevisionDisposition,
    DecisionRevisionPolicy,
    evaluate_decision_revision,
    initial_decision_revision_state,
)
from intergrax.contracts.decision_verification import (
    VerificationDisposition,
    VerificationResult,
)
from intergrax.runtime.decision_authorization import mint_validated_execution_authorization
from intergrax.runtime.decision_human_review import (
    request_decision_human_review,
    transition_lifecycle_for_human_review_request,
)
from intergrax.runtime.decision_revision import transition_lifecycle_for_revision
from intergrax.runtime.decision_verification import VerificationPipeline
from intergrax.runtime.execution.active_decision_lifecycle_host import (
    require_active_decision_lifecycle_host,
)
from intergrax.runtime.execution.decision_lifecycle_host import DecisionLifecycleHost

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

    verification_pipeline: VerificationPipeline[T]
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
    """Neutral reusable authority seam for Graph, UAEP, and future hosts."""

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


def decision_identity_from_seed(seed: DecisionFlowIdentitySeed) -> DecisionIdentity:
    """Mint or reuse decision identity from neutral seed inputs."""
    if type(seed) is not DecisionFlowIdentitySeed:
        raise TypeError("seed must be DecisionFlowIdentitySeed")
    resolved_id = seed.decision_id if seed.decision_id is not None else mint_decision_id()
    return DecisionIdentity(
        decision_id=resolved_id,
        version=initial_decision_version(),
        scope=seed.scope,
        tenant_id=seed.tenant_id,
        execution=seed.execution,
    )


def critic_hooks_have_authority(
    *,
    verify_node_partial: bool,
    verify_graph_final: bool,
    verify_uaep_step: bool,
) -> bool:
    """Return whether legacy critic hook flags claim production authority."""
    return verify_node_partial or verify_graph_final or verify_uaep_step


def validate_decision_critic_authority_config(
    *,
    decision_gate: DecisionFlowGate[T] | None,
    verify_node_partial: bool = False,
    verify_graph_final: bool = False,
    verify_uaep_step: bool = False,
    critic_shadow_only: bool = False,
) -> None:
    """Reject dual production authority unless critic is explicitly shadow-only."""
    critic_authority = critic_hooks_have_authority(
        verify_node_partial=verify_node_partial,
        verify_graph_final=verify_graph_final,
        verify_uaep_step=verify_uaep_step,
    )
    if decision_gate is None or not critic_authority or critic_shadow_only:
        return
    raise DecisionCriticAuthorityConflictError(
        "Decision flow gate and legacy critic authority cannot both control outcomes",
    )


def _transition_lifecycle_to_terminal_resolution(
    lifecycle_host: DecisionLifecycleHost,
    lifecycle_state: DecisionLifecycleState,
) -> DecisionLifecycleState:
    lifecycle_state = lifecycle_host.transition(
        lifecycle_state,
        DecisionLifecycleStage.RESOLUTION,
    )
    return lifecycle_host.transition(
        lifecycle_state,
        DecisionLifecycleStage.FINALIZATION,
    )


def _accepted_proposal_ref(
    accepted: AuthoritativeAcceptedDecision[T],
) -> DecisionProposalRef:
    return DecisionProposalRef(
        identity=accepted.identity,
        lineage_ref=accepted.lineage.current,
    )


@dataclass(frozen=True, slots=True)
class CanonicalDecisionFlowGate(Generic[T]):
    """Stateless gate delegating to canonical Decision runtime components."""

    capabilities: DecisionFlowGateCapabilities[T]

    @property
    def scopes(self) -> frozenset[DecisionFlowScope]:
        return self.capabilities.scopes

    def supports_scope(self, flow_scope: DecisionFlowScope) -> bool:
        if type(flow_scope) is not DecisionFlowScope:
            raise TypeError("flow_scope must be DecisionFlowScope")
        return flow_scope in self.capabilities.scopes

    async def evaluate(
        self,
        request: DecisionFlowRequest[T],
    ) -> DecisionFlowResult[T]:
        if type(request) is not DecisionFlowRequest:
            raise TypeError("request must be DecisionFlowRequest")
        if not self.supports_scope(request.flow_scope):
            raise ValueError(
                f"decision flow scope {request.flow_scope.value!r} is not configured",
            )
        lifecycle_host = require_active_decision_lifecycle_host()
        identity = decision_identity_from_seed(request.identity_seed)
        candidate = candidate_decision(
            identity=identity,
            artifact_kind=request.artifact_kind,
            payload=request.payload,
        )
        proposal_ref = candidate_decision_ref(candidate)
        lifecycle_state = lifecycle_host.start(identity)
        lifecycle_state = lifecycle_host.transition(
            lifecycle_state,
            DecisionLifecycleStage.VERIFICATION,
        )
        verification_result = await self.capabilities.verification_pipeline.verify(
            candidate,
        )
        revision_state = initial_decision_revision_state(proposal_ref)
        finalize_key = DecisionFinalizationKey(
            decision_id=identity.decision_id,
            scope=identity.scope,
            tenant_id=identity.tenant_id,
        )
        guard_state = (
            request.finalize_guard_state
            if request.finalize_guard_state is not None
            else initial_decision_finalize_guard(finalize_key)
        )
        if verification_result.disposition is VerificationDisposition.PASSED:
            return await self._resolve_passed(
                candidate=candidate,
                lifecycle_host=lifecycle_host,
                lifecycle_state=lifecycle_state,
                verification_result=verification_result,
                guard_state=guard_state,
                flow_scope=request.flow_scope,
            )
        revision_decision = evaluate_decision_revision(
            policy=self.capabilities.revision_policy,
            state=revision_state,
            verification_result=verification_result,
        )
        if revision_decision.disposition is DecisionRevisionDisposition.ALLOWED:
            lifecycle_state = transition_lifecycle_for_revision(
                lifecycle_state=lifecycle_state,
                verification_result=verification_result,
                revision_decision=revision_decision,
            )
            return DecisionFlowResult(
                host_action=DecisionFlowHostAction.BLOCK,
                flow_scope=request.flow_scope,
                candidate=candidate,
                verification_result=verification_result,
                lifecycle_state=lifecycle_state,
                revision_decision=revision_decision,
                authority_reason="decision_revision_required",
            )
        if (
            revision_decision.disposition is DecisionRevisionDisposition.EXHAUSTED
            and self.capabilities.request_human_on_revision_exhausted
            and self.capabilities.human_review_port is not None
        ):
            review_request = decision_human_review_request(
                proposal_ref=proposal_ref,
                reason_code=revision_exhausted_human_review_reason(),
            )
            pending = request_decision_human_review(review_request)
            self.capabilities.human_review_port.request_review(review_request)
            lifecycle_state = transition_lifecycle_for_human_review_request(
                lifecycle_state=lifecycle_state,
            )
            return DecisionFlowResult(
                host_action=DecisionFlowHostAction.PENDING_HUMAN,
                flow_scope=request.flow_scope,
                candidate=candidate,
                verification_result=verification_result,
                lifecycle_state=lifecycle_state,
                human_review_pending=pending,
                revision_decision=revision_decision,
                authority_reason="decision_human_review_pending",
            )
        lifecycle_state = _transition_lifecycle_to_terminal_resolution(
            lifecycle_host,
            lifecycle_state,
        )
        rejected = AuthoritativeResolutionRecord(
            identity=identity,
            resolution=DecisionResolution.REJECTED,
        )
        return DecisionFlowResult(
            host_action=DecisionFlowHostAction.BLOCK,
            flow_scope=request.flow_scope,
            candidate=candidate,
            verification_result=verification_result,
            lifecycle_state=lifecycle_state,
            resolution_record=rejected,
            revision_decision=revision_decision,
            authority_reason="decision_verification_rejected",
        )

    async def _resolve_passed(
        self,
        *,
        candidate: CandidateDecision[T],
        lifecycle_host: DecisionLifecycleHost,
        lifecycle_state: DecisionLifecycleState,
        verification_result: VerificationResult,
        guard_state: DecisionFinalizeGuardState[T],
        flow_scope: DecisionFlowScope,
    ) -> DecisionFlowResult[T]:
        lifecycle_state = lifecycle_host.transition(
            lifecycle_state,
            DecisionLifecycleStage.RESOLUTION,
        )
        accepted = AuthoritativeAcceptedDecision(
            identity=candidate.identity,
            artifact=candidate.artifact,
            lineage=candidate.lineage,
        )
        finalize_result = guard_decision_finalization(guard_state, accepted)
        lifecycle_state = lifecycle_host.transition(
            lifecycle_state,
            DecisionLifecycleStage.FINALIZATION,
        )
        governance_spec = self.capabilities.governance_spec
        if governance_spec is None:
            return DecisionFlowResult(
                host_action=DecisionFlowHostAction.CONTINUE,
                flow_scope=flow_scope,
                candidate=candidate,
                verification_result=verification_result,
                lifecycle_state=lifecycle_state,
                accepted_decision=accepted,
                finalize_disposition=finalize_result.disposition,
            )
        evaluation_input = DecisionGovernanceEvaluationInput(
            decision=accepted,
            action=governance_spec.action,
            policy_context=governance_spec.policy_context,
        )
        governance_decision = evaluate_decision_governance_with(
            evaluator=governance_spec.evaluator,
            evaluation_input=evaluation_input,
        )
        if governance_decision.disposition is DecisionGovernanceDisposition.ALLOW:
            authorization = mint_validated_execution_authorization(
                evaluation_input=evaluation_input,
                governance_decision=governance_decision,
            )
            _ = authoritative_decision_ref(accepted)
            return DecisionFlowResult(
                host_action=DecisionFlowHostAction.CONTINUE,
                flow_scope=flow_scope,
                candidate=candidate,
                verification_result=verification_result,
                lifecycle_state=lifecycle_state,
                accepted_decision=accepted,
                authorization=authorization,
                finalize_disposition=finalize_result.disposition,
            )
        if governance_decision.disposition is DecisionGovernanceDisposition.DENY:
            return DecisionFlowResult(
                host_action=DecisionFlowHostAction.BLOCK,
                flow_scope=flow_scope,
                candidate=candidate,
                verification_result=verification_result,
                lifecycle_state=lifecycle_state,
                accepted_decision=accepted,
                finalize_disposition=finalize_result.disposition,
                authority_reason="decision_governance_denied",
            )
        if governance_decision.disposition is DecisionGovernanceDisposition.REQUIRE_HUMAN:
            return self._resolve_governance_require_human(
                candidate=candidate,
                accepted=accepted,
                lifecycle_state=lifecycle_state,
                verification_result=verification_result,
                finalize_result=finalize_result,
                flow_scope=flow_scope,
            )
        raise ValueError(
            f"unsupported governance disposition: {governance_decision.disposition.value!r}",
        )

    def _resolve_governance_require_human(
        self,
        *,
        candidate: CandidateDecision[T],
        accepted: AuthoritativeAcceptedDecision[T],
        lifecycle_state: DecisionLifecycleState,
        verification_result: VerificationResult,
        finalize_result: DecisionFinalizeGuardResult[T],
        flow_scope: DecisionFlowScope,
    ) -> DecisionFlowResult[T]:
        human_review_port = self.capabilities.human_review_port
        if human_review_port is None:
            return DecisionFlowResult(
                host_action=DecisionFlowHostAction.BLOCK,
                flow_scope=flow_scope,
                candidate=candidate,
                verification_result=verification_result,
                lifecycle_state=lifecycle_state,
                accepted_decision=accepted,
                finalize_disposition=finalize_result.disposition,
                authority_reason="decision_governance_human_review_unavailable",
            )
        review_request = decision_human_review_request(
            proposal_ref=_accepted_proposal_ref(accepted),
            reason_code=governance_requires_human_review_reason(),
        )
        pending = request_decision_human_review(review_request)
        try:
            human_review_port.request_review(review_request)
        except Exception:
            return DecisionFlowResult(
                host_action=DecisionFlowHostAction.BLOCK,
                flow_scope=flow_scope,
                candidate=candidate,
                verification_result=verification_result,
                lifecycle_state=lifecycle_state,
                accepted_decision=accepted,
                finalize_disposition=finalize_result.disposition,
                authority_reason="decision_governance_human_review_unavailable",
            )
        return DecisionFlowResult(
            host_action=DecisionFlowHostAction.PENDING_HUMAN,
            flow_scope=flow_scope,
            candidate=candidate,
            verification_result=verification_result,
            lifecycle_state=lifecycle_state,
            accepted_decision=accepted,
            human_review_pending=pending,
            finalize_disposition=finalize_result.disposition,
            authority_reason="decision_governance_human_review_pending",
        )
