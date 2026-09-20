# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical MSE HITL effect gate after fresh Governance authorize (GR-10-R11-R2/R3).

Human judgment evidence and continuation grants never become Governance ALLOW.
``GovernedContinuationApprovalGrant`` is correlation / single-use evidence only.
Canonical pause / wait / resume authority remains ``ExecutionContinuationPort``.

Ordinary fresh ``ALLOW`` (no canonical HITL continuation for the current proposal)
proceeds without an approval grant.

Post-HITL physical effect may proceed only when:

* fresh Governance returns ``ALLOW``, and
* canonical human-governed continuation for the current proposal is ``RESUMED``
  (GR-5: only ``RESUMED`` clears the progress gate; ``RESUME_AUTHORIZED`` still
  blocks execution progress), and
* a scoped approval grant matches the current side-effect proposal (evidence), and
* continuation identity / ``continuation_request_id`` correlate with that grant.

Proposal identity for post-HITL classification requires exact
``GovernedContinuationCorrelation`` match (execution + operation + resource +
side-effect scope / digest). ``human_request_id`` alone never classifies the
current effect as post-HITL (GR-10-R11-R3).

Fresh ``REQUIRE_HUMAN`` is never ``PROCEED`` — matching grants cannot override it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from intergrax.contracts.collaborative_work import CollaborativeWorkEnforcementRequest
from intergrax.contracts.execution_continuation import (
    ExecutionContinuationError,
    ExecutionContinuationErrorCode,
    ExecutionContinuationIdentity,
    ExecutionContinuationLifecycleState,
    ExecutionContinuationLookup,
    ExecutionContinuationPort,
    PendingExecutionContinuation,
)
from intergrax.contracts.governed_continuation import GovernedContinuationRequest
from intergrax.contracts.governed_continuation_correlation import (
    GovernedContinuationCorrelation,
)
from intergrax.contracts.governed_continuation_grant import GovernedContinuationApprovalGrant
from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectRequest
from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationResult,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.execution.active_execution_continuation_store import (
    peek_active_execution_continuation_state_store,
)
from intergrax.runtime.execution.continuation.composition import (
    wire_execution_continuation_port,
)
from intergrax.runtime.human.governed_continuation_grant import (
    GovernedContinuationGrantCoordinator,
    grant_belongs_to_same_proposal_scope,
)
from intergrax.runtime.task.task import Task

_POST_HITL_EFFECT_LIFECYCLE: frozenset[ExecutionContinuationLifecycleState] = frozenset(
    {
        ExecutionContinuationLifecycleState.RESUMED,
    }
)


class MseHitlEffectGateDisposition(Enum):
    """Disposition after fresh MSE authorization under HITL semantics."""

    PROCEED = auto()
    BLOCK = auto()
    REQUIRE_HITL = auto()


class HumanGovernedProposalRelation(Enum):
    """Proposal-scoped relation between a pending continuation and the current effect.

    ``human_request_id`` alone is never ``MATCHED_HITL_PROPOSAL`` (GR-10-R11-R3).
    """

    MATCHED_HITL_PROPOSAL = auto()
    UNRELATED_HUMAN_CONTINUATION = auto()
    NON_HITL_CONTINUATION = auto()
    CORRELATION_INSUFFICIENT = auto()


class EffectContinuationClassification(Enum):
    """Typed classification of canonical continuation relative to the current effect."""

    NO_CANONICAL_CONTINUATION = auto()
    CANONICAL_NON_HITL_OR_NON_BLOCKING = auto()
    UNRELATED_HUMAN_CONTINUATION = auto()
    CORRELATION_INSUFFICIENT = auto()
    POST_HITL_RESUMED = auto()
    POST_HITL_NOT_RESUMED = auto()


_ORDINARY_ALLOW_CLASSIFICATIONS: frozenset[EffectContinuationClassification] = frozenset(
    {
        EffectContinuationClassification.CANONICAL_NON_HITL_OR_NON_BLOCKING,
        EffectContinuationClassification.UNRELATED_HUMAN_CONTINUATION,
        EffectContinuationClassification.CORRELATION_INSUFFICIENT,
    }
)


@dataclass(frozen=True, slots=True)
class MseHitlEffectGateOutcome:
    """Typed outcome of the MSE HITL effect gate."""

    disposition: MseHitlEffectGateDisposition
    authorization: MeaningfulSideEffectAuthorizationResult

    @property
    def governed_continuation_request(self) -> GovernedContinuationRequest | None:
        return self.authorization.governed_continuation_request


@dataclass(frozen=True, slots=True)
class CanonicalContinuationAuthorityView:
    """Typed snapshot of canonical continuation authority for post-HITL effect."""

    continuation_id: str
    lifecycle_state: ExecutionContinuationLifecycleState
    identity: ExecutionContinuationIdentity


@dataclass(frozen=True, slots=True)
class EffectContinuationContext:
    """Typed continuation context for one consequential effect proposal."""

    classification: EffectContinuationClassification
    pending: PendingExecutionContinuation | None = None


def _identity_from_side_effect(
    side_effect: MeaningfulSideEffectRequest,
) -> ExecutionContinuationIdentity:
    return ExecutionContinuationIdentity(
        task_id=side_effect.task_id,
        run_id=side_effect.run_id,
        attempt_id=side_effect.attempt_id,
        execution_id=side_effect.execution_id,
    )


def _load_pending(
    port: ExecutionContinuationPort,
    *,
    continuation_id: str | None = None,
    identity: ExecutionContinuationIdentity | None = None,
) -> PendingExecutionContinuation | None:
    try:
        return port.get_pending(
            ExecutionContinuationLookup(
                continuation_id=continuation_id,
                identity=identity,
            )
        )
    except ExecutionContinuationError as exc:
        if exc.code is ExecutionContinuationErrorCode.NOT_FOUND:
            return None
        raise


def _identity_matches_side_effect(
    identity: ExecutionContinuationIdentity,
    side_effect: MeaningfulSideEffectRequest,
) -> bool:
    return (
        identity.task_id == side_effect.task_id
        and identity.run_id == side_effect.run_id
        and identity.attempt_id == side_effect.attempt_id
        and identity.execution_id == side_effect.execution_id
    )


def _governed_correlation_matches_current_proposal(
    correlation: GovernedContinuationCorrelation,
    *,
    side_effect: MeaningfulSideEffectRequest,
    operation_id: str,
    resource_scope: str | None,
) -> bool:
    if not _identity_matches_side_effect(
        ExecutionContinuationIdentity(
            task_id=correlation.task_id,
            run_id=correlation.run_id,
            attempt_id=correlation.attempt_id,
            execution_id=correlation.execution_id,
        ),
        side_effect,
    ):
        return False
    normalized_operation = operation_id.strip()
    if not normalized_operation or correlation.operation_id != normalized_operation:
        return False
    if correlation.resource_scope is not None and correlation.resource_scope != resource_scope:
        return False
    if (
        correlation.side_effect_scope_id is not None
        and correlation.side_effect_scope_id != side_effect.side_effect_scope_id
    ):
        return False
    if (
        correlation.side_effect_scope_digest is not None
        and correlation.side_effect_scope_digest != side_effect.side_effect_scope_digest
    ):
        return False
    return True


def classify_human_governed_proposal_relation(
    pending: PendingExecutionContinuation,
    *,
    side_effect: MeaningfulSideEffectRequest,
    operation_id: str,
    resource_scope: str | None,
) -> HumanGovernedProposalRelation:
    """Classify whether pending is human-governed for *this* consequential proposal.

    Exact ``GovernedContinuationCorrelation`` match is the only positive proof of
    post-HITL authority for the current effect. ``human_request_id`` without
    proposal correlation is ``CORRELATION_INSUFFICIENT`` — never automatic HITL
    for every later effect on the same execution (GR-10-R11-R3).
    """
    correlation = pending.governed_correlation
    if correlation is not None:
        if _governed_correlation_matches_current_proposal(
            correlation,
            side_effect=side_effect,
            operation_id=operation_id,
            resource_scope=resource_scope,
        ):
            return HumanGovernedProposalRelation.MATCHED_HITL_PROPOSAL
        if pending.human_request_id is not None:
            return HumanGovernedProposalRelation.UNRELATED_HUMAN_CONTINUATION
        return HumanGovernedProposalRelation.NON_HITL_CONTINUATION
    if pending.human_request_id is not None:
        return HumanGovernedProposalRelation.CORRELATION_INSUFFICIENT
    return HumanGovernedProposalRelation.NON_HITL_CONTINUATION


def resolve_effect_continuation_context(
    port: ExecutionContinuationPort,
    *,
    side_effect: MeaningfulSideEffectRequest,
    operation_id: str,
    resource_scope: str | None = None,
    grant: GovernedContinuationApprovalGrant | None = None,
) -> EffectContinuationContext:
    """Classify canonical continuation state for the current consequential proposal."""
    identity = _identity_from_side_effect(side_effect)
    if grant is not None:
        pending = _load_pending(
            port,
            continuation_id=grant.continuation_request_id,
            identity=identity,
        )
        if pending is None or pending.continuation_id != grant.continuation_request_id:
            return EffectContinuationContext(
                classification=EffectContinuationClassification.NO_CANONICAL_CONTINUATION,
            )
    else:
        pending = _load_pending(port, identity=identity)
        if pending is None:
            return EffectContinuationContext(
                classification=EffectContinuationClassification.NO_CANONICAL_CONTINUATION,
            )

    if not _identity_matches_side_effect(pending.identity, side_effect):
        return EffectContinuationContext(
            classification=EffectContinuationClassification.NO_CANONICAL_CONTINUATION,
        )

    relation = classify_human_governed_proposal_relation(
        pending,
        side_effect=side_effect,
        operation_id=operation_id,
        resource_scope=resource_scope,
    )
    if relation is HumanGovernedProposalRelation.CORRELATION_INSUFFICIENT:
        return EffectContinuationContext(
            classification=EffectContinuationClassification.CORRELATION_INSUFFICIENT,
            pending=pending,
        )
    if relation is HumanGovernedProposalRelation.UNRELATED_HUMAN_CONTINUATION:
        return EffectContinuationContext(
            classification=EffectContinuationClassification.UNRELATED_HUMAN_CONTINUATION,
            pending=pending,
        )
    if relation is HumanGovernedProposalRelation.NON_HITL_CONTINUATION:
        return EffectContinuationContext(
            classification=EffectContinuationClassification.CANONICAL_NON_HITL_OR_NON_BLOCKING,
            pending=pending,
        )

    if pending.lifecycle_state in _POST_HITL_EFFECT_LIFECYCLE:
        return EffectContinuationContext(
            classification=EffectContinuationClassification.POST_HITL_RESUMED,
            pending=pending,
        )
    return EffectContinuationContext(
        classification=EffectContinuationClassification.POST_HITL_NOT_RESUMED,
        pending=pending,
    )


def _effective_attested_policy_bundle(
    decision: PolicyDecision,
) -> tuple[str, str, str] | None:
    if decision.has_attested_policy_bundle_refs():
        return (
            str(decision.policy_bundle_id),
            str(decision.policy_bundle_version),
            str(decision.policy_bundle_digest),
        )
    audit = decision.audit_payload
    if not isinstance(audit, dict):
        return None
    layers = audit.get("contributing_layers")
    if not isinstance(layers, dict):
        return None
    runtime = layers.get("runtime_policy")
    if not isinstance(runtime, dict):
        return None
    bundle_id = str(runtime.get("policy_bundle_id") or "").strip()
    bundle_version = str(runtime.get("policy_bundle_version") or "").strip()
    bundle_digest = str(runtime.get("policy_bundle_digest") or "").strip()
    if not bundle_id or not bundle_version or not bundle_digest:
        return None
    return (bundle_id, bundle_version, bundle_digest)


def _grant_policy_bundle_matches_decision(
    grant: GovernedContinuationApprovalGrant,
    decision: PolicyDecision,
) -> bool:
    if not grant.has_attested_policy_bundle_refs():
        return False
    effective = _effective_attested_policy_bundle(decision)
    if effective is None:
        return False
    bundle_id, bundle_version, bundle_digest = effective
    return (
        grant.policy_bundle_id == bundle_id
        and grant.policy_bundle_version == bundle_version
        and grant.policy_bundle_digest == bundle_digest
    )


def _grant_matches_post_hitl_allow_proposal(
    grant: GovernedContinuationApprovalGrant,
    *,
    side_effect: MeaningfulSideEffectRequest,
    operation_id: str,
    resource_scope: str | None,
    decision: PolicyDecision,
) -> bool:
    if not grant_belongs_to_same_proposal_scope(
        grant,
        side_effect=side_effect,
        operation_id=operation_id,
        resource_scope=resource_scope,
    ):
        return False
    return _grant_policy_bundle_matches_decision(grant, decision)


def resolve_continuation_port_for_mse_hitl_gate(
    continuation_port: ExecutionContinuationPort | None = None,
) -> ExecutionContinuationPort | None:
    if continuation_port is not None:
        return continuation_port
    store = peek_active_execution_continuation_state_store()
    if store is None:
        return None
    return wire_execution_continuation_port(state_store=store)


def resolve_canonical_continuation_authority(
    port: ExecutionContinuationPort,
    *,
    side_effect: MeaningfulSideEffectRequest,
    grant: GovernedContinuationApprovalGrant | None,
    operation_id: str = "",
    resource_scope: str | None = None,
) -> CanonicalContinuationAuthorityView | None:
    """Return RESUMED human-governed authority for the current proposal, or None."""
    if not operation_id and grant is not None:
        operation_id = grant.operation_id
    if resource_scope is None and grant is not None:
        resource_scope = grant.resource_scope
    # Classification is always proposal/identity based — never grant-keyed.
    context = resolve_effect_continuation_context(
        port,
        side_effect=side_effect,
        operation_id=operation_id,
        resource_scope=resource_scope,
        grant=None,
    )
    if context.classification is not EffectContinuationClassification.POST_HITL_RESUMED:
        return None
    pending = context.pending
    if pending is None:
        return None
    if grant is not None and pending.continuation_id != grant.continuation_request_id:
        return None
    return CanonicalContinuationAuthorityView(
        continuation_id=pending.continuation_id,
        lifecycle_state=pending.lifecycle_state,
        identity=pending.identity,
    )


def evaluate_mse_hitl_effect_gate(
    authorization: MeaningfulSideEffectAuthorizationResult,
    *,
    enforcement_request: CollaborativeWorkEnforcementRequest,
    task: Task | None = None,
    continuation_port: ExecutionContinuationPort | None = None,
) -> MseHitlEffectGateOutcome:
    action = authorization.decision.action
    enforcement = authorization.enforcement_result
    side_effect = enforcement_request.meaningful_side_effect_request
    resource_scope = enforcement_request.resource_scope or enforcement.authority_scope
    operation_id = enforcement.operation_id
    stored_grant = (
        task.runtime.governance.governed_continuation_grant if task is not None else None
    )

    if action is PolicyAction.DENY or action is PolicyAction.MODIFY:
        if task is not None and side_effect is not None:
            GovernedContinuationGrantCoordinator.clear_obsolete_grant_for_proposal(
                task,
                side_effect=side_effect,
                operation_id=operation_id,
                resource_scope=resource_scope,
            )
        return MseHitlEffectGateOutcome(
            disposition=MseHitlEffectGateDisposition.BLOCK,
            authorization=authorization,
        )

    if action is PolicyAction.ESCALATE:
        return MseHitlEffectGateOutcome(
            disposition=MseHitlEffectGateDisposition.REQUIRE_HITL,
            authorization=authorization,
        )

    if action is PolicyAction.REQUIRE_HUMAN:
        if task is not None and side_effect is not None:
            GovernedContinuationGrantCoordinator.clear_obsolete_grant_for_proposal(
                task,
                side_effect=side_effect,
                operation_id=operation_id,
                resource_scope=resource_scope,
            )
        return MseHitlEffectGateOutcome(
            disposition=MseHitlEffectGateDisposition.REQUIRE_HITL,
            authorization=authorization,
        )

    if action is PolicyAction.ALLOW and authorization.permitted:
        return _evaluate_allow_for_effect(
            authorization,
            task=task,
            side_effect=side_effect,
            operation_id=operation_id,
            resource_scope=resource_scope,
            stored_grant=stored_grant,
            continuation_port=continuation_port,
        )

    return MseHitlEffectGateOutcome(
        disposition=MseHitlEffectGateDisposition.BLOCK,
        authorization=authorization,
    )


def _proceed(
    authorization: MeaningfulSideEffectAuthorizationResult,
) -> MseHitlEffectGateOutcome:
    return MseHitlEffectGateOutcome(
        disposition=MseHitlEffectGateDisposition.PROCEED,
        authorization=authorization,
    )


def _block(
    authorization: MeaningfulSideEffectAuthorizationResult,
) -> MseHitlEffectGateOutcome:
    return MseHitlEffectGateOutcome(
        disposition=MseHitlEffectGateDisposition.BLOCK,
        authorization=authorization,
    )


def _evaluate_allow_for_effect(
    authorization: MeaningfulSideEffectAuthorizationResult,
    *,
    task: Task | None,
    side_effect: MeaningfulSideEffectRequest | None,
    operation_id: str,
    resource_scope: str | None,
    stored_grant: GovernedContinuationApprovalGrant | None,
    continuation_port: ExecutionContinuationPort | None,
) -> MseHitlEffectGateOutcome:
    # Ordinary ALLOW: no port / no side-effect identity → no post-HITL classification.
    if continuation_port is None or side_effect is None:
        if stored_grant is None:
            return _proceed(authorization)
        return _block(authorization)

    # Classification is always by identity + proposal scope — never by grant presence.
    context = resolve_effect_continuation_context(
        continuation_port,
        side_effect=side_effect,
        operation_id=operation_id,
        resource_scope=resource_scope,
        grant=None,
    )

    if context.classification is EffectContinuationClassification.NO_CANONICAL_CONTINUATION:
        if stored_grant is None:
            if task is not None:
                GovernedContinuationGrantCoordinator.clear_obsolete_grant_for_proposal(
                    task,
                    side_effect=side_effect,
                    operation_id=operation_id,
                    resource_scope=resource_scope,
                )
            return _proceed(authorization)
        # Unrelated stale grant must not block ordinary ALLOW for a different proposal.
        if not grant_belongs_to_same_proposal_scope(
            stored_grant,
            side_effect=side_effect,
            operation_id=operation_id,
            resource_scope=resource_scope,
        ):
            return _proceed(authorization)
        if task is not None:
            GovernedContinuationGrantCoordinator.clear_obsolete_grant_for_proposal(
                task,
                side_effect=side_effect,
                operation_id=operation_id,
                resource_scope=resource_scope,
            )
        return _block(authorization)

    if context.classification in _ORDINARY_ALLOW_CLASSIFICATIONS:
        # Non-HITL / unrelated human / correlation-insufficient: ordinary ALLOW.
        # Legacy human_request_id without proposal correlation must not force post-HITL.
        # Unrelated stale grant must neither authorize nor block this effect.
        if stored_grant is not None and task is not None:
            if grant_belongs_to_same_proposal_scope(
                stored_grant,
                side_effect=side_effect,
                operation_id=operation_id,
                resource_scope=resource_scope,
            ):
                # Same-proposal grant without human-governed continuation for this effect.
                GovernedContinuationGrantCoordinator.clear_obsolete_grant_for_proposal(
                    task,
                    side_effect=side_effect,
                    operation_id=operation_id,
                    resource_scope=resource_scope,
                )
                return _block(authorization)
        return _proceed(authorization)

    if context.classification is EffectContinuationClassification.POST_HITL_NOT_RESUMED:
        return _block(authorization)

    # POST_HITL_RESUMED — matching human approval evidence required.
    if stored_grant is None or task is None:
        return _block(authorization)

    if not _grant_matches_post_hitl_allow_proposal(
        stored_grant,
        side_effect=side_effect,
        operation_id=operation_id,
        resource_scope=resource_scope,
        decision=authorization.decision,
    ):
        GovernedContinuationGrantCoordinator.clear_obsolete_grant_for_proposal(
            task,
            side_effect=side_effect,
            operation_id=operation_id,
            resource_scope=resource_scope,
        )
        return _block(authorization)

    pending = context.pending
    if pending is None or pending.continuation_id != stored_grant.continuation_request_id:
        return _block(authorization)

    consumed = GovernedContinuationGrantCoordinator.consume_matching_grant(
        task,
        expected_grant_id=stored_grant.grant_id,
    )
    if consumed is None:
        return _block(authorization)
    return _proceed(authorization)


__all__ = [
    "CanonicalContinuationAuthorityView",
    "EffectContinuationClassification",
    "EffectContinuationContext",
    "HumanGovernedProposalRelation",
    "MseHitlEffectGateDisposition",
    "MseHitlEffectGateOutcome",
    "classify_human_governed_proposal_relation",
    "evaluate_mse_hitl_effect_gate",
    "resolve_canonical_continuation_authority",
    "resolve_continuation_port_for_mse_hitl_gate",
    "resolve_effect_continuation_context",
]
