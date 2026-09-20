# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical MSE HITL effect gate after fresh Governance authorize (GR-10-R11-R1).

Human judgment evidence and continuation grants never become Governance ALLOW.
``GovernedContinuationApprovalGrant`` is correlation / single-use evidence only.
Canonical pause / wait / resume authority remains ``ExecutionContinuationPort``.

Post-HITL physical effect may proceed only when:

* fresh Governance returns ``ALLOW``, and
* canonical continuation exists in ``RESUMED`` (GR-5: only ``RESUMED`` clears the
  progress gate; ``RESUME_AUTHORIZED`` still blocks execution progress), and
* a scoped approval grant matches the current side-effect proposal (evidence), and
* continuation identity / ``continuation_request_id`` correlate with that grant.

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
) -> CanonicalContinuationAuthorityView | None:
    identity = _identity_from_side_effect(side_effect)
    if grant is not None:
        pending = _load_pending(
            port,
            continuation_id=grant.continuation_request_id,
            identity=identity,
        )
        if pending is None or pending.continuation_id != grant.continuation_request_id:
            return None
    else:
        pending = _load_pending(port, identity=identity)
        if pending is None:
            return None

    if pending.identity.task_id != identity.task_id:
        return None
    if pending.identity.run_id != identity.run_id:
        return None
    if pending.identity.attempt_id != identity.attempt_id:
        return None
    if pending.identity.execution_id != identity.execution_id:
        return None
    if pending.lifecycle_state not in _POST_HITL_EFFECT_LIFECYCLE:
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
    if stored_grant is None:
        if continuation_port is not None and side_effect is not None:
            pending = _load_pending(
                continuation_port,
                identity=_identity_from_side_effect(side_effect),
            )
            if (
                pending is not None
                and pending.lifecycle_state
                is not ExecutionContinuationLifecycleState.RESUMED
            ):
                return MseHitlEffectGateOutcome(
                    disposition=MseHitlEffectGateDisposition.BLOCK,
                    authorization=authorization,
                )
        if task is not None and side_effect is not None:
            GovernedContinuationGrantCoordinator.clear_obsolete_grant_for_proposal(
                task,
                side_effect=side_effect,
                operation_id=operation_id,
                resource_scope=resource_scope,
            )
        return MseHitlEffectGateOutcome(
            disposition=MseHitlEffectGateDisposition.PROCEED,
            authorization=authorization,
        )

    if side_effect is None or continuation_port is None or task is None:
        return MseHitlEffectGateOutcome(
            disposition=MseHitlEffectGateDisposition.BLOCK,
            authorization=authorization,
        )

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
        return MseHitlEffectGateOutcome(
            disposition=MseHitlEffectGateDisposition.BLOCK,
            authorization=authorization,
        )

    authority = resolve_canonical_continuation_authority(
        continuation_port,
        side_effect=side_effect,
        grant=stored_grant,
    )
    if authority is None:
        return MseHitlEffectGateOutcome(
            disposition=MseHitlEffectGateDisposition.BLOCK,
            authorization=authorization,
        )

    consumed = GovernedContinuationGrantCoordinator.consume_matching_grant(
        task,
        expected_grant_id=stored_grant.grant_id,
    )
    if consumed is None:
        return MseHitlEffectGateOutcome(
            disposition=MseHitlEffectGateDisposition.BLOCK,
            authorization=authorization,
        )
    return MseHitlEffectGateOutcome(
        disposition=MseHitlEffectGateDisposition.PROCEED,
        authorization=authorization,
    )


__all__ = [
    "CanonicalContinuationAuthorityView",
    "MseHitlEffectGateDisposition",
    "MseHitlEffectGateOutcome",
    "evaluate_mse_hitl_effect_gate",
    "resolve_canonical_continuation_authority",
    "resolve_continuation_port_for_mse_hitl_gate",
]
