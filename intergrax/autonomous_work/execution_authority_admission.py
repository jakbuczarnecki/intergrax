# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Worker execution authority admission composition (AW-3B).

Orchestrates AW-3A identity binding with Collaborative Work effective authority
resolution. Produces immutable collaborative admission context for prospective
Execution intake without dispatching work (AW-5A owns dispatch).

Trust boundary:

    Collaborative Work — identity/membership/delegation/base authority resolution
    AW-3B — Worker → collaborative authority admission context
    Runtime/Governance — full policy evaluation + trusted execution authority
    AW-5A — Worker → Execution dispatch

Collaborative Work ALLOW is not global runtime ALLOW. AW-3B must not mint
``ParentExecutionAuthority``; later runtime policy may narrow scopes but never
widen beyond collaborative-approved requested scopes.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from intergrax.autonomous_work.principal_binding_resolver import WorkerPrincipalBindingResolver
from intergrax.contracts.autonomous_work.execution_authority import (
    WorkerExecutionAuthorityContext,
    WorkerExecutionAuthorityRequest,
)
from intergrax.contracts.autonomous_work.ids import WorkerInstanceId
from intergrax.contracts.autonomous_work.principal_binding import ResolvedWorkerPrincipal
from intergrax.contracts.collaborative_work import (
    AuthorityDelegation,
    DelegationStatus,
    EffectiveAuthorityDecision,
    EffectiveAuthorityRequest,
    MembershipResolutionMode,
)
from intergrax.contracts.runtime_policy import PolicyAction


def _utc_now() -> datetime:
    return datetime.now(UTC)


@runtime_checkable
class CollaborativeWorkAuthorityResolverPort(Protocol):
    """Stable authority resolver port — Collaborative Work owns semantics."""

    def resolve(self, request: EffectiveAuthorityRequest) -> EffectiveAuthorityDecision: ...


class WorkerExecutionAuthorityDenied(Exception):
    """Effective authority resolver denied Worker execution admission."""

    def __init__(
        self,
        *,
        worker_instance_id: WorkerInstanceId,
        decision: EffectiveAuthorityDecision,
    ) -> None:
        self.worker_instance_id = worker_instance_id
        self.decision = decision
        super().__init__(
            f"worker execution authority denied for {worker_instance_id}: "
            f"{decision.decision.reason}"
        )


class WorkerExecutionAdmissionService:
    """Prepare Worker execution authority context for canonical intake."""

    def __init__(
        self,
        *,
        binding_resolver: WorkerPrincipalBindingResolver,
        authority_resolver: CollaborativeWorkAuthorityResolverPort,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._binding_resolver = binding_resolver
        self._authority_resolver = authority_resolver
        self._clock = clock or _utc_now

    def prepare(
        self,
        request: WorkerExecutionAuthorityRequest,
    ) -> WorkerExecutionAuthorityContext:
        resolved_principal = self._binding_resolver.resolve(
            worker_instance_id=request.worker_instance_id,
        )
        effective_request = EffectiveAuthorityRequest(
            tenant_id=resolved_principal.tenant_id,
            workspace_id=resolved_principal.workspace_id,
            acting_principal_id=resolved_principal.principal_id,
            requested_authority_scopes=request.requested_authority_scopes,
            delegator_principal_id=request.delegator_principal_id,
            resource_scope=request.resource_scope,
            delegation=self._delegation_locator(
                request=request,
                resolved_principal=resolved_principal,
            ),
            membership_resolution_mode=MembershipResolutionMode.CANONICAL_PRINCIPAL,
        )
        decision = self._authority_resolver.resolve(effective_request)
        if decision.decision.action is not PolicyAction.ALLOW:
            raise WorkerExecutionAuthorityDenied(
                worker_instance_id=request.worker_instance_id,
                decision=decision,
            )
        return WorkerExecutionAuthorityContext(
            worker_instance_id=request.worker_instance_id,
            resolved_principal=resolved_principal,
            requested_authority_scopes=request.requested_authority_scopes,
            collaborative_authority_scopes=request.requested_authority_scopes,
            effective_authority_request=effective_request,
            effective_authority_decision=decision,
            evaluated_at=self._clock(),
        )

    @staticmethod
    def _delegation_locator(
        *,
        request: WorkerExecutionAuthorityRequest,
        resolved_principal: ResolvedWorkerPrincipal,
    ) -> AuthorityDelegation | None:
        if request.delegator_principal_id is None:
            return None
        assert request.delegation_id is not None
        return AuthorityDelegation(
            delegation_id=request.delegation_id,
            tenant_id=resolved_principal.tenant_id,
            workspace_id=resolved_principal.workspace_id,
            delegator_principal_id=request.delegator_principal_id,
            delegate_principal_id=resolved_principal.principal_id,
            authority_scopes=request.requested_authority_scopes,
            status=DelegationStatus.ACTIVE,
            revision=0,
        )
