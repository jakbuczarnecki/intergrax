# © Artur Czarnecki. All rights reserved.

"""AW-3B fixtures for UCA-6C qualified capability resume governance tests."""

from __future__ import annotations

from datetime import UTC, datetime

from intergrax.autonomous_work.execution_authority_admission import (
    WorkerExecutionAdmissionService,
)
from intergrax.contracts.admitted_root_governance_identity import (
    AdmittedRootGovernanceIdentity,
)
from intergrax.contracts.autonomous_work.execution_authority import (
    WorkerExecutionAuthorityRequest,
)
from intergrax.contracts.collaborative_work import EffectiveAuthorityDecision
from intergrax.autonomous_work.in_memory_repository import (
    InMemoryWorkerPrincipalBindingRepository,
)
from intergrax.autonomous_work.principal_binding_resolver import (
    WorkerPrincipalBindingResolver,
)
from intergrax.collaborative_work.authority import CollaborativeWorkAuthorityResolver
from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAuthorityDelegationRepository,
    InMemoryPrincipalAuthorityRepository,
    InMemoryWorkspaceMembershipRepository,
)
from intergrax.collaborative_work.repository import (
    CreatePrincipalAuthorityGrantCommand,
    CreateWorkspaceMembershipCommand,
)
from intergrax.contracts.autonomous_work.ids import WorkerInstanceId
from intergrax.contracts.collaborative_work import (
    AuthorityGrantStatus,
    MembershipStatus,
    WorkspaceMembershipRole,
)
from tests.unit.autonomous_work import repository_contracts as contract_suite

_UTC = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
_READ = "workspace.read"
_WRITE = "workspace.write"


def build_worker_execution_admission_for_uca6c(
    *,
    worker_instance_id: WorkerInstanceId,
    tenant_id: str,
    workspace_id: str,
    principal_id: str,
    granted_scopes: tuple[str, ...] = (_READ, _WRITE),
) -> WorkerExecutionAdmissionService:
    """Legal AW-3B admission wired for one worker principal binding."""
    binding_repo = InMemoryWorkerPrincipalBindingRepository()
    binding_repo.create(
        contract_suite.worker_principal_binding(
            worker_instance_id=worker_instance_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            principal_id=principal_id,
        ),
    )
    membership_repo = InMemoryWorkspaceMembershipRepository()
    membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            membership_id=f"membership-{principal_id}",
            principal_id=principal_id,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        ),
    )
    authority_repo = InMemoryPrincipalAuthorityRepository()
    authority_repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            authority_grant_id=f"grant-{principal_id}",
            principal_id=principal_id,
            authority_scopes=granted_scopes,
            status=AuthorityGrantStatus.ACTIVE,
        ),
    )
    delegation_repo = InMemoryAuthorityDelegationRepository()
    return WorkerExecutionAdmissionService(
        binding_resolver=WorkerPrincipalBindingResolver(binding_repo),
        authority_resolver=CollaborativeWorkAuthorityResolver(
            membership_repository=membership_repo,
            delegation_repository=delegation_repo,
            principal_authority_repository=authority_repo,
            clock=lambda: _UTC,
        ),
        clock=lambda: _UTC,
    )


def trusted_governance_from_admission(
    *,
    admission: WorkerExecutionAdmissionService,
    worker_instance_id: WorkerInstanceId,
    requested_scopes: tuple[str, ...],
) -> tuple[
    AdmittedRootGovernanceIdentity,
    EffectiveAuthorityDecision,
    tuple[str, ...],
]:
    """Legal AW-3B snapshot for qualified capability execution handoff tests."""
    context = admission.prepare(
        WorkerExecutionAuthorityRequest(
            worker_instance_id=worker_instance_id,
            requested_authority_scopes=requested_scopes,
        ),
    )
    principal = context.resolved_principal
    identity = AdmittedRootGovernanceIdentity(
        tenant_id=principal.tenant_id,
        workspace_id=principal.workspace_id,
        principal_id=principal.principal_id,
    )
    return (
        identity,
        context.effective_authority_decision,
        context.collaborative_authority_scopes,
    )


__all__ = [
    "build_worker_execution_admission_for_uca6c",
    "trusted_governance_from_admission",
    "_READ",
    "_WRITE",
]
