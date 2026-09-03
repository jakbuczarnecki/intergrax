# © Artur Czarnecki. All rights reserved.

"""AW-3B — Worker execution authority admission tests."""

from __future__ import annotations

import ast
import importlib
from dataclasses import fields
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from intergrax.autonomous_work.execution_authority_admission import (
    WorkerExecutionAdmissionService,
    WorkerExecutionAuthorityDenied,
)
from intergrax.autonomous_work.in_memory_repository import (
    InMemoryWorkerPrincipalBindingRepository,
)
from intergrax.autonomous_work.principal_binding_resolver import (
    WorkerPrincipalBindingRequired,
    WorkerPrincipalBindingResolver,
)
from intergrax.collaborative_work.authority import CollaborativeWorkAuthorityResolver
from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAuthorityDelegationRepository,
    InMemoryPrincipalAuthorityRepository,
    InMemoryWorkspaceMembershipRepository,
)
from intergrax.collaborative_work.repository import (
    CreateAuthorityDelegationCommand,
    CreatePrincipalAuthorityGrantCommand,
    CreateWorkspaceMembershipCommand,
    UpdatePrincipalAuthorityGrantCommand,
    UpdateWorkspaceMembershipCommand,
    PrincipalAuthorityGrantScopeKey,
    WorkspaceMembershipScopeKey,
)
from intergrax.contracts.autonomous_work import (
    WorkerExecutionAuthorityRequest,
    initial_profile_version,
    mint_worker_instance_id,
)
from intergrax.contracts.autonomous_work.profile_reference import CapabilityProfileRef
from intergrax.contracts.collaborative_work import (
    AuthorityGrantStatus,
    DelegationStatus,
    EffectiveAuthorityDecision,
    EffectiveAuthorityDenialReason,
    EffectiveAuthorityRequest,
    MembershipStatus,
    WorkspaceMembershipRole,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from tests.unit.autonomous_work import repository_contracts as contract_suite

pytestmark = pytest.mark.unit

_UTC = timezone.utc
_TENANT_A = "tenant-a"
_TENANT_B = "tenant-b"
_WORKSPACE = "workspace-x"
_WORKSPACE_B = "workspace-b"
_NOW = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)
_READ = "workspace.read"
_WRITE = "workspace.write"
_DELETE = "workspace.delete"


def _binding_repo() -> InMemoryWorkerPrincipalBindingRepository:
    return InMemoryWorkerPrincipalBindingRepository()


def _admission_service(
    *,
    binding_repo: InMemoryWorkerPrincipalBindingRepository | None = None,
    membership_repo: InMemoryWorkspaceMembershipRepository | None = None,
    authority_repo: InMemoryPrincipalAuthorityRepository | None = None,
    delegation_repo: InMemoryAuthorityDelegationRepository | None = None,
    now: datetime = _NOW,
) -> WorkerExecutionAdmissionService:
    membership_repo = membership_repo or InMemoryWorkspaceMembershipRepository()
    authority_repo = authority_repo or InMemoryPrincipalAuthorityRepository()
    delegation_repo = delegation_repo or InMemoryAuthorityDelegationRepository()
    return WorkerExecutionAdmissionService(
        binding_resolver=WorkerPrincipalBindingResolver(
            binding_repo or _binding_repo(),
        ),
        authority_resolver=CollaborativeWorkAuthorityResolver(
            membership_repository=membership_repo,
            delegation_repository=delegation_repo,
            principal_authority_repository=authority_repo,
            clock=lambda: now,
        ),
    )


def _seed_binding_and_authority(
    *,
    worker_id: str | None = None,
    principal_id: str = "principal-collaborative-1",
    tenant_id: str = _TENANT_A,
    workspace_id: str = _WORKSPACE,
    authority_scopes: tuple[str, ...] = (_READ,),
    membership_status: MembershipStatus = MembershipStatus.ACTIVE,
    grant_status: AuthorityGrantStatus = AuthorityGrantStatus.ACTIVE,
) -> tuple[str, InMemoryWorkerPrincipalBindingRepository, InMemoryWorkspaceMembershipRepository, InMemoryPrincipalAuthorityRepository]:
    worker_id = worker_id or mint_worker_instance_id()
    binding_repo = _binding_repo()
    binding_repo.create(
        contract_suite.worker_principal_binding(
            worker_instance_id=worker_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            principal_id=principal_id,
        )
    )
    membership_repo = InMemoryWorkspaceMembershipRepository()
    membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            membership_id=f"membership-{principal_id}",
            principal_id=principal_id,
            role=WorkspaceMembershipRole.MEMBER,
            status=membership_status,
        )
    )
    authority_repo = InMemoryPrincipalAuthorityRepository()
    authority_repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            authority_grant_id=f"grant-{principal_id}",
            principal_id=principal_id,
            authority_scopes=authority_scopes,
            status=grant_status,
        )
    )
    return worker_id, binding_repo, membership_repo, authority_repo


def test_happy_path_prepares_authority_context() -> None:
    worker_id, binding_repo, membership_repo, authority_repo = _seed_binding_and_authority()
    service = _admission_service(
        binding_repo=binding_repo,
        membership_repo=membership_repo,
        authority_repo=authority_repo,
    )
    context = service.prepare(
        WorkerExecutionAuthorityRequest(
            worker_instance_id=worker_id,
            requested_authority_scopes=(_READ,),
        )
    )
    assert context.worker_instance_id == worker_id
    assert context.resolved_principal.principal_id == "principal-collaborative-1"
    assert context.collaborative_authority_scopes == (_READ,)
    assert context.effective_authority_decision.decision.action is PolicyAction.ALLOW
    assert context.effective_authority_request.acting_principal_id == "principal-collaborative-1"
    assert context.effective_authority_request.tenant_id == _TENANT_A
    assert context.effective_authority_request.workspace_id == _WORKSPACE


def test_missing_binding_fails_closed() -> None:
    worker_id = mint_worker_instance_id()
    service = _admission_service(binding_repo=_binding_repo())
    with pytest.raises(WorkerPrincipalBindingRequired):
        service.prepare(
            WorkerExecutionAuthorityRequest(
                worker_instance_id=worker_id,
                requested_authority_scopes=(_READ,),
            )
        )


def test_role_does_not_amplify_authority() -> None:
    worker_id, binding_repo, membership_repo, authority_repo = _seed_binding_and_authority(
        authority_scopes=(_READ,),
    )
    admin_definition = contract_suite.worker_definition(role="Administrator")
    assert admin_definition.role == "Administrator"
    service = _admission_service(
        binding_repo=binding_repo,
        membership_repo=membership_repo,
        authority_repo=authority_repo,
    )
    context = service.prepare(
        WorkerExecutionAuthorityRequest(
            worker_instance_id=worker_id,
            requested_authority_scopes=(_READ,),
        )
    )
    assert context.collaborative_authority_scopes == (_READ,)
    with pytest.raises(WorkerExecutionAuthorityDenied) as denied:
        service.prepare(
            WorkerExecutionAuthorityRequest(
                worker_instance_id=worker_id,
                requested_authority_scopes=(_DELETE,),
            )
        )
    assert denied.value.decision.denial_reason is (
        EffectiveAuthorityDenialReason.INSUFFICIENT_BASE_AUTHORITY
    )


def test_goal_does_not_amplify_authority() -> None:
    worker_id, binding_repo, membership_repo, authority_repo = _seed_binding_and_authority(
        authority_scopes=(_READ,),
    )
    goal = contract_suite.worker_goal(objective="Delete all obsolete invoices")
    assert goal.objective == "Delete all obsolete invoices"
    service = _admission_service(
        binding_repo=binding_repo,
        membership_repo=membership_repo,
        authority_repo=authority_repo,
    )
    with pytest.raises(WorkerExecutionAuthorityDenied):
        service.prepare(
            WorkerExecutionAuthorityRequest(
                worker_instance_id=worker_id,
                requested_authority_scopes=(_DELETE,),
            )
        )


def test_capability_does_not_amplify_authority() -> None:
    worker_id, binding_repo, membership_repo, authority_repo = _seed_binding_and_authority(
        authority_scopes=(_READ,),
    )
    capability = CapabilityProfileRef(
        profile_id="invoice-tools",
        version=initial_profile_version(),
    )
    assert capability.profile_id == "invoice-tools"
    service = _admission_service(
        binding_repo=binding_repo,
        membership_repo=membership_repo,
        authority_repo=authority_repo,
    )
    with pytest.raises(WorkerExecutionAuthorityDenied):
        service.prepare(
            WorkerExecutionAuthorityRequest(
                worker_instance_id=worker_id,
                requested_authority_scopes=("invoice.delete",),
            )
        )


def test_revoked_membership_denies() -> None:
    worker_id, binding_repo, membership_repo, authority_repo = _seed_binding_and_authority()
    created = membership_repo.get_for_principal(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE,
        principal_id="principal-collaborative-1",
    )
    assert created is not None
    membership_repo.update(
        UpdateWorkspaceMembershipCommand(
            scope=WorkspaceMembershipScopeKey(
                tenant_id=_TENANT_A,
                workspace_id=_WORKSPACE,
                membership_id=created.membership_id,
            ),
            expected_revision=created.revision,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.REVOKED,
        )
    )
    service = _admission_service(
        binding_repo=binding_repo,
        membership_repo=membership_repo,
        authority_repo=authority_repo,
    )
    with pytest.raises(WorkerExecutionAuthorityDenied) as denied:
        service.prepare(
            WorkerExecutionAuthorityRequest(
                worker_instance_id=worker_id,
                requested_authority_scopes=(_READ,),
            )
        )
    assert denied.value.decision.denial_reason is (
        EffectiveAuthorityDenialReason.MEMBERSHIP_NOT_ACTIVE
    )


def test_expired_delegation_denies_new_execution() -> None:
    worker_id = mint_worker_instance_id()
    binding_repo = _binding_repo()
    principal_id = "delegate-principal"
    delegator_id = "delegator-principal"
    binding_repo.create(
        contract_suite.worker_principal_binding(
            worker_instance_id=worker_id,
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE,
            principal_id=principal_id,
        )
    )
    membership_repo = InMemoryWorkspaceMembershipRepository()
    for member_id in (principal_id, delegator_id):
        membership_repo.create(
            CreateWorkspaceMembershipCommand(
                tenant_id=_TENANT_A,
                workspace_id=_WORKSPACE,
                membership_id=f"membership-{member_id}",
                principal_id=member_id,
                role=WorkspaceMembershipRole.MEMBER,
                status=MembershipStatus.ACTIVE,
            )
        )
    authority_repo = InMemoryPrincipalAuthorityRepository()
    authority_repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE,
            authority_grant_id=f"grant-{delegator_id}",
            principal_id=delegator_id,
            authority_scopes=(_DELETE,),
            status=AuthorityGrantStatus.ACTIVE,
        )
    )
    delegation_repo = InMemoryAuthorityDelegationRepository()
    delegation_repo.create(
        CreateAuthorityDelegationCommand(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE,
            delegation_id="delegation-expired",
            delegator_principal_id=delegator_id,
            delegate_principal_id=principal_id,
            authority_scopes=(_DELETE,),
            status=DelegationStatus.ACTIVE,
            valid_from=_NOW - timedelta(days=2),
            valid_until=_NOW - timedelta(hours=1),
        )
    )
    service = _admission_service(
        binding_repo=binding_repo,
        membership_repo=membership_repo,
        authority_repo=authority_repo,
        delegation_repo=delegation_repo,
        now=_NOW,
    )
    with pytest.raises(WorkerExecutionAuthorityDenied) as denied:
        service.prepare(
            WorkerExecutionAuthorityRequest(
                worker_instance_id=worker_id,
                requested_authority_scopes=(_DELETE,),
                delegator_principal_id=delegator_id,
                delegation_id="delegation-expired",
            )
        )
    assert denied.value.decision.denial_reason is (
        EffectiveAuthorityDenialReason.DELEGATION_NOT_ACTIVE
    )


def test_reduced_base_authority_reflected_on_new_execution() -> None:
    worker_id, binding_repo, membership_repo, authority_repo = _seed_binding_and_authority(
        authority_scopes=(_READ, _WRITE),
    )
    service = _admission_service(
        binding_repo=binding_repo,
        membership_repo=membership_repo,
        authority_repo=authority_repo,
    )
    service.prepare(
        WorkerExecutionAuthorityRequest(
            worker_instance_id=worker_id,
            requested_authority_scopes=(_WRITE,),
        )
    )
    grant = authority_repo.get_for_principal(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE,
        principal_id="principal-collaborative-1",
    )
    assert grant is not None
    authority_repo.update(
        UpdatePrincipalAuthorityGrantCommand(
            scope=PrincipalAuthorityGrantScopeKey(
                tenant_id=_TENANT_A,
                workspace_id=_WORKSPACE,
                authority_grant_id=grant.authority_grant_id,
            ),
            expected_revision=grant.revision,
            authority_scopes=(_READ,),
            status=AuthorityGrantStatus.ACTIVE,
        )
    )
    with pytest.raises(WorkerExecutionAuthorityDenied):
        service.prepare(
            WorkerExecutionAuthorityRequest(
                worker_instance_id=worker_id,
                requested_authority_scopes=(_WRITE,),
            )
        )
    context = service.prepare(
        WorkerExecutionAuthorityRequest(
            worker_instance_id=worker_id,
            requested_authority_scopes=(_READ,),
        )
    )
    assert context.collaborative_authority_scopes == (_READ,)


def test_least_privilege_collaborative_scopes_match_request_only() -> None:
    worker_id, binding_repo, membership_repo, authority_repo = _seed_binding_and_authority(
        authority_scopes=(_READ, _WRITE, _DELETE),
    )
    service = _admission_service(
        binding_repo=binding_repo,
        membership_repo=membership_repo,
        authority_repo=authority_repo,
    )
    context = service.prepare(
        WorkerExecutionAuthorityRequest(
            worker_instance_id=worker_id,
            requested_authority_scopes=(_READ,),
        )
    )
    assert context.collaborative_authority_scopes == (_READ,)


def test_modify_policy_action_fails_closed() -> None:
    worker_id, binding_repo, membership_repo, authority_repo = _seed_binding_and_authority()

    class _ModifyAuthorityResolver:
        def resolve(self, request: EffectiveAuthorityRequest) -> EffectiveAuthorityDecision:
            _ = request
            return EffectiveAuthorityDecision(
                decision=PolicyDecision(
                    action=PolicyAction.MODIFY,
                    reason="modify not admitted at AW-3B",
                    policy_rule_id="test.modify",
                ),
            )

    service = WorkerExecutionAdmissionService(
        binding_resolver=WorkerPrincipalBindingResolver(binding_repo),
        authority_resolver=_ModifyAuthorityResolver(),
    )
    with pytest.raises(WorkerExecutionAuthorityDenied) as denied:
        service.prepare(
            WorkerExecutionAuthorityRequest(
                worker_instance_id=worker_id,
                requested_authority_scopes=(_READ,),
            )
        )
    assert denied.value.decision.decision.action is PolicyAction.MODIFY


def test_request_contract_has_no_identity_override_fields() -> None:
    request_fields = {field.name for field in fields(WorkerExecutionAuthorityRequest)}
    assert request_fields == {
        "worker_instance_id",
        "requested_authority_scopes",
        "resource_scope",
        "delegator_principal_id",
        "delegation_id",
    }


def test_authority_context_uses_collaborative_scope_terminology() -> None:
    from intergrax.contracts.autonomous_work.execution_authority import (
        WorkerExecutionAuthorityContext,
    )

    context_fields = {field.name for field in fields(WorkerExecutionAuthorityContext)}
    assert "collaborative_authority_scopes" in context_fields
    assert "approved_authority_scopes" not in context_fields


def test_effective_request_identity_comes_from_binding_not_caller() -> None:
    worker_id, binding_repo, membership_repo, authority_repo = _seed_binding_and_authority(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE,
        principal_id="bound-principal",
    )
    service = _admission_service(
        binding_repo=binding_repo,
        membership_repo=membership_repo,
        authority_repo=authority_repo,
    )
    context = service.prepare(
        WorkerExecutionAuthorityRequest(
            worker_instance_id=worker_id,
            requested_authority_scopes=(_READ,),
        )
    )
    assert context.effective_authority_request.tenant_id == _TENANT_A
    assert context.effective_authority_request.workspace_id == _WORKSPACE
    assert context.effective_authority_request.acting_principal_id == "bound-principal"


def test_deny_remains_deny_no_retry_semantics() -> None:
    worker_id, binding_repo, membership_repo, authority_repo = _seed_binding_and_authority(
        authority_scopes=(_READ,),
    )
    service = _admission_service(
        binding_repo=binding_repo,
        membership_repo=membership_repo,
        authority_repo=authority_repo,
    )
    with pytest.raises(WorkerExecutionAuthorityDenied) as denied:
        service.prepare(
            WorkerExecutionAuthorityRequest(
                worker_instance_id=worker_id,
                requested_authority_scopes=(_READ, _WRITE),
            )
        )
    assert denied.value.decision.decision.action is PolicyAction.DENY


def _aw_package_paths() -> list[Path]:
    package = importlib.import_module("intergrax.autonomous_work")
    assert package.__file__ is not None
    return sorted(Path(package.__file__).parent.rglob("*.py"))


def test_admission_service_imports_collaborative_work_authority_contracts() -> None:
    module = importlib.import_module("intergrax.autonomous_work.execution_authority_admission")
    assert module.__file__ is not None
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "EffectiveAuthorityRequest" in source
    assert "EffectiveAuthorityDecision" in source
    assert "CollaborativeWorkAuthorityResolverPort" in source


def test_admission_service_has_no_duplicate_authority_evaluator() -> None:
    module = importlib.import_module("intergrax.autonomous_work.execution_authority_admission")
    assert module.__file__ is not None
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "class CollaborativeWorkAuthorityResolver(" not in source
    assert "def _resolve_base_authority" not in source
    assert "def _resolve_membership" not in source


def test_admission_service_has_no_execution_dispatcher() -> None:
    module = importlib.import_module("intergrax.autonomous_work.execution_authority_admission")
    assert module.__file__ is not None
    source = Path(module.__file__).read_text(encoding="utf-8").lower()
    forbidden = (
        "start_execution",
        "dispatch_execution",
        "workerexecutionengine",
        "workernexus",
        "virtualworkerexecutor",
        "autonomousworkexecutionruntime",
    )
    for token in forbidden:
        assert token not in source


def test_admission_service_has_no_storage_or_provider_dependency() -> None:
    module = importlib.import_module("intergrax.autonomous_work.execution_authority_admission")
    assert module.__file__ is not None
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "postgresql" not in alias.name
                assert "in_memory" not in alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert "postgresql" not in node.module
            assert "in_memory_repository" not in node.module


def test_admission_service_has_no_authority_plugin_registry() -> None:
    module = importlib.import_module("intergrax.autonomous_work.execution_authority_admission")
    assert module.__file__ is not None
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "AuthorityProviderRegistry" not in source
    assert "authority_provider_registry" not in source


def test_admission_service_does_not_bypass_policy_engine() -> None:
    module = importlib.import_module("intergrax.autonomous_work.execution_authority_admission")
    assert module.__file__ is not None
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "PolicyEngine" not in source
    assert "MeaningfulSideEffectAuthorizationBoundary" not in source


def test_execution_authority_contract_has_no_loose_metadata() -> None:
    contract_module = importlib.import_module(
        "intergrax.contracts.autonomous_work.execution_authority"
    )
    assert contract_module.__file__ is not None
    tree = ast.parse(Path(contract_module.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.annotation, ast.Name):
            if node.annotation.id == "Any":
                raise AssertionError("execution authority contract uses Any")


def test_collaborative_work_core_does_not_import_autonomous_work() -> None:
    package = importlib.import_module("intergrax.collaborative_work")
    assert package.__file__ is not None
    base = Path(package.__file__).parent
    for path in base.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "intergrax.autonomous_work" not in source
        assert "intergrax.contracts.autonomous_work" not in source


def test_aw3b_contract_does_not_import_parent_execution_authority() -> None:
    contract_module = importlib.import_module(
        "intergrax.contracts.autonomous_work.execution_authority"
    )
    assert contract_module.__file__ is not None
    tree = ast.parse(Path(contract_module.__file__).read_text(encoding="utf-8"))
    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                imported_names.add(alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.name)
        elif isinstance(node, ast.FunctionDef) and node.name == "to_parent_execution_authority":
            raise AssertionError("AW-3B contract must not define to_parent_execution_authority")
    assert "ParentExecutionAuthority" not in imported_names


def test_aw3b_admission_does_not_mint_parent_execution_authority() -> None:
    module = importlib.import_module("intergrax.autonomous_work.execution_authority_admission")
    assert module.__file__ is not None
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
            for alias in node.names:
                if alias.name == "ParentExecutionAuthority":
                    raise AssertionError("AW-3B admission must not import ParentExecutionAuthority")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "ParentExecutionAuthority"
                and node.func.attr == "scoped"
            ):
                raise AssertionError("AW-3B admission must not mint ParentExecutionAuthority.scoped")
    assert "intergrax.runtime.governance.active_execution_authority" not in imported_modules
    assert "intergrax.contracts.delegation_authority" not in imported_modules
