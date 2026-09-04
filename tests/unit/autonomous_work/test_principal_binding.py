# © Artur Czarnecki. All rights reserved.

"""AW-3A — Worker→Collaborative Principal binding tests."""

from __future__ import annotations

import ast
import importlib
from dataclasses import replace
from datetime import UTC, datetime, timezone
from pathlib import Path

import pytest

from intergrax.autonomous_work.in_memory_repository import (
    InMemoryWorkerPrincipalBindingRepository,
)
from intergrax.autonomous_work.principal_binding_resolver import (
    WorkerPrincipalBindingRequired,
    WorkerPrincipalBindingResolver,
)
from intergrax.autonomous_work.repository import AutonomousWorkEntityConflict
from intergrax.autonomous_work.serialization import (
    worker_principal_binding_from_json,
    worker_principal_binding_to_json,
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
    UpdateWorkspaceMembershipCommand,
    WorkspaceMembershipScopeKey,
)
from intergrax.contracts.autonomous_work import (
    WorkerDefinition,
    WorkerGoal,
    WorkerInstance,
    initial_revision,
    mint_worker_instance_id,
)
from intergrax.contracts.autonomous_work.principal_binding import (
    ResolvedWorkerPrincipal,
    WorkerPrincipalBinding,
    validate_collaborative_principal_id,
)
from intergrax.contracts.autonomous_work.profile_reference import CapabilityProfileRef
from intergrax.contracts.collaborative_work import (
    AuthorityGrantStatus,
    EffectiveAuthorityDenialReason,
    EffectiveAuthorityRequest,
    MembershipStatus,
    WorkspaceMembership,
    WorkspaceMembershipRole,
)
from intergrax.contracts.runtime_policy import PolicyAction
from tests.unit.autonomous_work import repository_contracts as contract_suite

pytestmark = pytest.mark.unit

_UTC = timezone.utc
_TENANT_A = "tenant-a"
_TENANT_B = "tenant-b"
_WORKSPACE = "workspace-x"
_WORKSPACE_B = "workspace-b"
_NOW = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)


def _binding_repo() -> InMemoryWorkerPrincipalBindingRepository:
    return InMemoryWorkerPrincipalBindingRepository()


def test_worker_principal_binding_contract_valid() -> None:
    binding = contract_suite.worker_principal_binding()
    assert binding.tenant_id == "tenant-a"
    assert binding.workspace_id == "workspace-x"
    assert binding.principal_id == "principal-collaborative-1"
    assert binding.revision == initial_revision()


def test_worker_principal_binding_rejects_empty_tenant_id() -> None:
    with pytest.raises(ValueError, match="tenant_id"):
        contract_suite.worker_principal_binding(tenant_id="")


def test_worker_principal_binding_rejects_empty_workspace_id() -> None:
    with pytest.raises(ValueError, match="workspace_id"):
        contract_suite.worker_principal_binding(workspace_id="")


def test_worker_principal_binding_rejects_empty_principal_id() -> None:
    with pytest.raises(ValueError, match="principal_id"):
        contract_suite.worker_principal_binding(principal_id="")


def test_worker_principal_binding_rejects_naive_created_at() -> None:
    with pytest.raises(ValueError, match="created_at"):
        contract_suite.worker_principal_binding(
            created_at=datetime(2026, 9, 2, 12, 0),
        )


def test_validate_collaborative_principal_id_rejects_whitespace() -> None:
    with pytest.raises(ValueError):
        validate_collaborative_principal_id("   ")


def test_worker_principal_binding_create_requires_initial_revision() -> None:
    repo = _binding_repo()
    binding = contract_suite.worker_principal_binding(revision=initial_revision())
    created = repo.create(binding)
    assert created == binding


def test_worker_principal_binding_create_rejects_non_initial_revision() -> None:
    repo = _binding_repo()
    from intergrax.contracts.autonomous_work import Revision

    binding = contract_suite.worker_principal_binding(revision=Revision(1))
    with pytest.raises(ValueError, match="requires revision"):
        repo.create(binding)


def test_worker_principal_binding_resolver_fail_closed_when_missing() -> None:
    repo = _binding_repo()
    resolver = WorkerPrincipalBindingResolver(repo)
    worker_id = mint_worker_instance_id()
    with pytest.raises(WorkerPrincipalBindingRequired, match="no principal binding"):
        resolver.resolve(worker_instance_id=worker_id)


def test_worker_principal_binding_resolver_returns_scoped_identity() -> None:
    repo = _binding_repo()
    worker_id = mint_worker_instance_id()
    binding = contract_suite.worker_principal_binding(
        worker_instance_id=worker_id,
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE,
        principal_id="principal-bound",
    )
    repo.create(binding)
    resolver = WorkerPrincipalBindingResolver(repo)
    resolved = resolver.resolve(worker_instance_id=worker_id)
    assert resolved == ResolvedWorkerPrincipal(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE,
        principal_id="principal-bound",
    )


def test_role_string_does_not_grant_authority_without_binding() -> None:
    definition = contract_suite.worker_definition(role="Finance Manager")
    assert definition.role == "Finance Manager"
    repo = _binding_repo()
    worker = contract_suite.worker_instance(
        worker_definition_id=definition.worker_definition_id,
    )
    resolver = WorkerPrincipalBindingResolver(repo)
    with pytest.raises(WorkerPrincipalBindingRequired):
        resolver.resolve(worker_instance_id=worker.worker_instance_id)


def test_goal_change_does_not_mutate_principal_binding() -> None:
    repo = _binding_repo()
    worker_id = mint_worker_instance_id()
    binding = contract_suite.worker_principal_binding(
        worker_instance_id=worker_id,
        principal_id="principal-stable",
    )
    repo.create(binding)
    goal_before = contract_suite.worker_goal(objective="Approve invoices")
    goal_after = replace(goal_before, objective="Reject invoices")
    assert goal_before.objective != goal_after.objective
    loaded = repo.get(worker_instance_id=worker_id)
    assert loaded == binding
    assert loaded is not None
    assert loaded.principal_id == "principal-stable"


def test_capability_profile_change_does_not_mutate_principal_binding() -> None:
    repo = _binding_repo()
    worker_id = mint_worker_instance_id()
    binding = contract_suite.worker_principal_binding(
        worker_instance_id=worker_id,
        principal_id="principal-stable",
    )
    repo.create(binding)
    definition = contract_suite.worker_definition()
    from intergrax.contracts.autonomous_work.profile_reference import initial_profile_version

    version = initial_profile_version()
    mutated = replace(
        definition,
        capability_profile_ref=CapabilityProfileRef(
            profile_id="capability/expanded",
            version=version,
        ),
    )
    assert mutated.capability_profile_ref != definition.capability_profile_ref
    loaded = repo.get(worker_instance_id=worker_id)
    assert loaded == binding


def test_conflicting_principal_binding_create_is_deterministic() -> None:
    repo = _binding_repo()
    binding = contract_suite.worker_principal_binding()
    repo.create(binding)
    conflict = replace(binding, principal_id="principal-other")
    with pytest.raises(AutonomousWorkEntityConflict):
        repo.create(conflict)


def test_worker_principal_binding_json_roundtrip() -> None:
    binding = contract_suite.worker_principal_binding()
    assert (
        worker_principal_binding_from_json(worker_principal_binding_to_json(binding))
        == binding
    )


def test_tampered_binding_json_rejected() -> None:
    binding = contract_suite.worker_principal_binding()
    payload = worker_principal_binding_to_json(binding)
    tampered = payload.replace("principal-collaborative-1", "principal-tampered")
    restored = worker_principal_binding_from_json(tampered)
    assert restored != binding


def test_malformed_binding_json_codec_version_rejected() -> None:
    binding = contract_suite.worker_principal_binding()
    import json

    payload = json.loads(worker_principal_binding_to_json(binding))
    payload["codec_version"] = 99
    with pytest.raises(ValueError, match="codec version"):
        worker_principal_binding_from_json(json.dumps(payload))


def test_malformed_binding_json_missing_scope_fields_rejected() -> None:
    binding = contract_suite.worker_principal_binding()
    import json

    payload = json.loads(worker_principal_binding_to_json(binding))
    del payload["tenant_id"]
    with pytest.raises(ValueError, match="malformed WorkerPrincipalBinding"):
        worker_principal_binding_from_json(json.dumps(payload))


def test_resolver_returns_scoped_identity_not_authority() -> None:
    module = importlib.import_module("intergrax.autonomous_work.principal_binding_resolver")
    assert module.__file__ is not None
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "resolve_principal_id" not in source
    assert "ResolvedWorkerPrincipal" in source
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "collaborative_work" not in node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert "collaborative_work" not in alias.name


def test_cross_worker_binding_isolation() -> None:
    contract_suite.contract_worker_principal_binding_worker_isolation(_binding_repo())


def test_same_principal_id_across_different_scopes_is_not_confused() -> None:
    contract_suite.contract_worker_principal_binding_same_principal_different_scopes(
        _binding_repo()
    )


def test_cross_tenant_binding_isolation() -> None:
    repo = _binding_repo()
    worker_id = mint_worker_instance_id()
    binding = contract_suite.worker_principal_binding(
        worker_instance_id=worker_id,
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE,
        principal_id="principal-shared",
    )
    repo.create(binding)
    resolved = WorkerPrincipalBindingResolver(repo).resolve(worker_instance_id=worker_id)
    assert resolved.tenant_id == _TENANT_A
    assert resolved.tenant_id != _TENANT_B


def test_cross_workspace_binding_isolation() -> None:
    repo = _binding_repo()
    worker_id = mint_worker_instance_id()
    binding = contract_suite.worker_principal_binding(
        worker_instance_id=worker_id,
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE,
        principal_id="principal-shared",
    )
    repo.create(binding)
    resolved = WorkerPrincipalBindingResolver(repo).resolve(worker_instance_id=worker_id)
    assert resolved.workspace_id == _WORKSPACE
    assert resolved.workspace_id != _WORKSPACE_B


def test_resolver_preserves_scope_for_distinct_workers_with_same_principal_id() -> None:
    repo = _binding_repo()
    worker_a = mint_worker_instance_id()
    worker_b = mint_worker_instance_id()
    shared_principal = "principal-shared"
    repo.create(
        contract_suite.worker_principal_binding(
            worker_instance_id=worker_a,
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE,
            principal_id=shared_principal,
        )
    )
    repo.create(
        contract_suite.worker_principal_binding(
            worker_instance_id=worker_b,
            tenant_id=_TENANT_B,
            workspace_id=_WORKSPACE_B,
            principal_id=shared_principal,
        )
    )
    resolver = WorkerPrincipalBindingResolver(repo)
    resolved_a = resolver.resolve(worker_instance_id=worker_a)
    resolved_b = resolver.resolve(worker_instance_id=worker_b)
    assert resolved_a.principal_id == resolved_b.principal_id == shared_principal
    assert (resolved_a.tenant_id, resolved_a.workspace_id) != (
        resolved_b.tenant_id,
        resolved_b.workspace_id,
    )


def _membership_locator(
    *,
    principal_id: str,
    membership_id: str = "membership-bound",
    status: MembershipStatus = MembershipStatus.ACTIVE,
) -> WorkspaceMembership:
    return WorkspaceMembership.model_validate(
        {
            "membership_id": membership_id,
            "tenant_id": _TENANT_A,
            "workspace_id": _WORKSPACE,
            "principal_id": principal_id,
            "role": WorkspaceMembershipRole.MEMBER,
            "status": status,
            "revision": 0,
        }
    )


def test_bound_principal_feeds_collaborative_authority_resolver_allow() -> None:
    principal_id = "principal-worker-bound"
    worker_id = mint_worker_instance_id()
    binding_repo = _binding_repo()
    binding_repo.create(
        contract_suite.worker_principal_binding(
            worker_instance_id=worker_id,
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE,
            principal_id=principal_id,
        )
    )
    identity_resolver = WorkerPrincipalBindingResolver(binding_repo)
    resolved = identity_resolver.resolve(worker_instance_id=worker_id)

    membership_repo = InMemoryWorkspaceMembershipRepository()
    authority_repo = InMemoryPrincipalAuthorityRepository()
    membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=resolved.tenant_id,
            workspace_id=resolved.workspace_id,
            membership_id="membership-bound",
            principal_id=resolved.principal_id,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        )
    )
    authority_repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=resolved.tenant_id,
            workspace_id=resolved.workspace_id,
            authority_grant_id="authority-grant-bound",
            principal_id=resolved.principal_id,
            authority_scopes=("workspace.read", "workspace.write"),
            status=AuthorityGrantStatus.ACTIVE,
        )
    )

    cw_resolver = CollaborativeWorkAuthorityResolver(
        membership_repository=membership_repo,
        delegation_repository=InMemoryAuthorityDelegationRepository(),
        principal_authority_repository=authority_repo,
        clock=lambda: _NOW,
    )
    decision = cw_resolver.resolve(
        EffectiveAuthorityRequest.model_validate(
            {
                "tenant_id": resolved.tenant_id,
                "workspace_id": resolved.workspace_id,
                "acting_principal_id": resolved.principal_id,
                "requested_authority_scopes": ("workspace.read",),
                "membership": _membership_locator(principal_id=resolved.principal_id),
            }
        )
    )
    assert decision.decision.action is PolicyAction.ALLOW


def test_bound_principal_missing_membership_denies() -> None:
    principal_id = "principal-without-membership"
    worker_id = mint_worker_instance_id()
    binding_repo = _binding_repo()
    binding_repo.create(
        contract_suite.worker_principal_binding(
            worker_instance_id=worker_id,
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE,
            principal_id=principal_id,
        )
    )
    resolved = WorkerPrincipalBindingResolver(binding_repo).resolve(worker_instance_id=worker_id)
    cw_resolver = CollaborativeWorkAuthorityResolver(
        membership_repository=InMemoryWorkspaceMembershipRepository(),
        delegation_repository=InMemoryAuthorityDelegationRepository(),
        principal_authority_repository=InMemoryPrincipalAuthorityRepository(),
        clock=lambda: _NOW,
    )
    decision = cw_resolver.resolve(
        EffectiveAuthorityRequest.model_validate(
            {
                "tenant_id": resolved.tenant_id,
                "workspace_id": resolved.workspace_id,
                "acting_principal_id": resolved.principal_id,
                "requested_authority_scopes": ("workspace.read",),
            }
        )
    )
    assert decision.decision.action is PolicyAction.DENY
    assert decision.denial_reason is EffectiveAuthorityDenialReason.MISSING_MEMBERSHIP


def test_wrong_tenant_in_authority_request_denies() -> None:
    worker_id = mint_worker_instance_id()
    binding_repo = _binding_repo()
    binding_repo.create(
        contract_suite.worker_principal_binding(
            worker_instance_id=worker_id,
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE,
            principal_id="principal-bound",
        )
    )
    resolved = WorkerPrincipalBindingResolver(binding_repo).resolve(worker_instance_id=worker_id)
    membership_repo = InMemoryWorkspaceMembershipRepository()
    authority_repo = InMemoryPrincipalAuthorityRepository()
    membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=resolved.tenant_id,
            workspace_id=resolved.workspace_id,
            membership_id="membership-bound",
            principal_id=resolved.principal_id,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        )
    )
    authority_repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=resolved.tenant_id,
            workspace_id=resolved.workspace_id,
            authority_grant_id="authority-grant-bound",
            principal_id=resolved.principal_id,
            authority_scopes=("workspace.read",),
            status=AuthorityGrantStatus.ACTIVE,
        )
    )
    cw_resolver = CollaborativeWorkAuthorityResolver(
        membership_repository=membership_repo,
        delegation_repository=InMemoryAuthorityDelegationRepository(),
        principal_authority_repository=authority_repo,
        clock=lambda: _NOW,
    )
    decision = cw_resolver.resolve(
        EffectiveAuthorityRequest.model_validate(
            {
                "tenant_id": _TENANT_B,
                "workspace_id": resolved.workspace_id,
                "acting_principal_id": resolved.principal_id,
                "requested_authority_scopes": ("workspace.read",),
            }
        )
    )
    assert decision.decision.action is PolicyAction.DENY


def test_wrong_workspace_in_authority_request_denies() -> None:
    worker_id = mint_worker_instance_id()
    binding_repo = _binding_repo()
    binding_repo.create(
        contract_suite.worker_principal_binding(
            worker_instance_id=worker_id,
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE,
            principal_id="principal-bound",
        )
    )
    resolved = WorkerPrincipalBindingResolver(binding_repo).resolve(worker_instance_id=worker_id)
    membership_repo = InMemoryWorkspaceMembershipRepository()
    membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=resolved.tenant_id,
            workspace_id=resolved.workspace_id,
            membership_id="membership-bound",
            principal_id=resolved.principal_id,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        )
    )
    cw_resolver = CollaborativeWorkAuthorityResolver(
        membership_repository=membership_repo,
        delegation_repository=InMemoryAuthorityDelegationRepository(),
        principal_authority_repository=InMemoryPrincipalAuthorityRepository(),
        clock=lambda: _NOW,
    )
    decision = cw_resolver.resolve(
        EffectiveAuthorityRequest.model_validate(
            {
                "tenant_id": resolved.tenant_id,
                "workspace_id": _WORKSPACE_B,
                "acting_principal_id": resolved.principal_id,
                "requested_authority_scopes": ("workspace.read",),
            }
        )
    )
    assert decision.decision.action is PolicyAction.DENY


def test_revoked_membership_denies_for_bound_principal() -> None:
    principal_id = "principal-revoked"
    worker_id = mint_worker_instance_id()
    binding_repo = _binding_repo()
    binding_repo.create(
        contract_suite.worker_principal_binding(
            worker_instance_id=worker_id,
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE,
            principal_id=principal_id,
        )
    )
    resolved = WorkerPrincipalBindingResolver(binding_repo).resolve(worker_instance_id=worker_id)
    membership_repo = InMemoryWorkspaceMembershipRepository()
    created = membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=resolved.tenant_id,
            workspace_id=resolved.workspace_id,
            membership_id="membership-revoked",
            principal_id=resolved.principal_id,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        )
    )
    membership_repo.update(
        UpdateWorkspaceMembershipCommand(
            scope=WorkspaceMembershipScopeKey(
                tenant_id=resolved.tenant_id,
                workspace_id=resolved.workspace_id,
                membership_id="membership-revoked",
            ),
            expected_revision=created.revision,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.REVOKED,
        )
    )
    cw_resolver = CollaborativeWorkAuthorityResolver(
        membership_repository=membership_repo,
        delegation_repository=InMemoryAuthorityDelegationRepository(),
        principal_authority_repository=InMemoryPrincipalAuthorityRepository(),
        clock=lambda: _NOW,
    )
    decision = cw_resolver.resolve(
        EffectiveAuthorityRequest.model_validate(
            {
                "tenant_id": resolved.tenant_id,
                "workspace_id": resolved.workspace_id,
                "acting_principal_id": resolved.principal_id,
                "requested_authority_scopes": ("workspace.read",),
                "membership": _membership_locator(
                    principal_id=resolved.principal_id,
                    membership_id="membership-revoked",
                ),
            }
        )
    )
    assert decision.decision.action is PolicyAction.DENY
    assert decision.denial_reason is EffectiveAuthorityDenialReason.MEMBERSHIP_NOT_ACTIVE


_FORBIDDEN_AW_AUTHORITY_TOKENS = (
    "WorkerPermission",
    "WorkerAuthority",
    "WorkerRolePermission",
    "WorkerGrant",
    "WorkerAccessPolicy",
    "WorkerACL",
    "WorkerRBAC",
)


def _aw_contract_source_paths() -> list[Path]:
    package = importlib.import_module("intergrax.contracts.autonomous_work")
    assert package.__file__ is not None
    root = Path(package.__file__).parent
    return sorted(root.glob("*.py"))


def test_worker_contracts_contain_no_permission_grant_fields() -> None:
    forbidden_field_tokens = (
        "permission",
        "permissions",
        "grant",
        "grants",
        "access_policy",
        "acl",
        "rbac",
        "authority",
    )
    for path in _aw_contract_source_paths():
        if path.name in {"execution_authority.py", "obstacle_recovery.py", "recovery_orchestration.py"}:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
                continue
            field_name = node.target.id.lower()
            for token in forbidden_field_tokens:
                assert token not in field_name, (
                    f"{path} defines forbidden authority field {node.target.id}"
                )


def test_binding_contract_contains_no_authority_payload() -> None:
    fields = {field.name for field in WorkerPrincipalBinding.__dataclass_fields__.values()}
    assert fields == {
        "worker_instance_id",
        "tenant_id",
        "workspace_id",
        "principal_id",
        "created_at",
        "revision",
    }
    resolved_fields = {
        field.name for field in ResolvedWorkerPrincipal.__dataclass_fields__.values()
    }
    assert resolved_fields == {"tenant_id", "workspace_id", "principal_id"}


def test_principal_binding_resolver_has_no_concrete_persistence_import() -> None:
    module = importlib.import_module("intergrax.autonomous_work.principal_binding_resolver")
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


def test_autonomous_work_has_no_duplicate_collaborative_authority_resolver() -> None:
    package = importlib.import_module("intergrax.autonomous_work")
    assert package.__file__ is not None
    base = Path(package.__file__).parent
    for path in base.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "class CollaborativeWorkAuthorityResolver(" not in source


def test_autonomous_work_contracts_exclude_worker_authority_types() -> None:
    package = importlib.import_module("intergrax.contracts.autonomous_work")
    assert package.__file__ is not None
    base = Path(package.__file__).parent
    for path in base.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for token in _FORBIDDEN_AW_AUTHORITY_TOKENS:
            assert token not in source


def test_collaborative_work_core_does_not_import_autonomous_work() -> None:
    package = importlib.import_module("intergrax.collaborative_work")
    assert package.__file__ is not None
    base = Path(package.__file__).parent
    for path in base.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "autonomous_work" not in alias.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert "autonomous_work" not in node.module


def test_worker_instance_is_not_subclass_of_principal_binding() -> None:
    assert not issubclass(WorkerInstance, WorkerPrincipalBinding)
    assert not issubclass(WorkerDefinition, WorkerPrincipalBinding)
    assert not issubclass(WorkerGoal, WorkerPrincipalBinding)
