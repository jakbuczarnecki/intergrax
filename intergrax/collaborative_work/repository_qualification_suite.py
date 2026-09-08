# © Artur Czarnecki. All rights reserved.

"""Collaborative Work repository provider qualification suite (PROVIDER-QUAL-7)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from intergrax.collaborative_work.persistence import (
    CollaborativeWorkMaterializedRepositories,
    CollaborativeWorkRepositories,
    CollaborativeWorkRepositoriesWithArtifacts,
    CollaborativeWorkRepositoriesWithSharedWork,
)
from intergrax.collaborative_work.postgresql_artifact_cross_process_cas_proof import (
    CrossProcessArtifactPublicationProofFailure,
    run_postgresql_artifact_cross_process_cas_proof,
)
from intergrax.collaborative_work.postgresql_cross_process_cas_proof import (
    CrossProcessCasProofFailure,
    run_postgresql_work_item_cross_process_cas_proof,
)
from intergrax.collaborative_work.persistence_provider import (
    resolve_collaborative_work_repositories,
)
from intergrax.collaborative_work.postgresql_repository import PostgreSQLCollaborativeWorkStore
from intergrax.collaborative_work.serialization import (
    published_work_artifact_version_to_json,
    work_artifact_version_to_json,
)
from intergrax.collaborative_work.repository import (
    AssignmentAlreadyExists,
    AssignmentIdempotencyConflict,
    AssignmentRevisionConflict,
    AssignmentScopeKey,
    AuthorityDelegationScopeKey,
    CollaborativeOperationPolicyProfileScopeKey,
    CreateAssignmentCommand,
    CreateAuthorityDelegationCommand,
    CreateCollaborativeOperationPolicyProfileCommand,
    CreateCollaborativePolicyRuleCommand,
    CreatePrincipalAuthorityGrantCommand,
    CreateWorkItemCommand,
    CreateWorkItemExecutionLinkCommand,
    CreateWorkspaceMembershipCommand,
    CreateArtifactWithInitialVersionCommand,
    PublishWorkArtifactVersionCommand,
    ArtifactPublicationIdempotencyConflict,
    WorkArtifactAlreadyExists,
    WorkArtifactIdempotencyConflict,
    WorkArtifactNotFound,
    WorkArtifactRevisionConflict,
    WorkArtifactTemporalConflict,
    WorkArtifactVersionAlreadyExists,
    INITIAL_RECORD_REVISION,
    PrincipalAuthorityGrantAlreadyExists,
    UpdateAssignmentCommand,
    UpdateAuthorityDelegationCommand,
    UpdateCollaborativeOperationPolicyProfileCommand,
    UpdateWorkItemCommand,
    UpdateWorkspaceMembershipCommand,
    WorkItemAlreadyExists,
    WorkItemExecutionLinkAlreadyExists,
    WorkItemExecutionLinkIdempotencyConflict,
    WorkItemIdempotencyConflict,
    WorkItemRevisionConflict,
    WorkItemScopeKey,
    WorkspaceMembershipAlreadyExists,
    WorkspaceMembershipRevisionConflict,
    WorkspaceMembershipScopeKey,
    CollaborativePolicyRuleAlreadyExists,
    CollaborativeOperationPolicyProfileRevisionConflict,
)
from intergrax.contracts.collaborative_work import (
    ArtifactContentRef,
    AssignmentState,
    AuthorityGrantStatus,
    CollaborativeOperationPolicyProfileStatus,
    CollaborativePolicyRuleStatus,
    DelegationStatus,
    MembershipStatus,
    OperationPolicyRequirement,
    PolicyCompositionLayer,
    PolicyLayerApplicability,
    WorkItemState,
    WorkspaceMembershipRole,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_provenance import ExecutionProvenanceRef
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.core.qualification.evidence import QualificationEvidence
from intergrax.core.qualification.execution import (
    ProviderQualificationSubjectMismatchError,
    ProviderQualificationSuiteInfrastructureError,
)
from intergrax.core.qualification.provider import (
    ProviderQualificationEnvironmentMetadata,
    ProviderQualificationEvidenceKind,
    ProviderQualificationResultSummary,
    ProviderQualificationSubject,
)
from intergrax.core.qualification.status import QualificationStatus
from intergrax.core.qualification.suite import (
    ProviderQualificationMaterializationHandle,
    ProviderQualificationSuite,
    ProviderQualificationSuiteIdentity,
    ProviderQualificationSuiteOutcome,
)
from intergrax.integrations.registry.profile import IntegrationProfile

COLLABORATIVE_WORK_DOMAIN = "collaborative_work"
COLLABORATIVE_WORK_PERSISTENCE_CAPABILITY = "collaborative_work.persistence.v1"

CW_POSTGRESQL_REPOSITORY_SUITE_ID = "cw.postgresql.repository.v1"
CW_SQLITE_REPOSITORY_SUITE_ID = "cw.sqlite.repository.v1"
CW_REPOSITORY_SUITE_VERSION = "4.0.0"

_TENANT_A = "qual-tenant-a"
_TENANT_B = "qual-tenant-b"
_WORKSPACE_A = "qual-workspace-a"
_WORKSPACE_B = "qual-workspace-b"
_VALID_FROM = datetime(2026, 1, 1, tzinfo=UTC)
_VALID_UNTIL = datetime(2026, 12, 31, tzinfo=UTC)
_CREATED_AT = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
_UPDATED_AT = datetime(2026, 1, 1, 12, 30, tzinfo=UTC)
_PUBLISHED_AT = datetime(2026, 1, 1, 12, 45, tzinfo=UTC)
_LATER_PUBLISHED = datetime(2026, 1, 1, 13, 0, tzinfo=UTC)
_ARTIFACT_DIGEST = "sha256:" + ("a" * 64)


class _RepositorySemanticCheckFailure(Exception):
    """Domain semantic rejection surfaced by one repository contract check."""


@dataclass(frozen=True, slots=True)
class _CollaborativeWorkMaterializationHandle:
    _bundle: CollaborativeWorkMaterializedRepositories

    def close(self) -> None:
        self._bundle.close()


def _membership_command(**overrides: object) -> CreateWorkspaceMembershipCommand:
    payload = {
        "tenant_id": _TENANT_A,
        "workspace_id": _WORKSPACE_A,
        "membership_id": "qual-membership-1",
        "principal_id": "qual-principal-1",
        "role": WorkspaceMembershipRole.MEMBER,
        "status": MembershipStatus.ACTIVE,
    }
    payload.update(overrides)
    return CreateWorkspaceMembershipCommand(**payload)


def _run_core_repository_contract_checks(bundle: CollaborativeWorkRepositories) -> tuple[int, int]:
    passed = 0
    failed = 0

    def _record_success() -> None:
        nonlocal passed
        passed += 1

    def _record_failure() -> None:
        nonlocal failed
        failed += 1

    def _run_check(check: Callable[[], None]) -> None:
        try:
            check()
            _record_success()
        except _RepositorySemanticCheckFailure:
            _record_failure()

    membership_repo = bundle.membership
    delegation_repo = bundle.delegation
    authority_repo = bundle.principal_authority
    policy_repo = bundle.policy
    profile_repo = bundle.operation_profile

    def _membership_create_get_revision_and_isolation() -> None:
        created = membership_repo.create(_membership_command())
        if created.revision != INITIAL_RECORD_REVISION:
            raise _RepositorySemanticCheckFailure("membership revision mismatch")
        loaded = membership_repo.get(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            membership_id="qual-membership-1",
        )
        if loaded != created:
            raise _RepositorySemanticCheckFailure("membership round-trip mismatch")
        if (
            membership_repo.get(
                tenant_id=_TENANT_B,
                workspace_id=_WORKSPACE_B,
                membership_id="qual-membership-1",
            )
            is not None
        ):
            raise _RepositorySemanticCheckFailure("membership tenant isolation failed")

    def _membership_duplicate_and_stale_revision() -> None:
        created = membership_repo.create(
            _membership_command(
                membership_id="qual-membership-2",
                principal_id="qual-principal-2",
            ),
        )
        try:
            membership_repo.create(
                _membership_command(
                    membership_id="qual-membership-2",
                    principal_id="qual-principal-2",
                ),
            )
            raise _RepositorySemanticCheckFailure("expected WorkspaceMembershipAlreadyExists")
        except WorkspaceMembershipAlreadyExists:
            pass
        try:
            membership_repo.update(
                UpdateWorkspaceMembershipCommand(
                    scope=WorkspaceMembershipScopeKey(
                        tenant_id=_TENANT_A,
                        workspace_id=_WORKSPACE_A,
                        membership_id="qual-membership-2",
                    ),
                    expected_revision=created.revision + 1,
                    role=WorkspaceMembershipRole.ADMIN,
                    status=MembershipStatus.SUSPENDED,
                )
            )
            raise _RepositorySemanticCheckFailure("expected WorkspaceMembershipRevisionConflict")
        except WorkspaceMembershipRevisionConflict:
            pass

    def _delegation_create_update_idempotency() -> None:
        command = CreateAuthorityDelegationCommand(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            delegation_id="qual-delegation-1",
            delegator_principal_id="qual-delegator",
            delegate_principal_id="qual-delegate",
            authority_scopes=("workspace.read",),
            status=DelegationStatus.ACTIVE,
            idempotency_key="qual-delegation-idem",
        )
        created = delegation_repo.create(command)
        if delegation_repo.create(command) != created:
            raise _RepositorySemanticCheckFailure("delegation idempotency mismatch")
        updated = delegation_repo.update(
            UpdateAuthorityDelegationCommand(
                scope=AuthorityDelegationScopeKey(
                    tenant_id=_TENANT_A,
                    workspace_id=_WORKSPACE_A,
                    delegation_id="qual-delegation-1",
                ),
                expected_revision=created.revision,
                authority_scopes=("workspace.write",),
                resource_scope="resource-1",
                valid_from=_VALID_FROM,
                valid_until=_VALID_UNTIL,
                status=DelegationStatus.REVOKED,
            )
        )
        if updated.revision != created.revision + 1:
            raise _RepositorySemanticCheckFailure("delegation revision increment mismatch")
        if delegation_repo.create(command) != created:
            raise _RepositorySemanticCheckFailure("delegation idempotency after update failed")

    def _authority_grant_principal_uniqueness() -> None:
        command = CreatePrincipalAuthorityGrantCommand(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            authority_grant_id="qual-grant-1",
            principal_id="qual-principal-authority-1",
            authority_scopes=("workspace.read",),
            status=AuthorityGrantStatus.ACTIVE,
        )
        authority_repo.create(command)
        try:
            authority_repo.create(
                CreatePrincipalAuthorityGrantCommand(
                    tenant_id=_TENANT_A,
                    workspace_id=_WORKSPACE_A,
                    authority_grant_id="qual-grant-2",
                    principal_id="qual-principal-authority-1",
                    authority_scopes=("workspace.write",),
                    status=AuthorityGrantStatus.ACTIVE,
                )
            )
            raise _RepositorySemanticCheckFailure("expected PrincipalAuthorityGrantAlreadyExists")
        except PrincipalAuthorityGrantAlreadyExists:
            pass

    def _policy_exact_key_uniqueness() -> None:
        policy_repo.create(
            CreateCollaborativePolicyRuleCommand(
                tenant_id=_TENANT_A,
                workspace_id=_WORKSPACE_A,
                policy_rule_id="qual-rule-1",
                layer=PolicyCompositionLayer.WORKSPACE_POLICY,
                authority_scope="document.delete",
                action=PolicyAction.ALLOW,
                status=CollaborativePolicyRuleStatus.ACTIVE,
            )
        )
        try:
            policy_repo.create(
                CreateCollaborativePolicyRuleCommand(
                    tenant_id=_TENANT_A,
                    workspace_id=_WORKSPACE_A,
                    policy_rule_id="qual-rule-2",
                    layer=PolicyCompositionLayer.WORKSPACE_POLICY,
                    authority_scope="document.delete",
                    action=PolicyAction.DENY,
                    status=CollaborativePolicyRuleStatus.ACTIVE,
                )
            )
            raise _RepositorySemanticCheckFailure("expected CollaborativePolicyRuleAlreadyExists")
        except CollaborativePolicyRuleAlreadyExists:
            pass

    def _profile_revision_increment() -> None:
        created = profile_repo.create(
            CreateCollaborativeOperationPolicyProfileCommand(
                tenant_id=_TENANT_A,
                workspace_id=_WORKSPACE_A,
                operation_id="qual-operation-1",
                authority_scope="document.delete",
                workspace_policy_applicability=PolicyLayerApplicability.REQUIRED,
                resource_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
                runtime_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
                resource_requirement=OperationPolicyRequirement.NOT_APPLICABLE,
                meaningful_side_effect_requirement=OperationPolicyRequirement.NOT_APPLICABLE,
                status=CollaborativeOperationPolicyProfileStatus.ACTIVE,
            )
        )
        try:
            profile_repo.update(
                UpdateCollaborativeOperationPolicyProfileCommand(
                    scope=CollaborativeOperationPolicyProfileScopeKey(
                        tenant_id=_TENANT_A,
                        workspace_id=_WORKSPACE_A,
                        operation_id="qual-operation-1",
                    ),
                    expected_revision=created.revision + 1,
                    authority_scope="document.delete",
                    workspace_policy_applicability=PolicyLayerApplicability.REQUIRED,
                    resource_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
                    runtime_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
                    resource_requirement=OperationPolicyRequirement.NOT_APPLICABLE,
                    meaningful_side_effect_requirement=OperationPolicyRequirement.NOT_APPLICABLE,
                    status=CollaborativeOperationPolicyProfileStatus.DISABLED,
                )
            )
            raise _RepositorySemanticCheckFailure(
                "expected CollaborativeOperationPolicyProfileRevisionConflict",
            )
        except CollaborativeOperationPolicyProfileRevisionConflict:
            pass

    for check in (
        _membership_create_get_revision_and_isolation,
        _membership_duplicate_and_stale_revision,
        _delegation_create_update_idempotency,
        _authority_grant_principal_uniqueness,
        _policy_exact_key_uniqueness,
        _profile_revision_increment,
    ):
        _run_check(check)

    return passed, failed


def _work_item_command(**overrides: object) -> CreateWorkItemCommand:
    payload = {
        "tenant_id": _TENANT_A,
        "workspace_id": _WORKSPACE_A,
        "work_item_id": "qual-work-item-1",
        "created_by_principal_id": "qual-principal-creator",
        "state": WorkItemState.OPEN,
        "created_at": _CREATED_AT,
        "updated_at": _CREATED_AT,
        "title": "Qualification WorkItem",
        "description": "MP-2 qualification proof",
    }
    payload.update(overrides)
    return CreateWorkItemCommand(**payload)


def _assignment_command(**overrides: object) -> CreateAssignmentCommand:
    payload = {
        "tenant_id": _TENANT_A,
        "workspace_id": _WORKSPACE_A,
        "assignment_id": "qual-assignment-1",
        "work_item_id": "qual-work-item-1",
        "principal_id": "qual-principal-1",
        "created_by_principal_id": "qual-principal-creator",
        "state": AssignmentState.ACTIVE,
        "created_at": _CREATED_AT,
        "updated_at": _CREATED_AT,
    }
    payload.update(overrides)
    return CreateAssignmentCommand(**payload)


def _execution_provenance(**overrides: object) -> ExecutionProvenanceRef:
    payload = {
        "task_id": mint_task_id(),
        "run_id": mint_run_id(),
        "attempt_id": mint_attempt_id(),
        "execution_id": mint_execution_id(),
    }
    payload.update(overrides)
    return ExecutionProvenanceRef(**payload)


def _execution_link_command(**overrides: object) -> CreateWorkItemExecutionLinkCommand:
    payload = {
        "tenant_id": _TENANT_A,
        "workspace_id": _WORKSPACE_A,
        "execution_link_id": "qual-execution-link-1",
        "work_item_id": "qual-work-item-1",
        "execution": _execution_provenance(),
        "linked_at": _CREATED_AT,
    }
    payload.update(overrides)
    return CreateWorkItemExecutionLinkCommand(**payload)


def _artifact_content_ref(**overrides: object) -> ArtifactContentRef:
    payload = {
        "content_ref": "content://qual-tenant-a/qual-workspace-a/body-1",
        "media_type": "application/json",
        "integrity_digest": _ARTIFACT_DIGEST,
    }
    payload.update(overrides)
    return ArtifactContentRef.model_validate(payload)


def _artifact_create_command(**overrides: object) -> CreateArtifactWithInitialVersionCommand:
    payload = {
        "tenant_id": _TENANT_A,
        "workspace_id": _WORKSPACE_A,
        "work_item_id": "qual-work-item-artifact",
        "work_artifact_id": "qual-artifact-1",
        "work_artifact_version_id": "qual-artifact-version-1",
        "created_by_principal_id": "qual-principal-creator",
        "published_by_principal_id": "qual-principal-publisher",
        "content_ref": _artifact_content_ref(),
        "artifact_created_at": _CREATED_AT,
        "artifact_updated_at": _UPDATED_AT,
        "version_created_at": _CREATED_AT,
        "version_published_at": _PUBLISHED_AT,
        "execution": None,
    }
    payload.update(overrides)
    return CreateArtifactWithInitialVersionCommand(**payload)


def _artifact_publish_command(**overrides: object) -> PublishWorkArtifactVersionCommand:
    payload = {
        "tenant_id": _TENANT_A,
        "workspace_id": _WORKSPACE_A,
        "work_item_id": "qual-work-item-artifact",
        "work_artifact_id": "qual-artifact-1",
        "work_artifact_version_id": "qual-artifact-version-2",
        "expected_revision": INITIAL_RECORD_REVISION,
        "created_by_principal_id": "qual-principal-creator",
        "published_by_principal_id": "qual-principal-publisher",
        "content_ref": _artifact_content_ref(content_ref="content://qual-tenant-a/qual-workspace-a/body-2"),
        "created_at": _UPDATED_AT,
        "published_at": _LATER_PUBLISHED,
        "artifact_updated_at": _LATER_PUBLISHED,
        "execution": None,
    }
    payload.update(overrides)
    return PublishWorkArtifactVersionCommand(**payload)


def _run_artifact_repository_contract_checks(
    bundle: CollaborativeWorkRepositoriesWithArtifacts,
) -> tuple[int, int]:
    passed = 0
    failed = 0

    def _record_success() -> None:
        nonlocal passed
        passed += 1

    def _record_failure() -> None:
        nonlocal failed
        failed += 1

    def _run_check(check: Callable[[], None]) -> None:
        try:
            check()
            _record_success()
        except _RepositorySemanticCheckFailure:
            _record_failure()

    artifact_repo = bundle.artifact
    version_repo = bundle.version
    publication_repo = bundle.publication

    def _artifact_create_read_history_isolation() -> None:
        created = publication_repo.create_artifact_with_initial_version(_artifact_create_command())
        if created.artifact.revision != INITIAL_RECORD_REVISION:
            raise _RepositorySemanticCheckFailure("artifact revision mismatch")
        if created.artifact.current_version_id != "qual-artifact-version-1":
            raise _RepositorySemanticCheckFailure("artifact pointer mismatch")
        loaded_artifact = artifact_repo.get(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            work_artifact_id="qual-artifact-1",
        )
        loaded_version = version_repo.get(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            work_artifact_version_id="qual-artifact-version-1",
        )
        if loaded_artifact != created.artifact or loaded_version != created.version:
            raise _RepositorySemanticCheckFailure("artifact round-trip mismatch")
        history = version_repo.list_for_artifact(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            work_artifact_id="qual-artifact-1",
        )
        if history != (created.version,):
            raise _RepositorySemanticCheckFailure("artifact history mismatch")
        if (
            artifact_repo.get(
                tenant_id=_TENANT_B,
                workspace_id=_WORKSPACE_B,
                work_artifact_id="qual-artifact-1",
            )
            is not None
        ):
            raise _RepositorySemanticCheckFailure("artifact tenant isolation failed")

    def _artifact_duplicate_and_idempotency() -> None:
        command = _artifact_create_command(
            work_artifact_id="qual-artifact-2",
            work_artifact_version_id="qual-artifact-version-2a",
            idempotency_key="qual-artifact-create-idem",
        )
        created = publication_repo.create_artifact_with_initial_version(command)
        try:
            publication_repo.create_artifact_with_initial_version(
                _artifact_create_command(
                    work_artifact_id="qual-artifact-2",
                    work_artifact_version_id="qual-artifact-version-2b",
                ),
            )
            raise _RepositorySemanticCheckFailure("expected WorkArtifactAlreadyExists")
        except WorkArtifactAlreadyExists:
            pass
        try:
            publication_repo.create_artifact_with_initial_version(
                _artifact_create_command(
                    work_artifact_id="qual-artifact-3",
                    work_artifact_version_id="qual-artifact-version-2a",
                ),
            )
            raise _RepositorySemanticCheckFailure("expected WorkArtifactVersionAlreadyExists")
        except WorkArtifactVersionAlreadyExists:
            pass
        replay = publication_repo.create_artifact_with_initial_version(command)
        if replay != created:
            raise _RepositorySemanticCheckFailure("artifact create idempotency replay mismatch")
        changed_replay = publication_repo.create_artifact_with_initial_version(
            _artifact_create_command(
                work_artifact_id="qual-artifact-2",
                work_artifact_version_id="qual-artifact-version-2a",
                idempotency_key="qual-artifact-create-idem",
                artifact_updated_at=_LATER_PUBLISHED,
                version_published_at=_LATER_PUBLISHED,
            ),
        )
        if changed_replay != created:
            raise _RepositorySemanticCheckFailure("artifact create idempotency timestamp drift")
        try:
            publication_repo.create_artifact_with_initial_version(
                _artifact_create_command(
                    work_artifact_id="qual-artifact-4",
                    idempotency_key="qual-artifact-create-idem",
                ),
            )
            raise _RepositorySemanticCheckFailure("expected WorkArtifactIdempotencyConflict")
        except WorkArtifactIdempotencyConflict:
            pass

    def _artifact_publish_revision_temporal_idempotency() -> None:
        publication_repo.create_artifact_with_initial_version(
            _artifact_create_command(
                work_artifact_id="qual-artifact-3",
                work_artifact_version_id="qual-artifact-version-3a",
            ),
        )
        published = publication_repo.publish_version(
            _artifact_publish_command(
                work_artifact_id="qual-artifact-3",
                work_artifact_version_id="qual-artifact-version-3b",
                expected_revision=INITIAL_RECORD_REVISION,
            ),
        )
        if published.artifact.revision != INITIAL_RECORD_REVISION + 1:
            raise _RepositorySemanticCheckFailure("artifact publish revision increment mismatch")
        history = version_repo.list_for_artifact(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            work_artifact_id="qual-artifact-3",
        )
        if len(history) != 2:
            raise _RepositorySemanticCheckFailure("artifact append-only history mismatch")
        try:
            publication_repo.publish_version(
                _artifact_publish_command(
                    work_artifact_id="qual-artifact-3",
                    work_artifact_version_id="qual-artifact-version-3c",
                    expected_revision=INITIAL_RECORD_REVISION,
                ),
            )
            raise _RepositorySemanticCheckFailure("expected WorkArtifactRevisionConflict")
        except WorkArtifactRevisionConflict:
            pass
        try:
            publication_repo.publish_version(
                _artifact_publish_command(
                    work_artifact_id="qual-artifact-3",
                    work_artifact_version_id="qual-artifact-version-3d",
                    expected_revision=INITIAL_RECORD_REVISION + 1,
                    artifact_updated_at=_CREATED_AT - timedelta(minutes=1),
                ),
            )
            raise _RepositorySemanticCheckFailure("expected WorkArtifactTemporalConflict")
        except WorkArtifactTemporalConflict:
            pass
        first_publish = publication_repo.publish_version(
            _artifact_publish_command(
                work_artifact_id="qual-artifact-3",
                work_artifact_version_id="qual-artifact-version-3e",
                expected_revision=INITIAL_RECORD_REVISION + 1,
                idempotency_key="qual-artifact-publish-idem",
            ),
        )
        publication_repo.publish_version(
            _artifact_publish_command(
                work_artifact_id="qual-artifact-3",
                work_artifact_version_id="qual-artifact-version-3f",
                expected_revision=first_publish.artifact.revision,
            ),
        )
        replay = publication_repo.publish_version(
            _artifact_publish_command(
                work_artifact_id="qual-artifact-3",
                work_artifact_version_id="qual-artifact-version-3e",
                expected_revision=INITIAL_RECORD_REVISION + 1,
                idempotency_key="qual-artifact-publish-idem",
            ),
        )
        if replay != first_publish:
            raise _RepositorySemanticCheckFailure("artifact publish idempotency replay mismatch")
        try:
            publication_repo.publish_version(
                _artifact_publish_command(
                    work_artifact_id="qual-artifact-3",
                    work_artifact_version_id="qual-artifact-version-3g",
                    expected_revision=first_publish.artifact.revision + 1,
                    idempotency_key="qual-artifact-publish-idem",
                ),
            )
            raise _RepositorySemanticCheckFailure("expected ArtifactPublicationIdempotencyConflict")
        except ArtifactPublicationIdempotencyConflict:
            pass
        try:
            publication_repo.publish_version(
                _artifact_publish_command(
                    tenant_id=_TENANT_B,
                    workspace_id=_WORKSPACE_B,
                    work_artifact_id="qual-artifact-3",
                ),
            )
            raise _RepositorySemanticCheckFailure("expected WorkArtifactNotFound")
        except WorkArtifactNotFound:
            pass

    def _artifact_execution_serialization_round_trip() -> None:
        execution = _execution_provenance()
        created = publication_repo.create_artifact_with_initial_version(
            _artifact_create_command(
                work_artifact_id="qual-artifact-4",
                work_artifact_version_id="qual-artifact-version-4a",
                execution=execution,
            ),
        )
        loaded = version_repo.get(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            work_artifact_version_id="qual-artifact-version-4a",
        )
        if loaded is None or loaded.execution != execution:
            raise _RepositorySemanticCheckFailure("artifact execution round-trip mismatch")
        encoded = published_work_artifact_version_to_json(created)
        if work_artifact_version_to_json(created.version) not in encoded:
            raise _RepositorySemanticCheckFailure("artifact serialization mismatch")

    for check in (
        _artifact_create_read_history_isolation,
        _artifact_duplicate_and_idempotency,
        _artifact_publish_revision_temporal_idempotency,
        _artifact_execution_serialization_round_trip,
    ):
        _run_check(check)

    return passed, failed


def _run_shared_work_repository_contract_checks(
    bundle: CollaborativeWorkRepositoriesWithSharedWork | CollaborativeWorkRepositoriesWithArtifacts,
) -> tuple[int, int]:
    passed = 0
    failed = 0

    def _record_success() -> None:
        nonlocal passed
        passed += 1

    def _record_failure() -> None:
        nonlocal failed
        failed += 1

    def _run_check(check: Callable[[], None]) -> None:
        try:
            check()
            _record_success()
        except _RepositorySemanticCheckFailure:
            _record_failure()

    work_item_repo = bundle.work_item
    assignment_repo = bundle.assignment
    execution_link_repo = bundle.execution_link

    def _work_item_create_get_isolation() -> None:
        created = work_item_repo.create(_work_item_command())
        if created.revision != INITIAL_RECORD_REVISION:
            raise _RepositorySemanticCheckFailure("work item revision mismatch")
        loaded = work_item_repo.get(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            work_item_id="qual-work-item-1",
        )
        if loaded != created:
            raise _RepositorySemanticCheckFailure("work item round-trip mismatch")
        if (
            work_item_repo.get(
                tenant_id=_TENANT_B,
                workspace_id=_WORKSPACE_B,
                work_item_id="qual-work-item-1",
            )
            is not None
        ):
            raise _RepositorySemanticCheckFailure("work item tenant isolation failed")

    def _work_item_duplicate_stale_and_idempotency() -> None:
        created = work_item_repo.create(
            _work_item_command(
                work_item_id="qual-work-item-2",
                idempotency_key="qual-work-item-idem",
            ),
        )
        try:
            work_item_repo.create(_work_item_command(work_item_id="qual-work-item-2"))
            raise _RepositorySemanticCheckFailure("expected WorkItemAlreadyExists")
        except WorkItemAlreadyExists:
            pass
        if work_item_repo.create(
            _work_item_command(
                work_item_id="qual-work-item-2",
                idempotency_key="qual-work-item-idem",
            ),
        ) != created:
            raise _RepositorySemanticCheckFailure("work item idempotency replay mismatch")
        updated = work_item_repo.update(
            UpdateWorkItemCommand(
                scope=WorkItemScopeKey(
                    tenant_id=_TENANT_A,
                    workspace_id=_WORKSPACE_A,
                    work_item_id="qual-work-item-2",
                ),
                expected_revision=created.revision,
                state=WorkItemState.ACTIVE,
                updated_at=_UPDATED_AT,
            ),
        )
        if updated.revision != created.revision + 1:
            raise _RepositorySemanticCheckFailure("work item revision increment mismatch")
        if work_item_repo.create(
            _work_item_command(
                work_item_id="qual-work-item-2",
                idempotency_key="qual-work-item-idem",
            ),
        ) != created:
            raise _RepositorySemanticCheckFailure(
                "work item idempotency replay after update failed",
            )
        try:
            work_item_repo.update(
                UpdateWorkItemCommand(
                    scope=WorkItemScopeKey(
                        tenant_id=_TENANT_A,
                        workspace_id=_WORKSPACE_A,
                        work_item_id="qual-work-item-2",
                    ),
                    expected_revision=created.revision,
                        state=WorkItemState.CANCELLED,
                    updated_at=_UPDATED_AT,
                ),
            )
            raise _RepositorySemanticCheckFailure("expected WorkItemRevisionConflict")
        except WorkItemRevisionConflict:
            pass
        try:
            work_item_repo.create(
                _work_item_command(
                    work_item_id="qual-work-item-3",
                    idempotency_key="qual-work-item-idem",
                ),
            )
            raise _RepositorySemanticCheckFailure("expected WorkItemIdempotencyConflict")
        except WorkItemIdempotencyConflict:
            pass

    def _assignment_create_update_idempotency() -> None:
        command = _assignment_command(
            assignment_id="qual-assignment-2",
            idempotency_key="qual-assignment-idem",
        )
        created = assignment_repo.create(command)
        if assignment_repo.create(command) != created:
            raise _RepositorySemanticCheckFailure("assignment idempotency mismatch")
        second = assignment_repo.create(
            _assignment_command(
                assignment_id="qual-assignment-3",
                principal_id="qual-principal-2",
            ),
        )
        if second.work_item_id != created.work_item_id:
            raise _RepositorySemanticCheckFailure("assignment work_item linkage mismatch")
        updated = assignment_repo.update(
            UpdateAssignmentCommand(
                scope=AssignmentScopeKey(
                    tenant_id=_TENANT_A,
                    workspace_id=_WORKSPACE_A,
                    assignment_id="qual-assignment-2",
                ),
                expected_revision=created.revision,
                state=AssignmentState.REVOKED,
                updated_at=_UPDATED_AT,
            ),
        )
        if updated.revision != created.revision + 1:
            raise _RepositorySemanticCheckFailure("assignment revision increment mismatch")
        if updated.principal_id != created.principal_id:
            raise _RepositorySemanticCheckFailure("assignment principal_id mutated")
        if updated.work_item_id != created.work_item_id:
            raise _RepositorySemanticCheckFailure("assignment work_item_id mutated")
        if assignment_repo.create(command) != created:
            raise _RepositorySemanticCheckFailure("assignment idempotency after update failed")
        try:
            assignment_repo.create(_assignment_command(assignment_id="qual-assignment-2"))
            raise _RepositorySemanticCheckFailure("expected AssignmentAlreadyExists")
        except AssignmentAlreadyExists:
            pass
        try:
            assignment_repo.update(
                UpdateAssignmentCommand(
                    scope=AssignmentScopeKey(
                        tenant_id=_TENANT_A,
                        workspace_id=_WORKSPACE_A,
                        assignment_id="qual-assignment-2",
                    ),
                    expected_revision=created.revision,
                    state=AssignmentState.ACTIVE,
                    updated_at=_UPDATED_AT,
                ),
            )
            raise _RepositorySemanticCheckFailure("expected AssignmentRevisionConflict")
        except AssignmentRevisionConflict:
            pass
        try:
            assignment_repo.create(
                _assignment_command(
                    assignment_id="qual-assignment-4",
                    idempotency_key="qual-assignment-idem",
                    principal_id="qual-principal-3",
                ),
            )
            raise _RepositorySemanticCheckFailure("expected AssignmentIdempotencyConflict")
        except AssignmentIdempotencyConflict:
            pass

    def _execution_link_create_get_isolation_and_provenance() -> None:
        created = execution_link_repo.create(
            _execution_link_command(work_item_id="qual-work-item-1"),
        )
        if str(created.execution.task_id).startswith("task_") is False:
            raise _RepositorySemanticCheckFailure("execution provenance task_id invalid")
        loaded = execution_link_repo.get(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            execution_link_id="qual-execution-link-1",
        )
        if loaded != created:
            raise _RepositorySemanticCheckFailure("execution link round-trip mismatch")
        if (
            execution_link_repo.get(
                tenant_id=_TENANT_B,
                workspace_id=_WORKSPACE_B,
                execution_link_id="qual-execution-link-1",
            )
            is not None
        ):
            raise _RepositorySemanticCheckFailure("execution link tenant isolation failed")

    def _execution_link_idempotency_duplicate_and_conflict() -> None:
        work_item_repo.create(_work_item_command(work_item_id="qual-work-item-exec-2"))
        command = _execution_link_command(
            execution_link_id="qual-execution-link-2",
            work_item_id="qual-work-item-exec-2",
            idempotency_key="qual-execution-link-idem",
        )
        created = execution_link_repo.create(command)
        replay_command = _execution_link_command(
            execution_link_id="qual-execution-link-2",
            work_item_id="qual-work-item-exec-2",
            idempotency_key="qual-execution-link-idem",
            linked_at=_UPDATED_AT,
            execution=command.execution,
        )
        replayed = execution_link_repo.create(replay_command)
        if replayed != created:
            raise _RepositorySemanticCheckFailure("execution link idempotency replay mismatch")
        if replayed.linked_at != created.linked_at:
            raise _RepositorySemanticCheckFailure(
                "execution link idempotency replay must preserve original linked_at",
            )
        try:
            execution_link_repo.create(
                _execution_link_command(
                    execution_link_id="qual-execution-link-2",
                    work_item_id="qual-work-item-exec-2",
                ),
            )
            raise _RepositorySemanticCheckFailure("expected WorkItemExecutionLinkAlreadyExists")
        except WorkItemExecutionLinkAlreadyExists:
            pass
        try:
            execution_link_repo.create(
                _execution_link_command(
                    execution_link_id="qual-execution-link-3",
                    work_item_id="qual-work-item-exec-2",
                    idempotency_key="qual-execution-link-idem",
                    execution=_execution_provenance(),
                ),
            )
            raise _RepositorySemanticCheckFailure("expected WorkItemExecutionLinkIdempotencyConflict")
        except WorkItemExecutionLinkIdempotencyConflict:
            pass

    def _execution_link_list_multiple_and_ordering() -> None:
        work_item_repo.create(_work_item_command(work_item_id="qual-work-item-exec-3"))
        first = execution_link_repo.create(
            _execution_link_command(
                execution_link_id="qual-execution-link-a",
                work_item_id="qual-work-item-exec-3",
                linked_at=_CREATED_AT,
                execution=_execution_provenance(),
            ),
        )
        second = execution_link_repo.create(
            _execution_link_command(
                execution_link_id="qual-execution-link-b",
                work_item_id="qual-work-item-exec-3",
                linked_at=_UPDATED_AT,
                execution=_execution_provenance(),
            ),
        )
        third = execution_link_repo.create(
            _execution_link_command(
                execution_link_id="qual-execution-link-c",
                work_item_id="qual-work-item-exec-3",
                linked_at=_CREATED_AT,
                execution=_execution_provenance(),
            ),
        )
        listed = execution_link_repo.list_for_work_item(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            work_item_id="qual-work-item-exec-3",
        )
        if len(listed) != 3:
            raise _RepositorySemanticCheckFailure("execution link list count mismatch")
        expected_order = sorted(
            (first, second, third),
            key=lambda record: (record.linked_at, record.execution_link_id),
        )
        if listed != tuple(expected_order):
            raise _RepositorySemanticCheckFailure("execution link ordering mismatch")

    for check in (
        _work_item_create_get_isolation,
        _work_item_duplicate_stale_and_idempotency,
        _assignment_create_update_idempotency,
        _execution_link_create_get_isolation_and_provenance,
        _execution_link_idempotency_duplicate_and_conflict,
        _execution_link_list_multiple_and_ordering,
    ):
        _run_check(check)

    return passed, failed


def _run_shared_work_cross_process_concurrency_check(
    bundle: CollaborativeWorkRepositoriesWithSharedWork | CollaborativeWorkRepositoriesWithArtifacts,
) -> tuple[int, int]:
    store = bundle.store
    if not isinstance(store, PostgreSQLCollaborativeWorkStore):
        raise _RepositorySemanticCheckFailure(
            "cross-process concurrency requires PostgreSQLCollaborativeWorkStore",
        )

    work_item_id = "qual-work-item-concurrency"
    created = bundle.work_item.create(_work_item_command(work_item_id=work_item_id))
    try:
        run_postgresql_work_item_cross_process_cas_proof(
            config=store.config,
            schema_name=store.schema_name,
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            work_item_id=work_item_id,
            expected_revision=created.revision,
            updated_at=_UPDATED_AT,
        )
    except CrossProcessCasProofFailure as exc:
        raise _RepositorySemanticCheckFailure(str(exc)) from exc

    return 1, 0


def _run_artifact_cross_process_concurrency_check(
    bundle: CollaborativeWorkRepositoriesWithArtifacts,
) -> tuple[int, int]:
    store = bundle.store
    if not isinstance(store, PostgreSQLCollaborativeWorkStore):
        raise _RepositorySemanticCheckFailure(
            "artifact cross-process concurrency requires PostgreSQLCollaborativeWorkStore",
        )

    initial = bundle.publication.create_artifact_with_initial_version(
        _artifact_create_command(
            work_artifact_id="qual-artifact-concurrency",
            work_artifact_version_id="qual-artifact-version-concurrency-initial",
            work_item_id="qual-work-item-artifact-concurrency",
        ),
    )
    try:
        run_postgresql_artifact_cross_process_cas_proof(
            config=store.config,
            schema_name=store.schema_name,
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            work_item_id="qual-work-item-artifact-concurrency",
            work_artifact_id="qual-artifact-concurrency",
            initial_version_id="qual-artifact-version-concurrency-initial",
            winning_version_id="qual-artifact-version-concurrency-win",
            losing_version_id="qual-artifact-version-concurrency-lose",
            expected_revision=initial.artifact.revision,
            published_at=_LATER_PUBLISHED,
            artifact_updated_at=_LATER_PUBLISHED,
            content_ref=_artifact_content_ref(
                content_ref="content://qual-tenant-a/qual-workspace-a/concurrency-body",
            ),
        )
    except CrossProcessArtifactPublicationProofFailure as exc:
        raise _RepositorySemanticCheckFailure(str(exc)) from exc

    return 1, 0


@dataclass(frozen=True, slots=True)
class CollaborativeWorkRepositoryQualificationSuite:
    """Domain-owned repository qualification suite for Collaborative Work persistence."""

    _identity: ProviderQualificationSuiteIdentity
    _qualified_status: QualificationStatus
    _environment_metadata: ProviderQualificationEnvironmentMetadata
    _limitations: tuple[str, ...]
    _reproducibility: str
    _requires_shared_work: bool = True
    _requires_artifacts: bool = False
    _requires_concurrency_proof: bool = False
    _requires_artifact_concurrency_proof: bool = False

    @property
    def identity(self) -> ProviderQualificationSuiteIdentity:
        return self._identity

    def execute(self, capability: object) -> ProviderQualificationSuiteOutcome:
        concurrency_evidence: tuple[QualificationEvidence, ...] = ()
        artifact_evidence: tuple[QualificationEvidence, ...] = ()

        if isinstance(capability, CollaborativeWorkRepositoriesWithArtifacts):
            core_passed, core_failed = _run_core_repository_contract_checks(capability.core)
            shared_passed, shared_failed = _run_shared_work_repository_contract_checks(capability)
            passed = core_passed + shared_passed
            failed = core_failed + shared_failed
            if self._requires_artifacts:
                art_passed, art_failed = _run_artifact_repository_contract_checks(capability)
                passed += art_passed
                failed += art_failed
                artifact_evidence = (
                    QualificationEvidence(
                        kind=ProviderQualificationEvidenceKind.SUITE_EXECUTION,
                        code="shared_work.artifacts",
                        label="work_artifact,work_artifact_version,publication",
                    ),
                    QualificationEvidence(
                        kind=ProviderQualificationEvidenceKind.SUITE_EXECUTION,
                        code="shared_work.artifact.atomic_publication",
                        label="initial_create,publish,cas,idempotency",
                    ),
                )
            if self._requires_concurrency_proof:
                try:
                    conc_passed, conc_failed = _run_shared_work_cross_process_concurrency_check(
                        capability,
                    )
                except _RepositorySemanticCheckFailure:
                    conc_passed, conc_failed = 0, 1
                passed += conc_passed
                failed += conc_failed
                concurrency_evidence = concurrency_evidence + (
                    QualificationEvidence(
                        kind=ProviderQualificationEvidenceKind.SUITE_EXECUTION,
                        code="shared_work.concurrency.cross_process",
                        label="transactional_cas",
                    ),
                )
            if self._requires_artifact_concurrency_proof:
                try:
                    art_conc_passed, art_conc_failed = _run_artifact_cross_process_concurrency_check(
                        capability,
                    )
                except _RepositorySemanticCheckFailure:
                    art_conc_passed, art_conc_failed = 0, 1
                passed += art_conc_passed
                failed += art_conc_failed
                artifact_evidence = artifact_evidence + (
                    QualificationEvidence(
                        kind=ProviderQualificationEvidenceKind.SUITE_EXECUTION,
                        code="shared_work.artifact.concurrency.cross_process",
                        label="transactional_publication_cas",
                    ),
                )
        elif isinstance(capability, CollaborativeWorkRepositoriesWithSharedWork):
            core_passed, core_failed = _run_core_repository_contract_checks(capability.core)
            shared_passed, shared_failed = _run_shared_work_repository_contract_checks(capability)
            passed = core_passed + shared_passed
            failed = core_failed + shared_failed
            if self._requires_concurrency_proof:
                try:
                    conc_passed, conc_failed = _run_shared_work_cross_process_concurrency_check(
                        capability,
                    )
                except _RepositorySemanticCheckFailure:
                    conc_passed, conc_failed = 0, 1
                passed += conc_passed
                failed += conc_failed
                concurrency_evidence = (
                    QualificationEvidence(
                        kind=ProviderQualificationEvidenceKind.SUITE_EXECUTION,
                        code="shared_work.concurrency.cross_process",
                        label="transactional_cas",
                    ),
                )
        elif isinstance(capability, CollaborativeWorkRepositories):
            if self._requires_shared_work:
                raise ProviderQualificationSuiteInfrastructureError(
                    "capability must include MP-2 Shared Work repositories",
                )
            passed, failed = _run_core_repository_contract_checks(capability)
        else:
            raise ProviderQualificationSuiteInfrastructureError(
                "capability must be CollaborativeWorkRepositories, "
                "CollaborativeWorkRepositoriesWithSharedWork, "
                "or CollaborativeWorkRepositoriesWithArtifacts",
            )

        skipped = 0
        status = self._qualified_status if failed == 0 else QualificationStatus.REJECTED
        evidence = (
            QualificationEvidence(
                kind=ProviderQualificationEvidenceKind.SUITE_EXECUTION,
                code="suite.passed" if failed == 0 else "suite.failed",
                ref=self._identity.qualification_suite_id,
                label=self._identity.qualification_suite_version,
            ),
            QualificationEvidence(
                kind=ProviderQualificationEvidenceKind.LIVE_BACKEND,
                code="backend.live",
                label=self._identity.qualification_suite_id,
            ),
            QualificationEvidence(
                kind=ProviderQualificationEvidenceKind.SUITE_EXECUTION,
                code="shared_work.mp2",
                label="work_item,assignment,execution_link",
            ),
            QualificationEvidence(
                kind=ProviderQualificationEvidenceKind.SUITE_EXECUTION,
                code="shared_work.execution_link",
                label="append_only_provenance",
            ),
            *artifact_evidence,
            *concurrency_evidence,
        )
        return ProviderQualificationSuiteOutcome(
            status=status,
            result_summary=ProviderQualificationResultSummary(
                passed=passed,
                failed=failed,
                skipped=skipped,
                label=self._identity.qualification_suite_id,
            ),
            evidence=evidence,
            environment_metadata=self._environment_metadata,
            limitations=self._limitations,
            reproducibility=self._reproducibility,
        )


def collaborative_work_postgresql_repository_qualification_suite() -> (
    CollaborativeWorkRepositoryQualificationSuite
):
    identity = ProviderQualificationSuiteIdentity(
        domain=COLLABORATIVE_WORK_DOMAIN,
        capability_id=COLLABORATIVE_WORK_PERSISTENCE_CAPABILITY,
        qualification_suite_id=CW_POSTGRESQL_REPOSITORY_SUITE_ID,
        qualification_suite_version=CW_REPOSITORY_SUITE_VERSION,
    )
    return CollaborativeWorkRepositoryQualificationSuite(
        _identity=identity,
        _qualified_status=QualificationStatus.PRODUCTION_QUALIFIED,
        _environment_metadata=ProviderQualificationEnvironmentMetadata(
            real_backend=True,
            mocks=False,
            sqlite_substitution=False,
            bounded_environment="docker-postgres-qual-host",
        ),
        _limitations=("bounded qualification schema",),
        _reproducibility=(
            "uv run pytest "
            "tests/integration/core/qualification/"
            "test_provider_qualification_execution_postgresql.py"
        ),
        _requires_shared_work=True,
        _requires_artifacts=True,
        _requires_concurrency_proof=True,
        _requires_artifact_concurrency_proof=True,
    )


def collaborative_work_sqlite_repository_qualification_suite() -> (
    CollaborativeWorkRepositoryQualificationSuite
):
    identity = ProviderQualificationSuiteIdentity(
        domain=COLLABORATIVE_WORK_DOMAIN,
        capability_id=COLLABORATIVE_WORK_PERSISTENCE_CAPABILITY,
        qualification_suite_id=CW_SQLITE_REPOSITORY_SUITE_ID,
        qualification_suite_version=CW_REPOSITORY_SUITE_VERSION,
    )
    return CollaborativeWorkRepositoryQualificationSuite(
        _identity=identity,
        _qualified_status=QualificationStatus.QUALIFIED,
        _environment_metadata=ProviderQualificationEnvironmentMetadata(
            real_backend=True,
            mocks=False,
            sqlite_substitution=False,
            bounded_environment="local-sqlite-lab",
        ),
        _limitations=("local sqlite lab qualification",),
        _reproducibility=(
            "uv run pytest tests/unit/core/qualification/"
            "test_provider_qualification_execution_runner.py::test_sqlite_provider_execution"
        ),
        _requires_shared_work=True,
        _requires_artifacts=True,
        _requires_concurrency_proof=False,
    )


@dataclass(frozen=True, slots=True)
class CollaborativeWorkRepositoryQualificationBinding:
    """Typed domain binding for Collaborative Work repository qualification."""

    _suite: CollaborativeWorkRepositoryQualificationSuite
    _expected_provider_id: str

    @property
    def suite(self) -> ProviderQualificationSuite:
        return self._suite

    def validate_resolved_provider(
        self,
        subject: object,
        *,
        resolved_provider_id: str,
    ) -> None:
        if not isinstance(subject, ProviderQualificationSubject):
            raise TypeError("subject must be ProviderQualificationSubject")
        if subject.domain != COLLABORATIVE_WORK_DOMAIN:
            raise ProviderQualificationSubjectMismatchError(
                "qualification subject domain does not match Collaborative Work binding",
            )
        if subject.capability_id != COLLABORATIVE_WORK_PERSISTENCE_CAPABILITY:
            raise ProviderQualificationSubjectMismatchError(
                "qualification subject capability_id does not match Collaborative Work binding",
            )
        if resolved_provider_id != self._expected_provider_id:
            raise ProviderQualificationSubjectMismatchError(
                "resolved provider_id does not match qualification subject provider_id",
            )
        if subject.provider_id != resolved_provider_id:
            raise ProviderQualificationSubjectMismatchError(
                "qualification subject provider_id does not match resolved provider",
            )

    def materialize(
        self,
        profile: object,
        *,
        resolved_provider_id: str,
    ) -> tuple[object, ProviderQualificationMaterializationHandle]:
        if not isinstance(profile, IntegrationProfile):
            raise TypeError("profile must be IntegrationProfile")
        if resolved_provider_id != self._expected_provider_id:
            raise ProviderQualificationSubjectMismatchError(
                "resolved provider_id does not match Collaborative Work binding",
            )

        bundle = resolve_collaborative_work_repositories(profile)
        return bundle, _CollaborativeWorkMaterializationHandle(bundle)


def collaborative_work_postgresql_repository_qualification_binding() -> (
    CollaborativeWorkRepositoryQualificationBinding
):
    return CollaborativeWorkRepositoryQualificationBinding(
        _suite=collaborative_work_postgresql_repository_qualification_suite(),
        _expected_provider_id="postgresql",
    )


def collaborative_work_sqlite_repository_qualification_binding() -> (
    CollaborativeWorkRepositoryQualificationBinding
):
    return CollaborativeWorkRepositoryQualificationBinding(
        _suite=collaborative_work_sqlite_repository_qualification_suite(),
        _expected_provider_id="sqlite",
    )
