# © Artur Czarnecki. All rights reserved.

"""Collaborative Work repository provider qualification suite (PROVIDER-QUAL-7)."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from intergrax.collaborative_work.persistence import (
    CollaborativeWorkMaterializedRepositories,
    CollaborativeWorkRepositories,
    CollaborativeWorkRepositoriesWithSharedWork,
    open_postgresql_collaborative_work_repositories,
)
from intergrax.collaborative_work.persistence_provider import (
    resolve_collaborative_work_repositories,
)
from intergrax.collaborative_work.postgresql_repository import PostgreSQLCollaborativeWorkStore
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
    CreateWorkspaceMembershipCommand,
    INITIAL_RECORD_REVISION,
    PrincipalAuthorityGrantAlreadyExists,
    UpdateAssignmentCommand,
    UpdateAuthorityDelegationCommand,
    UpdateCollaborativeOperationPolicyProfileCommand,
    UpdateWorkItemCommand,
    UpdateWorkspaceMembershipCommand,
    WorkItemAlreadyExists,
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
CW_REPOSITORY_SUITE_VERSION = "2.0.0"

_TENANT_A = "qual-tenant-a"
_TENANT_B = "qual-tenant-b"
_WORKSPACE_A = "qual-workspace-a"
_WORKSPACE_B = "qual-workspace-b"
_VALID_FROM = datetime(2026, 1, 1, tzinfo=UTC)
_VALID_UNTIL = datetime(2026, 12, 31, tzinfo=UTC)
_CREATED_AT = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
_UPDATED_AT = datetime(2026, 1, 1, 12, 30, tzinfo=UTC)


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


def _run_shared_work_repository_contract_checks(
    bundle: CollaborativeWorkRepositoriesWithSharedWork,
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

    for check in (
        _work_item_create_get_isolation,
        _work_item_duplicate_stale_and_idempotency,
        _assignment_create_update_idempotency,
    ):
        _run_check(check)

    return passed, failed


def _run_shared_work_cross_connection_concurrency_check(
    bundle: CollaborativeWorkRepositoriesWithSharedWork,
) -> tuple[int, int]:
    store = bundle.store
    if not isinstance(store, PostgreSQLCollaborativeWorkStore):
        raise _RepositorySemanticCheckFailure(
            "cross-connection concurrency requires PostgreSQLCollaborativeWorkStore",
        )

    work_item_repo = bundle.work_item
    created = work_item_repo.create(_work_item_command(work_item_id="qual-work-item-concurrency"))
    bundle_b = open_postgresql_collaborative_work_repositories(
        config=store.config,
        schema_name=store.schema_name,
    )
    try:
        read_a = work_item_repo.get(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            work_item_id="qual-work-item-concurrency",
        )
        read_b = bundle_b.work_item.get(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            work_item_id="qual-work-item-concurrency",
        )
        if read_a is None or read_b is None:
            raise _RepositorySemanticCheckFailure("concurrency pre-read missing work item")
        if read_a.revision != read_b.revision != created.revision:
            raise _RepositorySemanticCheckFailure("concurrency revision baseline mismatch")

        errors: list[BaseException] = []
        barrier = threading.Barrier(2)

        def attempt(target: CollaborativeWorkRepositoriesWithSharedWork) -> None:
            try:
                barrier.wait(timeout=5)
                target.work_item.update(
                    UpdateWorkItemCommand(
                        scope=WorkItemScopeKey(
                            tenant_id=_TENANT_A,
                            workspace_id=_WORKSPACE_A,
                            work_item_id="qual-work-item-concurrency",
                        ),
                        expected_revision=created.revision,
                        state=WorkItemState.ACTIVE,
                        updated_at=_UPDATED_AT,
                    ),
                )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=attempt, args=(bundle,)),
            threading.Thread(target=attempt, args=(bundle_b,)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        if len(errors) != 1:
            raise _RepositorySemanticCheckFailure("expected exactly one revision conflict")
        if not isinstance(errors[0], WorkItemRevisionConflict):
            raise _RepositorySemanticCheckFailure("expected WorkItemRevisionConflict")
        final = work_item_repo.get(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            work_item_id="qual-work-item-concurrency",
        )
        if final is None or final.revision != created.revision + 1:
            raise _RepositorySemanticCheckFailure("concurrency final revision mismatch")
    finally:
        bundle_b.close()

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
    _requires_concurrency_proof: bool = False

    @property
    def identity(self) -> ProviderQualificationSuiteIdentity:
        return self._identity

    def execute(self, capability: object) -> ProviderQualificationSuiteOutcome:
        if isinstance(capability, CollaborativeWorkRepositoriesWithSharedWork):
            core_passed, core_failed = _run_core_repository_contract_checks(capability.core)
            shared_passed, shared_failed = _run_shared_work_repository_contract_checks(capability)
            passed = core_passed + shared_passed
            failed = core_failed + shared_failed
            concurrency_evidence: tuple[QualificationEvidence, ...] = ()
            if self._requires_concurrency_proof:
                try:
                    conc_passed, conc_failed = _run_shared_work_cross_connection_concurrency_check(
                        capability,
                    )
                except _RepositorySemanticCheckFailure:
                    conc_passed, conc_failed = 0, 1
                passed += conc_passed
                failed += conc_failed
                concurrency_evidence = (
                    QualificationEvidence(
                        kind=ProviderQualificationEvidenceKind.SUITE_EXECUTION,
                        code="shared_work.concurrency.cross_connection",
                        label="transactional_cas",
                    ),
                )
        elif isinstance(capability, CollaborativeWorkRepositories):
            if self._requires_shared_work:
                raise ProviderQualificationSuiteInfrastructureError(
                    "capability must include MP-2 Shared Work repositories",
                )
            passed, failed = _run_core_repository_contract_checks(capability)
            concurrency_evidence = ()
        else:
            raise ProviderQualificationSuiteInfrastructureError(
                "capability must be CollaborativeWorkRepositories "
                "or CollaborativeWorkRepositoriesWithSharedWork",
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
                label="work_item,assignment",
            ),
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
        _requires_concurrency_proof=True,
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
