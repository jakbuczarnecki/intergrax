# © Artur Czarnecki. All rights reserved.

"""Collaborative Work repository provider qualification suite (PROVIDER-QUAL-7)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from intergrax.collaborative_work.persistence import CollaborativeWorkRepositories
from intergrax.collaborative_work.persistence_provider import (
    resolve_collaborative_work_repositories,
)
from intergrax.collaborative_work.repository import (
    AuthorityDelegationScopeKey,
    CollaborativeOperationPolicyProfileScopeKey,
    CreateAuthorityDelegationCommand,
    CreateCollaborativeOperationPolicyProfileCommand,
    CreateCollaborativePolicyRuleCommand,
    CreatePrincipalAuthorityGrantCommand,
    CreateWorkspaceMembershipCommand,
    INITIAL_RECORD_REVISION,
    PrincipalAuthorityGrantAlreadyExists,
    UpdateAuthorityDelegationCommand,
    UpdateCollaborativeOperationPolicyProfileCommand,
    UpdateWorkspaceMembershipCommand,
    WorkspaceMembershipAlreadyExists,
    WorkspaceMembershipRevisionConflict,
    WorkspaceMembershipScopeKey,
    CollaborativePolicyRuleAlreadyExists,
    CollaborativeOperationPolicyProfileRevisionConflict,
)
from intergrax.contracts.collaborative_work import (
    AuthorityGrantStatus,
    CollaborativeOperationPolicyProfileStatus,
    CollaborativePolicyRuleStatus,
    DelegationStatus,
    MembershipStatus,
    OperationPolicyRequirement,
    PolicyCompositionLayer,
    PolicyLayerApplicability,
    WorkspaceMembershipRole,
)
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.core.qualification.evidence import QualificationEvidence
from intergrax.core.qualification.execution import ProviderQualificationSubjectMismatchError
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
from intergrax.integrations.registry.factory import resolve_slug

COLLABORATIVE_WORK_DOMAIN = "collaborative_work"
COLLABORATIVE_WORK_PERSISTENCE_CAPABILITY = "collaborative_work.persistence.v1"

CW_POSTGRESQL_REPOSITORY_SUITE_ID = "cw.postgresql.repository.v1"
CW_SQLITE_REPOSITORY_SUITE_ID = "cw.sqlite.repository.v1"
CW_REPOSITORY_SUITE_VERSION = "1.0.0"

_TENANT_A = "qual-tenant-a"
_TENANT_B = "qual-tenant-b"
_WORKSPACE_A = "qual-workspace-a"
_WORKSPACE_B = "qual-workspace-b"
_VALID_FROM = datetime(2026, 1, 1, tzinfo=UTC)
_VALID_UNTIL = datetime(2026, 12, 31, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class PostgreSQLQualificationMaterializationOptions:
    """Provider-owned PostgreSQL qualification materialization options."""

    schema_name: str

    @property
    def postgresql_schema_name(self) -> str:
        return self.schema_name


@dataclass(frozen=True, slots=True)
class _CollaborativeWorkMaterializationHandle:
    _bundle: CollaborativeWorkRepositories

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


def _run_repository_contract_checks(bundle: CollaborativeWorkRepositories) -> tuple[int, int]:
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
        except Exception:
            _record_failure()

    membership_repo = bundle.membership
    delegation_repo = bundle.delegation
    authority_repo = bundle.principal_authority
    policy_repo = bundle.policy
    profile_repo = bundle.operation_profile

    def _membership_create_get_revision_and_isolation() -> None:
        created = membership_repo.create(_membership_command())
        assert created.revision == INITIAL_RECORD_REVISION
        loaded = membership_repo.get(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            membership_id="qual-membership-1",
        )
        assert loaded == created
        assert (
            membership_repo.get(
                tenant_id=_TENANT_B,
                workspace_id=_WORKSPACE_B,
                membership_id="qual-membership-1",
            )
            is None
        )

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
            raise AssertionError("expected WorkspaceMembershipAlreadyExists")
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
            raise AssertionError("expected WorkspaceMembershipRevisionConflict")
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
        assert delegation_repo.create(command) == created
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
        assert updated.revision == created.revision + 1
        assert delegation_repo.create(command) == created

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
            raise AssertionError("expected PrincipalAuthorityGrantAlreadyExists")
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
            raise AssertionError("expected CollaborativePolicyRuleAlreadyExists")
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
            raise AssertionError("expected CollaborativeOperationPolicyProfileRevisionConflict")
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


@dataclass(frozen=True, slots=True)
class CollaborativeWorkRepositoryQualificationSuite:
    """Domain-owned repository qualification suite for Collaborative Work persistence."""

    _identity: ProviderQualificationSuiteIdentity
    _qualified_status: QualificationStatus
    _environment_metadata: ProviderQualificationEnvironmentMetadata
    _limitations: tuple[str, ...]
    _reproducibility: str

    @property
    def identity(self) -> ProviderQualificationSuiteIdentity:
        return self._identity

    def execute(self, capability: object) -> ProviderQualificationSuiteOutcome:
        if not isinstance(capability, CollaborativeWorkRepositories):
            raise TypeError("capability must be CollaborativeWorkRepositories")

        passed, failed = _run_repository_contract_checks(capability)
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
    )


@dataclass(frozen=True, slots=True)
class CollaborativeWorkRepositoryQualificationBinding:
    """Typed domain binding for Collaborative Work repository qualification."""

    _suite: CollaborativeWorkRepositoryQualificationSuite
    _expected_provider_id: str
    _materialization_options: PostgreSQLQualificationMaterializationOptions | None = None

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

        options = self._materialization_options
        if (
            options is not None
            and options.postgresql_schema_name is not None
            and resolved_provider_id == "postgresql"
        ):
            from intergrax.collaborative_work.persistence import (
                open_postgresql_collaborative_work_repositories,
            )
            from intergrax.integrations._shared.config import merge_config
            from intergrax.integrations.contracts.base import IntegrationCategory
            from intergrax.integrations.providers.relational_store.postgresql.bundle import (
                resolve_postgresql_config,
            )

            slug = resolve_slug(IntegrationCategory.RELATIONAL_STORE, profile=profile)
            merged = merge_config(profile.options_for_slug(slug), None)
            config = resolve_postgresql_config(
                **{key: value for key, value in merged.items() if value is not None},
            )
            bundle = open_postgresql_collaborative_work_repositories(
                config=config,
                schema_name=options.postgresql_schema_name,
            )
        else:
            bundle = resolve_collaborative_work_repositories(profile)

        return bundle, _CollaborativeWorkMaterializationHandle(bundle)


def collaborative_work_postgresql_repository_qualification_binding(
    *,
    materialization_options: PostgreSQLQualificationMaterializationOptions | None = None,
) -> CollaborativeWorkRepositoryQualificationBinding:
    return CollaborativeWorkRepositoryQualificationBinding(
        _suite=collaborative_work_postgresql_repository_qualification_suite(),
        _expected_provider_id="postgresql",
        _materialization_options=materialization_options,
    )


def collaborative_work_sqlite_repository_qualification_binding() -> (
    CollaborativeWorkRepositoryQualificationBinding
):
    return CollaborativeWorkRepositoryQualificationBinding(
        _suite=collaborative_work_sqlite_repository_qualification_suite(),
        _expected_provider_id="sqlite",
    )
