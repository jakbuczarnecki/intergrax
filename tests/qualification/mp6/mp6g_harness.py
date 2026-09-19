# © Artur Czarnecki. All rights reserved.

"""MP-6G composition harness — production factories only, no store bypass."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

from intergrax.collaborative_work.artifact_service import (
    CollaborativeWorkArtifactService,
    TRUSTED_OPERATION_WORK_ARTIFACT_CREATE,
    TRUSTED_OPERATION_WORK_ARTIFACT_PUBLISH,
)
from intergrax.collaborative_work.authority import CollaborativeWorkAuthorityResolver
from intergrax.collaborative_work.collaborative_activity_composition import (
    build_collaborative_activity_ingestion_service,
    build_collaborative_activity_read_service,
)
from intergrax.collaborative_work.collaborative_activity_read import CollaborativeActivityReadService
from intergrax.collaborative_work.collaborative_activity_publisher_resolution import (
    DefaultCollaborativeActivityPublisherContextResolver,
    MappingCollaborativeActivityPublisherAuthoritySource,
)
from intergrax.collaborative_work.collaborative_activity_source_adapters import (
    CollaborativeDecisionBindingServiceWithActivityPublication,
    CollaborativeWorkArtifactServiceWithActivityPublication,
    CollaborativeWorkServiceWithActivityPublication,
    ContextViewComposerWithActivityPublication,
)
from intergrax.collaborative_work.collaborative_activity_source_mapping import (
    FixedCollaborativeActivityActorPrincipalKindResolver,
)
from intergrax.collaborative_work.collaborative_activity_source_wiring import (
    wire_collaborative_decision_binding_service_with_activity_publication,
    wire_collaborative_work_artifact_service_with_activity_publication,
    wire_collaborative_work_service_with_activity_publication,
    wire_context_view_composer_with_activity_publication,
)
from intergrax.collaborative_work.context_view_composition import DefaultContextViewComposer
from intergrax.collaborative_work.decision_binding_service import (
    CollaborativeDecisionBindingService,
    TRUSTED_OPERATION_COLLABORATIVE_DECISION_BINDING_CREATE,
)
from intergrax.collaborative_work.collaborative_activity_ingestion import (
    CollaborativeActivityIngestionService,
)
from intergrax.collaborative_work.enforcement_gate import CollaborativeWorkEnforcementGate
from intergrax.collaborative_work.persistence import (
    CollaborativeWorkRepositoriesWithArtifacts,
    collaborative_activity_append_store_from_postgresql_bundle,
    collaborative_activity_append_store_from_sqlite_bundle,
    collaborative_activity_read_store_from_postgresql_bundle,
    collaborative_activity_read_store_from_sqlite_bundle,
)
from intergrax.collaborative_work.policy_source import CollaborativePolicyEvaluator
from intergrax.collaborative_work.repository import (
    CreateCollaborativeOperationPolicyProfileCommand,
    CreatePrincipalAuthorityGrantCommand,
    CreateWorkspaceMembershipCommand,
)
from intergrax.collaborative_work.service import (
    CollaborativeWorkService,
    TRUSTED_OPERATION_ASSIGNMENT_CREATE,
    TRUSTED_OPERATION_ASSIGNMENT_TRANSITION,
    TRUSTED_OPERATION_WORK_ITEM_CREATE,
    TRUSTED_OPERATION_WORK_ITEM_TRANSITION,
)
from intergrax.contracts.collaborative_activity_ingestion import CollaborativeActivityIngestionPolicy
from intergrax.contracts.collaborative_activity_publisher_authority import (
    CollaborativeActivityPublisherRegistration,
)
from intergrax.contracts.context_view_source_ports import (
    ContextViewMemorySourceCandidatesResult,
    ContextViewMemorySourceRequest,
    ContextViewSourceOutcome,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.collaborative_activity import (
    CollaborativeActivityAppendStore,
    CollaborativeActivityPageCursor,
    CollaborativeActivityReadPort,
)
from intergrax.contracts.collaborative_activity_publisher_authority import (
    VerifiedCollaborativeActivityPublisherIdentity,
    verified_collaborative_activity_publisher_identity_from_request_identity,
)
from intergrax.contracts.collaborative_activity_read import (
    COLLABORATIVE_ACTIVITY_READ_AUTHORITY_SCOPE,
)
from intergrax.contracts.collaborative_work import (
    AuthorityGrantStatus,
    CollaborativeOperationPolicyProfileStatus,
    MembershipResolutionMode,
    MembershipStatus,
    OperationPolicyRequirement,
    PolicyLayerApplicability,
    PrincipalKind,
    WorkspaceMembershipRole,
)
from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectRequest
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from tests.unit.collaborative_work.mp6c_publisher_authority_test_support import (
    platform_publisher_registration,
)

TENANT_A: Final = "tenant-a"
TENANT_B: Final = "tenant-b"
WS_A: Final = "workspace-a"
WS_B: Final = "workspace-b"
PUBLISHER_PRINCIPAL: Final = "platform-activity-publisher"
SOURCE_ACTOR: Final = "human-user-123"
READ_CONSUMER: Final = "read-consumer-456"
_WORK_AUTHORITY: Final = "collaborative_work.manage"


class _UnusedRuntimeEvaluator:
    def evaluate_meaningful_side_effect(
        self,
        request: MeaningfulSideEffectRequest,
    ) -> PolicyDecision:
        _ = request
        return PolicyDecision(
            action=PolicyAction.DENY,
            reason="runtime evaluator must not run for internal shared-work mutations",
            policy_rule_id="test.unexpected_runtime",
        )


def verified_platform_publisher(tenant_id: str, producer_principal_id: str) -> VerifiedCollaborativeActivityPublisherIdentity:
    return verified_collaborative_activity_publisher_identity_from_request_identity(
        RequestIdentity(
            tenant_id=tenant_id,
            auth_subject=producer_principal_id,
            principal_type=PrincipalType.SERVICE,
        ),
    )


def seed_workspace_principal_authority(
    bundle: CollaborativeWorkRepositoriesWithArtifacts,
    *,
    tenant_id: str,
    workspace_id: str,
    principal_id: str,
    membership_id: str,
    grant_id: str,
    authority_scopes: tuple[str, ...],
    operation_ids: tuple[str, ...],
    clock: Callable[[], datetime],
) -> None:
    _ = clock
    if bundle.membership.get_for_principal(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        principal_id=principal_id,
    ) is None:
        bundle.membership.create(
            CreateWorkspaceMembershipCommand(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                membership_id=membership_id,
                principal_id=principal_id,
                role=WorkspaceMembershipRole.MEMBER,
                status=MembershipStatus.ACTIVE,
            ),
        )
    if bundle.principal_authority.get_for_principal(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        principal_id=principal_id,
    ) is None:
        bundle.principal_authority.create(
            CreatePrincipalAuthorityGrantCommand(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                authority_grant_id=grant_id,
                principal_id=principal_id,
                authority_scopes=authority_scopes,
                status=AuthorityGrantStatus.ACTIVE,
            ),
        )
    for operation_id in operation_ids:
        if (
            bundle.operation_profile.get_for_operation(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                operation_id=operation_id,
            )
            is None
        ):
            bundle.operation_profile.create(
                CreateCollaborativeOperationPolicyProfileCommand(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    operation_id=operation_id,
                    authority_scope=authority_scopes[0],
                    workspace_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
                    resource_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
                    runtime_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
                    resource_requirement=OperationPolicyRequirement.NOT_APPLICABLE,
                    meaningful_side_effect_requirement=OperationPolicyRequirement.NOT_APPLICABLE,
                    status=CollaborativeOperationPolicyProfileStatus.ACTIVE,
                ),
            )


@dataclass(frozen=True, slots=True)
class _Mp6gEmptyMemorySource:
    """Empty Memory port for DefaultContextViewComposer in qualification wiring."""

    def list_candidates(
        self,
        request: ContextViewMemorySourceRequest,
    ) -> ContextViewMemorySourceCandidatesResult:
        _ = request
        return ContextViewMemorySourceCandidatesResult(
            outcome=ContextViewSourceOutcome.OK,
            candidates=(),
        )


def build_enforcement_gate(
    bundle: CollaborativeWorkRepositoriesWithArtifacts,
    *,
    clock: Callable[[], datetime],
) -> CollaborativeWorkEnforcementGate:
    return CollaborativeWorkEnforcementGate(
        profile_repository=bundle.operation_profile,
        authority_resolver=CollaborativeWorkAuthorityResolver(
            membership_repository=bundle.membership,
            delegation_repository=bundle.delegation,
            principal_authority_repository=bundle.principal_authority,
            clock=clock,
        ),
        policy_evaluator=CollaborativePolicyEvaluator(bundle.policy),
        runtime_policy_evaluator=_UnusedRuntimeEvaluator(),
    )


def build_ingestion_service(
    *,
    append_store: CollaborativeActivityAppendStore,
    authority_source: MappingCollaborativeActivityPublisherAuthoritySource,
    verified_publisher: VerifiedCollaborativeActivityPublisherIdentity,
) -> CollaborativeActivityIngestionService:
    resolver = DefaultCollaborativeActivityPublisherContextResolver(authority_source)
    return build_collaborative_activity_ingestion_service(
        verified_publisher_identity=verified_publisher,
        publisher_context_resolver=resolver,
        append_store=append_store,
    )


@dataclass(slots=True)
class Mp6gHarness:
    """Contract-level surface for provider-neutral MP-6G qualification."""

    bundle: CollaborativeWorkRepositoriesWithArtifacts
    append_store: CollaborativeActivityAppendStore
    read_port: CollaborativeActivityReadPort
    ingestion: CollaborativeActivityIngestionService
    ingestion_by_tenant: dict[str, CollaborativeActivityIngestionService]
    read_service: CollaborativeActivityReadService
    enforcement_gate: CollaborativeWorkEnforcementGate
    raw_work_service: CollaborativeWorkService
    principal_kind_resolver: FixedCollaborativeActivityActorPrincipalKindResolver
    work_service: CollaborativeWorkServiceWithActivityPublication
    work_services_by_tenant: dict[str, CollaborativeWorkServiceWithActivityPublication]
    artifact_service: CollaborativeWorkArtifactServiceWithActivityPublication
    artifact_services_by_tenant: dict[str, CollaborativeWorkArtifactServiceWithActivityPublication]
    decision_binding_service: CollaborativeDecisionBindingServiceWithActivityPublication
    decision_binding_services_by_tenant: dict[
        str,
        CollaborativeDecisionBindingServiceWithActivityPublication,
    ]
    context_view_composer: ContextViewComposerWithActivityPublication
    context_view_composers_by_tenant: dict[str, ContextViewComposerWithActivityPublication]
    authority_resolver: CollaborativeWorkAuthorityResolver
    verified_publisher: VerifiedCollaborativeActivityPublisherIdentity
    clock: Callable[[], datetime]
    utc_now: Callable[[], datetime]

    def work_service_for(self, tenant_id: str) -> CollaborativeWorkServiceWithActivityPublication:
        return self.work_services_by_tenant[tenant_id]

    def artifact_service_for(
        self,
        tenant_id: str,
    ) -> CollaborativeWorkArtifactServiceWithActivityPublication:
        return self.artifact_services_by_tenant[tenant_id]

    def decision_binding_service_for(
        self,
        tenant_id: str,
    ) -> CollaborativeDecisionBindingServiceWithActivityPublication:
        return self.decision_binding_services_by_tenant[tenant_id]

    def context_view_composer_for(self, tenant_id: str) -> ContextViewComposerWithActivityPublication:
        return self.context_view_composers_by_tenant[tenant_id]

    def close(self) -> None:
        self.bundle.close()


def build_mp6g_harness_from_sqlite_bundle(
    bundle: CollaborativeWorkRepositoriesWithArtifacts,
    *,
    utc_now: Callable[[], datetime],
    clock: Callable[[], datetime],
    extra_publisher_registrations: tuple[CollaborativeActivityPublisherRegistration, ...] = (),
    publisher_principal_id: str = PUBLISHER_PRINCIPAL,
    publisher_registrations: tuple[CollaborativeActivityPublisherRegistration, ...] | None = None,
    ingestion_policy: CollaborativeActivityIngestionPolicy | None = None,
) -> Mp6gHarness:
    append_store = collaborative_activity_append_store_from_sqlite_bundle(bundle, utc_now=utc_now)
    read_port = collaborative_activity_read_store_from_sqlite_bundle(bundle)
    return _assemble_harness(
        bundle=bundle,
        append_store=append_store,
        read_port=read_port,
        utc_now=utc_now,
        clock=clock,
        extra_publisher_registrations=extra_publisher_registrations,
        publisher_principal_id=publisher_principal_id,
        publisher_registrations=publisher_registrations,
        ingestion_policy=ingestion_policy,
    )


def build_mp6g_harness_from_postgresql_bundle(
    bundle: CollaborativeWorkRepositoriesWithArtifacts,
    *,
    utc_now: Callable[[], datetime],
    clock: Callable[[], datetime],
    extra_publisher_registrations: tuple[CollaborativeActivityPublisherRegistration, ...] = (),
    publisher_principal_id: str = PUBLISHER_PRINCIPAL,
    publisher_registrations: tuple[CollaborativeActivityPublisherRegistration, ...] | None = None,
    ingestion_policy: CollaborativeActivityIngestionPolicy | None = None,
) -> Mp6gHarness:
    append_store = collaborative_activity_append_store_from_postgresql_bundle(bundle, utc_now=utc_now)
    read_port = collaborative_activity_read_store_from_postgresql_bundle(bundle)
    return _assemble_harness(
        bundle=bundle,
        append_store=append_store,
        read_port=read_port,
        utc_now=utc_now,
        clock=clock,
        extra_publisher_registrations=extra_publisher_registrations,
        publisher_principal_id=publisher_principal_id,
        publisher_registrations=publisher_registrations,
        ingestion_policy=ingestion_policy,
    )


def _assemble_harness(
    *,
    bundle: CollaborativeWorkRepositoriesWithArtifacts,
    append_store: CollaborativeActivityAppendStore,
    read_port: CollaborativeActivityReadPort,
    utc_now: Callable[[], datetime],
    clock: Callable[[], datetime],
    extra_publisher_registrations: tuple[CollaborativeActivityPublisherRegistration, ...],
    publisher_principal_id: str = PUBLISHER_PRINCIPAL,
    publisher_registrations: tuple[CollaborativeActivityPublisherRegistration, ...] | None = None,
    ingestion_policy: CollaborativeActivityIngestionPolicy | None = None,
) -> Mp6gHarness:
    seed_workspace_principal_authority(
        bundle,
        tenant_id=TENANT_A,
        workspace_id=WS_A,
        principal_id=SOURCE_ACTOR,
        membership_id="mp6g-membership-actor-a",
        grant_id="mp6g-grant-actor-a",
        authority_scopes=(_WORK_AUTHORITY,),
        operation_ids=(
            TRUSTED_OPERATION_WORK_ITEM_CREATE,
            TRUSTED_OPERATION_WORK_ITEM_TRANSITION,
            TRUSTED_OPERATION_ASSIGNMENT_CREATE,
            TRUSTED_OPERATION_ASSIGNMENT_TRANSITION,
            TRUSTED_OPERATION_WORK_ARTIFACT_CREATE,
            TRUSTED_OPERATION_WORK_ARTIFACT_PUBLISH,
            TRUSTED_OPERATION_COLLABORATIVE_DECISION_BINDING_CREATE,
        ),
        clock=clock,
    )
    seed_workspace_principal_authority(
        bundle,
        tenant_id=TENANT_A,
        workspace_id=WS_A,
        principal_id=READ_CONSUMER,
        membership_id="mp6g-membership-reader-a",
        grant_id="mp6g-grant-reader-a",
        authority_scopes=(COLLABORATIVE_ACTIVITY_READ_AUTHORITY_SCOPE,),
        operation_ids=(),
        clock=clock,
    )
    seed_workspace_principal_authority(
        bundle,
        tenant_id=TENANT_B,
        workspace_id=WS_B,
        principal_id=SOURCE_ACTOR,
        membership_id="mp6g-membership-actor-b",
        grant_id="mp6g-grant-actor-b",
        authority_scopes=(_WORK_AUTHORITY,),
        operation_ids=(TRUSTED_OPERATION_WORK_ITEM_CREATE,),
        clock=clock,
    )
    seed_workspace_principal_authority(
        bundle,
        tenant_id=TENANT_A,
        workspace_id=WS_B,
        principal_id=SOURCE_ACTOR,
        membership_id="mp6g-membership-actor-ws-b",
        grant_id="mp6g-grant-actor-ws-b",
        authority_scopes=(_WORK_AUTHORITY,),
        operation_ids=(TRUSTED_OPERATION_WORK_ITEM_CREATE,),
        clock=clock,
    )
    seed_workspace_principal_authority(
        bundle,
        tenant_id=TENANT_A,
        workspace_id=WS_B,
        principal_id=READ_CONSUMER,
        membership_id="mp6g-membership-reader-ws-b",
        grant_id="mp6g-grant-reader-ws-b",
        authority_scopes=(COLLABORATIVE_ACTIVITY_READ_AUTHORITY_SCOPE,),
        operation_ids=(),
        clock=clock,
    )
    seed_workspace_principal_authority(
        bundle,
        tenant_id=TENANT_B,
        workspace_id=WS_B,
        principal_id=READ_CONSUMER,
        membership_id="mp6g-membership-reader-b",
        grant_id="mp6g-grant-reader-b",
        authority_scopes=(COLLABORATIVE_ACTIVITY_READ_AUTHORITY_SCOPE,),
        operation_ids=(),
        clock=clock,
    )

    if publisher_registrations is None:
        publisher_registrations = (
            platform_publisher_registration(TENANT_A, publisher_principal_id),
            platform_publisher_registration(TENANT_B, publisher_principal_id),
            *extra_publisher_registrations,
        )
    authority_source = MappingCollaborativeActivityPublisherAuthoritySource(
        registrations=publisher_registrations,
    )
    resolver = DefaultCollaborativeActivityPublisherContextResolver(authority_source)
    if publisher_registrations is not None:
        tenant_ids = tuple({registration.tenant_id for registration in publisher_registrations})
    else:
        tenant_ids = (TENANT_A, TENANT_B)
    ingestion_by_tenant: dict[str, CollaborativeActivityIngestionService] = {}
    for tenant_id in tenant_ids:
        verified = verified_platform_publisher(tenant_id, publisher_principal_id)
        if ingestion_policy is None:
            ingestion_by_tenant[tenant_id] = build_collaborative_activity_ingestion_service(
                verified_publisher_identity=verified,
                publisher_context_resolver=resolver,
                append_store=append_store,
            )
        else:
            ingestion_by_tenant[tenant_id] = build_collaborative_activity_ingestion_service(
                verified_publisher_identity=verified,
                publisher_context_resolver=resolver,
                append_store=append_store,
                ingestion_policy=ingestion_policy,
            )
    ingestion = ingestion_by_tenant[TENANT_A]

    gate = build_enforcement_gate(bundle, clock=clock)
    raw_work_service = CollaborativeWorkService(
        work_item_repository=bundle.work_item,
        assignment_repository=bundle.assignment,
        enforcement_gate=gate,
        clock=clock,
    )
    raw_artifact_service = CollaborativeWorkArtifactService(
        work_item_repository=bundle.work_item,
        work_artifact_repository=bundle.artifact,
        artifact_publication_repository=bundle.publication,
        enforcement_gate=gate,
        clock=clock,
    )
    raw_decision_binding_service = CollaborativeDecisionBindingService(
        work_item_repository=bundle.work_item,
        work_artifact_version_repository=bundle.version,
        binding_repository=bundle.decision_binding,
        enforcement_gate=gate,
        clock=clock,
    )
    raw_context_view_composer = DefaultContextViewComposer(
        memory_source=_Mp6gEmptyMemorySource(),
    )
    principal_kind_resolver = FixedCollaborativeActivityActorPrincipalKindResolver(
        principal_kind=PrincipalKind.HUMAN,
    )
    work_services_by_tenant = {
        tenant_id: wire_collaborative_work_service_with_activity_publication(
            inner=raw_work_service,
            publication_port=ingestion_by_tenant[tenant_id],
            principal_kind_resolver=principal_kind_resolver,
        )
        for tenant_id in tenant_ids
    }
    work_service = work_services_by_tenant[TENANT_A]
    artifact_services_by_tenant = {
        tenant_id: wire_collaborative_work_artifact_service_with_activity_publication(
            inner=raw_artifact_service,
            publication_port=ingestion_by_tenant[tenant_id],
            principal_kind_resolver=principal_kind_resolver,
        )
        for tenant_id in tenant_ids
    }
    artifact_service = artifact_services_by_tenant[TENANT_A]
    decision_binding_services_by_tenant = {
        tenant_id: wire_collaborative_decision_binding_service_with_activity_publication(
            inner=raw_decision_binding_service,
            publication_port=ingestion_by_tenant[tenant_id],
            principal_kind_resolver=principal_kind_resolver,
        )
        for tenant_id in tenant_ids
    }
    decision_binding_service = decision_binding_services_by_tenant[TENANT_A]
    context_view_composers_by_tenant = {
        tenant_id: wire_context_view_composer_with_activity_publication(
            inner=raw_context_view_composer,
            publication_port=ingestion_by_tenant[tenant_id],
            principal_kind_resolver=principal_kind_resolver,
        )
        for tenant_id in tenant_ids
    }
    context_view_composer = context_view_composers_by_tenant[TENANT_A]

    authority_resolver = CollaborativeWorkAuthorityResolver(
        membership_repository=bundle.membership,
        delegation_repository=bundle.delegation,
        principal_authority_repository=bundle.principal_authority,
        clock=clock,
    )
    read_service = build_collaborative_activity_read_service(
        authority_resolver=authority_resolver,
        read_port=read_port,
    )

    return Mp6gHarness(
        bundle=bundle,
        append_store=append_store,
        read_port=read_port,
        ingestion=ingestion,
        ingestion_by_tenant=ingestion_by_tenant,
        read_service=read_service,
        enforcement_gate=gate,
        raw_work_service=raw_work_service,
        principal_kind_resolver=principal_kind_resolver,
        work_service=work_service,
        work_services_by_tenant=work_services_by_tenant,
        artifact_service=artifact_service,
        artifact_services_by_tenant=artifact_services_by_tenant,
        decision_binding_service=decision_binding_service,
        decision_binding_services_by_tenant=decision_binding_services_by_tenant,
        context_view_composer=context_view_composer,
        context_view_composers_by_tenant=context_view_composers_by_tenant,
        authority_resolver=authority_resolver,
        verified_publisher=verified_platform_publisher(TENANT_A, publisher_principal_id),
        clock=clock,
        utc_now=utc_now,
    )


def count_provider_visible_activities(harness: Mp6gHarness, *, tenant_id: str, workspace_id: str) -> int:
    from intergrax.contracts.collaborative_activity import CollaborativeActivityQuery

    page = harness.read_port.query(
        CollaborativeActivityQuery(tenant_id=tenant_id, workspace_id=workspace_id, limit=500),
    )
    return len(page.activities)


def read_authorized_page(
    harness: Mp6gHarness,
    *,
    tenant_id: str = TENANT_A,
    workspace_id: str = WS_A,
    acting_principal_id: str = READ_CONSUMER,
    limit: int = 50,
    cursor: CollaborativeActivityPageCursor | None = None,
):
    from intergrax.contracts.collaborative_activity import CollaborativeActivityPage, CollaborativeActivityQuery
    from intergrax.contracts.collaborative_activity_read import CollaborativeActivityReadRequest

    membership = harness.bundle.membership.get_for_principal(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        principal_id=acting_principal_id,
    )
    assert membership is not None
    service = harness.read_service
    query = CollaborativeActivityQuery(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        limit=limit,
        cursor=cursor,
    )
    page: CollaborativeActivityPage = service.read_page(
        CollaborativeActivityReadRequest(
            query=query,
            acting_principal_id=acting_principal_id,
            membership_resolution_mode=MembershipResolutionMode.LOCATOR,
            membership=membership,
        ),
    )
    return page


MP6G_FIXED_CLOCK = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)
MP6G_FIXED_UTC_NOW = datetime(2026, 9, 19, 12, 5, tzinfo=UTC)


def mp6g_fixed_clock() -> datetime:
    return MP6G_FIXED_CLOCK


def mp6g_fixed_utc_now() -> datetime:
    return MP6G_FIXED_UTC_NOW
