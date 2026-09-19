# © Artur Czarnecki. All rights reserved.

"""Provider-neutral MP-6G E2E qualification contract."""

from __future__ import annotations

import threading
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest

from intergrax.collaborative_work.collaborative_activity_composition import (
    build_collaborative_activity_ingestion_service,
    build_collaborative_activity_read_service,
)
from intergrax.collaborative_work.collaborative_activity_page_cursor_codec import (
    encode_collaborative_activity_page_cursor,
)
from intergrax.collaborative_work.collaborative_activity_publisher_resolution import (
    DefaultCollaborativeActivityPublisherContextResolver,
)
from intergrax.collaborative_work.collaborative_activity_source_adapters import (
    CollaborativeActivitySourcePublicationSideEffect,
    CollaborativeWorkServiceWithActivityPublication,
)
from intergrax.collaborative_work.collaborative_activity_source_mapping import (
    CollaborativeActivitySourceMappingError,
    DefaultCollaborativeWorkActivitySourceMapper,
)
from intergrax.collaborative_work.collaborative_activity_source_wiring import (
    wire_collaborative_work_service_with_activity_publication,
)
from intergrax.contracts.collaborative_activity import (
    ArtifactVersionActivityProvenanceRef,
    AssignmentActivityTargetRef,
    CollaborativeActivityBuiltinType,
    CollaborativeActivityPublication,
    CollaborativeActivityPublicationPort,
    CollaborativeActivityQuery,
    CollaborativeActivityReadPort,
    CollaborativeDecisionBindingActivityTargetRef,
    ContextViewActivityProvenanceRef,
    ContextViewActivityTargetRef,
    WorkArtifactVersionActivityTargetRef,
    WorkItemActivityTargetRef,
)
from intergrax.contracts.collaborative_activity_read import (
    CollaborativeActivityCursorInvalid,
    CollaborativeActivityReadDenied,
)
from intergrax.contracts.collaborative_activity_ingestion import (
    CollaborativeActivityAdmissionRejected,
    CollaborativeActivityIngestionDenialReason,
    CollaborativeActivityIngestionPolicy,
    CollaborativeActivityIngestionRequest,
    CollaborativeActivityIngestionDecision,
)
from intergrax.contracts.collaborative_activity_publisher_authority import (
    CollaborativeActivityPublisherResolutionError,
)
from intergrax.contracts.agent_run import PrincipalType, RequestIdentity
from intergrax.contracts.collaborative_decision_binding import CreateCollaborativeDecisionBindingRequest
from intergrax.contracts.collaborative_work import (
    ArtifactContentRef,
    AssignmentState,
    CreateAssignmentRequest,
    CreateWorkArtifactRequest,
    CreateWorkItemRequest,
    MembershipResolutionMode,
    PublishWorkArtifactVersionRequest,
    TransitionAssignmentRequest,
    TransitionWorkItemRequest,
    WorkItem,
    WorkItemState,
)
from intergrax.contracts.context_view import (
    ContextViewCategory,
    ContextViewRequest,
    ContextViewScope,
    ContextViewVisibilityClass,
)
from intergrax.contracts.context_view_composition import ContextViewCompositionRequest
from intergrax.contracts.context_view_visibility_policy import (
    ContextViewPolicyDecision,
    ContextViewPolicyOutcome,
    DEFAULT_CONTEXT_VIEW_VISIBILITY_POLICY_ID,
)
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    initial_decision_version,
    mint_decision_id,
)
from intergrax.contracts.decision_record import DecisionProposalRef, decision_lineage_ref
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.collaborative_work.repository import INITIAL_RECORD_REVISION
from tests.qualification.mp6.mp6g_harness import (
    PUBLISHER_PRINCIPAL,
    READ_CONSUMER,
    SOURCE_ACTOR,
    TENANT_A,
    TENANT_B,
    WS_A,
    WS_B,
    Mp6gHarness,
    count_provider_visible_activities,
    read_authorized_page,
    verified_platform_publisher,
)
from tests.unit.collaborative_work.mp6c_publisher_authority_test_support import (
    mapping_authority_source,
    platform_publisher_registration,
)


def _create_work_item(
    harness: Mp6gHarness,
    *,
    tenant_id: str = TENANT_A,
    workspace_id: str = WS_A,
    work_item_id: str,
    idempotency_key: str,
) -> None:
    harness.work_service_for(tenant_id).create_work_item(
        CreateWorkItemRequest(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            work_item_id=work_item_id,
            acting_principal_id=SOURCE_ACTOR,
            idempotency_key=idempotency_key,
            membership_resolution_mode=MembershipResolutionMode.CANONICAL_PRINCIPAL,
        ),
    )


def _assert_happy_path_readback(harness: Mp6gHarness, *, stable_id: str, work_item_id: str) -> None:
    page = read_authorized_page(harness)
    assert len(page.activities) == 1
    activity = page.activities[0]
    assert activity.activity_type == CollaborativeActivityBuiltinType.WORK_ITEM_CREATED
    assert activity.idempotency_key.source_stable_id == stable_id
    assert activity.actor.principal_id == SOURCE_ACTOR
    assert activity.actor.principal_id != PUBLISHER_PRINCIPAL
    assert activity.scope.tenant_id == TENANT_A
    assert activity.scope.workspace_id == WS_A
    assert isinstance(activity.target, WorkItemActivityTargetRef)
    assert activity.target.work_item_id == work_item_id
    assert activity.append_position >= 1
    assert activity.recorded_at.tzinfo is not None
    assert activity.occurred_at.tzinfo is not None


def run_mp6g_work_item_create_happy_path(harness: Mp6gHarness) -> None:
    stable = "mp6g-wi-create-1"
    wi = "wi-mp6g-1"
    _create_work_item(harness, work_item_id=wi, idempotency_key=stable)
    _assert_happy_path_readback(harness, stable_id=stable, work_item_id=wi)


def run_mp6g_work_item_create_replay(harness: Mp6gHarness) -> None:
    stable = "mp6g-replay-key"
    wi = "wi-mp6g-replay"
    _create_work_item(harness, work_item_id=wi, idempotency_key=stable)
    page1 = read_authorized_page(harness)
    first = page1.activities[0]

    _create_work_item(harness, work_item_id=wi, idempotency_key=stable)
    page2 = read_authorized_page(harness)
    assert len(page2.activities) == 1
    replay = page2.activities[0]
    assert replay.activity_id == first.activity_id
    assert replay.append_position == first.append_position
    assert replay.recorded_at == first.recorded_at
    assert replay.durability_class == first.durability_class


def run_mp6g_cross_tenant_isolation(harness: Mp6gHarness) -> None:
    stable = "shared-stable-x"
    _create_work_item(harness, tenant_id=TENANT_A, workspace_id=WS_A, work_item_id="wi-a", idempotency_key=stable)
    _create_work_item(harness, tenant_id=TENANT_B, workspace_id=WS_B, work_item_id="wi-b", idempotency_key=stable)

    page_a = read_authorized_page(harness, tenant_id=TENANT_A, workspace_id=WS_A)
    page_b = read_authorized_page(harness, tenant_id=TENANT_B, workspace_id=WS_B)

    assert len(page_a.activities) == 1
    assert len(page_b.activities) == 1
    assert page_a.activities[0].activity_id != page_b.activities[0].activity_id


def run_mp6g_cross_workspace_isolation(harness: Mp6gHarness) -> None:
    stable = "shared-stable-ws"
    _create_work_item(harness, workspace_id=WS_A, work_item_id="wi-ws-a", idempotency_key=stable)
    _create_work_item(harness, workspace_id=WS_B, work_item_id="wi-ws-b", idempotency_key=stable)

    page_a = read_authorized_page(harness, workspace_id=WS_A)
    page_b = read_authorized_page(harness, workspace_id=WS_B)
    assert len(page_a.activities) == 1
    assert len(page_b.activities) == 1
    assert page_a.activities[0].activity_id != page_b.activities[0].activity_id
    assert page_a.activities[0].append_position == page_b.activities[0].append_position == 1


def run_mp6g_unknown_publisher_denied(harness: Mp6gHarness) -> None:
    unregistered = verified_platform_publisher(TENANT_A, "unregistered-publisher")
    with pytest.raises(CollaborativeActivityPublisherResolutionError):
        build_collaborative_activity_ingestion_service(
            verified_publisher_identity=unregistered,
            publisher_context_resolver=DefaultCollaborativeActivityPublisherContextResolver(
                mapping_authority_source(
                    platform_publisher_registration(TENANT_A, PUBLISHER_PRINCIPAL),
                ),
            ),
            append_store=harness.append_store,
        )
    assert count_provider_visible_activities(harness, tenant_id=TENANT_A, workspace_id=WS_A) == 0


def run_mp6g_wrong_workspace_publisher_denied(
    harness: Mp6gHarness,
    *,
    build_restricted_harness: Callable[[], Mp6gHarness],
) -> None:
    restricted = build_restricted_harness()
    try:
        restricted.work_service_for(TENANT_A).create_work_item(
            CreateWorkItemRequest(
                tenant_id=TENANT_A,
                workspace_id=WS_B,
                work_item_id="wi-denied-ws",
                acting_principal_id=SOURCE_ACTOR,
                idempotency_key="denied-ws-pub",
                membership_resolution_mode=MembershipResolutionMode.CANONICAL_PRINCIPAL,
            ),
        )
        pytest.fail("expected ingestion denial for workspace-restricted publisher")
    except CollaborativeActivityAdmissionRejected as exc:
        assert exc.denial_reason == CollaborativeActivityIngestionDenialReason.WORKSPACE_RESTRICTED
    assert count_provider_visible_activities(restricted, tenant_id=TENANT_A, workspace_id=WS_B) == 0
    restricted.close()


def run_mp6g_unauthorized_read_denied(harness: Mp6gHarness) -> None:
    _create_work_item(harness, work_item_id="wi-unauth", idempotency_key="unauth-read")
    spy = MagicMock(spec=CollaborativeActivityReadPort)
    read_service = build_collaborative_activity_read_service(
        authority_resolver=harness.authority_resolver,
        read_port=spy,
    )
    membership = harness.bundle.membership.get_for_principal(
        tenant_id=TENANT_A, workspace_id=WS_A, principal_id=SOURCE_ACTOR,
    )
    assert membership is not None
    from intergrax.contracts.collaborative_activity import CollaborativeActivityQuery
    from intergrax.contracts.collaborative_activity_read import CollaborativeActivityReadRequest

    with pytest.raises(CollaborativeActivityReadDenied):
        read_service.read_page(
            CollaborativeActivityReadRequest(
                query=CollaborativeActivityQuery(
                    tenant_id=TENANT_A, workspace_id=WS_A, limit=10,
                ),
                acting_principal_id=SOURCE_ACTOR,
                membership_resolution_mode=MembershipResolutionMode.LOCATOR,
                membership=membership,
            ),
        )
    spy.query.assert_not_called()


def run_mp6g_pagination_no_duplicate_skip(harness: Mp6gHarness) -> None:
    for index in range(4):
        _create_work_item(
            harness,
            work_item_id=f"wi-page-{index}",
            idempotency_key=f"page-stable-{index}",
        )
    page1 = read_authorized_page(harness, limit=2)
    assert [a.append_position for a in page1.activities] == [1, 2]
    assert page1.next_cursor is not None

    page2 = read_authorized_page(harness, limit=2, cursor=page1.next_cursor)
    assert [a.append_position for a in page2.activities] == [3, 4]

    all_positions = [a.append_position for a in page1.activities + page2.activities]
    assert all_positions == [1, 2, 3, 4]


def run_mp6g_work_item_transition_e2e(harness: Mp6gHarness) -> None:
    wi = "wi-transition"
    _create_work_item(harness, work_item_id=wi, idempotency_key="transition-create")
    created = harness.bundle.work_item.get(tenant_id=TENANT_A, workspace_id=WS_A, work_item_id=wi)
    assert created is not None
    harness.work_service.transition_work_item(
        TransitionWorkItemRequest(
            tenant_id=TENANT_A,
            workspace_id=WS_A,
            work_item_id=wi,
            expected_revision=created.revision,
            target_state=WorkItemState.ACTIVE,
            acting_principal_id=SOURCE_ACTOR,
            idempotency_key="transition-op",
            membership_resolution_mode=MembershipResolutionMode.CANONICAL_PRINCIPAL,
        ),
    )
    page = read_authorized_page(harness)
    activity_types = [a.activity_type for a in page.activities]
    assert CollaborativeActivityBuiltinType.WORK_ITEM_CREATED in activity_types
    assert CollaborativeActivityBuiltinType.WORK_ITEM_STATE_CHANGED in activity_types


def run_mp6g_publication_failure_recovery(
    harness: Mp6gHarness,
    *,
    rebuild_work_service: Callable[[CollaborativeActivityPublication], Mp6gHarness],
) -> None:
    """Source committed → publication fails → retry converges to one activity."""
    stable = "pub-fail-recover"
    wi = "wi-pub-fail"

    class _FailingPort:
        def __init__(self, delegate: CollaborativeActivityPublicationPort) -> None:
            self.calls = 0
            self._delegate = delegate

        def publish(self, publication: CollaborativeActivityPublication):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("simulated publication port failure")
            return self._delegate.publish(publication)

    failing_port = _FailingPort(harness.ingestion)
    wrapped = wire_collaborative_work_service_with_activity_publication(
        inner=harness.raw_work_service,
        publication_port=failing_port,
        principal_kind_resolver=harness.principal_kind_resolver,
    )
    with pytest.raises(RuntimeError, match="simulated publication port failure"):
        wrapped.create_work_item(
            CreateWorkItemRequest(
                tenant_id=TENANT_A,
                workspace_id=WS_A,
                work_item_id=wi,
                acting_principal_id=SOURCE_ACTOR,
                idempotency_key=stable,
                membership_resolution_mode=MembershipResolutionMode.CANONICAL_PRINCIPAL,
            ),
        )
    assert failing_port.calls == 1
    assert count_provider_visible_activities(harness, tenant_id=TENANT_A, workspace_id=WS_A) == 0

    harness.work_service.create_work_item(
        CreateWorkItemRequest(
            tenant_id=TENANT_A,
            workspace_id=WS_A,
            work_item_id=wi,
            acting_principal_id=SOURCE_ACTOR,
            idempotency_key=stable,
            membership_resolution_mode=MembershipResolutionMode.CANONICAL_PRINCIPAL,
        ),
    )
    assert count_provider_visible_activities(harness, tenant_id=TENANT_A, workspace_id=WS_A) == 1
    stored = harness.bundle.work_item.get(tenant_id=TENANT_A, workspace_id=WS_A, work_item_id=wi)
    assert stored is not None


def run_mp6g_mapping_failure_no_activity(harness: Mp6gHarness) -> None:
    class _FailingMapper(DefaultCollaborativeWorkActivitySourceMapper):
        def map_work_item_created(
            self,
            *,
            request: CreateWorkItemRequest,
            work_item: WorkItem,
        ) -> CollaborativeActivityPublication:
            raise CollaborativeActivitySourceMappingError("mapper failed for qualification")

    wrapped = CollaborativeWorkServiceWithActivityPublication(
        inner=harness.raw_work_service,
        side_effect=CollaborativeActivitySourcePublicationSideEffect(
            publication_port=harness.ingestion,
        ),
        mapper=_FailingMapper(principal_kind_resolver=harness.principal_kind_resolver),
    )
    with pytest.raises(CollaborativeActivitySourceMappingError):
        wrapped.create_work_item(
            CreateWorkItemRequest(
                tenant_id=TENANT_A,
                workspace_id=WS_A,
                work_item_id="wi-map-fail",
                acting_principal_id=SOURCE_ACTOR,
                idempotency_key="map-fail-key",
                membership_resolution_mode=MembershipResolutionMode.CANONICAL_PRINCIPAL,
            ),
        )
    assert count_provider_visible_activities(harness, tenant_id=TENANT_A, workspace_id=WS_A) == 0


_ARTIFACT_DIGEST = "sha256:" + "a" * 64


def _artifact_content_ref(
    *,
    content_ref: str = "content://tenant-a/workspace-a/mp6g-artifact-body",
) -> ArtifactContentRef:
    return ArtifactContentRef(
        content_ref=content_ref,
        media_type="application/json",
        integrity_digest=_ARTIFACT_DIGEST,
    )


def _decision_proposal() -> DecisionProposalRef:
    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="mp6g", subject="qualification"),
        tenant_id=TENANT_A,
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
    )
    return DecisionProposalRef(
        identity=identity,
        lineage_ref=decision_lineage_ref(identity.version),
    )


def _context_view_composition_request() -> ContextViewCompositionRequest:
    scope = ContextViewScope(tenant_id=TENANT_A, workspace_id=WS_A)
    cv_request = ContextViewRequest(
        scope=scope,
        acting_principal_id=SOURCE_ACTOR,
        operation_id="collaborative_work.context_view.compose",
        requested_categories=(ContextViewCategory.MEMORY,),
    )
    decision = ContextViewPolicyDecision(
        outcome=ContextViewPolicyOutcome.ALLOW,
        policy_id=DEFAULT_CONTEXT_VIEW_VISIBILITY_POLICY_ID,
        effective_scope=scope,
        eligible_categories=(ContextViewCategory.MEMORY,),
        eligible_visibility_classes=(ContextViewVisibilityClass.PRIVATE_TO_PRINCIPAL,),
        private_visibility_principal_id=SOURCE_ACTOR,
    )
    return ContextViewCompositionRequest(
        request=cv_request,
        policy_decision=decision,
        principal_identity=RequestIdentity(
            tenant_id=TENANT_A,
            user_id=SOURCE_ACTOR,
            principal_type=PrincipalType.USER,
            auth_subject=SOURCE_ACTOR,
        ),
    )


def run_mp6g_shared_work_source_coverage(harness: Mp6gHarness) -> None:
    wi = "wi-mp6g-assignment-base"
    _create_work_item(harness, work_item_id=wi, idempotency_key="assignment-base-wi")
    assignment_id = "asg-mp6g-1"
    harness.work_service.create_assignment(
        CreateAssignmentRequest(
            tenant_id=TENANT_A,
            workspace_id=WS_A,
            assignment_id=assignment_id,
            work_item_id=wi,
            principal_id=SOURCE_ACTOR,
            acting_principal_id=SOURCE_ACTOR,
            idempotency_key="assignment-create-stable",
            membership_resolution_mode=MembershipResolutionMode.CANONICAL_PRINCIPAL,
        ),
    )
    page = read_authorized_page(harness)
    created = next(
        a for a in page.activities if a.activity_type == CollaborativeActivityBuiltinType.ASSIGNMENT_CREATED
    )
    assert created.idempotency_key.source_stable_id == "assignment-create-stable"
    assert created.actor.principal_id == SOURCE_ACTOR
    assert created.scope.tenant_id == TENANT_A
    assert created.scope.workspace_id == WS_A
    assert isinstance(created.target, AssignmentActivityTargetRef)
    assert created.target.assignment_id == assignment_id
    assert created.occurred_at.tzinfo is not None
    assert created.durability_class is not None

    assignment = harness.bundle.assignment.get(
        tenant_id=TENANT_A,
        workspace_id=WS_A,
        assignment_id=assignment_id,
    )
    assert assignment is not None
    harness.work_service.transition_assignment(
        TransitionAssignmentRequest(
            tenant_id=TENANT_A,
            workspace_id=WS_A,
            assignment_id=assignment_id,
            work_item_id=wi,
            expected_revision=assignment.revision,
            target_state=AssignmentState.COMPLETED,
            acting_principal_id=SOURCE_ACTOR,
            idempotency_key="assignment-transition-stable",
            membership_resolution_mode=MembershipResolutionMode.CANONICAL_PRINCIPAL,
        ),
    )
    page2 = read_authorized_page(harness)
    assert any(
        a.activity_type == CollaborativeActivityBuiltinType.ASSIGNMENT_STATE_CHANGED
        for a in page2.activities
    )


def run_mp6g_artifact_source_coverage(harness: Mp6gHarness) -> None:
    wi = "wi-mp6g-artifact"
    _create_work_item(harness, work_item_id=wi, idempotency_key="artifact-base-wi")
    artifact_id = "artifact-mp6g-1"
    version_1 = "artifact-ver-1"
    harness.artifact_service.create_artifact(
        CreateWorkArtifactRequest(
            tenant_id=TENANT_A,
            workspace_id=WS_A,
            work_item_id=wi,
            work_artifact_id=artifact_id,
            work_artifact_version_id=version_1,
            acting_principal_id=SOURCE_ACTOR,
            content_ref=_artifact_content_ref(),
            idempotency_key="artifact-create-stable",
            membership_resolution_mode=MembershipResolutionMode.CANONICAL_PRINCIPAL,
        ),
    )
    page = read_authorized_page(harness)
    created = next(
        a
        for a in page.activities
        if a.activity_type == CollaborativeActivityBuiltinType.WORK_ARTIFACT_CREATED
    )
    assert created.actor.principal_id == SOURCE_ACTOR
    assert created.scope.workspace_id == WS_A
    assert any(isinstance(ref, ArtifactVersionActivityProvenanceRef) for ref in created.provenance_refs)

    version_2 = "artifact-ver-2"
    harness.artifact_service.publish_version(
        PublishWorkArtifactVersionRequest(
            tenant_id=TENANT_A,
            workspace_id=WS_A,
            work_item_id=wi,
            work_artifact_id=artifact_id,
            work_artifact_version_id=version_2,
            expected_revision=INITIAL_RECORD_REVISION,
            acting_principal_id=SOURCE_ACTOR,
            content_ref=_artifact_content_ref(
                content_ref="content://tenant-a/workspace-a/mp6g-artifact-body-2",
            ),
            idempotency_key="artifact-publish-stable",
            membership_resolution_mode=MembershipResolutionMode.CANONICAL_PRINCIPAL,
        ),
    )
    page2 = read_authorized_page(harness)
    published = next(
        a
        for a in page2.activities
        if a.activity_type == CollaborativeActivityBuiltinType.WORK_ARTIFACT_VERSION_PUBLISHED
    )
    assert published.idempotency_key.source_stable_id == "artifact-publish-stable"
    assert isinstance(published.target, WorkArtifactVersionActivityTargetRef)
    assert published.target.version_ref.work_artifact_version_id == version_2


def run_mp6g_decision_source_coverage(harness: Mp6gHarness) -> None:
    wi = "wi-mp6g-decision"
    _create_work_item(harness, work_item_id=wi, idempotency_key="decision-base-wi")
    harness.decision_binding_service.create_binding(
        CreateCollaborativeDecisionBindingRequest(
            tenant_id=TENANT_A,
            workspace_id=WS_A,
            work_item_id=wi,
            decision_proposal=_decision_proposal(),
            acting_principal_id=SOURCE_ACTOR,
            idempotency_key="decision-binding-stable",
            membership_resolution_mode=MembershipResolutionMode.CANONICAL_PRINCIPAL,
        ),
    )
    page = read_authorized_page(harness)
    created = next(
        a
        for a in page.activities
        if a.activity_type == CollaborativeActivityBuiltinType.COLLABORATIVE_DECISION_BINDING_CREATED
    )
    assert created.actor.principal_id == SOURCE_ACTOR
    assert created.scope.workspace_id == WS_A
    assert isinstance(created.target, CollaborativeDecisionBindingActivityTargetRef)
    assert created.target.binding_id is not None


def run_mp6g_context_view_source_coverage(harness: Mp6gHarness) -> None:
    composition = _context_view_composition_request()
    view = harness.context_view_composer.compose(composition)
    page = read_authorized_page(harness)
    composed = next(
        a for a in page.activities if a.activity_type == CollaborativeActivityBuiltinType.CONTEXT_VIEW_COMPOSED
    )
    assert composed.actor.principal_id == SOURCE_ACTOR
    assert isinstance(composed.target, ContextViewActivityTargetRef)
    assert composed.target.view_id == view.view_id
    assert any(isinstance(ref, ContextViewActivityProvenanceRef) for ref in composed.provenance_refs)
    assert composed.scope.workspace_id == WS_A
    assert composed.occurred_at.tzinfo is not None


def run_mp6g_cursor_scope_mismatch(harness: Mp6gHarness) -> None:
    _create_work_item(harness, work_item_id="wi-cursor-scope", idempotency_key="cursor-scope-1")
    page = read_authorized_page(harness, workspace_id=WS_A)
    foreign_cursor = encode_collaborative_activity_page_cursor(
        query=CollaborativeActivityQuery(
            tenant_id=TENANT_A,
            workspace_id=WS_B,
            limit=10,
        ),
        after_append_position=page.activities[0].append_position,
    )
    membership = harness.bundle.membership.get_for_principal(
        tenant_id=TENANT_A,
        workspace_id=WS_A,
        principal_id=READ_CONSUMER,
    )
    assert membership is not None
    from intergrax.contracts.collaborative_activity_read import CollaborativeActivityReadRequest

    with pytest.raises(CollaborativeActivityCursorInvalid):
        harness.read_service.read_page(
            CollaborativeActivityReadRequest(
                query=CollaborativeActivityQuery(
                    tenant_id=TENANT_A,
                    workspace_id=WS_A,
                    limit=10,
                    cursor=foreign_cursor,
                ),
                acting_principal_id=READ_CONSUMER,
                membership_resolution_mode=MembershipResolutionMode.LOCATOR,
                membership=membership,
            ),
        )


def run_mp6g_cursor_not_authorization_token(harness: Mp6gHarness) -> None:
    for index in range(2):
        _create_work_item(
            harness,
            work_item_id=f"wi-cursor-auth-{index}",
            idempotency_key=f"cursor-auth-{index}",
        )
    page = read_authorized_page(harness, limit=1)
    assert page.next_cursor is not None
    spy = MagicMock(spec=CollaborativeActivityReadPort)
    read_service = build_collaborative_activity_read_service(
        authority_resolver=harness.authority_resolver,
        read_port=spy,
    )
    membership = harness.bundle.membership.get_for_principal(
        tenant_id=TENANT_A,
        workspace_id=WS_A,
        principal_id=SOURCE_ACTOR,
    )
    assert membership is not None
    from intergrax.contracts.collaborative_activity_read import CollaborativeActivityReadRequest

    with pytest.raises(CollaborativeActivityReadDenied):
        read_service.read_page(
            CollaborativeActivityReadRequest(
                query=CollaborativeActivityQuery(
                    tenant_id=TENANT_A,
                    workspace_id=WS_A,
                    limit=10,
                    cursor=page.next_cursor,
                ),
                acting_principal_id=SOURCE_ACTOR,
                membership_resolution_mode=MembershipResolutionMode.LOCATOR,
                membership=membership,
            ),
        )
    spy.query.assert_not_called()


def run_mp6g_late_occurred_at_pagination(harness: Mp6gHarness) -> None:
    """READ/PAGINATION qualification — ingestion publications, not source E2E."""
    from datetime import UTC, datetime

    from tests.unit.collaborative_work.collaborative_activity_append_store_contract import (
        make_intent,
        make_publication,
    )

    early = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    late = datetime(2026, 1, 1, 13, 0, tzinfo=UTC)
    first = harness.ingestion.publish(
        make_intent(
            make_publication(stable_id="late-occurred-1", workspace_id=WS_A).model_copy(
                update={"occurred_at": late},
            ),
        ).publication,
    )
    second = harness.ingestion.publish(
        make_intent(
            make_publication(stable_id="late-occurred-2", workspace_id=WS_A).model_copy(
                update={"occurred_at": early},
            ),
        ).publication,
    )
    assert first.append_position == 1
    assert second.append_position == 2
    page = read_authorized_page(harness, limit=10)
    assert [a.activity_id for a in page.activities] == [first.activity_id, second.activity_id]


def run_mp6g_concurrent_append_between_pages(harness: Mp6gHarness) -> None:
    for index in range(3):
        _create_work_item(
            harness,
            work_item_id=f"wi-between-{index}",
            idempotency_key=f"between-pages-{index}",
        )
    page1 = read_authorized_page(harness, limit=2)
    assert [a.append_position for a in page1.activities] == [1, 2]
    assert page1.next_cursor is not None
    _create_work_item(harness, work_item_id="wi-between-3", idempotency_key="between-pages-3")
    page2 = read_authorized_page(harness, limit=2, cursor=page1.next_cursor)
    assert [a.append_position for a in page2.activities] == [3, 4]


class _DenyAllIngestionPolicy(CollaborativeActivityIngestionPolicy):
    @property
    def policy_id(self) -> str:
        return "mp6g.test.deny_all"

    def evaluate(self, request: CollaborativeActivityIngestionRequest) -> CollaborativeActivityIngestionDecision:
        from intergrax.contracts.collaborative_activity_ingestion import (
            fail_closed_collaborative_activity_ingestion_decision,
        )

        return fail_closed_collaborative_activity_ingestion_decision(
            policy_id=self.policy_id,
            denial_reason=CollaborativeActivityIngestionDenialReason.POLICY_AMBIGUITY,
        )


def run_mp6g_custom_ingestion_policy_pluginability(
    harness: Mp6gHarness,
    *,
    build_with_policy: Callable[[CollaborativeActivityIngestionPolicy], Mp6gHarness],
) -> None:
    custom = build_with_policy(_DenyAllIngestionPolicy())
    try:
        custom.work_service.create_work_item(
            CreateWorkItemRequest(
                tenant_id=TENANT_A,
                workspace_id=WS_A,
                work_item_id="wi-custom-policy",
                acting_principal_id=SOURCE_ACTOR,
                idempotency_key="custom-policy-key",
                membership_resolution_mode=MembershipResolutionMode.CANONICAL_PRINCIPAL,
            ),
        )
        pytest.fail("expected custom policy denial")
    except CollaborativeActivityAdmissionRejected:
        pass
    assert count_provider_visible_activities(custom, tenant_id=TENANT_A, workspace_id=WS_A) == 0
    custom.close()


def run_mp6g_concurrent_duplicate_ingestion(
    harness: Mp6gHarness,
    *,
    open_second_harness: Callable[[], Mp6gHarness],
) -> None:
    from tests.unit.collaborative_work.collaborative_activity_append_store_contract import make_intent, make_publication

    second = open_second_harness()
    pub = make_publication(stable_id="mp6g-race-dup", workspace_id=WS_A)
    intent = make_intent(pub)
    results: list = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def attempt(ingestion) -> None:
        try:
            barrier.wait(timeout=5)
            results.append(ingestion.publish(intent.publication))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=attempt, args=(harness.ingestion,)),
        threading.Thread(target=attempt, args=(second.ingestion,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    second.close()

    assert not errors, errors
    assert len(results) == 2
    assert results[0] == results[1]
    assert results[0].append_position == 1


def run_mp6g_concurrent_distinct_ingestion(
    harness: Mp6gHarness,
    *,
    open_second_harness: Callable[[], Mp6gHarness],
) -> None:
    """MP-6C/MP-6D concurrent distinct append qualification (ingestion path)."""
    from tests.unit.collaborative_work.collaborative_activity_append_store_contract import make_intent, make_publication

    second = open_second_harness()
    pub_a = make_intent(make_publication(stable_id="mp6g-race-distinct-a", workspace_id=WS_A))
    pub_b = make_intent(make_publication(stable_id="mp6g-race-distinct-b", workspace_id=WS_A))
    results: list = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def attempt(ingestion, intent) -> None:
        try:
            barrier.wait(timeout=5)
            results.append(ingestion.publish(intent.publication))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=attempt, args=(harness.ingestion, pub_a)),
        threading.Thread(target=attempt, args=(second.ingestion, pub_b)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    second.close()

    assert not errors, errors
    assert len(results) == 2
    assert results[0].activity_id != results[1].activity_id
    first_pos, second_pos = sorted((results[0].append_position, results[1].append_position))
    assert second_pos - first_pos == 1
    assert results[0].scope.workspace_id == WS_A
    assert results[1].scope.workspace_id == WS_A


def _with_fresh_harness(
    harness_factory: Callable[[], Mp6gHarness],
    scenario: Callable[[Mp6gHarness], None],
) -> None:
    harness = harness_factory()
    try:
        scenario(harness)
    finally:
        harness.close()


def run_mp6g_e2e_contract_suite(
    harness_factory: Callable[[], Mp6gHarness],
    *,
    restricted_harness_factory: Callable[[], Mp6gHarness] | None = None,
    policy_harness_factory: Callable[[CollaborativeActivityIngestionPolicy], Mp6gHarness] | None = None,
    concurrent_pair_harness_factory: Callable[[], Mp6gHarness] | None = None,
    concurrent_distinct_pair_harness_factory: Callable[[], Mp6gHarness] | None = None,
) -> None:
    _with_fresh_harness(harness_factory, run_mp6g_work_item_create_happy_path)
    _with_fresh_harness(harness_factory, run_mp6g_work_item_create_replay)
    _with_fresh_harness(harness_factory, run_mp6g_cross_tenant_isolation)
    _with_fresh_harness(harness_factory, run_mp6g_cross_workspace_isolation)
    _with_fresh_harness(harness_factory, run_mp6g_unknown_publisher_denied)
    if restricted_harness_factory is not None:
        _with_fresh_harness(
            harness_factory,
            lambda h: run_mp6g_wrong_workspace_publisher_denied(
                h,
                build_restricted_harness=restricted_harness_factory,
            ),
        )
    _with_fresh_harness(harness_factory, run_mp6g_unauthorized_read_denied)
    _with_fresh_harness(harness_factory, run_mp6g_pagination_no_duplicate_skip)
    _with_fresh_harness(harness_factory, run_mp6g_work_item_transition_e2e)
    _with_fresh_harness(harness_factory, run_mp6g_shared_work_source_coverage)
    _with_fresh_harness(harness_factory, run_mp6g_artifact_source_coverage)
    _with_fresh_harness(harness_factory, run_mp6g_decision_source_coverage)
    _with_fresh_harness(harness_factory, run_mp6g_context_view_source_coverage)
    _with_fresh_harness(harness_factory, run_mp6g_cursor_scope_mismatch)
    _with_fresh_harness(harness_factory, run_mp6g_cursor_not_authorization_token)
    _with_fresh_harness(harness_factory, run_mp6g_late_occurred_at_pagination)
    _with_fresh_harness(harness_factory, run_mp6g_concurrent_append_between_pages)
    _with_fresh_harness(
        harness_factory,
        lambda h: run_mp6g_publication_failure_recovery(h, rebuild_work_service=lambda _p: h),
    )
    _with_fresh_harness(harness_factory, run_mp6g_mapping_failure_no_activity)
    if policy_harness_factory is not None:
        _with_fresh_harness(
            harness_factory,
            lambda h: run_mp6g_custom_ingestion_policy_pluginability(
                h,
                build_with_policy=policy_harness_factory,
            ),
        )
    if concurrent_pair_harness_factory is not None:
        primary_duplicate = concurrent_pair_harness_factory()
        try:
            run_mp6g_concurrent_duplicate_ingestion(
                primary_duplicate,
                open_second_harness=concurrent_pair_harness_factory,
            )
        finally:
            primary_duplicate.close()
    distinct_factory = concurrent_distinct_pair_harness_factory or concurrent_pair_harness_factory
    if distinct_factory is not None:
        primary_distinct = distinct_factory()
        try:
            run_mp6g_concurrent_distinct_ingestion(
                primary_distinct,
                open_second_harness=distinct_factory,
            )
        finally:
            primary_distinct.close()
