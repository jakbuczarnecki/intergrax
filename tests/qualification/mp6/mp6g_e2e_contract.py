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
    FixedCollaborativeActivityActorPrincipalKindResolver,
)
from intergrax.collaborative_work.collaborative_activity_source_wiring import (
    wire_collaborative_work_service_with_activity_publication,
)
from intergrax.contracts.collaborative_activity import (
    CollaborativeActivityBuiltinType,
    CollaborativeActivityPublication,
    CollaborativeActivityReadPort,
)
from intergrax.contracts.collaborative_activity_ingestion import (
    CollaborativeActivityAdmissionRejected,
    CollaborativeActivityIngestionDenialReason,
    CollaborativeActivityIngestionPolicy,
    CollaborativeActivityIngestionRequest,
    CollaborativeActivityIngestionOutcome,
    CollaborativeActivityIngestionDecision,
)
from intergrax.contracts.collaborative_activity_publisher_authority import (
    CollaborativeActivityPublisherResolutionError,
    restricted_collaborative_activity_workspace_authority,
)
from intergrax.contracts.collaborative_activity_read import CollaborativeActivityReadDenied
from intergrax.contracts.collaborative_work import (
    CreateWorkItemRequest,
    MembershipResolutionMode,
    PrincipalKind,
    TransitionWorkItemRequest,
    WorkItemState,
)
from tests.qualification.mp6.mp6g_harness import (
    PUBLISHER_PRINCIPAL,
    READ_CONSUMER,
    SOURCE_ACTOR,
    TENANT_A,
    TENANT_B,
    WS_A,
    WS_B,
    Mp6gHarness,
    count_durable_activities,
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
    assert count_durable_activities(harness, tenant_id=TENANT_A, workspace_id=WS_A) == 0


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
    assert count_durable_activities(restricted, tenant_id=TENANT_A, workspace_id=WS_B) == 0
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
    types = {a.activity_type for a in page.activities}
    assert CollaborativeActivityBuiltinType.WORK_ITEM_CREATED in types
    assert CollaborativeActivityBuiltinType.WORK_ITEM_STATE_CHANGED in types


def run_mp6g_publication_failure_recovery(
    harness: Mp6gHarness,
    *,
    rebuild_work_service: Callable[[CollaborativeActivityPublication], Mp6gHarness],
) -> None:
    """Source committed → publication fails → retry converges to one activity."""
    stable = "pub-fail-recover"
    wi = "wi-pub-fail"

    class _FailingPort:
        def __init__(self) -> None:
            self.calls = 0
            self._inner = harness.ingestion

        def publish(self, publication: CollaborativeActivityPublication):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("simulated publication port failure")
            return self._inner.publish(publication)

    gate = harness.work_service._inner._enforcement_gate  # noqa: SLF001 — qualification orchestration
    from intergrax.collaborative_work.service import CollaborativeWorkService

    inner = CollaborativeWorkService(
        work_item_repository=harness.bundle.work_item,
        assignment_repository=harness.bundle.assignment,
        enforcement_gate=gate,
        clock=harness.clock,
    )
    resolver = FixedCollaborativeActivityActorPrincipalKindResolver(principal_kind=PrincipalKind.HUMAN)
    failing_port = _FailingPort()
    wrapped = wire_collaborative_work_service_with_activity_publication(
        inner=inner,
        publication_port=failing_port,
        principal_kind_resolver=resolver,
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
    assert count_durable_activities(harness, tenant_id=TENANT_A, workspace_id=WS_A) == 0

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
    assert count_durable_activities(harness, tenant_id=TENANT_A, workspace_id=WS_A) == 1


def run_mp6g_mapping_failure_no_activity(harness: Mp6gHarness) -> None:
    class _FailingMapper(DefaultCollaborativeWorkActivitySourceMapper):
        def map_work_item_created(self, *, request, work_item):  # type: ignore[no-untyped-def]
            raise CollaborativeActivitySourceMappingError("mapper failed for qualification")

    gate = harness.work_service._inner._enforcement_gate  # noqa: SLF001
    from intergrax.collaborative_work.service import CollaborativeWorkService

    inner = CollaborativeWorkService(
        work_item_repository=harness.bundle.work_item,
        assignment_repository=harness.bundle.assignment,
        enforcement_gate=gate,
        clock=harness.clock,
    )
    resolver = FixedCollaborativeActivityActorPrincipalKindResolver(principal_kind=PrincipalKind.HUMAN)
    wrapped = CollaborativeWorkServiceWithActivityPublication(
        inner=inner,
        side_effect=CollaborativeActivitySourcePublicationSideEffect(
            publication_port=harness.ingestion,
        ),
        mapper=_FailingMapper(principal_kind_resolver=resolver),
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
    assert count_durable_activities(harness, tenant_id=TENANT_A, workspace_id=WS_A) == 0


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
    assert count_durable_activities(custom, tenant_id=TENANT_A, workspace_id=WS_A) == 0
    custom.close()


def run_mp6g_concurrent_duplicate_ingestion(
    harness: Mp6gHarness,
    *,
    open_second_harness: Callable[[], Mp6gHarness],
) -> None:
    from intergrax.contracts.collaborative_activity import ActivityIdempotencyKey, CollaborativeActivityBuiltinSource
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
        primary = concurrent_pair_harness_factory()
        try:
            run_mp6g_concurrent_duplicate_ingestion(
                primary,
                open_second_harness=concurrent_pair_harness_factory,
            )
        finally:
            primary.close()
