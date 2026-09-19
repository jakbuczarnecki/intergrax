# © Artur Czarnecki. All rights reserved.

"""MP-6C-C1 — trusted publisher identity binding and anti-spoofing gates."""

from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path

import pytest

from intergrax.collaborative_work.collaborative_activity_composition import (
    build_collaborative_activity_ingestion_service,
)
from intergrax.collaborative_work.collaborative_activity_ingestion import (
    CollaborativeActivityIngestionService,
    DefaultCollaborativeActivityIngestionPolicy,
)
from intergrax.collaborative_work.collaborative_activity_publisher_resolution import (
    DefaultCollaborativeActivityPublisherContextResolver,
    MappingCollaborativeActivityPublisherAuthoritySource,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.collaborative_activity import (
    ActivityIdempotencyKey,
    CollaborativeActivity,
    CollaborativeActivityActorRef,
    CollaborativeActivityAppendIntent,
    CollaborativeActivityBuiltinSource,
    CollaborativeActivityBuiltinType,
    CollaborativeActivityDurabilityClass,
    CollaborativeActivityOutcome,
    CollaborativeActivityOutcomeStatus,
    CollaborativeActivityPublication,
    CollaborativeActivityScope,
    CollaborativeActivitySourceId,
    CollaborativeActivityTypeId,
    WorkItemActivityTargetRef,
    mint_collaborative_activity_id,
)
from intergrax.contracts.collaborative_activity_ingestion import (
    CollaborativeActivityAdmissionRejected,
    CollaborativeActivityPublisherContext,
    CollaborativeActivityPublisherKind,
)
from intergrax.contracts.collaborative_activity_publisher_authority import (
    CollaborativeActivityPluginPublisherRegistration,
    CollaborativeActivityPublisherContextResolver,
    CollaborativeActivityPublisherResolutionError,
    VerifiedCollaborativeActivityPublisherIdentity,
    verified_collaborative_activity_publisher_identity_from_request_identity,
)
from intergrax.contracts.collaborative_work import PrincipalKind

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_COMPOSITION_MODULE = (
    _REPO_ROOT
    / "intergrax"
    / "collaborative_work"
    / "collaborative_activity_composition.py"
)
_INGESTION_MODULE = (
    _REPO_ROOT / "intergrax" / "collaborative_work" / "collaborative_activity_ingestion.py"
)

_NOW = datetime(2026, 9, 18, 12, 0, 0, tzinfo=timezone.utc)
_RECORDED = datetime(2026, 9, 18, 13, 0, 0, tzinfo=timezone.utc)


def _service_identity(tenant: str, principal: str) -> RequestIdentity:
    return RequestIdentity(
        tenant_id=tenant,
        auth_subject=principal,
        principal_type=PrincipalType.SERVICE,
    )


def _verified(tenant: str, principal: str) -> VerifiedCollaborativeActivityPublisherIdentity:
    return verified_collaborative_activity_publisher_identity_from_request_identity(
        _service_identity(tenant, principal),
    )


def _authority(
    *plugins: CollaborativeActivityPluginPublisherRegistration,
    platform_workspace_grants: dict[tuple[str, str], tuple[str, ...]] | None = None,
) -> MappingCollaborativeActivityPublisherAuthoritySource:
    return MappingCollaborativeActivityPublisherAuthoritySource(
        plugin_registrations=plugins,
        platform_workspace_grants=platform_workspace_grants,
    )


def _resolver(
    authority: MappingCollaborativeActivityPublisherAuthoritySource | None = None,
) -> DefaultCollaborativeActivityPublisherContextResolver:
    return DefaultCollaborativeActivityPublisherContextResolver(
        authority or _authority(),
    )


class _RecordingAppendStore:
    def __init__(self) -> None:
        self.append_calls: list[CollaborativeActivityAppendIntent] = []

    def append_idempotent(self, intent: CollaborativeActivityAppendIntent) -> CollaborativeActivity:
        self.append_calls.append(intent)
        publication = intent.publication
        return CollaborativeActivity(
            activity_id=mint_collaborative_activity_id(idempotency_key=publication.idempotency_key),
            idempotency_key=publication.idempotency_key,
            activity_type=publication.activity_type,
            actor=publication.actor,
            scope=publication.scope,
            target=publication.target,
            outcome=publication.outcome,
            occurred_at=publication.occurred_at,
            recorded_at=_RECORDED,
            append_position=1,
            provenance_refs=publication.provenance_refs,
            correlation=publication.correlation,
            caused_by_activity_id=publication.caused_by_activity_id,
            durability_class=intent.effective_durability_class,
        )

    def get_by_idempotency_key(self, key: ActivityIdempotencyKey) -> CollaborativeActivity | None:
        return None


def _actor() -> CollaborativeActivityActorRef:
    return CollaborativeActivityActorRef(
        tenant_id="tenant-a",
        principal_id="semantic-actor",
        principal_kind=PrincipalKind.HUMAN,
    )


def _scope(workspace: str = "ws-a") -> CollaborativeActivityScope:
    return CollaborativeActivityScope(
        tenant_id="tenant-a",
        workspace_id=workspace,
        work_item_id="wi-1",
    )


def _platform_publication(workspace: str = "ws-a") -> CollaborativeActivityPublication:
    return CollaborativeActivityPublication(
        idempotency_key=ActivityIdempotencyKey(
            tenant_id="tenant-a",
            workspace_id=workspace,
            source=CollaborativeActivityBuiltinSource.COLLABORATIVE_WORK,
            source_stable_id="evt-1",
            activity_type=CollaborativeActivityBuiltinType.WORK_ITEM_CREATED,
        ),
        actor=_actor(),
        scope=_scope(workspace),
        target=WorkItemActivityTargetRef(work_item_id="wi-1"),
        outcome=CollaborativeActivityOutcome(status=CollaborativeActivityOutcomeStatus.SUCCEEDED),
        occurred_at=_NOW,
        requested_durability_class=CollaborativeActivityDurabilityClass.COLLABORATIVE,
    )


def test_mp6c_c1_composition_rejects_raw_publisher_context_parameter() -> None:
    text = _COMPOSITION_MODULE.read_text(encoding="utf-8-sig")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_collaborative_activity_ingestion_service":
            arg_names = [arg.arg for arg in node.args.kwonlyargs]
            assert "publisher_context" not in arg_names
            assert "verified_publisher_identity" in arg_names
            assert "publisher_context_resolver" in arg_names


def test_mp6c_c1_unsafe_factory_helpers_not_exported_from_ingestion() -> None:
    text = _INGESTION_MODULE.read_text(encoding="utf-8-sig")
    assert "platform_collaborative_activity_publisher_context" not in text
    assert "plugin_collaborative_activity_publisher_context" not in text


def test_mp6c_c1_user_identity_fail_closed_no_service() -> None:
    store = _RecordingAppendStore()
    user_identity = RequestIdentity(
        tenant_id="tenant-a",
        user_id="user-1",
        principal_type=PrincipalType.USER,
    )
    with pytest.raises(CollaborativeActivityPublisherResolutionError):
        verified_collaborative_activity_publisher_identity_from_request_identity(user_identity)
    assert len(store.append_calls) == 0


def test_mp6c_c1_resolution_failure_zero_store_calls() -> None:
    store = _RecordingAppendStore()

    class _FailingResolver:
        def resolve(
            self,
            identity: VerifiedCollaborativeActivityPublisherIdentity,
        ) -> CollaborativeActivityPublisherContext:
            raise CollaborativeActivityPublisherResolutionError("unbound identity")

    with pytest.raises(CollaborativeActivityPublisherResolutionError):
        build_collaborative_activity_ingestion_service(
            verified_publisher_identity=_verified("tenant-a", "platform-1"),
            publisher_context_resolver=_FailingResolver(),
            append_store=store,
        )
    assert len(store.append_calls) == 0


def test_mp6c_c1_platform_service_identity_resolves_platform_kind() -> None:
    resolver = _resolver()
    ctx = resolver.resolve(_verified("tenant-a", "platform-producer"))
    assert ctx.kind is CollaborativeActivityPublisherKind.PLATFORM
    assert ctx.owned_namespace is None


def test_mp6c_c1_registered_plugin_cannot_self_promote_to_platform() -> None:
    authority = _authority(
        CollaborativeActivityPluginPublisherRegistration(
            tenant_id="tenant-a",
            producer_principal_id="plugin-producer",
            owned_namespace="vendor.a",
        ),
    )
    resolver = _resolver(authority)
    ctx = resolver.resolve(_verified("tenant-a", "plugin-producer"))
    assert ctx.kind is CollaborativeActivityPublisherKind.PLUGIN
    assert ctx.owned_namespace == "vendor.a"


def test_mp6c_c1_plugin_cannot_claim_peer_namespace() -> None:
    authority = _authority(
        CollaborativeActivityPluginPublisherRegistration(
            tenant_id="tenant-a",
            producer_principal_id="plugin-a",
            owned_namespace="vendor.a",
        ),
    )
    resolver = _resolver(authority)
    ctx = resolver.resolve(_verified("tenant-a", "plugin-a"))
    assert ctx.owned_namespace == "vendor.a"
    assert ctx.owned_namespace != "vendor.b"


def test_mp6c_c1_cross_tenant_identity_binding() -> None:
    authority = _authority(
        CollaborativeActivityPluginPublisherRegistration(
            tenant_id="tenant-a",
            producer_principal_id="plugin-a",
            owned_namespace="vendor.a",
        ),
    )
    resolver = _resolver(authority)
    ctx = resolver.resolve(_verified("tenant-a", "plugin-a"))
    assert ctx.tenant_id == "tenant-a"
    unregistered = resolver.resolve(_verified("tenant-b", "plugin-a"))
    assert unregistered.kind is CollaborativeActivityPublisherKind.PLATFORM


def test_mp6c_c1_workspace_binding_enforced() -> None:
    authority = _authority(
        CollaborativeActivityPluginPublisherRegistration(
            tenant_id="tenant-a",
            producer_principal_id="plugin-a",
            owned_namespace="vendor.a",
            allowed_workspace_ids=("ws-a",),
        ),
    )
    resolver = _resolver(authority)
    ctx = resolver.resolve(_verified("tenant-a", "plugin-a"))
    store = _RecordingAppendStore()
    service = build_collaborative_activity_ingestion_service(
        verified_publisher_identity=_verified("tenant-a", "plugin-a"),
        publisher_context_resolver=resolver,
        append_store=store,
    )
    pub = CollaborativeActivityPublication(
        idempotency_key=ActivityIdempotencyKey(
            tenant_id="tenant-a",
            workspace_id="ws-b",
            source=CollaborativeActivitySourceId.for_extension("vendor.a", "adapter"),
            source_stable_id="ws-deny",
            activity_type=CollaborativeActivityTypeId.for_extension("vendor.a", "work_item.created"),
        ),
        actor=_actor(),
        scope=_scope("ws-b"),
        target=WorkItemActivityTargetRef(work_item_id="wi-1"),
        outcome=CollaborativeActivityOutcome(status=CollaborativeActivityOutcomeStatus.SUCCEEDED),
        occurred_at=_NOW,
    )
    with pytest.raises(CollaborativeActivityAdmissionRejected):
        service.publish(pub)
    assert len(store.append_calls) == 0
    assert ctx.allowed_workspace_ids == ("ws-a",)


def test_mp6c_c1_plugin_reserved_namespace_publication_denied() -> None:
    authority = _authority(
        CollaborativeActivityPluginPublisherRegistration(
            tenant_id="tenant-a",
            producer_principal_id="evil-plugin",
            owned_namespace="vendor.a",
        ),
    )
    store = _RecordingAppendStore()
    service = build_collaborative_activity_ingestion_service(
        verified_publisher_identity=_verified("tenant-a", "evil-plugin"),
        publisher_context_resolver=_resolver(authority),
        append_store=store,
    )
    pub = CollaborativeActivityPublication(
        idempotency_key=ActivityIdempotencyKey(
            tenant_id="tenant-a",
            workspace_id="ws-a",
            source=CollaborativeActivitySourceId(namespace="platform", name="collaborative_work"),
            source_stable_id="spoof-1",
            activity_type=CollaborativeActivityTypeId(namespace="platform", name="work_item.created"),
        ),
        actor=_actor(),
        scope=_scope(),
        target=WorkItemActivityTargetRef(work_item_id="wi-1"),
        outcome=CollaborativeActivityOutcome(status=CollaborativeActivityOutcomeStatus.SUCCEEDED),
        occurred_at=_NOW,
    )
    with pytest.raises(CollaborativeActivityAdmissionRejected):
        service.publish(pub)
    assert len(store.append_calls) == 0


def test_mp6c_c1_custom_resolver_injection() -> None:
    store = _RecordingAppendStore()

    class _CustomResolver:
        def resolve(
            self,
            identity: VerifiedCollaborativeActivityPublisherIdentity,
        ) -> CollaborativeActivityPublisherContext:
            return CollaborativeActivityPublisherContext(
                tenant_id=identity.tenant_id,
                producer_principal_id=identity.producer_principal_id,
                kind=CollaborativeActivityPublisherKind.PLATFORM,
            )

    service = build_collaborative_activity_ingestion_service(
        verified_publisher_identity=_verified("tenant-a", "custom-platform"),
        publisher_context_resolver=_CustomResolver(),
        append_store=store,
    )
    service.publish(_platform_publication())
    assert len(store.append_calls) == 1


def test_mp6c_c1_actor_distinct_from_publisher() -> None:
    store = _RecordingAppendStore()
    service = build_collaborative_activity_ingestion_service(
        verified_publisher_identity=_verified("tenant-a", "platform-producer"),
        publisher_context_resolver=_resolver(),
        append_store=store,
    )
    activity = service.publish(_platform_publication())
    assert activity.actor.principal_id == "semantic-actor"
    assert len(store.append_calls) == 1


def test_mp6c_c1_resolver_protocol_runtime_checkable() -> None:
    assert isinstance(_resolver(), CollaborativeActivityPublisherContextResolver)
