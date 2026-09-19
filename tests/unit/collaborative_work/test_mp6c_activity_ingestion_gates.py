# © Artur Czarnecki. All rights reserved.

"""MP-6C — publication / ingestion boundary gates."""

from __future__ import annotations

import ast
from collections import defaultdict
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
    CollaborativeActivityIngestionDecision,
    CollaborativeActivityIngestionDenialReason,
    CollaborativeActivityIngestionOutcome,
    CollaborativeActivityIngestionPolicy,
    CollaborativeActivityIngestionPolicyError,
    CollaborativeActivityIngestionRequest,
    CollaborativeActivityPublisherContext,
    CollaborativeActivityPublisherKind,
    fail_closed_collaborative_activity_ingestion_decision,
)
from intergrax.contracts.collaborative_activity_publisher_authority import (
    CollaborativeActivityPluginPublisherRegistration,
    verified_collaborative_activity_publisher_identity_from_request_identity,
)
from intergrax.contracts.collaborative_work import PrincipalKind

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_INGESTION_MODULE = (
    _REPO_ROOT / "intergrax" / "collaborative_work" / "collaborative_activity_ingestion.py"
)
_INGESTION_CONTRACT = (
    _REPO_ROOT / "intergrax" / "contracts" / "collaborative_activity_ingestion.py"
)

_NOW = datetime(2026, 9, 18, 12, 0, 0, tzinfo=timezone.utc)
_RECORDED = datetime(2026, 9, 18, 13, 0, 0, tzinfo=timezone.utc)

_FORBIDDEN_IMPORT_MARKERS = (
    "sqlalchemy",
    "psycopg",
    "kafka",
    "openai",
    "anthropic",
    "agents.",
    "applications.",
)


def _actor(tenant: str = "tenant-a", principal: str = "actor-principal") -> CollaborativeActivityActorRef:
    return CollaborativeActivityActorRef(
        tenant_id=tenant,
        principal_id=principal,
        principal_kind=PrincipalKind.HUMAN,
    )


def _scope(tenant: str = "tenant-a", workspace: str = "ws-a") -> CollaborativeActivityScope:
    return CollaborativeActivityScope(
        tenant_id=tenant,
        workspace_id=workspace,
        work_item_id="wi-1",
    )


def _platform_publication(
    *,
    tenant: str = "tenant-a",
    workspace: str = "ws-a",
    source_stable_id: str = "stable-1",
    activity_type: CollaborativeActivityTypeId = CollaborativeActivityBuiltinType.WORK_ITEM_CREATED,
    requested_durability: CollaborativeActivityDurabilityClass = (
        CollaborativeActivityDurabilityClass.COLLABORATIVE
    ),
    actor_principal: str = "actor-principal",
) -> CollaborativeActivityPublication:
    return CollaborativeActivityPublication(
        idempotency_key=ActivityIdempotencyKey(
            tenant_id=tenant,
            workspace_id=workspace,
            source=CollaborativeActivityBuiltinSource.COLLABORATIVE_WORK,
            source_stable_id=source_stable_id,
            activity_type=activity_type,
        ),
        actor=_actor(tenant, actor_principal),
        scope=_scope(tenant, workspace),
        target=WorkItemActivityTargetRef(work_item_id="wi-1"),
        outcome=CollaborativeActivityOutcome(status=CollaborativeActivityOutcomeStatus.SUCCEEDED),
        occurred_at=_NOW,
        requested_durability_class=requested_durability,
    )


class _RecordingAppendStore:
    def __init__(self, *, recorded_at: datetime = _RECORDED) -> None:
        self._recorded_at = recorded_at
        self._by_key: dict[str, CollaborativeActivity] = {}
        self._next_position: dict[tuple[str, str], int] = defaultdict(lambda: 1)
        self.append_calls: list[CollaborativeActivityAppendIntent] = []

    def append_idempotent(self, intent: CollaborativeActivityAppendIntent) -> CollaborativeActivity:
        self.append_calls.append(intent)
        publication = intent.publication
        activity_id = mint_collaborative_activity_id(idempotency_key=publication.idempotency_key)
        existing = self._by_key.get(activity_id)
        if existing is not None:
            return existing
        workspace_key = (publication.scope.tenant_id, publication.scope.workspace_id)
        position = self._next_position[workspace_key]
        self._next_position[workspace_key] = position + 1
        materialized = CollaborativeActivity(
            activity_id=activity_id,
            idempotency_key=publication.idempotency_key,
            activity_type=publication.activity_type,
            actor=publication.actor,
            scope=publication.scope,
            target=publication.target,
            outcome=publication.outcome,
            occurred_at=publication.occurred_at,
            recorded_at=self._recorded_at,
            append_position=position,
            provenance_refs=publication.provenance_refs,
            correlation=publication.correlation,
            caused_by_activity_id=publication.caused_by_activity_id,
            durability_class=intent.effective_durability_class,
        )
        self._by_key[activity_id] = materialized
        return materialized

    def get_by_idempotency_key(self, key: ActivityIdempotencyKey) -> CollaborativeActivity | None:
        activity_id = mint_collaborative_activity_id(idempotency_key=key)
        return self._by_key.get(activity_id)


def _service_request_identity(tenant: str, producer: str) -> RequestIdentity:
    return RequestIdentity(
        tenant_id=tenant,
        auth_subject=producer,
        principal_type=PrincipalType.SERVICE,
    )


def _default_resolver(
    *plugins: CollaborativeActivityPluginPublisherRegistration,
) -> DefaultCollaborativeActivityPublisherContextResolver:
    return DefaultCollaborativeActivityPublisherContextResolver(
        MappingCollaborativeActivityPublisherAuthoritySource(plugin_registrations=plugins),
    )


def _platform_service(
    store: _RecordingAppendStore,
    *,
    tenant: str = "tenant-a",
    producer: str = "platform-producer-1",
    policy: CollaborativeActivityIngestionPolicy | None = None,
) -> CollaborativeActivityIngestionService:
    verified = verified_collaborative_activity_publisher_identity_from_request_identity(
        _service_request_identity(tenant, producer),
    )
    return build_collaborative_activity_ingestion_service(
        verified_publisher_identity=verified,
        publisher_context_resolver=_default_resolver(),
        append_store=store,
        ingestion_policy=policy,
    )


def _direct_publisher_context(
    *,
    tenant_id: str,
    producer_principal_id: str,
    kind: CollaborativeActivityPublisherKind,
    owned_namespace: str | None = None,
    allowed_workspace_ids: tuple[str, ...] = (),
) -> CollaborativeActivityPublisherContext:
    """Test-only policy fixture — not a trusted production boundary."""
    return CollaborativeActivityPublisherContext(
        tenant_id=tenant_id,
        producer_principal_id=producer_principal_id,
        kind=kind,
        owned_namespace=owned_namespace,
        allowed_workspace_ids=allowed_workspace_ids,
    )


def test_mp6c_happy_path_platform_publish() -> None:
    store = _RecordingAppendStore()
    service = _platform_service(store)
    pub = _platform_publication()
    activity = service.publish(pub)
    assert len(store.append_calls) == 1
    assert activity.durability_class == CollaborativeActivityDurabilityClass.COLLABORATIVE
    assert activity.actor.principal_id == "actor-principal"


def test_mp6c_policy_deny_zero_store_calls() -> None:
    store = _RecordingAppendStore()
    service = _platform_service(store, tenant="tenant-b")
    with pytest.raises(CollaborativeActivityAdmissionRejected) as exc_info:
        service.publish(_platform_publication(tenant="tenant-a"))
    assert exc_info.value.denial_reason is CollaborativeActivityIngestionDenialReason.TENANT_MISMATCH
    assert len(store.append_calls) == 0


class _RaisingPolicy:
    @property
    def policy_id(self) -> str:
        return "test.raising"

    def evaluate(self, request: CollaborativeActivityIngestionRequest) -> CollaborativeActivityIngestionDecision:
        raise RuntimeError("policy exploded")


def test_mp6c_policy_exception_fail_closed() -> None:
    store = _RecordingAppendStore()
    service = _platform_service(store, policy=_RaisingPolicy())
    with pytest.raises(CollaborativeActivityIngestionPolicyError):
        service.publish(_platform_publication())
    assert len(store.append_calls) == 0


class _BrokenStore(_RecordingAppendStore):
    def append_idempotent(self, intent: CollaborativeActivityAppendIntent) -> CollaborativeActivity:
        raise OSError("store down")


def test_mp6c_append_store_failure_propagates() -> None:
    store = _BrokenStore()
    service = _platform_service(store)
    with pytest.raises(Exception, match="append store failed"):
        service.publish(_platform_publication())


def test_mp6c_requested_informational_upgraded_to_audit_critical() -> None:
    store = _RecordingAppendStore()
    service = _platform_service(store)
    pub = _platform_publication(
        activity_type=CollaborativeActivityBuiltinType.AUTHORITY_RELEVANT_ACTION,
        requested_durability=CollaborativeActivityDurabilityClass.INFORMATIONAL,
    )
    activity = service.publish(pub)
    assert activity.durability_class == CollaborativeActivityDurabilityClass.AUDIT_CRITICAL
    assert store.append_calls[0].effective_durability_class == CollaborativeActivityDurabilityClass.AUDIT_CRITICAL


def test_mp6c_replay_policy_reevaluated_store_idempotent() -> None:
    store = _RecordingAppendStore()
    service = _platform_service(store)
    pub = _platform_publication(source_stable_id="replay-1")
    first = service.publish(pub)
    second = service.publish(pub)
    assert first.activity_id == second.activity_id
    assert len(store.append_calls) == 2


def test_mp6c_unauthorized_replay_denied_no_store() -> None:
    store = _RecordingAppendStore()
    authorized = _platform_service(store, producer="producer-a")
    pub = _platform_publication(source_stable_id="replay-deny")
    authorized.publish(pub)
    assert len(store.append_calls) == 1

    unauthorized = build_collaborative_activity_ingestion_service(
        verified_publisher_identity=verified_collaborative_activity_publisher_identity_from_request_identity(
            _service_request_identity("tenant-b", "producer-b"),
        ),
        publisher_context_resolver=_default_resolver(),
        append_store=store,
    )
    with pytest.raises(CollaborativeActivityAdmissionRejected):
        unauthorized.publish(pub)
    assert len(store.append_calls) == 1


def test_mp6c_plugin_own_namespace_allowed() -> None:
    store = _RecordingAppendStore()
    service = build_collaborative_activity_ingestion_service(
        verified_publisher_identity=verified_collaborative_activity_publisher_identity_from_request_identity(
            _service_request_identity("tenant-a", "plugin-producer"),
        ),
        publisher_context_resolver=_default_resolver(
            CollaborativeActivityPluginPublisherRegistration(
                tenant_id="tenant-a",
                producer_principal_id="plugin-producer",
                owned_namespace="vendor.a",
            ),
        ),
        append_store=store,
    )
    pub = CollaborativeActivityPublication(
        idempotency_key=ActivityIdempotencyKey(
            tenant_id="tenant-a",
            workspace_id="ws-a",
            source=CollaborativeActivitySourceId.for_extension("vendor.a", "adapter"),
            source_stable_id="evt-1",
            activity_type=CollaborativeActivityTypeId.for_extension("vendor.a", "work_item.created"),
        ),
        actor=_actor(),
        scope=_scope(),
        target=WorkItemActivityTargetRef(work_item_id="wi-1"),
        outcome=CollaborativeActivityOutcome(status=CollaborativeActivityOutcomeStatus.SUCCEEDED),
        occurred_at=_NOW,
    )
    service.publish(pub)
    assert len(store.append_calls) == 1


def test_mp6c_reserved_namespace_spoof_denied() -> None:
    store = _RecordingAppendStore()
    service = build_collaborative_activity_ingestion_service(
        verified_publisher_identity=verified_collaborative_activity_publisher_identity_from_request_identity(
            _service_request_identity("tenant-a", "evil-plugin"),
        ),
        publisher_context_resolver=_default_resolver(
            CollaborativeActivityPluginPublisherRegistration(
                tenant_id="tenant-a",
                producer_principal_id="evil-plugin",
                owned_namespace="vendor.a",
            ),
        ),
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
    with pytest.raises(CollaborativeActivityAdmissionRejected) as exc_info:
        service.publish(pub)
    assert exc_info.value.denial_reason is CollaborativeActivityIngestionDenialReason.RESERVED_NAMESPACE_SPOOF
    assert len(store.append_calls) == 0


def test_mp6c_source_type_namespace_mismatch_denied() -> None:
    store = _RecordingAppendStore()
    service = build_collaborative_activity_ingestion_service(
        verified_publisher_identity=verified_collaborative_activity_publisher_identity_from_request_identity(
            _service_request_identity("tenant-a", "plugin-producer"),
        ),
        publisher_context_resolver=_default_resolver(
            CollaborativeActivityPluginPublisherRegistration(
                tenant_id="tenant-a",
                producer_principal_id="plugin-producer",
                owned_namespace="vendor.a",
            ),
        ),
        append_store=store,
    )
    pub = CollaborativeActivityPublication(
        idempotency_key=ActivityIdempotencyKey(
            tenant_id="tenant-a",
            workspace_id="ws-a",
            source=CollaborativeActivitySourceId.for_extension("vendor.a", "adapter"),
            source_stable_id="mismatch-1",
            activity_type=CollaborativeActivityTypeId.for_extension("vendor.b", "work_item.created"),
        ),
        actor=_actor(),
        scope=_scope(),
        target=WorkItemActivityTargetRef(work_item_id="wi-1"),
        outcome=CollaborativeActivityOutcome(status=CollaborativeActivityOutcomeStatus.SUCCEEDED),
        occurred_at=_NOW,
    )
    with pytest.raises(CollaborativeActivityAdmissionRejected) as exc_info:
        service.publish(pub)
    assert (
        exc_info.value.denial_reason
        is CollaborativeActivityIngestionDenialReason.TYPE_NAMESPACE_UNAUTHORIZED
    )
    assert len(store.append_calls) == 0


def test_mp6c_actor_distinct_from_publisher_allowed() -> None:
    store = _RecordingAppendStore()
    service = _platform_service(store, producer="platform-service-caller")
    pub = _platform_publication(actor_principal="semantic-actor-only")
    activity = service.publish(pub)
    assert activity.actor.principal_id == "semantic-actor-only"
    assert len(store.append_calls) == 1


def test_mp6c_custom_policy_injection() -> None:
    store = _RecordingAppendStore()

    class _AlwaysDenyPolicy:
        @property
        def policy_id(self) -> str:
            return "test.always_deny"

        def evaluate(
            self,
            request: CollaborativeActivityIngestionRequest,
        ) -> CollaborativeActivityIngestionDecision:
            return fail_closed_collaborative_activity_ingestion_decision(
                policy_id=self.policy_id,
                denial_reason=CollaborativeActivityIngestionDenialReason.POLICY_AMBIGUITY,
            )

    service = _platform_service(store, policy=_AlwaysDenyPolicy())
    with pytest.raises(CollaborativeActivityAdmissionRejected):
        service.publish(_platform_publication())
    assert len(store.append_calls) == 0


def test_mp6c_deterministic_policy_decision() -> None:
    policy = DefaultCollaborativeActivityIngestionPolicy()
    ctx = _direct_publisher_context(
        tenant_id="tenant-a",
        producer_principal_id="p1",
        kind=CollaborativeActivityPublisherKind.PLATFORM,
    )
    pub = _platform_publication()
    request = CollaborativeActivityIngestionRequest(publication=pub, publisher_context=ctx)
    first = policy.evaluate(request)
    second = policy.evaluate(request)
    assert first == second
    assert first.outcome is CollaborativeActivityIngestionOutcome.ALLOW


def test_mp6c_deny_reason_code_stable() -> None:
    policy = DefaultCollaborativeActivityIngestionPolicy()
    ctx = _direct_publisher_context(
        tenant_id="tenant-a",
        producer_principal_id="p1",
        kind=CollaborativeActivityPublisherKind.PLATFORM,
    )
    pub = _platform_publication(tenant="tenant-b")
    decision = policy.evaluate(
        CollaborativeActivityIngestionRequest(publication=pub, publisher_context=ctx),
    )
    assert decision.denial_reason is CollaborativeActivityIngestionDenialReason.TENANT_MISMATCH


def test_mp6c_no_forbidden_imports_in_ingestion_modules() -> None:
    for path in (_INGESTION_MODULE, _INGESTION_CONTRACT):
        text = path.read_text(encoding="utf-8-sig").lower()
        for marker in _FORBIDDEN_IMPORT_MARKERS:
            assert marker not in text, f"{path.name}: forbidden {marker}"


def test_mp6c_service_depends_on_protocols_only() -> None:
    tree = ast.parse(_INGESTION_MODULE.read_text(encoding="utf-8-sig"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    forbidden = [m for m in imports if m.startswith("intergrax.collaborative_work.repository")]
    assert not forbidden


def test_mp6c_no_any_dict_on_decision_contract() -> None:
    text = _INGESTION_CONTRACT.read_text(encoding="utf-8-sig")
    assert "Any" not in text
    assert "dict[" not in text
