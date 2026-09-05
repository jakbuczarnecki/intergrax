# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Conformance harness for Decision finalization persistence backends (DS-REC-01)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import Barrier
from typing import Generic, TypeVar

from intergrax.contracts.decision_finalization import (
    DecisionFinalizationKey,
    decision_finalization_key,
)
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionId,
    DecisionIdentity,
    DecisionScope,
    DecisionVersion,
    initial_decision_version,
    mint_decision_id,
    next_decision_version,
)
from intergrax.contracts.decision_record import (
    AuthoritativeAcceptedDecision,
    DecisionArtifact,
    DecisionVersionLineage,
    decision_lineage_ref,
    validate_decision_artifact_kind,
)
from intergrax.contracts.decision_resolution import (
    AuthoritativeResolutionRecord,
    DecisionResolution,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.decision_record import validate_decision_artifact_kind
from intergrax.knowledge.contracts.validation import JsonValue
from intergrax.runtime.execution.decision_artifact_payload_codec import (
    DecisionArtifactPayloadCodec,
    DecisionArtifactPayloadCodecRegistry,
    decision_artifact_payload_codec_registry,
)
from intergrax.runtime.execution.decision_finalization_persistence import (
    DecisionDurableFinalizationDisposition,
    DecisionFinalizationPersistence,
)

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class IncidentDecisionPayload:
    recommendation: str


@dataclass(frozen=True, slots=True)
class IncidentDecisionPayloadCodec:
    """Explicit durable codec for conformance incident-resolution payloads."""

    def encode(self, payload: object) -> JsonValue:
        if type(payload) is not IncidentDecisionPayload:
            raise TypeError("incident_resolution codec expects IncidentDecisionPayload")
        return {"recommendation": payload.recommendation}

    def decode(self, payload: JsonValue) -> IncidentDecisionPayload:
        if type(payload) is not dict:
            raise TypeError("incident_resolution payload must be a JSON object")
        recommendation = payload.get("recommendation")
        if type(recommendation) is not str:
            raise TypeError("incident_resolution payload recommendation must be str")
        return IncidentDecisionPayload(recommendation=recommendation)


def conformance_artifact_payload_codec_registry() -> DecisionArtifactPayloadCodecRegistry:
    """Registry for Decision durable conformance and SQLite proof scenarios."""
    kind = validate_decision_artifact_kind("incident_resolution")
    codec: DecisionArtifactPayloadCodec[object] = IncidentDecisionPayloadCodec()
    return decision_artifact_payload_codec_registry(codecs={kind: codec})


def _execution_lineage() -> DecisionExecutionLineage:
    return DecisionExecutionLineage(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


def _identity(
    *,
    decision_id: DecisionId | None = None,
    tenant_id: str = "tenant-a",
    namespace: str = "incident",
    subject: str = "incident-123",
    version: DecisionVersion | None = None,
) -> DecisionIdentity:
    return DecisionIdentity(
        decision_id=mint_decision_id() if decision_id is None else decision_id,
        version=version or initial_decision_version(),
        scope=DecisionScope(namespace=namespace, subject=subject),
        tenant_id=tenant_id,
        execution=_execution_lineage(),
    )


def _artifact(recommendation: str = "escalate") -> DecisionArtifact[IncidentDecisionPayload]:
    return DecisionArtifact(
        kind=validate_decision_artifact_kind("incident_resolution"),
        content=IncidentDecisionPayload(recommendation=recommendation),
    )


def _accepted(
    *,
    identity: DecisionIdentity,
    recommendation: str = "escalate",
    lineage: DecisionVersionLineage | None = None,
) -> AuthoritativeAcceptedDecision[IncidentDecisionPayload]:
    resolved_lineage = lineage or DecisionVersionLineage(
        current=decision_lineage_ref(identity.version),
    )
    return AuthoritativeAcceptedDecision(
        identity=identity,
        artifact=_artifact(recommendation=recommendation),
        lineage=resolved_lineage,
    )


def _resolution(
    *,
    identity: DecisionIdentity,
    resolution: DecisionResolution,
) -> AuthoritativeResolutionRecord:
    return AuthoritativeResolutionRecord(identity=identity, resolution=resolution)


def assert_decision_finalization_persistence_conformance(
    factory: Callable[[], DecisionFinalizationPersistence[IncidentDecisionPayload]],
    *,
    label: str,
) -> None:
    """Exercise required DS-REC-01 scenarios against one backend factory."""
    persistence = factory()
    identity = _identity()
    key = decision_finalization_key(identity)
    accepted = _accepted(identity=identity)

    first = persistence.commit_authoritative_outcome(key=key, requested_outcome=accepted)
    assert first.disposition is DecisionDurableFinalizationDisposition.COMMITTED
    assert first.guard_state.authoritative_outcome == accepted

    replay = persistence.commit_authoritative_outcome(key=key, requested_outcome=accepted)
    assert replay.disposition is DecisionDurableFinalizationDisposition.IDEMPOTENT_REPLAY
    assert replay.guard_state.authoritative_outcome == accepted

    fixed_id = mint_decision_id()
    scope = DecisionScope(namespace="incident", subject="incident-123")
    identity_v1 = _identity(
        decision_id=fixed_id,
        namespace=scope.namespace,
        subject=scope.subject,
        version=initial_decision_version(),
    )
    identity_v2 = _identity(
        decision_id=fixed_id,
        namespace=scope.namespace,
        subject=scope.subject,
        version=next_decision_version(initial_decision_version()),
    )
    conflict_key = decision_finalization_key(identity_v1)
    accepted_v1 = _accepted(
        identity=identity_v1,
        lineage=DecisionVersionLineage(current=decision_lineage_ref(identity_v1.version)),
    )
    accepted_v2 = _accepted(
        identity=identity_v2,
        lineage=DecisionVersionLineage(
            current=decision_lineage_ref(identity_v2.version),
            parents=(decision_lineage_ref(identity_v1.version),),
        ),
    )
    conflict_store = factory()
    conflict_store.commit_authoritative_outcome(
        key=conflict_key,
        requested_outcome=accepted_v1,
    )
    conflict = conflict_store.commit_authoritative_outcome(
        key=conflict_key,
        requested_outcome=accepted_v2,
    )
    assert conflict.disposition is DecisionDurableFinalizationDisposition.CONFLICT

    accepted_vs_rejected = factory()
    accepted_identity = _identity()
    accepted_key = decision_finalization_key(accepted_identity)
    accepted_vs_rejected.commit_authoritative_outcome(
        key=accepted_key,
        requested_outcome=_accepted(identity=accepted_identity),
    )
    rejected_conflict = accepted_vs_rejected.commit_authoritative_outcome(
        key=accepted_key,
        requested_outcome=_resolution(
            identity=accepted_identity,
            resolution=DecisionResolution.REJECTED,
        ),
    )
    assert rejected_conflict.disposition is DecisionDurableFinalizationDisposition.CONFLICT

    rejected_vs_unresolved = factory()
    unresolved_identity = _identity()
    unresolved_key = decision_finalization_key(unresolved_identity)
    rejected_vs_unresolved.commit_authoritative_outcome(
        key=unresolved_key,
        requested_outcome=_resolution(
            identity=unresolved_identity,
            resolution=DecisionResolution.REJECTED,
        ),
    )
    unresolved_conflict = rejected_vs_unresolved.commit_authoritative_outcome(
        key=unresolved_key,
        requested_outcome=_resolution(
            identity=unresolved_identity,
            resolution=DecisionResolution.UNRESOLVED,
        ),
    )
    assert unresolved_conflict.disposition is DecisionDurableFinalizationDisposition.CONFLICT

    tenant_a = factory()
    tenant_b = factory()
    shared_scope = DecisionScope(namespace="incident", subject="shared-subject")
    shared_id = mint_decision_id()
    tenant_a_identity = _identity(
        decision_id=shared_id,
        tenant_id="tenant-a",
        namespace=shared_scope.namespace,
        subject=shared_scope.subject,
    )
    tenant_b_identity = _identity(
        decision_id=shared_id,
        tenant_id="tenant-b",
        namespace=shared_scope.namespace,
        subject=shared_scope.subject,
    )
    tenant_a_key = decision_finalization_key(tenant_a_identity)
    tenant_b_key = decision_finalization_key(tenant_b_identity)
    tenant_a.commit_authoritative_outcome(
        key=tenant_a_key,
        requested_outcome=_accepted(identity=tenant_a_identity, recommendation="a"),
    )
    tenant_b_result = tenant_b.commit_authoritative_outcome(
        key=tenant_b_key,
        requested_outcome=_accepted(identity=tenant_b_identity, recommendation="b"),
    )
    assert tenant_b_result.disposition is DecisionDurableFinalizationDisposition.COMMITTED

    scope_store = factory()
    shared_tenant = "tenant-scope"
    shared_decision = mint_decision_id()
    scope_x_identity = _identity(
        decision_id=shared_decision,
        tenant_id=shared_tenant,
        namespace="namespace-x",
        subject="subject-1",
    )
    scope_y_identity = _identity(
        decision_id=shared_decision,
        tenant_id=shared_tenant,
        namespace="namespace-y",
        subject="subject-1",
    )
    scope_store.commit_authoritative_outcome(
        key=decision_finalization_key(scope_x_identity),
        requested_outcome=_accepted(identity=scope_x_identity, recommendation="x"),
    )
    scope_y_result = scope_store.commit_authoritative_outcome(
        key=decision_finalization_key(scope_y_identity),
        requested_outcome=_accepted(identity=scope_y_identity, recommendation="y"),
    )
    assert scope_y_result.disposition is DecisionDurableFinalizationDisposition.COMMITTED

    decision_isolation = factory()
    shared_tenant_scope = DecisionScope(namespace="incident", subject="shared")
    decision_one = mint_decision_id()
    decision_two = mint_decision_id()
    identity_one = _identity(
        decision_id=decision_one,
        tenant_id="tenant-isolation",
        namespace=shared_tenant_scope.namespace,
        subject=shared_tenant_scope.subject,
    )
    identity_two = _identity(
        decision_id=decision_two,
        tenant_id="tenant-isolation",
        namespace=shared_tenant_scope.namespace,
        subject=shared_tenant_scope.subject,
    )
    decision_isolation.commit_authoritative_outcome(
        key=decision_finalization_key(identity_one),
        requested_outcome=_accepted(identity=identity_one, recommendation="one"),
    )
    decision_two_result = decision_isolation.commit_authoritative_outcome(
        key=decision_finalization_key(identity_two),
        requested_outcome=_accepted(identity=identity_two, recommendation="two"),
    )
    assert decision_two_result.disposition is DecisionDurableFinalizationDisposition.COMMITTED


def assert_concurrent_finalization_race(
    factory: Callable[[], DecisionFinalizationPersistence[IncidentDecisionPayload]],
    *,
    label: str,
) -> None:
    """Two writers race on one key; exactly one commits and one conflicts."""
    shared_key_holder: list[DecisionFinalizationKey] = []
    fixed_id = mint_decision_id()
    identity_a = _identity(decision_id=fixed_id, tenant_id="tenant-race")
    identity_b = _identity(
        decision_id=fixed_id,
        tenant_id="tenant-race",
        version=next_decision_version(initial_decision_version()),
    )
    key = decision_finalization_key(identity_a)
    shared_key_holder.append(key)
    outcome_a = _accepted(identity=identity_a, recommendation="winner-a")
    outcome_b = _resolution(identity=identity_b, resolution=DecisionResolution.REJECTED)
    barrier = Barrier(2)
    results: list[DecisionDurableFinalizationDisposition] = []

    def _worker(outcome: AuthoritativeAcceptedDecision[IncidentDecisionPayload] | AuthoritativeResolutionRecord) -> None:
        store = factory()
        barrier.wait()
        result = store.commit_authoritative_outcome(key=key, requested_outcome=outcome)
        results.append(result.disposition)

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(_worker, outcome_a)
        future_b = executor.submit(_worker, outcome_b)
        future_a.result()
        future_b.result()

    assert DecisionDurableFinalizationDisposition.COMMITTED in results
    assert DecisionDurableFinalizationDisposition.CONFLICT in results
    assert results.count(DecisionDurableFinalizationDisposition.COMMITTED) == 1
    assert results.count(DecisionDurableFinalizationDisposition.CONFLICT) == 1

    verify_store = factory()
    try:
        loaded = verify_store.load_guard_state(key=key)
        assert loaded is not None
        assert loaded.authoritative_outcome is not None
    finally:
        verify_store.close()


def assert_concurrent_idempotent_replay(
    factory: Callable[[], DecisionFinalizationPersistence[IncidentDecisionPayload]],
    *,
    label: str,
) -> None:
    """Two writers replay the same outcome; one commits and one idempotent replays."""
    identity = _identity(tenant_id="tenant-idempotent-race")
    key = decision_finalization_key(identity)
    accepted = _accepted(identity=identity)
    barrier = Barrier(2)
    results: list[DecisionDurableFinalizationDisposition] = []

    def _worker() -> None:
        store = factory()
        barrier.wait()
        result = store.commit_authoritative_outcome(key=key, requested_outcome=accepted)
        results.append(result.disposition)

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(_worker)
        future_b = executor.submit(_worker)
        future_a.result()
        future_b.result()

    assert results.count(DecisionDurableFinalizationDisposition.COMMITTED) == 1
    assert results.count(DecisionDurableFinalizationDisposition.IDEMPOTENT_REPLAY) == 1
