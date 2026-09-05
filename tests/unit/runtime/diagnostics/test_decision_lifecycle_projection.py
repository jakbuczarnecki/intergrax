# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace

import pytest

from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    DecisionVersion,
    initial_decision_version,
    mint_decision_id,
    next_decision_version,
)
from intergrax.contracts.decision_lifecycle import (
    DecisionLifecycleStage,
    validate_lifecycle_transition,
)
from intergrax.contracts.decision_record import (
    AuthoritativeAcceptedDecision,
    CandidateDecision,
    candidate_decision,
    candidate_decision_ref,
    decision_lineage_ref,
    decision_version_lineage,
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
from intergrax.runtime.decision_lifecycle_observability import (
    CanonicalRuntimeEventDecisionLifecycleObserver,
    observe_decision_resolution,
    observe_durable_decision_finalization,
    register_decision_lifecycle_domain_signals,
)
from intergrax.runtime.diagnostics.decision_lifecycle_projection import (
    DecisionLifecycleDiagnosticSnapshot,
    DecisionLifecycleReconstructionError,
    project_decision_lifecycle_snapshot,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.event_kind_registry import clear_event_kind_registry
from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.execution.decision_finalization_persistence import (
    DecisionDurableFinalizationDisposition,
    DecisionDurableFinalizationResult,
)
from intergrax.runtime.execution.decision_lifecycle_host import CanonicalDecisionLifecycleHost
from intergrax.contracts.decision_finalization import initial_decision_finalize_guard, decision_finalization_key
from testing_support.runtime_events import emit_context_test_identity

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@dataclass(frozen=True, slots=True)
class IncidentPayload:
    recommendation: str


def _legal_transitions() -> list[tuple[DecisionLifecycleStage, DecisionLifecycleStage]]:
    pairs: list[tuple[DecisionLifecycleStage, DecisionLifecycleStage]] = []
    for current in DecisionLifecycleStage:
        for target in DecisionLifecycleStage:
            if current is target:
                continue
            try:
                validate_lifecycle_transition(from_stage=current, to_stage=target)
            except ValueError:
                continue
            pairs.append((current, target))
    return pairs


def _identity(
    *,
    decision_id: str | None = None,
    version: DecisionVersion | None = None,
    tenant_id: str = "tenant-a",
) -> DecisionIdentity:
    return DecisionIdentity(
        decision_id=mint_decision_id() if decision_id is None else decision_id,
        version=initial_decision_version() if version is None else version,
        scope=DecisionScope(namespace="incident", subject="incident-123"),
        tenant_id=tenant_id,
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
    )


def _path_to(target: DecisionLifecycleStage) -> tuple[DecisionLifecycleStage, ...]:
    if target is DecisionLifecycleStage.PROPOSAL:
        return ()
    if target is DecisionLifecycleStage.VERIFICATION:
        return (DecisionLifecycleStage.VERIFICATION,)
    if target is DecisionLifecycleStage.DELIBERATION:
        return (DecisionLifecycleStage.DELIBERATION,)
    if target is DecisionLifecycleStage.REVISION:
        return (
            DecisionLifecycleStage.VERIFICATION,
            DecisionLifecycleStage.REVISION,
        )
    if target is DecisionLifecycleStage.ADJUDICATION:
        return (
            DecisionLifecycleStage.VERIFICATION,
            DecisionLifecycleStage.ADJUDICATION,
        )
    if target is DecisionLifecycleStage.RESOLUTION:
        return (
            DecisionLifecycleStage.VERIFICATION,
            DecisionLifecycleStage.RESOLUTION,
        )
    if target is DecisionLifecycleStage.FINALIZATION:
        return (
            DecisionLifecycleStage.VERIFICATION,
            DecisionLifecycleStage.RESOLUTION,
            DecisionLifecycleStage.FINALIZATION,
        )
    if target is DecisionLifecycleStage.TERMINAL:
        return (
            DecisionLifecycleStage.VERIFICATION,
            DecisionLifecycleStage.RESOLUTION,
            DecisionLifecycleStage.FINALIZATION,
            DecisionLifecycleStage.TERMINAL,
        )
    raise ValueError(f"unsupported target stage: {target}")


def _record_normal_lifecycle(
    identity: DecisionIdentity,
    *,
    include_revision_loop: bool = False,
    include_adjudication: bool = False,
) -> tuple[RuntimeEventBus, tuple[RuntimeEvent, ...]]:
    bus = RuntimeEventBus(record_history=True)
    execution = identity.execution
    ctx = emit_context_test_identity(
        task_id=execution.task_id,
        run_id=execution.run_id,
        attempt_id=execution.attempt_id,
        execution_id=execution.execution_id,
        tenant_id=identity.tenant_id,
        bus=bus,
    )
    observer = CanonicalRuntimeEventDecisionLifecycleObserver(ctx=ctx)
    host = CanonicalDecisionLifecycleHost(observer=observer)
    state = host.start(identity)
    if include_revision_loop:
        state = host.transition(state, DecisionLifecycleStage.VERIFICATION)
        state = host.transition(state, DecisionLifecycleStage.REVISION)
        state = host.transition(state, DecisionLifecycleStage.VERIFICATION)
    elif include_adjudication:
        state = host.transition(state, DecisionLifecycleStage.VERIFICATION)
        state = host.transition(state, DecisionLifecycleStage.ADJUDICATION)
    else:
        state = host.transition(state, DecisionLifecycleStage.VERIFICATION)
    state = host.transition(state, DecisionLifecycleStage.RESOLUTION)
    candidate = candidate_decision(
        identity=identity,
        artifact_kind=validate_decision_artifact_kind("incident_resolution"),
        payload=IncidentPayload(recommendation="accept"),
        lineage=(
            decision_version_lineage(
                current=decision_lineage_ref(identity.version),
                parents=(decision_lineage_ref(initial_decision_version()),),
            )
            if identity.version.value > 1
            else None
        ),
    )
    accepted = AuthoritativeAcceptedDecision(
        identity=identity,
        artifact=candidate.artifact,
        lineage=candidate.lineage,
    )
    observe_decision_resolution(
        observer,
        lifecycle_state=state,
        outcome=accepted,
    )
    state = host.transition(state, DecisionLifecycleStage.FINALIZATION)
    guard = initial_decision_finalize_guard(decision_finalization_key(identity))
    observe_durable_decision_finalization(
        observer,
        lifecycle_state=state,
        result=DecisionDurableFinalizationResult(
            disposition=DecisionDurableFinalizationDisposition.COMMITTED,
            guard_state=guard,
        ),
    )
    state = host.transition(state, DecisionLifecycleStage.TERMINAL)
    return bus, tuple(bus.history)


@pytest.fixture(autouse=True)
def _register_lifecycle_domain_signals() -> Iterator[None]:
    clear_event_kind_registry()
    register_decision_lifecycle_domain_signals()
    yield
    clear_event_kind_registry()


def test_happy_reconstruction_proof_a() -> None:
    identity = _identity()
    _, events = _record_normal_lifecycle(identity)
    snapshot = project_decision_lifecycle_snapshot(events)
    assert snapshot.current_stage is DecisionLifecycleStage.TERMINAL
    assert snapshot.transition_count == 4
    assert snapshot.resolution_outcome is DecisionResolution.ACCEPTED
    assert snapshot.finalization_disposition is DecisionDurableFinalizationDisposition.COMMITTED
    assert snapshot.terminal is True


def test_revision_loop_reconstruction_proof_b() -> None:
    identity = _identity()
    _, events = _record_normal_lifecycle(identity, include_revision_loop=True)
    snapshot = project_decision_lifecycle_snapshot(events)
    assert snapshot.transition_count == 6
    assert snapshot.resolution_outcome is DecisionResolution.ACCEPTED
    assert snapshot.terminal is True


def test_adjudication_path_visible_proof_c() -> None:
    identity = _identity()
    _, events = _record_normal_lifecycle(identity, include_adjudication=True)
    snapshot = project_decision_lifecycle_snapshot(events)
    assert snapshot.transition_count == 5
    assert snapshot.resolution_outcome is DecisionResolution.ACCEPTED


def test_conflict_finalization_visible_proof_d() -> None:
    identity = _identity()
    bus = RuntimeEventBus(record_history=True)
    execution = identity.execution
    ctx = emit_context_test_identity(
        task_id=execution.task_id,
        run_id=execution.run_id,
        attempt_id=execution.attempt_id,
        execution_id=execution.execution_id,
        tenant_id=identity.tenant_id,
        bus=bus,
    )
    observer = CanonicalRuntimeEventDecisionLifecycleObserver(ctx=ctx)
    host = CanonicalDecisionLifecycleHost(observer=observer)
    state = host.start(identity)
    state = host.transition(state, DecisionLifecycleStage.VERIFICATION)
    state = host.transition(state, DecisionLifecycleStage.RESOLUTION)
    state = host.transition(state, DecisionLifecycleStage.FINALIZATION)
    observe_durable_decision_finalization(
        observer,
        lifecycle_state=state,
        result=DecisionDurableFinalizationResult(
            disposition=DecisionDurableFinalizationDisposition.CONFLICT,
            guard_state=initial_decision_finalize_guard(decision_finalization_key(identity)),
        ),
    )
    snapshot = project_decision_lifecycle_snapshot(tuple(bus.history))
    assert snapshot.finalization_disposition is DecisionDurableFinalizationDisposition.CONFLICT
    assert snapshot.resolution_outcome is None


def test_version_isolation_proof_e() -> None:
    shared = mint_decision_id()
    identity_v1 = _identity(decision_id=shared, version=initial_decision_version())
    identity_v2 = _identity(
        decision_id=shared,
        version=next_decision_version(initial_decision_version()),
    )
    _, events_v1 = _record_normal_lifecycle(identity_v1)
    _, events_v2 = _record_normal_lifecycle(identity_v2)
    snapshot_v1 = project_decision_lifecycle_snapshot(events_v1)
    snapshot_v2 = project_decision_lifecycle_snapshot(events_v2)
    assert snapshot_v1.decision_version == 1
    assert snapshot_v2.decision_version == 2


def test_tenant_isolation_proof_f() -> None:
    identity_a = _identity(tenant_id="tenant-a")
    identity_b = DecisionIdentity(
        decision_id=identity_a.decision_id,
        version=identity_a.version,
        scope=identity_a.scope,
        tenant_id="tenant-b",
        execution=identity_a.execution,
    )
    _, events_a = _record_normal_lifecycle(identity_a)
    _, events_b = _record_normal_lifecycle(identity_b)
    mixed = events_a + events_b
    with pytest.raises(DecisionLifecycleReconstructionError, match="mixes"):
        project_decision_lifecycle_snapshot(mixed)


def test_gap_fails_closed_proof_g() -> None:
    identity = _identity()
    _, events = _record_normal_lifecycle(identity)
    list_events = list(events)
    list_events = [
        event
        for event in list_events
        if not (
            event.payload.get("data", {}).get("phase") == "transitioned"
            and event.payload.get("data", {}).get("transition_index") == 2
        )
    ]
    with pytest.raises(DecisionLifecycleReconstructionError):
        project_decision_lifecycle_snapshot(tuple(list_events))


def test_duplicate_event_id_is_idempotent() -> None:
    identity = _identity()
    _, events = _record_normal_lifecycle(identity)
    duplicate = events + (events[0],)
    snapshot = project_decision_lifecycle_snapshot(duplicate)
    assert snapshot.transition_count == 4


def test_conflicting_duplicate_fails_closed() -> None:
    identity = _identity()
    _, events = _record_normal_lifecycle(identity)
    first = events[1]
    second = events[2]
    conflicting = RuntimeEvent.model_validate(
        {
            **second.model_dump(mode="json"),
            "event_id": first.event_id,
        },
    )
    with pytest.raises(DecisionLifecycleReconstructionError, match="event_id with conflicting payload"):
        project_decision_lifecycle_snapshot((events[0], first, conflicting, *events[2:]))


def test_cross_version_mix_fails_closed() -> None:
    shared = mint_decision_id()
    identity_v1 = _identity(decision_id=shared, version=initial_decision_version())
    identity_v2 = _identity(
        decision_id=shared,
        version=next_decision_version(initial_decision_version()),
    )
    _, events_v1 = _record_normal_lifecycle(identity_v1)
    _, events_v2 = _record_normal_lifecycle(identity_v2)
    with pytest.raises(DecisionLifecycleReconstructionError, match="mixes"):
        project_decision_lifecycle_snapshot(events_v1[:1] + events_v2[:1])


def test_illegal_observed_transition_sequence_fails_closed() -> None:
    identity = _identity()
    _, events = _record_normal_lifecycle(identity)
    list_events = list(events)
    for index, event in enumerate(list_events):
        data = event.payload.get("data")
        if type(data) is dict and data.get("phase") == "transitioned":
            if data.get("transition_index") == 1:
                mutated = dict(event.payload)
                mutated_data = dict(data)
                mutated_data["to_stage"] = DecisionLifecycleStage.REVISION.value
                mutated["data"] = mutated_data
                list_events[index] = RuntimeEvent.model_validate(
                    {
                        **event.model_dump(mode="json"),
                        "payload": mutated,
                    },
                )
                break
    with pytest.raises(DecisionLifecycleReconstructionError):
        project_decision_lifecycle_snapshot(tuple(list_events))


@pytest.mark.parametrize(("from_stage", "to_stage"), _legal_transitions())
def test_projection_accepts_canonical_transition_table(
    from_stage: DecisionLifecycleStage,
    to_stage: DecisionLifecycleStage,
) -> None:
    identity = _identity()
    bus = RuntimeEventBus(record_history=True)
    execution = identity.execution
    ctx = emit_context_test_identity(
        task_id=execution.task_id,
        run_id=execution.run_id,
        attempt_id=execution.attempt_id,
        execution_id=execution.execution_id,
        tenant_id=identity.tenant_id,
        bus=bus,
    )
    observer = CanonicalRuntimeEventDecisionLifecycleObserver(ctx=ctx)
    host = CanonicalDecisionLifecycleHost(observer=observer)
    state = host.start(identity)
    for stage in _path_to(from_stage):
        state = host.transition(state, stage)
    state = host.transition(state, to_stage)
    snapshot = project_decision_lifecycle_snapshot(tuple(bus.history))
    assert isinstance(snapshot, DecisionLifecycleDiagnosticSnapshot)
    assert snapshot.current_stage is to_stage
