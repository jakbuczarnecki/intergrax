# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

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
    DecisionLifecycleState,
    initial_decision_lifecycle_state,
    validate_lifecycle_transition,
)
from intergrax.contracts.decision_record import (
    AuthoritativeAcceptedDecision,
    CandidateDecision,
    DecisionArtifact,
    DecisionVersionLineage,
    candidate_decision,
    candidate_decision_ref,
    decision_lineage_ref,
    validate_decision_artifact_kind,
)
from intergrax.contracts.decision_resolution import (
    AuthoritativeResolutionRecord,
    DecisionResolution,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.decision_lifecycle_observability import (
    DECISION_LIFECYCLE_FINALIZED_EVENT_KIND,
    DECISION_LIFECYCLE_RESOLVED_EVENT_KIND,
    DECISION_LIFECYCLE_SIGNAL_PAYLOAD_SCHEMA_ID,
    DECISION_LIFECYCLE_STARTED_EVENT_KIND,
    DECISION_LIFECYCLE_TERMINAL_EVENT_KIND,
    DECISION_LIFECYCLE_TRANSITIONED_EVENT_KIND,
    CanonicalRuntimeEventDecisionLifecycleObserver,
    DecisionLifecycleSignalPhase,
    observe_decision_resolution,
    observe_durable_decision_finalization,
    register_decision_lifecycle_domain_signals,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.event_kind_registry import clear_event_kind_registry
from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.execution.decision_finalization_persistence import (
    DecisionDurableFinalizationDisposition,
    DecisionDurableFinalizationResult,
)
from intergrax.runtime.execution.decision_lifecycle_host import CanonicalDecisionLifecycleHost
from intergrax.runtime.execution.in_memory_decision_finalization_persistence import (
    InMemoryDecisionFinalizationPersistence,
)
from intergrax.runtime.execution.decision_finalization_persistence import (
    commit_durable_authoritative_outcome,
)
from intergrax.contracts.decision_finalization import decision_finalization_key
from testing_support.runtime_events import emit_context_test_identity

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_SECRET_MARKER = "SUPER_SECRET_COT_AND_PROMPT"
_OBSERVABILITY_MODULE = (
    Path(__file__).resolve().parents[3]
    / "intergrax"
    / "runtime"
    / "decision_lifecycle_observability.py"
)
_FORBIDDEN_PATTERNS = (
    ": Any",
    "dict[str, Any]",
    "cast(",
    "type: ignore",
    "pyright: ignore",
    "getattr(",
    "setattr(",
    "hasattr(",
    "inspect.",
    "exec(",
    "eval(",
    "object.__setattr__",
)


@dataclass(frozen=True, slots=True)
class IncidentPayload:
    recommendation: str


def _lineage(
    *,
    task_id: TaskId | None = None,
    run_id: RunId | None = None,
    attempt_id: AttemptId | None = None,
    execution_id: ExecutionId | None = None,
) -> DecisionExecutionLineage:
    return DecisionExecutionLineage(
        task_id=mint_task_id() if task_id is None else task_id,
        run_id=mint_run_id() if run_id is None else run_id,
        attempt_id=mint_attempt_id() if attempt_id is None else attempt_id,
        execution_id=mint_execution_id() if execution_id is None else execution_id,
    )


def _identity(
    *,
    decision_id: str | None = None,
    version: DecisionVersion | None = None,
    tenant_id: str = "tenant-a",
    execution: DecisionExecutionLineage | None = None,
) -> DecisionIdentity:
    return DecisionIdentity(
        decision_id=mint_decision_id() if decision_id is None else decision_id,
        version=initial_decision_version() if version is None else version,
        scope=DecisionScope(namespace="incident", subject="incident-123"),
        tenant_id=tenant_id,
        execution=execution or _lineage(),
    )


def _candidate(
    identity: DecisionIdentity,
    *,
    secret: bool = False,
) -> CandidateDecision[IncidentPayload]:
    return candidate_decision(
        identity=identity,
        artifact_kind=validate_decision_artifact_kind("incident_resolution"),
        payload=IncidentPayload(
            recommendation=_SECRET_MARKER if secret else "escalate",
        ),
    )


def _accepted(
    candidate: CandidateDecision[IncidentPayload],
) -> AuthoritativeAcceptedDecision[IncidentPayload]:
    return AuthoritativeAcceptedDecision(
        identity=candidate.identity,
        artifact=candidate.artifact,
        lineage=candidate.lineage,
    )


def _host_setup(
    identity: DecisionIdentity,
) -> tuple[
    CanonicalDecisionLifecycleHost,
    CanonicalRuntimeEventDecisionLifecycleObserver,
    RuntimeEventBus,
]:
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
    return host, observer, bus


def _host_with_bus(
    identity: DecisionIdentity,
) -> tuple[CanonicalDecisionLifecycleHost, RuntimeEventBus]:
    host, _, bus = _host_setup(identity)
    return host, bus


def _payload_data(event: RuntimeEvent) -> dict[str, object]:
    payload = event.payload
    if type(payload) is not dict:
        raise TypeError("expected dict payload envelope")
    data = payload.get("data")
    if type(data) is not dict:
        raise TypeError("expected typed payload data dict")
    return data


def _advance(
    host: CanonicalDecisionLifecycleHost,
    state: DecisionLifecycleState,
    *stages: DecisionLifecycleStage,
) -> DecisionLifecycleState:
    current = state
    for stage in stages:
        current = host.transition(current, stage)
    return current


@pytest.fixture(autouse=True)
def _register_lifecycle_domain_signals() -> Iterator[None]:
    clear_event_kind_registry()
    register_decision_lifecycle_domain_signals()
    yield
    clear_event_kind_registry()


def test_lifecycle_start_emits_started_with_proposal_index_zero() -> None:
    identity = _identity()
    host, bus = _host_with_bus(identity)
    state = host.start(identity)
    assert state.stage is DecisionLifecycleStage.PROPOSAL
    assert state.transition_index == 0
    events = bus.history
    assert len(events) == 1
    assert events[0].event_kind == DECISION_LIFECYCLE_STARTED_EVENT_KIND
    data = _payload_data(events[0])
    assert data["phase"] == DecisionLifecycleSignalPhase.STARTED.value
    assert data["to_stage"] == DecisionLifecycleStage.PROPOSAL.value
    assert data["transition_index"] == 0
    assert data["decision_id"] == str(identity.decision_id)
    assert data["decision_version"] == identity.version.value
    assert data["tenant_id"] == identity.tenant_id


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


@pytest.mark.parametrize(("from_stage", "to_stage"), _legal_transitions())
def test_every_legal_transition_emits_transitioned(
    from_stage: DecisionLifecycleStage,
    to_stage: DecisionLifecycleStage,
) -> None:
    identity = _identity()
    host, bus = _host_with_bus(identity)
    state = initial_decision_lifecycle_state(identity)
    if from_stage is not DecisionLifecycleStage.PROPOSAL:
        state = _advance(host, state, *_path_to(from_stage))
    previous_index = state.transition_index
    updated = host.transition(state, to_stage)
    assert updated.stage is to_stage
    assert updated.transition_index == previous_index + 1
    transitioned = [
        event
        for event in bus.history
        if event.event_kind == DECISION_LIFECYCLE_TRANSITIONED_EVENT_KIND
    ]
    last = _payload_data(transitioned[-1])
    assert last["from_stage"] == from_stage.value
    assert last["to_stage"] == to_stage.value
    assert last["transition_index"] == updated.transition_index


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


def test_illegal_transition_raises_without_transition_event() -> None:
    identity = _identity()
    host, bus = _host_with_bus(identity)
    state = host.start(identity)
    with pytest.raises(ValueError, match="Unsupported lifecycle transition"):
        host.transition(state, DecisionLifecycleStage.TERMINAL)
    assert not any(
        event.event_kind == DECISION_LIFECYCLE_TRANSITIONED_EVENT_KIND for event in bus.history
    )


def test_execution_lineage_mismatch_rejects_emission() -> None:
    identity = _identity()
    bus = RuntimeEventBus(record_history=True)
    execution = identity.execution
    ctx = emit_context_test_identity(
        task_id=mint_task_id(),
        run_id=execution.run_id,
        attempt_id=execution.attempt_id,
        execution_id=execution.execution_id,
        tenant_id=identity.tenant_id,
        bus=bus,
    )
    observer = CanonicalRuntimeEventDecisionLifecycleObserver(ctx=ctx)
    host = CanonicalDecisionLifecycleHost(observer=observer)
    with pytest.raises(ValueError, match="task_id must match"):
        host.start(identity)


@pytest.mark.parametrize(
    "resolution",
    [
        DecisionResolution.ACCEPTED,
        DecisionResolution.REJECTED,
        DecisionResolution.UNRESOLVED,
    ],
)
def test_resolution_observation_emits_resolved(
    resolution: DecisionResolution,
) -> None:
    identity = _identity()
    host, observer, bus = _host_setup(identity)
    state = _advance(
        host,
        host.start(identity),
        DecisionLifecycleStage.VERIFICATION,
        DecisionLifecycleStage.RESOLUTION,
    )
    candidate = _candidate(identity)
    if resolution is DecisionResolution.ACCEPTED:
        observe_decision_resolution(
            observer,
            lifecycle_state=state,
            outcome=_accepted(candidate),
        )
    else:
        observe_decision_resolution(
            observer,
            lifecycle_state=state,
            outcome=AuthoritativeResolutionRecord(
                identity=identity,
                resolution=resolution,
            ),
            proposal_branch_id=str(candidate_decision_ref(candidate).lineage_ref.branch_id),
        )
    resolved_events = [
        event
        for event in bus.history
        if event.event_kind == DECISION_LIFECYCLE_RESOLVED_EVENT_KIND
    ]
    assert len(resolved_events) == 1
    data = _payload_data(resolved_events[0])
    assert data["resolution_outcome"] == resolution.value


def test_finalization_committed_only_after_persistence_returns() -> None:
    identity = _identity()
    candidate = _candidate(identity)
    accepted = _accepted(candidate)
    persistence: InMemoryDecisionFinalizationPersistence[IncidentPayload] = (
        InMemoryDecisionFinalizationPersistence()
    )
    key = decision_finalization_key(identity)
    host, bus = _host_with_bus(identity)
    state = _advance(
        host,
        host.start(identity),
        DecisionLifecycleStage.VERIFICATION,
        DecisionLifecycleStage.RESOLUTION,
        DecisionLifecycleStage.FINALIZATION,
    )
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

    def _observe_after_commit() -> None:
        result = commit_durable_authoritative_outcome(
            persistence,
            key=key,
            requested_outcome=accepted,
        )
        observe_durable_decision_finalization(
            observer,
            lifecycle_state=state,
            result=result,
        )

    _observe_after_commit()
    finalized = [
        event
        for event in bus.history
        if event.event_kind == DECISION_LIFECYCLE_FINALIZED_EVENT_KIND
    ]
    assert len(finalized) == 1
    assert _payload_data(finalized[0])["finalization_disposition"] == (
        DecisionDurableFinalizationDisposition.COMMITTED.value
    )


@pytest.mark.parametrize(
    "disposition",
    [
        DecisionDurableFinalizationDisposition.COMMITTED,
        DecisionDurableFinalizationDisposition.IDEMPOTENT_REPLAY,
        DecisionDurableFinalizationDisposition.CONFLICT,
    ],
)
def test_finalization_dispositions_are_observable(
    disposition: DecisionDurableFinalizationDisposition,
) -> None:
    identity = _identity()
    host, bus = _host_with_bus(identity)
    state = _advance(
        host,
        host.start(identity),
        DecisionLifecycleStage.VERIFICATION,
        DecisionLifecycleStage.RESOLUTION,
        DecisionLifecycleStage.FINALIZATION,
    )
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
    from intergrax.contracts.decision_finalization import initial_decision_finalize_guard

    guard = initial_decision_finalize_guard(decision_finalization_key(identity))
    result = DecisionDurableFinalizationResult(
        disposition=disposition,
        guard_state=guard,
    )
    observe_durable_decision_finalization(
        observer,
        lifecycle_state=state,
        result=result,
    )
    finalized = [
        event
        for event in bus.history
        if event.event_kind == DECISION_LIFECYCLE_FINALIZED_EVENT_KIND
    ]
    assert _payload_data(finalized[0])["finalization_disposition"] == disposition.value


def test_terminal_emitted_only_after_terminal_transition() -> None:
    identity = _identity()
    host, bus = _host_with_bus(identity)
    state = _advance(
        host,
        host.start(identity),
        DecisionLifecycleStage.VERIFICATION,
        DecisionLifecycleStage.RESOLUTION,
        DecisionLifecycleStage.FINALIZATION,
    )
    assert not any(
        event.event_kind == DECISION_LIFECYCLE_TERMINAL_EVENT_KIND for event in bus.history
    )
    host.transition(state, DecisionLifecycleStage.TERMINAL)
    terminal = [
        event
        for event in bus.history
        if event.event_kind == DECISION_LIFECYCLE_TERMINAL_EVENT_KIND
    ]
    assert len(terminal) == 1
    assert _payload_data(terminal[0])["to_stage"] == DecisionLifecycleStage.TERMINAL.value


def test_version_binding_keeps_distinct_streams() -> None:
    shared_id = mint_decision_id()
    identity_v1 = _identity(decision_id=shared_id, version=initial_decision_version())
    identity_v2 = _identity(
        decision_id=shared_id,
        version=next_decision_version(initial_decision_version()),
    )
    host_v1, bus_v1 = _host_with_bus(identity_v1)
    host_v2, bus_v2 = _host_with_bus(identity_v2)
    host_v1.start(identity_v1)
    host_v2.start(identity_v2)
    assert _payload_data(bus_v1.history[0])["decision_version"] == 1
    assert _payload_data(bus_v2.history[0])["decision_version"] == 2


def test_redaction_excludes_secret_artifact_content() -> None:
    identity = _identity()
    host, bus = _host_with_bus(identity)
    host.start(identity)
    serialized = json.dumps([event.model_dump(mode="json") for event in bus.history])
    assert _SECRET_MARKER not in serialized
    assert "recommendation" not in serialized


def test_payload_survives_runtime_event_serialization() -> None:
    identity = _identity()
    host, bus = _host_with_bus(identity)
    host.start(identity)
    roundtrip = RuntimeEvent.model_validate(bus.history[0].model_dump(mode="json"))
    assert roundtrip.payload["payload_schema_id"] == DECISION_LIFECYCLE_SIGNAL_PAYLOAD_SCHEMA_ID


def test_no_forbidden_typing_patterns_in_observability_module() -> None:
    source = _OBSERVABILITY_MODULE.read_text(encoding="utf-8")
    for pattern in _FORBIDDEN_PATTERNS:
        assert pattern not in source


def test_register_decision_lifecycle_domain_signals_is_idempotent() -> None:
    register_decision_lifecycle_domain_signals()
    register_decision_lifecycle_domain_signals()
