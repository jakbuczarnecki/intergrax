# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Deterministic Decision Lifecycle diagnostic projection from RuntimeEvent evidence (DS-OBS-02)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.contracts.decision_lifecycle import (
    DecisionLifecycleStage,
    validate_lifecycle_transition,
)
from intergrax.contracts.decision_resolution import DecisionResolution
from intergrax.contracts.execution_identity import EventId, validate_event_id
from intergrax.runtime.decision_lifecycle_observability import (
    DECISION_LIFECYCLE_SIGNAL_PAYLOAD_SCHEMA_ID,
    DecisionLifecycleSignalPayloadV1,
    DecisionLifecycleSignalPhase,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.execution.decision_finalization_persistence import (
    DecisionDurableFinalizationDisposition,
)


class DecisionLifecycleReconstructionError(Exception):
    """Raised when lifecycle audit evidence cannot be reconstructed safely."""


@dataclass(frozen=True, slots=True)
class DecisionLifecycleDiagnosticSnapshot:
    """Compact operator-facing lifecycle projection — observational only."""

    decision_id: str
    decision_version: int
    tenant_id: str
    current_stage: DecisionLifecycleStage
    transition_count: int
    resolution_outcome: DecisionResolution | None
    finalization_disposition: DecisionDurableFinalizationDisposition | None
    terminal: bool
    evidence_event_ids: tuple[EventId, ...]


@dataclass(frozen=True, slots=True)
class _LifecycleStreamKey:
    decision_id: str
    decision_version: int
    tenant_id: str


@dataclass(frozen=True, slots=True)
class _ParsedLifecycleEvidence:
    event_id: EventId
    payload: DecisionLifecycleSignalPayloadV1


def _payload_from_runtime_event(event: RuntimeEvent) -> DecisionLifecycleSignalPayloadV1:
    envelope = event.payload
    if type(envelope) is not dict:
        raise DecisionLifecycleReconstructionError(
            "lifecycle evidence event payload must be a typed envelope dict",
        )
    schema_id = envelope.get("payload_schema_id")
    if schema_id != DECISION_LIFECYCLE_SIGNAL_PAYLOAD_SCHEMA_ID:
        raise DecisionLifecycleReconstructionError(
            "lifecycle evidence event payload schema_id mismatch",
        )
    return DecisionLifecycleSignalPayloadV1.from_envelope(envelope)


def _stream_key_from_payload(payload: DecisionLifecycleSignalPayloadV1) -> _LifecycleStreamKey:
    return _LifecycleStreamKey(
        decision_id=payload.decision_id,
        decision_version=payload.decision_version,
        tenant_id=payload.tenant_id,
    )


def _parse_lifecycle_events(
    events: tuple[RuntimeEvent, ...],
) -> tuple[_LifecycleStreamKey, tuple[_ParsedLifecycleEvidence, ...]]:
    if not events:
        raise DecisionLifecycleReconstructionError("lifecycle evidence stream must not be empty")
    parsed: list[_ParsedLifecycleEvidence] = []
    stream_key: _LifecycleStreamKey | None = None
    for event in events:
        payload = _payload_from_runtime_event(event)
        key = _stream_key_from_payload(payload)
        if stream_key is None:
            stream_key = key
        elif stream_key != key:
            raise DecisionLifecycleReconstructionError(
                "lifecycle evidence stream mixes decision_id, decision_version, or tenant_id",
            )
        parsed.append(
            _ParsedLifecycleEvidence(
                event_id=validate_event_id(event.event_id),
                payload=payload,
            ),
        )
    assert stream_key is not None
    return stream_key, tuple(parsed)


def project_decision_lifecycle_snapshot(
    events: tuple[RuntimeEvent, ...],
) -> DecisionLifecycleDiagnosticSnapshot:
    """Reconstruct one exact-version lifecycle snapshot from lifecycle RuntimeEvents."""
    stream_key, parsed_events = _parse_lifecycle_events(events)
    seen_event_ids: dict[EventId, DecisionLifecycleSignalPayloadV1] = {}
    seen_transition_indexes: dict[int, DecisionLifecycleSignalPayloadV1] = {}
    current_stage = DecisionLifecycleStage.PROPOSAL
    transition_count = 0
    resolution_outcome: DecisionResolution | None = None
    finalization_disposition: DecisionDurableFinalizationDisposition | None = None
    terminal = False
    evidence_event_ids: list[EventId] = []
    started_seen = False
    expected_next_transition_index = 1

    ordered = sorted(
        parsed_events,
        key=lambda item: (
            item.payload.transition_index,
            _PHASE_ORDER[DecisionLifecycleSignalPhase(item.payload.phase)],
            str(item.event_id),
        ),
    )

    for evidence in ordered:
        payload = evidence.payload
        if evidence.event_id in seen_event_ids:
            if seen_event_ids[evidence.event_id] != payload:
                raise DecisionLifecycleReconstructionError(
                    "duplicate lifecycle evidence event_id with conflicting payload",
                )
            continue
        seen_event_ids[evidence.event_id] = payload
        phase = DecisionLifecycleSignalPhase(payload.phase)

        if phase is DecisionLifecycleSignalPhase.STARTED:
            if started_seen:
                raise DecisionLifecycleReconstructionError(
                    "lifecycle evidence contains duplicate started signal",
                )
            started_seen = True
            if payload.transition_index != 0:
                raise DecisionLifecycleReconstructionError(
                    "started signal requires transition_index 0",
                )
            current_stage = DecisionLifecycleStage.PROPOSAL
            transition_count = 0
            evidence_event_ids.append(evidence.event_id)
            continue

        if not started_seen:
            raise DecisionLifecycleReconstructionError(
                "lifecycle evidence must begin with started signal",
            )

        if phase is DecisionLifecycleSignalPhase.TRANSITIONED:
            if payload.from_stage is None or payload.to_stage is None:
                raise DecisionLifecycleReconstructionError(
                    "transitioned signal missing from_stage or to_stage",
                )
            if payload.transition_index != expected_next_transition_index:
                raise DecisionLifecycleReconstructionError(
                    "lifecycle evidence transition_index gap or reorder detected",
                )
            from_stage = DecisionLifecycleStage(payload.from_stage)
            to_stage = DecisionLifecycleStage(payload.to_stage)
            if from_stage is not current_stage:
                raise DecisionLifecycleReconstructionError(
                    "lifecycle evidence transitioned from_stage does not match "
                    "reconstructed current stage",
                )
            try:
                validate_lifecycle_transition(from_stage=from_stage, to_stage=to_stage)
            except ValueError as exc:
                raise DecisionLifecycleReconstructionError(str(exc)) from exc
            prior = seen_transition_indexes.get(payload.transition_index)
            if prior is not None and prior != payload:
                raise DecisionLifecycleReconstructionError(
                    "conflicting duplicate transitioned evidence at same transition_index",
                )
            seen_transition_indexes[payload.transition_index] = payload
            current_stage = to_stage
            transition_count = payload.transition_index
            expected_next_transition_index = payload.transition_index + 1
            evidence_event_ids.append(evidence.event_id)
            continue

        if phase is DecisionLifecycleSignalPhase.RESOLVED:
            if payload.resolution_outcome is None:
                raise DecisionLifecycleReconstructionError(
                    "resolved signal missing resolution_outcome",
                )
            if payload.transition_index != transition_count:
                raise DecisionLifecycleReconstructionError(
                    "resolved signal transition_index does not match reconstructed count",
                )
            resolution_outcome = DecisionResolution(payload.resolution_outcome)
            evidence_event_ids.append(evidence.event_id)
            continue

        if phase is DecisionLifecycleSignalPhase.FINALIZED:
            if payload.finalization_disposition is None:
                raise DecisionLifecycleReconstructionError(
                    "finalized signal missing finalization_disposition",
                )
            if payload.transition_index != transition_count:
                raise DecisionLifecycleReconstructionError(
                    "finalized signal transition_index does not match reconstructed count",
                )
            finalization_disposition = DecisionDurableFinalizationDisposition(
                payload.finalization_disposition,
            )
            evidence_event_ids.append(evidence.event_id)
            continue

        if phase is DecisionLifecycleSignalPhase.TERMINAL:
            if current_stage is not DecisionLifecycleStage.TERMINAL:
                raise DecisionLifecycleReconstructionError(
                    "terminal signal requires reconstructed stage terminal",
                )
            if payload.transition_index != transition_count:
                raise DecisionLifecycleReconstructionError(
                    "terminal signal transition_index does not match reconstructed count",
                )
            terminal = True
            evidence_event_ids.append(evidence.event_id)
            continue

        raise DecisionLifecycleReconstructionError(
            f"unsupported lifecycle evidence phase: {payload.phase!r}",
        )

    if not started_seen:
        raise DecisionLifecycleReconstructionError(
            "lifecycle evidence must include started signal",
        )

    return DecisionLifecycleDiagnosticSnapshot(
        decision_id=stream_key.decision_id,
        decision_version=stream_key.decision_version,
        tenant_id=stream_key.tenant_id,
        current_stage=current_stage,
        transition_count=transition_count,
        resolution_outcome=resolution_outcome,
        finalization_disposition=finalization_disposition,
        terminal=terminal,
        evidence_event_ids=tuple(evidence_event_ids),
    )


_PHASE_ORDER: dict[DecisionLifecycleSignalPhase, int] = {
    DecisionLifecycleSignalPhase.STARTED: 0,
    DecisionLifecycleSignalPhase.TRANSITIONED: 1,
    DecisionLifecycleSignalPhase.RESOLVED: 2,
    DecisionLifecycleSignalPhase.FINALIZED: 3,
    DecisionLifecycleSignalPhase.TERMINAL: 4,
}
