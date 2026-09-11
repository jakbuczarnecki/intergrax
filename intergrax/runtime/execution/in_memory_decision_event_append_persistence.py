# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""In-memory reference decision event append store (W3-C)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from threading import Lock

from intergrax.contracts.decision_event_append import (
    DecisionEvent,
    DuplicateDecisionEventError,
    StaleDecisionEventAppendError,
)
from intergrax.contracts.decision_finalization import DecisionFinalizationKey
from intergrax.contracts.decision_identity import validate_decision_id
from intergrax.runtime.execution.decision_event_payload_codec import (
    DecisionEventPayloadCodecRegistry,
)
from intergrax.runtime.execution.decision_event_record import DecisionEventRecord


@dataclass(frozen=True, slots=True)
class _StoredEventWire:
    event_type: str
    occurred_at_utc: str
    payload_type: str
    payload_json: str
    event_sequence: int


class InMemoryDecisionEventAppendPersistence:
    """Reference compare-and-append event stream for unit tests."""

    __slots__ = ("_codec", "_events_by_id", "_last_sequence", "_lock")

    def __init__(self, *, payload_codecs: DecisionEventPayloadCodecRegistry) -> None:
        self._codec = payload_codecs
        self._lock = Lock()
        self._last_sequence: dict[DecisionFinalizationKey, int] = {}
        self._events_by_id: dict[tuple[DecisionFinalizationKey, str], _StoredEventWire] = {}

    def last_sequence(self, *, key: DecisionFinalizationKey) -> int:
        with self._lock:
            return self._last_sequence.get(key, 0)

    def append(
        self,
        *,
        key: DecisionFinalizationKey,
        event: DecisionEvent,
        expected_last_sequence: int,
    ) -> DecisionEvent:
        if str(event.decision_id) != str(key.decision_id):
            raise ValueError("decision event decision_id does not match finalization key")
        payload_type, payload_wire = self._codec.encode(event.payload)
        payload_json = json.dumps(payload_wire, separators=(",", ":"), sort_keys=True)
        id_key = (key, event.event_id)

        with self._lock:
            existing = self._events_by_id.get(id_key)
            if existing is not None:
                if (
                    existing.event_type == event.event_type
                    and existing.occurred_at_utc == event.occurred_at_utc
                    and existing.payload_type == payload_type
                    and existing.payload_json == payload_json
                ):
                    return self._wire_to_record(key=key, event_id=event.event_id, wire=existing)
                raise DuplicateDecisionEventError(
                    f"event_id={event.event_id!r} already exists with different payload",
                )

            current_last = self._last_sequence.get(key, 0)
            if current_last != expected_last_sequence:
                raise StaleDecisionEventAppendError(
                    f"expected last_sequence={expected_last_sequence}, actual={current_last}",
                )
            next_sequence = expected_last_sequence + 1
            self._last_sequence[key] = next_sequence
            self._events_by_id[id_key] = _StoredEventWire(
                event_type=event.event_type,
                occurred_at_utc=event.occurred_at_utc,
                payload_type=payload_type,
                payload_json=payload_json,
                event_sequence=next_sequence,
            )
            wire = self._events_by_id[id_key]

        return self._wire_to_record(key=key, event_id=event.event_id, wire=wire)

    def _wire_to_record(
        self,
        *,
        key: DecisionFinalizationKey,
        event_id: str,
        wire: _StoredEventWire,
    ) -> DecisionEventRecord:
        payload = self._codec.decode(
            payload_type=wire.payload_type,
            payload=json.loads(wire.payload_json),
        )
        return DecisionEventRecord(
            event_id=event_id,
            decision_id=validate_decision_id(str(key.decision_id)),
            event_sequence=wire.event_sequence,
            event_type=wire.event_type,
            occurred_at_utc=wire.occurred_at_utc,
            payload=payload,
        )
