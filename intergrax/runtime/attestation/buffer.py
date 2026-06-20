# © Artur Czarnecki. All rights reserved.

"""In-memory per-run buffer for boundary events surfaced in Tier-3 API responses."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from intergrax.runtime.attestation.execution_boundary_event import ExecutionBoundaryEventV1
from intergrax.runtime.attestation.host_attestation import HostAttestationEnvelopeV1

if TYPE_CHECKING:
    from intergrax.runtime.attestation.host_attestation import HostAttestationSealer


@dataclass(frozen=True, slots=True)
class StoredBoundaryEvent:
    event: ExecutionBoundaryEventV1
    host_attestation: HostAttestationEnvelopeV1 | None = None


class BoundaryEventBuffer:
    """Thread-safe store keyed by ``run_id`` (PoC response sink)."""

    def __init__(
        self,
        *,
        host_attestation_sealer: HostAttestationSealer | None = None,
    ) -> None:
        self._rows: dict[str, list[StoredBoundaryEvent]] = {}
        self._sequences: dict[str, int] = {}
        self._lock = threading.Lock()
        self._host_attestation_sealer = host_attestation_sealer

    @property
    def host_signing_enabled(self) -> bool:
        return self._host_attestation_sealer is not None

    def append(self, run_id: str, event: ExecutionBoundaryEventV1) -> ExecutionBoundaryEventV1:
        if not run_id:
            return event
        with self._lock:
            sequence = self._sequences.get(run_id, 0) + 1
            self._sequences[run_id] = sequence
            sequenced = event.model_copy(update={"event_sequence": sequence})
            stored = self._maybe_seal(sequenced)
            self._rows.setdefault(run_id, []).append(stored)
            return stored.event

    def _maybe_seal(self, event: ExecutionBoundaryEventV1) -> StoredBoundaryEvent:
        sealer = self._host_attestation_sealer
        if sealer is None:
            return StoredBoundaryEvent(event=event.model_copy(update={"signed": False}))
        signed_event, envelope = sealer.seal_event(event)
        return StoredBoundaryEvent(event=signed_event, host_attestation=envelope)

    def list_for_run(self, run_id: str) -> list[StoredBoundaryEvent]:
        with self._lock:
            return list(self._rows.get(run_id, ()))

    def snapshot_for_run(self, run_id: str) -> list[dict[str, object]]:
        events = sorted(
            self.list_for_run(run_id),
            key=lambda stored: stored.event.event_sequence,
        )
        deliveries: list[dict[str, object]] = []
        for stored in events:
            payload = stored.event.model_dump(mode="json")
            payload["host_attestation"] = (
                stored.host_attestation.model_dump(mode="json")
                if stored.host_attestation is not None
                else None
            )
            deliveries.append(payload)
        return deliveries

    def all_run_ids(self) -> Sequence[str]:
        with self._lock:
            return tuple(self._rows.keys())
