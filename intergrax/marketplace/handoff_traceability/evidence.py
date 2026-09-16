# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""In-memory handoff trace evidence consumer (reference provider)."""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.contracts.marketplace.handoff_traceability import CapabilityHandoffEnvelope


@dataclass
class InMemoryCapabilityHandoffTraceEvidenceConsumer:
    """Records handoff envelopes once per ``handoff_id`` — idempotent trace facts."""

    _by_handoff_id: dict[str, CapabilityHandoffEnvelope] = field(default_factory=dict)

    def record_handoff(self, envelope: CapabilityHandoffEnvelope) -> bool:
        existing = self._by_handoff_id.get(envelope.handoff_id)
        if existing is not None:
            if existing != envelope:
                raise ValueError(
                    "duplicate handoff_id with differing envelope payload",
                )
            return False
        self._by_handoff_id[envelope.handoff_id] = envelope
        return True

    def get(self, handoff_id: str) -> CapabilityHandoffEnvelope | None:
        return self._by_handoff_id.get(handoff_id)

    def recorded_handoff_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_handoff_id))


__all__ = ["InMemoryCapabilityHandoffTraceEvidenceConsumer"]
