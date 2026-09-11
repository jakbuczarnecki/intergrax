# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Concrete immutable decision event records for durable append stores (W3-C)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.decision_event_append import DecisionEventPayload
from intergrax.contracts.decision_identity import DecisionId


@dataclass(frozen=True, slots=True)
class DecisionEventRecord:
    """One persisted or pending decision stream event."""

    event_id: str
    decision_id: DecisionId
    event_sequence: int
    event_type: str
    occurred_at_utc: str
    payload: DecisionEventPayload
