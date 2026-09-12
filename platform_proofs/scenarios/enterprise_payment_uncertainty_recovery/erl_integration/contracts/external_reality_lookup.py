"""Port — read authoritative external effect reality without ERL or payment imports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ExternalRealitySnapshot:
    """Normalized SoR row — payment semantics stay in scenario adapters only."""

    correlation_id: str
    external_effect_reference: str
    terminal_outcome: str
    funds_captured: bool
    truth_availability_state: str
    sor_transaction_ref: str | None


class ExternalRealityLookupPort(Protocol):
    """Replaceable external truth source (PostgreSQL lab, in-memory tests)."""

    def lookup_by_correlation_id(self, correlation_id: str) -> ExternalRealitySnapshot:
        """Return authoritative reality for the generic external-effect correlation id."""
        ...
