"""Failure model for external-reality lookup at the scenario integration boundary."""

from __future__ import annotations


class ExternalRealityLookupError(Exception):
    """Lookup could not produce a reconciliation snapshot."""


class ExternalRealitySourceUnavailable(ExternalRealityLookupError):
    """PostgreSQL or configured reality source is unreachable."""


class ExternalRealityRecordMissing(ExternalRealityLookupError):
    """No authoritative external-reality row for the supplied correlation reference."""


class ExternalRealityInconsistentState(ExternalRealityLookupError):
    """SoR columns contradict each other — cannot derive a definitive verdict."""
