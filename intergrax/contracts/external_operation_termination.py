# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider native termination port (W4-D) — signal only, no CAS or permits."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.contracts.external_operation_identity import ExternalOperationIdentity


@dataclass(frozen=True, slots=True)
class ExternalOperationCapabilities:
    """Declared provider / boundary cancellation affordances."""

    supports_native_cancel: bool
    supports_stream_abort: bool
    supports_remote_termination: bool


class TerminationOutcome(StrEnum):
    """Observed result of a physical termination attempt (not durable state)."""

    PHYSICAL_STOP_CONFIRMED = "physical_stop_confirmed"
    TRANSPORT_CLOSED = "transport_closed"
    NOT_SUPPORTED = "not_supported"
    SIGNAL_FAILED = "signal_failed"


@dataclass(frozen=True, slots=True)
class TerminationResult:
    """Provider-side termination observation — input to CAS terminalization."""

    outcome: TerminationOutcome

    @staticmethod
    def physical_stop_confirmed() -> TerminationResult:
        return TerminationResult(outcome=TerminationOutcome.PHYSICAL_STOP_CONFIRMED)

    @staticmethod
    def transport_closed() -> TerminationResult:
        return TerminationResult(outcome=TerminationOutcome.TRANSPORT_CLOSED)

    @staticmethod
    def not_supported() -> TerminationResult:
        return TerminationResult(outcome=TerminationOutcome.NOT_SUPPORTED)

    @staticmethod
    def signal_failed() -> TerminationResult:
        return TerminationResult(outcome=TerminationOutcome.SIGNAL_FAILED)


@runtime_checkable
class ExternalOperationTerminationPort(Protocol):
    """Native cancel, transport close, stream abort — no retry or durable CAS."""

    async def terminate(
        self,
        identity: ExternalOperationIdentity,
    ) -> TerminationResult:
        """Send provider-specific termination for one physical attempt."""
        ...
