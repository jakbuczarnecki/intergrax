# © Artur Czarnecki. All rights reserved.

"""Cooperative cancellation helpers — import submodules directly to avoid import cycles."""

from intergrax.runtime.cancellation.coordinator import (
    CANCELLATION_REASON_KEY,
    CANCELLATION_REQUESTED_KEY,
    CancellationCoordinator,
    CooperativeCancellationAbort,
    cooperative_delay_seconds,
)

__all__ = [
    "CANCELLATION_REASON_KEY",
    "CANCELLATION_REQUESTED_KEY",
    "CancellationCoordinator",
    "CooperativeCancellationAbort",
    "cooperative_delay_seconds",
]
