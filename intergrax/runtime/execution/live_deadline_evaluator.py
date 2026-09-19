# © Artur Czarnecki. All rights reserved.

"""Live execution deadline evaluation from monotonic projection (HARNESS-02)."""

from __future__ import annotations

from intergrax.contracts.execution_deadline.clock import MonotonicClockPort
from intergrax.contracts.execution_deadline.projection import ExecutionDeadlineProjection


def execution_live_remaining_seconds(
    projection: ExecutionDeadlineProjection,
    monotonic_clock: MonotonicClockPort,
) -> float | None:
    """Remaining execution time at ``monotonic_clock.now``; ``None`` if unbounded."""
    bound = projection.global_deadline_monotonic
    if bound is None:
        return None
    return max(0.0, bound - monotonic_clock.monotonic())


def execution_is_expired_now(
    projection: ExecutionDeadlineProjection,
    monotonic_clock: MonotonicClockPort,
) -> bool:
    """Whether the process-local monotonic deadline has been reached."""
    bound = projection.global_deadline_monotonic
    if bound is None:
        return False
    return monotonic_clock.monotonic() >= bound
