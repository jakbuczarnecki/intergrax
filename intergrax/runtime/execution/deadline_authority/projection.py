# © Artur Czarnecki. All rights reserved.

"""Derive process-local deadline projection from durable authority."""

from __future__ import annotations

from datetime import datetime, timedelta

from intergrax.contracts.execution_deadline.clock import MonotonicClockPort, UtcClockPort
from intergrax.contracts.execution_deadline.projection import ExecutionDeadlineProjection
from intergrax.contracts.execution_deadline.snapshot import ExecutionDeadlineAuthoritySnapshot


def project_execution_deadline(
    snapshot: ExecutionDeadlineAuthoritySnapshot,
    *,
    utc_clock: UtcClockPort,
    monotonic_clock: MonotonicClockPort,
) -> ExecutionDeadlineProjection:
    now_utc = utc_clock.now_utc()
    deadline_at_utc = snapshot.deadline_at_utc
    if deadline_at_utc is None:
        return ExecutionDeadlineProjection(
            deadline_at_utc=None,
            remaining_seconds=float("inf"),
            is_expired=False,
            global_deadline_monotonic=None,
        )
    if deadline_at_utc <= now_utc:
        return ExecutionDeadlineProjection(
            deadline_at_utc=deadline_at_utc,
            remaining_seconds=0.0,
            is_expired=True,
            global_deadline_monotonic=monotonic_clock.monotonic(),
        )
    remaining = (deadline_at_utc - now_utc).total_seconds()
    if remaining < 0:
        remaining = 0.0
    return ExecutionDeadlineProjection(
        deadline_at_utc=deadline_at_utc,
        remaining_seconds=remaining,
        is_expired=False,
        global_deadline_monotonic=monotonic_clock.monotonic() + remaining,
    )


def project_deadline_at_utc(
    deadline_at_utc: datetime | None,
    *,
    utc_clock: UtcClockPort,
    monotonic_clock: MonotonicClockPort,
) -> ExecutionDeadlineProjection:
    if deadline_at_utc is None:
        return ExecutionDeadlineProjection(
            deadline_at_utc=None,
            remaining_seconds=float("inf"),
            is_expired=False,
            global_deadline_monotonic=None,
        )
    now_utc = utc_clock.now_utc()
    if deadline_at_utc <= now_utc:
        return ExecutionDeadlineProjection(
            deadline_at_utc=deadline_at_utc,
            remaining_seconds=0.0,
            is_expired=True,
            global_deadline_monotonic=monotonic_clock.monotonic(),
        )
    remaining = (deadline_at_utc - now_utc).total_seconds()
    return ExecutionDeadlineProjection(
        deadline_at_utc=deadline_at_utc,
        remaining_seconds=remaining,
        is_expired=False,
        global_deadline_monotonic=monotonic_clock.monotonic() + remaining,
    )


def narrow_child_deadline_at_utc(
    parent_deadline_at_utc: datetime | None,
    *,
    child_max_wall_time_seconds: float | None,
    utc_clock: UtcClockPort,
) -> datetime | None:
    if child_max_wall_time_seconds is None:
        return parent_deadline_at_utc
    now_utc = utc_clock.now_utc()
    child_cap = now_utc + timedelta(seconds=child_max_wall_time_seconds)
    if parent_deadline_at_utc is None:
        return child_cap
    return min(parent_deadline_at_utc, child_cap)
