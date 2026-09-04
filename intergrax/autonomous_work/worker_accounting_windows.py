# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""UTC accounting window calculation for worker budget windows (AW-5B)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from intergrax.contracts.autonomous_work.ids import WorkerInstanceId
from intergrax.contracts.autonomous_work.worker_budget_accounting import (
    WorkerAccountingWindow,
    WorkerAccountingWindowKind,
)


def _require_utc(at: datetime) -> datetime:
    if at.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware UTC")
    normalized = at.astimezone(UTC)
    if normalized.utcoffset() != timedelta(0):
        raise ValueError("timestamp must be UTC")
    return normalized


def daily_window_bounds(at: datetime) -> tuple[datetime, datetime]:
    """Return inclusive UTC day start and exclusive next-day boundary."""
    normalized = _require_utc(at)
    window_start = normalized.replace(hour=0, minute=0, second=0, microsecond=0)
    return window_start, window_start + timedelta(days=1)


def monthly_window_bounds(at: datetime) -> tuple[datetime, datetime]:
    """Return inclusive UTC month start and exclusive next-month boundary."""
    normalized = _require_utc(at)
    window_start = normalized.replace(
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    if window_start.month == 12:
        window_end = window_start.replace(year=window_start.year + 1, month=1)
    else:
        window_end = window_start.replace(month=window_start.month + 1)
    return window_start, window_end


def worker_accounting_window(
    *,
    worker_instance_id: WorkerInstanceId,
    window_kind: WorkerAccountingWindowKind,
    at: datetime,
) -> WorkerAccountingWindow:
    """Build canonical immutable window identity for ``at`` (admission/start time)."""
    if window_kind is WorkerAccountingWindowKind.DAILY:
        window_start, window_end = daily_window_bounds(at)
    elif window_kind is WorkerAccountingWindowKind.MONTHLY:
        window_start, window_end = monthly_window_bounds(at)
    else:
        raise ValueError(f"unsupported window kind: {window_kind}")
    return WorkerAccountingWindow(
        worker_instance_id=worker_instance_id,
        window_kind=window_kind,
        window_start=window_start,
        window_end=window_end,
    )


def window_identity_key(window: WorkerAccountingWindow) -> tuple[str, str, str]:
    """Stable persistence key for one immutable window row."""
    if window.window_kind is WorkerAccountingWindowKind.DAILY:
        day_key = window.window_start.date().isoformat()
        return (
            window.worker_instance_id.strip(),
            window.window_kind.value,
            day_key,
        )
    if window.window_kind is WorkerAccountingWindowKind.MONTHLY:
        month_key = f"{window.window_start.year:04d}-{window.window_start.month:02d}"
        return (
            window.worker_instance_id.strip(),
            window.window_kind.value,
            month_key,
        )
    raise ValueError(f"unsupported window kind: {window.window_kind}")
