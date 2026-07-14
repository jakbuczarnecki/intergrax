# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from intergrax.hosting.contracts.lifecycle import (
    HostedApplicationLifecycleSnapshot,
    HostedApplicationLifecycleState,
    HostedApplicationShutdownRequestSnapshot,
)

pytestmark = pytest.mark.unit

_AWARE = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)
_NAIVE = datetime(2026, 7, 14, 12, 0)


def test_lifecycle_snapshot_allows_empty_reason_code() -> None:
    snapshot = HostedApplicationLifecycleSnapshot(
        state=HostedApplicationLifecycleState.READY,
        accepting_new_work=True,
        shutdown_requested=False,
        last_transition_at=_AWARE,
        reason_code="",
    )
    assert snapshot.reason_code == ""


def test_lifecycle_snapshot_validates_non_empty_reason_code() -> None:
    snapshot = HostedApplicationLifecycleSnapshot(
        state=HostedApplicationLifecycleState.READY,
        accepting_new_work=True,
        shutdown_requested=False,
        last_transition_at=_AWARE,
        reason_code="ready",
    )
    assert snapshot.reason_code == "ready"


def test_lifecycle_snapshot_rejects_invalid_reason_code() -> None:
    with pytest.raises(ValidationError, match="reason_code"):
        HostedApplicationLifecycleSnapshot(
            state=HostedApplicationLifecycleState.READY,
            accepting_new_work=True,
            shutdown_requested=False,
            last_transition_at=_AWARE,
            reason_code="bad reason",
        )


def test_lifecycle_snapshot_rejects_naive_last_transition_at() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        HostedApplicationLifecycleSnapshot(
            state=HostedApplicationLifecycleState.READY,
            accepting_new_work=True,
            shutdown_requested=False,
            last_transition_at=_NAIVE,
        )


def test_shutdown_request_requires_non_empty_reason_code() -> None:
    with pytest.raises(ValidationError, match="reason_code"):
        HostedApplicationShutdownRequestSnapshot(
            reason_code="",
            requested_at=_AWARE,
        )


def test_shutdown_request_rejects_naive_requested_at() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        HostedApplicationShutdownRequestSnapshot(
            reason_code="operator_stop",
            requested_at=_NAIVE,
        )


def test_shutdown_request_rejects_naive_deadline_at() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        HostedApplicationShutdownRequestSnapshot(
            reason_code="operator_stop",
            requested_at=_AWARE,
            deadline_at=_NAIVE,
        )


def test_shutdown_request_rejects_deadline_before_requested_at() -> None:
    with pytest.raises(ValidationError, match="deadline_at"):
        HostedApplicationShutdownRequestSnapshot(
            reason_code="operator_stop",
            requested_at=_AWARE,
            deadline_at=_AWARE - timedelta(seconds=1),
        )
