# © Artur Czarnecki. All rights reserved.

"""MEM-6.1: retention helper for memory stores."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from intergrax.runtime.task_memory.retention import is_record_expired

pytestmark = pytest.mark.unit


def test_is_record_expired_disabled_when_retention_days_none() -> None:
    updated = datetime.now(timezone.utc).isoformat()
    assert is_record_expired(updated, retention_days=None) is False


def test_is_record_expired_disabled_when_retention_days_zero() -> None:
    updated = datetime.now(timezone.utc).isoformat()
    assert is_record_expired(updated, retention_days=0) is False


def test_is_record_expired_true_for_stale_record() -> None:
    stale = datetime.now(timezone.utc) - timedelta(days=40)
    assert is_record_expired(stale.isoformat(), retention_days=30) is True


def test_is_record_expired_false_for_recent_record() -> None:
    recent = datetime.now(timezone.utc) - timedelta(days=2)
    assert is_record_expired(recent.isoformat(), retention_days=30) is False


def test_is_record_expired_treats_naive_timestamp_as_utc() -> None:
    stale_naive = (datetime.now(timezone.utc) - timedelta(days=10)).replace(tzinfo=None)
    assert is_record_expired(stale_naive.isoformat(), retention_days=7) is True


def test_is_record_expired_false_for_invalid_timestamp() -> None:
    assert is_record_expired("not-a-timestamp", retention_days=30) is False
