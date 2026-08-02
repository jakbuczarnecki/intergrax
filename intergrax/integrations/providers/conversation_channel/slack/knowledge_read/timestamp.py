# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Strict Slack timestamp validation for knowledge-read surfaces."""

from __future__ import annotations

import re
from decimal import Decimal

_SLACK_TS_RE = re.compile(r"^[0-9]+\.[0-9]{6}$")
_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_MAX_SLACK_TIMESTAMP_LEN = 32
_INVALID_SLACK_TIMESTAMP = "invalid Slack timestamp"


def _slack_timestamp_decimal(value: str) -> Decimal:
    seconds, micros = value.split(".", 1)
    return Decimal(seconds) + (Decimal(micros) / Decimal("1000000"))


def compare_slack_timestamps(left: str, right: str) -> int:
    """Compare canonical Slack timestamps without floating-point rounding."""
    left_value = _slack_timestamp_decimal(left)
    right_value = _slack_timestamp_decimal(right)
    if left_value < right_value:
        return -1
    if left_value > right_value:
        return 1
    return 0


def slack_timestamp_in_window(*, value: str, oldest: str, latest: str) -> bool:
    """Return whether ``value`` lies within the inclusive provider window."""
    return (
        compare_slack_timestamps(value, oldest) >= 0
        and compare_slack_timestamps(value, latest) <= 0
    )


def validate_slack_timestamp(value: object) -> str:
    """Validate canonical Slack ``seconds.microseconds`` timestamps byte-exactly."""
    if not isinstance(value, str):
        raise ValueError(_INVALID_SLACK_TIMESTAMP)
    if value == "":
        raise ValueError(_INVALID_SLACK_TIMESTAMP)
    if value != value.strip():
        raise ValueError(_INVALID_SLACK_TIMESTAMP)
    if _ASCII_CONTROL.search(value):
        raise ValueError(_INVALID_SLACK_TIMESTAMP)
    if len(value) > _MAX_SLACK_TIMESTAMP_LEN:
        raise ValueError(_INVALID_SLACK_TIMESTAMP)
    if not _SLACK_TS_RE.match(value):
        raise ValueError(_INVALID_SLACK_TIMESTAMP)
    try:
        parsed = float(value)
    except ValueError:
        raise ValueError(_INVALID_SLACK_TIMESTAMP) from None
    if parsed <= 0:
        raise ValueError(_INVALID_SLACK_TIMESTAMP)
    return value


__all__ = [
    "compare_slack_timestamps",
    "slack_timestamp_in_window",
    "validate_slack_timestamp",
]
