# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Strict Slack timestamp validation for knowledge-read surfaces."""

from __future__ import annotations

import re

_SLACK_TS_RE = re.compile(r"^[0-9]+\.[0-9]{6}$")
_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_MAX_SLACK_TIMESTAMP_LEN = 32
_INVALID_SLACK_TIMESTAMP = "invalid Slack timestamp"


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


__all__ = ["validate_slack_timestamp"]
