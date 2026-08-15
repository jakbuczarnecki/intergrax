# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical execution identity carriers (OBSERVABILITY §5, TRACE-1A)."""

from __future__ import annotations

import re
from typing import NewType
from uuid import uuid4

TaskId = NewType("TaskId", str)
RunId = NewType("RunId", str)
AttemptId = NewType("AttemptId", str)

_CANONICAL_SUFFIX = re.compile(r"^[0-9a-f]{32}$")


def _validate_canonical_id(value: object, prefix: str, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be str, got {type(value).__name__}")
    if not value or not value.strip():
        raise ValueError(f"{label} must be non-empty and not whitespace-only")
    if value != value.strip():
        raise ValueError(f"{label} must not contain leading or trailing whitespace")
    if not value.startswith(prefix):
        raise ValueError(f"{label} must start with {prefix!r}")
    suffix = value[len(prefix):]
    if not _CANONICAL_SUFFIX.fullmatch(suffix):
        raise ValueError(f"{label} suffix must match [0-9a-f]{{32}}")
    return value


def validate_task_id(value: object) -> TaskId:
    return TaskId(_validate_canonical_id(value, "task_", "TaskId"))


def validate_run_id(value: object) -> RunId:
    return RunId(_validate_canonical_id(value, "run_", "RunId"))


def validate_attempt_id(value: object) -> AttemptId:
    return AttemptId(_validate_canonical_id(value, "attempt_", "AttemptId"))


def mint_task_id() -> TaskId:
    return TaskId(f"task_{uuid4().hex}")


def mint_run_id() -> RunId:
    return RunId(f"run_{uuid4().hex}")


def mint_attempt_id() -> AttemptId:
    return AttemptId(f"attempt_{uuid4().hex}")
