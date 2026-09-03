# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Stable typed identifiers for Autonomous Work domain entities (AW-1A)."""

from __future__ import annotations

import re
from typing import NewType
from uuid import uuid4

WorkerDefinitionId = NewType("WorkerDefinitionId", str)
WorkerInstanceId = NewType("WorkerInstanceId", str)
ResponsibilityId = NewType("ResponsibilityId", str)
WorkerGoalId = NewType("WorkerGoalId", str)
WakeUpId = NewType("WakeUpId", str)

_CANONICAL_SUFFIX = re.compile(r"^[0-9a-f]{32}$")


def _validate_canonical_id(value: object, *, prefix: str, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be str, got {type(value).__name__}")
    if not value or not value.strip():
        raise ValueError(f"{label} must be non-empty and not whitespace-only")
    if value != value.strip():
        raise ValueError(f"{label} must not contain leading or trailing whitespace")
    if not value.startswith(prefix):
        raise ValueError(f"{label} must start with {prefix!r}")
    suffix = value[len(prefix) :]
    if not _CANONICAL_SUFFIX.fullmatch(suffix):
        raise ValueError(f"{label} suffix must match [0-9a-f]{{32}}")
    return value


def validate_worker_definition_id(value: object) -> WorkerDefinitionId:
    return WorkerDefinitionId(
        _validate_canonical_id(value, prefix="wdef_", label="WorkerDefinitionId"),
    )


def validate_worker_instance_id(value: object) -> WorkerInstanceId:
    return WorkerInstanceId(
        _validate_canonical_id(value, prefix="winst_", label="WorkerInstanceId"),
    )


def validate_responsibility_id(value: object) -> ResponsibilityId:
    return ResponsibilityId(
        _validate_canonical_id(value, prefix="resp_", label="ResponsibilityId"),
    )


def validate_worker_goal_id(value: object) -> WorkerGoalId:
    return WorkerGoalId(
        _validate_canonical_id(value, prefix="wgoal_", label="WorkerGoalId"),
    )


def validate_wake_up_id(value: object) -> WakeUpId:
    return WakeUpId(
        _validate_canonical_id(value, prefix="wkup_", label="WakeUpId"),
    )


def mint_worker_definition_id() -> WorkerDefinitionId:
    return WorkerDefinitionId(f"wdef_{uuid4().hex}")


def mint_worker_instance_id() -> WorkerInstanceId:
    return WorkerInstanceId(f"winst_{uuid4().hex}")


def mint_responsibility_id() -> ResponsibilityId:
    return ResponsibilityId(f"resp_{uuid4().hex}")


def mint_worker_goal_id() -> WorkerGoalId:
    return WorkerGoalId(f"wgoal_{uuid4().hex}")


def mint_wake_up_id() -> WakeUpId:
    return WakeUpId(f"wkup_{uuid4().hex}")
