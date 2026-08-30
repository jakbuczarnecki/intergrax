# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical Decision System identity contracts (DS-CORE-01)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import NewType
from uuid import uuid4

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    validate_attempt_id,
    validate_execution_id,
    validate_run_id,
    validate_task_id,
)

DecisionId = NewType("DecisionId", str)

_DECISION_ID_PREFIX = "decision_"
_CANONICAL_SUFFIX = re.compile(r"^[0-9a-f]{32}$")


def _validate_canonical_decision_id(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be str, got {type(value).__name__}")
    if not value or not value.strip():
        raise ValueError(f"{label} must be non-empty and not whitespace-only")
    if value != value.strip():
        raise ValueError(f"{label} must not contain leading or trailing whitespace")
    if not value.startswith(_DECISION_ID_PREFIX):
        raise ValueError(f"{label} must start with {_DECISION_ID_PREFIX!r}")
    suffix = value[len(_DECISION_ID_PREFIX) :]
    if not _CANONICAL_SUFFIX.fullmatch(suffix):
        raise ValueError(f"{label} suffix must match [0-9a-f]{{32}}")
    return value


def validate_decision_id(value: object) -> DecisionId:
    return DecisionId(_validate_canonical_decision_id(value, "DecisionId"))


def mint_decision_id() -> DecisionId:
    return DecisionId(f"{_DECISION_ID_PREFIX}{uuid4().hex}")


def validate_decision_version(value: object) -> int:
    if isinstance(value, DecisionVersion):
        return value.value
    if type(value) is not int or isinstance(value, bool):
        raise TypeError(
            f"DecisionVersion must be int, got {type(value).__name__}",
        )
    if value < 1:
        raise ValueError("DecisionVersion must be a positive int >= 1")
    return value


@dataclass(frozen=True, slots=True)
class DecisionVersion:
    """Immutable positive decision version starting at 1."""

    value: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", validate_decision_version(self.value))


def initial_decision_version() -> DecisionVersion:
    return DecisionVersion(1)


def next_decision_version(current: DecisionVersion) -> DecisionVersion:
    if type(current) is not DecisionVersion:
        raise TypeError("current must be DecisionVersion")
    return DecisionVersion(current.value + 1)


def _validate_scope_field(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be str, got {type(value).__name__}")
    if not value or not value.strip():
        raise ValueError(f"{label} must be non-empty and not whitespace-only")
    if value != value.strip():
        raise ValueError(f"{label} must not contain leading or trailing whitespace")
    return value


@dataclass(frozen=True, slots=True)
class DecisionScope:
    """Domain-neutral authority scope: namespace + subject."""

    namespace: str
    subject: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "namespace",
            _validate_scope_field(self.namespace, "DecisionScope.namespace"),
        )
        object.__setattr__(
            self,
            "subject",
            _validate_scope_field(self.subject, "DecisionScope.subject"),
        )


def _validate_tenant_id(value: object) -> str:
    if type(value) is not str:
        raise TypeError(
            f"tenant_id must be str, got {type(value).__name__}",
        )
    if not value or value != value.strip():
        raise ValueError(
            "tenant_id must be non-empty without surrounding whitespace",
        )
    return value


@dataclass(frozen=True, slots=True)
class DecisionExecutionLineage:
    """Binds a decision to the Nexus execution tree.

  ``execution_id`` is optional when a decision is created before ExecutionId mint.
    """

    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", validate_task_id(self.task_id))
        object.__setattr__(self, "run_id", validate_run_id(self.run_id))
        object.__setattr__(self, "attempt_id", validate_attempt_id(self.attempt_id))
        if self.execution_id is not None:
            object.__setattr__(
                self,
                "execution_id",
                validate_execution_id(self.execution_id),
            )


@dataclass(frozen=True, slots=True)
class DecisionIdentity:
    """Canonical authority envelope for one decision version in execution context."""

    decision_id: DecisionId
    version: DecisionVersion
    scope: DecisionScope
    tenant_id: str
    execution: DecisionExecutionLineage

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_id", validate_decision_id(self.decision_id))
        if type(self.version) is not DecisionVersion:
            raise TypeError("DecisionIdentity.version must be DecisionVersion")
        if type(self.scope) is not DecisionScope:
            raise TypeError("DecisionIdentity.scope must be DecisionScope")
        object.__setattr__(self, "tenant_id", _validate_tenant_id(self.tenant_id))
        if type(self.execution) is not DecisionExecutionLineage:
            raise TypeError("DecisionIdentity.execution must be DecisionExecutionLineage")
