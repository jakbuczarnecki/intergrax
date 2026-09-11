# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Stable identity for distributed external operation termination (W4-C)."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from intergrax.contracts.dependency_concurrency_admission import DependencyConcurrencyKind
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    validate_attempt_id,
    validate_execution_id,
)

_OPERATION_ID_SAFE = re.compile(r"[^a-zA-Z0-9._-]+")


@dataclass(frozen=True, slots=True)
class ExternalOperationIdentity:
    """Logical external operation plus one physical attempt binding."""

    execution_id: ExecutionId
    attempt_id: AttemptId
    dependency_kind: DependencyConcurrencyKind
    dependency_identity: str
    operation_id: str
    physical_attempt_sequence: int = 1

    def __post_init__(self) -> None:
        validate_execution_id(self.execution_id)
        validate_attempt_id(self.attempt_id)
        if type(self.dependency_kind) is not DependencyConcurrencyKind:
            raise TypeError("dependency_kind must be DependencyConcurrencyKind")
        if type(self.dependency_identity) is not str or not self.dependency_identity.strip():
            raise ValueError("dependency_identity must be a non-empty str")
        if type(self.operation_id) is not str or not self.operation_id.strip():
            raise ValueError("operation_id must be a non-empty str")
        if type(self.physical_attempt_sequence) is not int or self.physical_attempt_sequence < 1:
            raise ValueError("physical_attempt_sequence must be int >= 1")


def mint_stable_operation_id(
    *,
    execution_id: ExecutionId,
    dependency_kind: DependencyConcurrencyKind,
    dependency_identity: str,
    logical_scope: str,
) -> str:
    """Derive a stable operation_id for retries (not a per-attempt uuid4)."""
    validate_execution_id(execution_id)
    if type(logical_scope) is not str or not logical_scope.strip():
        raise ValueError("logical_scope must be a non-empty str")
    kind = dependency_kind.value
    dep = _sanitize_token(dependency_identity)
    scope = _sanitize_token(logical_scope)
    exec_suffix = str(execution_id).removeprefix("exec_")[:16]
    digest = hashlib.sha256(
        f"{execution_id}|{kind}|{dependency_identity}|{logical_scope}".encode()
    ).hexdigest()[:16]
    return f"extop_{exec_suffix}_{kind}_{dep}_{scope}_{digest}"


def _sanitize_token(value: str) -> str:
    collapsed = _OPERATION_ID_SAFE.sub("_", value.strip())
    return collapsed[:48] if collapsed else "x"
