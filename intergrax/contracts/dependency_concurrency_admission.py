# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Dependency concurrency admission contracts (W2-ADR).

This controls in-flight dependency concurrency only.

It does not implement:
- rate limiting
- circuit breaking
- retry
- tenant fairness
- root execution admission
- graph concurrent work width

Distributed enforcement is pluggable via async ports; first implementation is expected
process-local (W2-B).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DependencyConcurrencyAdmissionError(RuntimeError):
    """Base error for dependency concurrency admission."""


class DependencyConcurrencyExceededError(DependencyConcurrencyAdmissionError):
    """No slot available under REJECT overload policy."""


class DependencyConcurrencyAdmissionTimeoutError(DependencyConcurrencyAdmissionError):
    """No slot became available before WAIT_WITH_TIMEOUT elapsed."""


class DependencyConcurrencyPolicyMissingError(DependencyConcurrencyAdmissionError):
    """Admission is active but no policy is configured for the dependency identity."""


def is_dependency_concurrency_admission_error(exc: BaseException) -> bool:
    """True for W2 dependency admission failures (not provider physical errors)."""
    return isinstance(exc, DependencyConcurrencyAdmissionError)


class DependencyConcurrencyOverloadMode(StrEnum):
    """Overload behavior when dependency concurrency slots are saturated."""

    REJECT = "REJECT"
    WAIT_WITH_TIMEOUT = "WAIT_WITH_TIMEOUT"


class DependencyConcurrencyKind(StrEnum):
    """Typed external dependency failure domain (not tenant, not root execution)."""

    TOOL = "TOOL"
    LLM_PROVIDER = "LLM_PROVIDER"
    INTEGRATION = "INTEGRATION"
    RETRIEVER = "RETRIEVER"


def _validate_dependency_value(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("dependency value must be str")
    if value == "":
        raise ValueError("dependency value must not be empty")
    if value != value.strip():
        raise ValueError("dependency value must equal its stripped form")
    return value


@dataclass(frozen=True, slots=True)
class DependencyConcurrencyIdentity:
    """Immutable typed key for one external dependency concurrency pool."""

    kind: DependencyConcurrencyKind
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, DependencyConcurrencyKind):
            raise TypeError("kind must be DependencyConcurrencyKind")
        object.__setattr__(self, "value", _validate_dependency_value(self.value))


@dataclass(frozen=True, slots=True)
class DependencyConcurrencyAdmissionRequest:
    """Identity for one dependency concurrency slot (capacity metadata only)."""

    dependency: DependencyConcurrencyIdentity
    tenant_id: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.dependency, DependencyConcurrencyIdentity):
            raise TypeError("dependency must be DependencyConcurrencyIdentity")
        if self.tenant_id is not None and not isinstance(self.tenant_id, str):
            raise TypeError("tenant_id must be str or None")


class DependencyConcurrencyPolicy(BaseModel):
    """Explicit dependency concurrency policy (no platform default capacity)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_concurrent_calls: int = Field(ge=1)
    overload_mode: DependencyConcurrencyOverloadMode
    wait_timeout_seconds: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_wait_timeout(self) -> DependencyConcurrencyPolicy:
        if self.overload_mode is DependencyConcurrencyOverloadMode.WAIT_WITH_TIMEOUT:
            if self.wait_timeout_seconds is None:
                raise ValueError(
                    "wait_timeout_seconds must be set when overload_mode is WAIT_WITH_TIMEOUT"
                )
        elif self.wait_timeout_seconds is not None:
            raise ValueError(
                "wait_timeout_seconds applies only when overload_mode is WAIT_WITH_TIMEOUT"
            )
        return self


@runtime_checkable
class DependencyConcurrencyPermit(Protocol):
    """Held slot for one in-flight external dependency attempt; release exactly once."""

    async def release(self) -> None:
        """Return the slot to the admission pool."""
        ...


@runtime_checkable
class DependencyConcurrencyAdmissionPort(Protocol):
    """Pluginable dependency concurrency admission (local, distributed, etc.)."""

    async def acquire(
        self,
        request: DependencyConcurrencyAdmissionRequest,
    ) -> DependencyConcurrencyPermit:
        """Reserve one dependency slot or fail fast / time out per policy."""
        ...
