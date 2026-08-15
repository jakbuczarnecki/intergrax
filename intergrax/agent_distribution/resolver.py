# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Dependency resolver port and deterministic in-memory adapter (AP-7 §15.4)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.agent_distribution.dependency import (
    DependencyResolverInput,
    MaterializedAgentClosureEntry,
    MaterializedLockPackage,
)

_NON_EMPTY = Field(min_length=1)


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


class ResolvedDependencyClosure(BaseModel):
    """Normalized resolver output — resolved facts only (§15.4)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    resolver_algorithm_id: str = _NON_EMPTY
    resolver_algorithm_version: str = _NON_EMPTY
    python_version: str = _NON_EMPTY
    packages: tuple[MaterializedLockPackage, ...]
    transitive_agent_closure: tuple[MaterializedAgentClosureEntry, ...] = ()

    @field_validator("resolver_algorithm_id", "resolver_algorithm_version", "python_version")
    @classmethod
    def _strip_fields(cls, value: str) -> str:
        return _strip_required(value)


class DependencyResolver(Protocol):
    """Implementation-agnostic resolver boundary — no package manager I/O."""

    def resolve(self, resolver_input: DependencyResolverInput) -> ResolvedDependencyClosure:
        """Resolve a fully pinned dependency closure for one resolver input."""


class CallableDependencyResolver:
    """Test and adapter hook — delegates to one deterministic callable."""

    def __init__(
        self,
        resolver: Callable[[DependencyResolverInput], ResolvedDependencyClosure],
    ) -> None:
        self._resolver = resolver

    def resolve(self, resolver_input: DependencyResolverInput) -> ResolvedDependencyClosure:
        return self._resolver(resolver_input)
