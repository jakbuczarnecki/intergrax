# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Scoped typed service registry port for hosted application contexts."""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


class HostedApplicationServiceRegistryError(RuntimeError):
    """Base error for hosted application service registry operations."""


class HostedApplicationServiceRegistryDuplicateError(HostedApplicationServiceRegistryError):
    """Raised when a duplicate service registration is attempted."""


class HostedApplicationServiceRegistryStateError(HostedApplicationServiceRegistryError):
    """Raised when registry state forbids an operation."""


class HostedApplicationServiceRegistryCompatibilityError(HostedApplicationServiceRegistryError):
    """Raised when a service is incompatible with its registration type."""


class HostedApplicationServiceRegistryMissingError(HostedApplicationServiceRegistryError):
    """Raised when a required service is not registered."""


@runtime_checkable
class HostedApplicationServiceRegistryPort(Protocol):
    """Structural port for instance-scoped typed service registries."""

    @property
    def is_sealed(self) -> bool: ...

    @property
    def is_closed(self) -> bool: ...

    def register(
        self,
        service_type: type[T],
        service: T,
        *,
        replace: bool = False,
    ) -> None: ...

    def optional(self, service_type: type[T]) -> T | None: ...

    def require(self, service_type: type[T]) -> T: ...

    def contains(self, service_type: type[object]) -> bool: ...

    def seal(self) -> None: ...

    def close(self) -> None: ...

    def diagnostic_view(self) -> tuple[str, ...]: ...
