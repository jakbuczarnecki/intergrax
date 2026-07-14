# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Scoped typed service registry for hosted application contexts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar

from intergrax.hosting.contracts.public_data import stable_service_type_id

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


@dataclass
class HostedApplicationServiceRegistry:
    """Instance-scoped typed service registry for hosted application contexts."""

    _services: dict[type[object], object] = field(default_factory=dict)
    _sealed: bool = False
    _closed: bool = False

    @property
    def is_sealed(self) -> bool:
        return self._sealed

    @property
    def is_closed(self) -> bool:
        return self._closed

    def register(
        self,
        service_type: type[T],
        service: T,
        *,
        replace: bool = False,
    ) -> None:
        if self._closed:
            raise HostedApplicationServiceRegistryStateError("registry is closed")
        if self._sealed:
            raise HostedApplicationServiceRegistryStateError("registry is sealed")
        if not isinstance(service, service_type):
            raise HostedApplicationServiceRegistryCompatibilityError(
                f"service is not compatible with {stable_service_type_id(service_type)}"
            )
        if service_type in self._services and not replace:
            raise HostedApplicationServiceRegistryDuplicateError(
                f"service already registered for {stable_service_type_id(service_type)}"
            )
        self._services[service_type] = service

    def optional(self, service_type: type[T]) -> T | None:
        if self._closed:
            raise HostedApplicationServiceRegistryStateError("registry is closed")
        service = self._services.get(service_type)
        return service  # type: ignore[return-value]

    def require(self, service_type: type[T]) -> T:
        if self._closed:
            raise HostedApplicationServiceRegistryStateError("registry is closed")
        service = self._services.get(service_type)
        if service is None:
            raise HostedApplicationServiceRegistryMissingError(
                f"required service missing: {stable_service_type_id(service_type)}"
            )
        return service  # type: ignore[return-value]

    def contains(self, service_type: type[object]) -> bool:
        if self._closed:
            raise HostedApplicationServiceRegistryStateError("registry is closed")
        return service_type in self._services

    def seal(self) -> None:
        if self._closed:
            raise HostedApplicationServiceRegistryStateError("registry is closed")
        self._sealed = True

    def close(self) -> None:
        self._closed = True
        self._sealed = True
        self._services.clear()

    def diagnostic_view(self) -> tuple[str, ...]:
        """Return deterministic service type identifiers without service objects."""
        return tuple(
            stable_service_type_id(service_type)
            for service_type in sorted(self._services, key=stable_service_type_id)
        )
