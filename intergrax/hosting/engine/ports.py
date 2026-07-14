# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Engine dependency-inversion ports (APP-HOST-W2)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from intergrax.hosting.instance.contracts import (
    HostedApplicationInstanceIdentity,
    HostedApplicationInstanceLeasePublicView,
)

if TYPE_CHECKING:
    from intergrax.hosting.contracts.context import HostedApplicationContext


@runtime_checkable
class HostedApplicationRuntime(Protocol):
    """Opaque hosted application runtime lifecycle protocol."""

    async def start(self, context: HostedApplicationContext) -> None: ...

    async def stop(self, context: HostedApplicationContext) -> None: ...

    async def ready(self, context: HostedApplicationContext) -> bool: ...


@runtime_checkable
class HostedApplicationInstanceLeasePort(Protocol):
    """Lease handle returned by the instance guard port."""

    def is_valid(self) -> bool: ...

    def verify_ownership(self) -> None: ...

    def public_view(self) -> HostedApplicationInstanceLeasePublicView: ...

    async def release(self) -> None: ...


@runtime_checkable
class HostedApplicationInstanceGuardPort(Protocol):
    """Dependency-inversion port for single-instance ownership."""

    async def acquire(
        self,
        identity: HostedApplicationInstanceIdentity,
    ) -> HostedApplicationInstanceLeasePort: ...
