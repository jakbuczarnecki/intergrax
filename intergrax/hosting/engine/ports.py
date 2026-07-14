# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Engine dependency-inversion ports (APP-HOST-W2)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from intergrax.hosting.instance.contracts import (
    HostedApplicationInstanceAcquisitionResult,
    HostedApplicationInstanceIdentity,
    HostedApplicationInstanceLeasePort,
)

# Backward-compatible re-exports for engine internals.
__all__ = [
    "HostedApplicationInstanceAcquisitionResult",
    "HostedApplicationInstanceGuardPort",
    "HostedApplicationInstanceIdentity",
    "HostedApplicationInstanceLeasePort",
    "HostedApplicationRuntime",
]

if TYPE_CHECKING:
    from intergrax.hosting.contracts.context import HostedApplicationContext


@runtime_checkable
class HostedApplicationRuntime(Protocol):
    """Opaque hosted application runtime lifecycle protocol."""

    async def start(self, context: HostedApplicationContext) -> None: ...

    async def stop(self, context: HostedApplicationContext) -> None: ...

    async def ready(self, context: HostedApplicationContext) -> bool: ...


@runtime_checkable
class HostedApplicationInstanceGuardPort(Protocol):
    """Dependency-inversion port for single-instance ownership."""

    async def acquire(
        self,
        identity: HostedApplicationInstanceIdentity,
    ) -> HostedApplicationInstanceAcquisitionResult: ...
