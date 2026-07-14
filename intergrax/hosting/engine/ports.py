# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Engine dependency-inversion ports (APP-HOST-W2)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from intergrax.hosting.contracts.context import HostedApplicationProcessIdentity

if TYPE_CHECKING:
    from intergrax.hosting.contracts.context import HostedApplicationContext


@runtime_checkable
class HostedApplicationRuntime(Protocol):
    """Opaque hosted application runtime lifecycle protocol."""

    async def start(self, context: HostedApplicationContext) -> None: ...

    async def stop(self, context: HostedApplicationContext) -> None: ...

    async def ready(self, context: HostedApplicationContext) -> bool: ...


class HostedApplicationInstanceIdentity(BaseModel):
    """Immutable instance identity used for lease acquisition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    application_id: str
    instance_id: str
    profile_digest: str
    process_identity: HostedApplicationProcessIdentity


@runtime_checkable
class HostedApplicationInstanceLeasePort(Protocol):
    """Lease handle returned by the instance guard port."""

    def is_valid(self) -> bool: ...

    async def release(self) -> None: ...


@runtime_checkable
class HostedApplicationInstanceGuardPort(Protocol):
    """Dependency-inversion port for single-instance ownership."""

    async def acquire(
        self,
        identity: HostedApplicationInstanceIdentity,
    ) -> HostedApplicationInstanceLeasePort: ...
