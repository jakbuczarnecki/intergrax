# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical sandbox isolation gate (P0-SAFETY-7)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from intergrax.runtime.sandbox.contracts import SandboxExecCapable
from intergrax.runtime.sandbox.isolation_errors import (
    SandboxIsolationFailureReason,
    SandboxIsolationRequiredError,
)
from intergrax.runtime.sandbox.session import SandboxSession
from intergrax.tools.core.contracts import ToolContract, contract_requires_sandbox_isolation
from intergrax.tools.registry.wiring import ToolWiringContext


@dataclass(frozen=True, slots=True)
class SandboxIsolationAvailability:
    """Resolved sandbox substrate availability for enforcement."""

    session_configured: bool
    host_configured: bool
    healthy: bool = True

    @property
    def available(self) -> bool:
        if not self.healthy:
            return False
        return self.session_configured or self.host_configured


SandboxAvailabilityProvider = Callable[[], SandboxIsolationAvailability]


def _session_configured(raw: object | None) -> bool:
    if raw is None:
        return False
    return isinstance(raw, (SandboxSession, SandboxExecCapable))


def sandbox_availability_from_wiring(
    ctx: ToolWiringContext,
    *,
    healthy: bool = True,
) -> SandboxIsolationAvailability:
    """Derive availability from canonical ``ToolWiringContext`` bindings."""
    raw_session = ctx.sandbox_session or ctx.extras.get("sandbox_session")
    return SandboxIsolationAvailability(
        session_configured=_session_configured(raw_session),
        host_configured=ctx.sandbox_host is not None,
        healthy=healthy,
    )


def sandbox_availability_provider(
    ctx: ToolWiringContext,
    *,
    healthy: bool = True,
) -> SandboxAvailabilityProvider:
    """Build a provider closure for :class:`~intergrax.runtime.nexus.tools.invoker.RuntimeToolInvoker`."""

    def _check() -> SandboxIsolationAvailability:
        return sandbox_availability_from_wiring(ctx, healthy=healthy)

    return _check


def require_sandbox_isolation(
    *,
    contract: ToolContract,
    availability: SandboxIsolationAvailability,
    run_id: str,
    agent_id: str,
) -> None:
    """Fail closed when a catalog tool requires isolation but none is available."""
    if not contract_requires_sandbox_isolation(contract):
        return
    if not availability.healthy:
        raise SandboxIsolationRequiredError(
            run_id=run_id,
            agent_id=agent_id,
            tool_id=contract.tool_id,
            reason=SandboxIsolationFailureReason.UNHEALTHY,
        )
    if not availability.available:
        raise SandboxIsolationRequiredError(
            run_id=run_id,
            agent_id=agent_id,
            tool_id=contract.tool_id,
            reason=SandboxIsolationFailureReason.NOT_CONFIGURED,
        )
