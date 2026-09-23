# © Artur Czarnecki. All rights reserved.

"""Sandbox session manager-backed durable wiring binding resolver (composition root)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.runtime.sandbox.manager import SandboxSessionManager
from intergrax.tools.durable_invocation_wiring_binding_resolver import (
    DurableToolInvocationWiringBindingResolutionError,
    DurableToolInvocationWiringBindingResolver,
)
from intergrax.tools.invocation_wiring import (
    FixedSandboxSessionWiringResolver,
    ToolInvocationWiringResolver,
)


@dataclass(frozen=True, slots=True)
class SandboxSessionManagerDurableWiringBindingResolver:
    """Materialize ``fixed_sandbox_session`` wiring from durable session identifiers."""

    sandbox_manager: SandboxSessionManager

    def resolve_fixed_sandbox_session_wiring(
        self,
        *,
        sandbox_session_id: str,
        tenant_id: str,
        task_id: str,
    ) -> ToolInvocationWiringResolver:
        session = self.sandbox_manager.resolve_session(
            session_id=sandbox_session_id,
            tenant_id=tenant_id,
            task_id=task_id,
        )
        if session is None:
            raise DurableToolInvocationWiringBindingResolutionError(
                "sandbox_session_unavailable",
                "sandbox session could not be resolved for durable reference",
            )
        return FixedSandboxSessionWiringResolver(sandbox_session=session)


def as_durable_wiring_binding_resolver(
    sandbox_manager: SandboxSessionManager,
) -> DurableToolInvocationWiringBindingResolver:
    return SandboxSessionManagerDurableWiringBindingResolver(
        sandbox_manager=sandbox_manager,
    )


__all__ = [
    "SandboxSessionManagerDurableWiringBindingResolver",
    "as_durable_wiring_binding_resolver",
]
