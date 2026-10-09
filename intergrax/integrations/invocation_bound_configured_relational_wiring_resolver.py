# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Invocation-bound wiring resolver capturing execution port A (TRACE-X-P5-R2-P3)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.integrations.contracts.configured_relational_store_execution import (
    ConfiguredRelationalStoreExecutionPort,
)
from intergrax.tools.invocation_wiring import (
    ToolInvocationContext,
    ToolInvocationWiring,
    ToolRegistrationWiringView,
)


@dataclass(frozen=True, slots=True)
class InvocationBoundConfiguredRelationalStoreWiringResolver:
    """Directly captures port A — no cache lookup, no materialization."""

    configured_relational_store_execution: ConfiguredRelationalStoreExecutionPort

    def resolve(
        self,
        *,
        tool_id: str,
        invocation_context: ToolInvocationContext,
        registration_wiring: ToolRegistrationWiringView,
    ) -> ToolInvocationWiring:
        _ = tool_id, invocation_context, registration_wiring
        return ToolInvocationWiring(
            configured_relational_store_execution=self.configured_relational_store_execution,
        )


__all__ = ["InvocationBoundConfiguredRelationalStoreWiringResolver"]
