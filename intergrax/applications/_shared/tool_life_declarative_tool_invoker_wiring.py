# © Artur Czarnecki. All rights reserved.

"""Forward TOOL-LIFE durable external-operation store into production tool invoker wiring."""

from __future__ import annotations

from intergrax.applications._shared.tool_life_execution_recovery_wiring import (
    ApplicationToolLifeExecutionRecoveryWiring,
)
from intergrax.runtime.external_operations.external_operation_state_store import (
    ExternalOperationStateStore,
)


def resolve_external_operation_state_store_for_tool_invoker(
    tool_life_execution_recovery: ApplicationToolLifeExecutionRecoveryWiring | None,
) -> ExternalOperationStateStore | None:
    """Materialized durable store only when G-A07 host bootstrap composition is active."""
    if tool_life_execution_recovery is None:
        return None
    return tool_life_execution_recovery.composition.bundle.external_operation_state


__all__ = ["resolve_external_operation_state_store_for_tool_invoker"]
