# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Active task registry read adapter for recovery fulfillment task correlation."""

from __future__ import annotations

from intergrax.autonomous_work.worker_recovery_capability_fulfillment_episode_context_ports import (
    WorkerRecoveryFulfillmentTaskContext,
    WorkerRecoveryFulfillmentTaskContextReadPort,
)
from intergrax.contracts.execution_identity import RunId, validate_run_id
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry


class ActiveTaskRegistryFulfillmentTaskContextReader:
    """Process-local task/tenant projection from in-flight execution bindings."""

    def resolve_task_context(
        self,
        *,
        run_id: RunId | None,
    ) -> WorkerRecoveryFulfillmentTaskContext | None:
        if run_id is None:
            return None
        validated_run_id = validate_run_id(run_id)
        task_id = ActiveTaskRegistry.peek_task_id_for_run(validated_run_id)
        if task_id is None:
            return None
        binding = ActiveTaskRegistry.peek_binding(task_id)
        if binding is None:
            return None
        if binding.run_id != validated_run_id:
            return None
        return WorkerRecoveryFulfillmentTaskContext(
            task_id=binding.task_id,
            tenant_id=binding.task.tenant_id,
            run_id=validated_run_id,
            attempt_id=None,
        )


__all__ = [
    "ActiveTaskRegistryFulfillmentTaskContextReader",
]
