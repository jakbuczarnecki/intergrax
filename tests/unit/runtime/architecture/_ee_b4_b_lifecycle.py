# © Artur Czarnecki. All rights reserved.

"""Shared EE-B4-B reference lifecycle fixtures."""

from __future__ import annotations

from intergrax.contracts.execution_capacity_admission import (
    ExecutionCapacityOverloadMode,
    ExecutionCapacityPolicy,
)
from testing_support.shutdown.reference_lifecycle import (
    ReferenceExecutionShutdownLifecycle,
)


def make_lifecycle(
    *,
    max_roots: int = 2,
    drain_timeout_seconds: float = 30.0,
) -> ReferenceExecutionShutdownLifecycle:
    return ReferenceExecutionShutdownLifecycle(
        capacity_policy=ExecutionCapacityPolicy(
            max_concurrent_root_executions=max_roots,
            overload_mode=ExecutionCapacityOverloadMode.REJECT,
        ),
        drain_timeout_seconds=drain_timeout_seconds,
    )
