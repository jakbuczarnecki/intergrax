# © Artur Czarnecki. All rights reserved.

from intergrax.runtime.execution.continuation.composition import (
    ExecutionEngineContinuationDependencies,
    wire_execution_continuation_port,
    wire_execution_engine_continuation_dependencies,
)
from intergrax.runtime.execution.continuation.persistence import (
    InMemoryExecutionContinuationStateStore,
    wire_execution_continuation_state_store,
)
from intergrax.runtime.execution.continuation.lifecycle_driver import (
    ExecutionContinuationLifecycleDriver,
)
from intergrax.runtime.execution.continuation.progress_gate import (
    assert_canonical_execution_may_progress,
    execute_canonical_work_when_unblocked,
)
from intergrax.runtime.execution.continuation.service import ExecutionContinuationService

__all__ = [
    "ExecutionContinuationLifecycleDriver",
    "ExecutionContinuationService",
    "ExecutionEngineContinuationDependencies",
    "InMemoryExecutionContinuationStateStore",
    "assert_canonical_execution_may_progress",
    "execute_canonical_work_when_unblocked",
    "wire_execution_continuation_port",
    "wire_execution_continuation_state_store",
    "wire_execution_engine_continuation_dependencies",
]
