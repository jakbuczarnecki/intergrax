# © Artur Czarnecki. All rights reserved.

"""Execution Engine reliability contracts (EE-B1.1)."""

from intergrax.contracts.execution_reliability.failure_classification_contract import (
    ExecutionFailureClassifier,
    ExecutionFailureContext,
    ExecutionFailureDecision,
    ExecutionFailureSemanticCategory,
)
from intergrax.contracts.execution_reliability.persistence_failure_semantics import (
    PersistenceFailurePolicy,
    PersistenceFailureSurface,
    resolve_persistence_failure_policy,
)
from intergrax.contracts.execution_reliability.shutdown_contract import (
    EXECUTION_RUNTIME_SHUTDOWN_PHASE_ORDER,
    ExecutionRuntimeShutdownPhase,
)

__all__ = [
    "EXECUTION_RUNTIME_SHUTDOWN_PHASE_ORDER",
    "ExecutionFailureClassifier",
    "ExecutionFailureContext",
    "ExecutionFailureDecision",
    "ExecutionFailureSemanticCategory",
    "ExecutionRuntimeShutdownPhase",
    "PersistenceFailurePolicy",
    "PersistenceFailureSurface",
    "resolve_persistence_failure_policy",
]
