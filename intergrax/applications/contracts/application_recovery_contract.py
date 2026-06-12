# © Artur Czarnecki. All rights reserved.

"""Application recovery contract for Tier-3 hosts (APP-EVOL-5 · §49.5)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

APPLICATION_RECOVERY_CONTRACT_KEY = "application_recovery_contract.v1"


class HostRestartRecoveryAction(StrEnum):
    """Host process restart behavior."""

    RESUME_SCHEDULER = "resume_scheduler"
    COLD_START_ONLY = "cold_start_only"


class TaskInterruptedRecoveryAction(StrEnum):
    """Interrupted task recovery behavior."""

    RESUME = "resume"
    RESTART = "restart"
    ESCALATE_HITL = "escalate_hitl"


class GraphNodeFailureRecoveryAction(StrEnum):
    """Graph node failure behavior."""

    RETRY_NODE = "retry_node"
    SKIP_WITH_AUDIT = "skip_with_audit"
    ABORT_GRAPH = "abort_graph"


class CorruptCheckpointRecoveryAction(StrEnum):
    """Corrupt checkpoint handling."""

    REPLAY_FROM_SNAPSHOT = "replay_from_snapshot"
    ABORT_WITH_INCIDENT = "abort_with_incident"


class ApplicationRecoveryContract(BaseModel):
    """Explicit host recovery guarantees after failure (§49.5.2)."""

    model_config = ConfigDict(extra="forbid")

    on_host_restart: HostRestartRecoveryAction = HostRestartRecoveryAction.RESUME_SCHEDULER
    on_task_interrupted: TaskInterruptedRecoveryAction = TaskInterruptedRecoveryAction.RESUME
    on_graph_node_failure: GraphNodeFailureRecoveryAction = (
        GraphNodeFailureRecoveryAction.RETRY_NODE
    )
    on_corrupt_checkpoint: CorruptCheckpointRecoveryAction = (
        CorruptCheckpointRecoveryAction.REPLAY_FROM_SNAPSHOT
    )
    max_resume_attempts: int = Field(default=3, ge=1, le=32)
    preserve_snapshot_id: bool = True


def standard_strict_product_recovery_contract() -> ApplicationRecoveryContract:
    """Default recovery posture for STRICT mutating product hosts."""
    return ApplicationRecoveryContract()
