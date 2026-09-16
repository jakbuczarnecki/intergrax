# © Artur Czarnecki. All rights reserved.

"""Governance domain invariant probe contract (GR-3 read-only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from intergrax.contracts.execution_identity import AttemptId, ExecutionId, RunId, TaskId


@dataclass(frozen=True, slots=True)
class GovernanceMeaningfulSideEffectBinding:
    """Optional four-ID binding evidence for inner governance rules."""

    request_task_id: TaskId
    request_run_id: RunId
    request_attempt_id: AttemptId
    request_execution_id: ExecutionId
    active_task_id: TaskId
    active_run_id: RunId
    active_attempt_id: AttemptId
    active_execution_id: ExecutionId


@dataclass(frozen=True, slots=True)
class GovernanceInvariantFacts:
    meaningful_side_effect_binding: GovernanceMeaningfulSideEffectBinding | None = None


class GovernanceInvariantProbe(Protocol):
    def read_facts(self) -> GovernanceInvariantFacts:
        ...


__all__ = [
    "GovernanceInvariantFacts",
    "GovernanceInvariantProbe",
    "GovernanceMeaningfulSideEffectBinding",
]
