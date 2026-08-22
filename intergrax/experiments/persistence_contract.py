# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider-neutral experiment persistence port (PBA-FIX-E, §35)."""

from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable

from intergrax.experiments.models import (
    ExperimentDecision,
    ExperimentRecord,
    RegisterExperimentRequest,
)

__all__ = [
    "ExperimentPersistence",
    "ExperimentReader",
]


@runtime_checkable
class ExperimentReader(Protocol):
    """Read-only experiment registry access for debug and lab surfaces."""

    def list_experiments(
        self,
        *,
        limit: int = 50,
        decision: Optional[ExperimentDecision] = None,
    ) -> List[ExperimentRecord]:
        ...

    def get(self, experiment_id: str) -> ExperimentRecord:
        ...


@runtime_checkable
class ExperimentPersistence(ExperimentReader, Protocol):
    """Experiment registry persistence consumed by lab workflow and debug API."""

    def register(self, request: RegisterExperimentRequest) -> ExperimentRecord:
        ...

    def set_decision(
        self,
        experiment_id: str,
        decision: ExperimentDecision,
        *,
        notes: Optional[str] = None,
    ) -> ExperimentRecord:
        ...

    def link_run(self, experiment_id: str, run_id: str) -> ExperimentRecord:
        ...
