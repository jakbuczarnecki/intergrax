# © Artur Czarnecki. All rights reserved.

"""Minimal governance for lab strict-harness mode."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.runtime.governance.execution_guard import GovernanceEvaluation


@dataclass(slots=True)
class LabAllowGovernanceService:
    """Satisfies ``RuntimeContext.build`` when ``production_mode=True``."""

    def evaluate(self, run_id: str, agent_id: str) -> GovernanceEvaluation | None:
        _ = run_id, agent_id
        return None


def create_lab_allow_governance_service() -> LabAllowGovernanceService:
    return LabAllowGovernanceService()
