# © Artur Czarnecki. All rights reserved.

"""Authoritative stimulus state projection for controlled alignment (qualification only)."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.ai_incident_investigation.application.scenario_contract import (
    COMPLETION_SUPPORTED_DIAGNOSIS,
)


@dataclass(frozen=True, slots=True)
class ControlledAlignmentStimulusState:
    """
    Deterministic MODEL_OVERCOMMIT preconditions:

    - completion_mode: supported_diagnosis (model claims confirmed diagnosis)
    - supported_state: absent (no evidence-backed supported diagnosis)
    """

    completion_mode: str
    supported_state_present: bool

    @classmethod
    def model_overcommit(cls) -> ControlledAlignmentStimulusState:
        return cls(
            completion_mode=COMPLETION_SUPPORTED_DIAGNOSIS,
            supported_state_present=False,
        )


__all__ = ["ControlledAlignmentStimulusState"]
