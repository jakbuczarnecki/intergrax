# © Artur Czarnecki. All rights reserved.

"""Controlled completion-alignment qualification stimulus (DS-E2E-15J-L1.R4.R4)."""

from testing_support.decision_e2e.controlled_alignment.scenario import (
    ControlledAlignmentScenario,
)
from testing_support.decision_e2e.controlled_alignment.source_freeze import (
    TASK_ID,
    verify_controlled_alignment_source_freeze,
)

__all__ = [
    "ControlledAlignmentScenario",
    "TASK_ID",
    "verify_controlled_alignment_source_freeze",
]
