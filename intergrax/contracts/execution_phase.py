# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime execution phases (architecture §42.31)."""

from __future__ import annotations

from enum import Enum


class ExecutionPhase(str, Enum):
    INTAKE = "intake"
    CLASSIFICATION = "classification"
    PLANNING = "planning"
    CONTEXT_BUILDING = "context_building"
    AGENT_SELECTION = "agent_selection"
    STEP_EXECUTION = "step_execution"
    VALIDATION = "validation"
    INTERRUPT_HANDLING = "interrupt_handling"
    RETRY_HANDLING = "retry_handling"
    HUMAN_APPROVAL = "human_approval"
    FINALIZATION = "finalization"
    TRACE_PERSISTENCE = "trace_persistence"
    COMPLETION = "completion"
    APPLICATION_HOSTING = "application_hosting"
