# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""UAEP thin-step wiring for Legal Agent (Phase E, §42.32)."""

from legal.uaep.thin_steps import (
    FINAL_SEQUENTIAL_STEP_ID,
    LEGAL_SEQUENTIAL_STEP_DEFS,
    legal_sequential_agent_steps,
    run_legal_uaep_step,
    run_sequential_pipeline_on_state,
)

__all__ = [
    "FINAL_SEQUENTIAL_STEP_ID",
    "LEGAL_SEQUENTIAL_STEP_DEFS",
    "legal_sequential_agent_steps",
    "run_legal_uaep_step",
    "run_sequential_pipeline_on_state",
]
