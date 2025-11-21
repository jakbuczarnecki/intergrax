# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import random
from intergrax.supervisor.supervisor_components import ComponentContext, ComponentResult, PipelineState, component


@component(
    name="Project Manager",
    description="Reviews the UX report and randomly approves with comments or rejects the proposal (mock decision).",
    use_when="Use when a PM decision is required before proceeding to the next execution stage.",
    examples=["The PM reviews UX changes and approves or rejects them before further processing."],
    available=True
)
def project_manager(state: PipelineState, ctx: ComponentContext) -> ComponentResult:
    # Mock decision model (70% chance the PM accepts)
    decision = random.choices(["accepted", "rejected"], weights=[0.7, 0.3], k=1)[0]

    notes = (
        "Please ensure proper contrast and consistency in color usage."
        if decision == "accepted"
        else "Too high of a risk. Return to the design phase."
    )

    produces = {
        "pm_decision": decision,
        "pm_notes": notes
    }

    logs = [f"PM: decision = {decision}"]

    # If rejected, stop the pipeline execution at this step
    meta = {}
    if decision == "rejected":
        meta = {
            "stop": True,
            "reason": "Project Manager rejected the proposal."
        }

    return ComponentResult(
        ok=True,
        produces=produces,
        output={"decision": decision, "notes": notes},
        logs=logs,
        meta=meta
    )
