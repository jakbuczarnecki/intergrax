# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import random
from intergrax.supervisor.supervisor_components import ComponentContext, ComponentResult, PipelineState, component


@component(
    name="Final Report",
    description="Generates a complete summary of the entire execution pipeline using all collected artifacts.",
    use_when="Executed always at the final stage (finally block).",
    examples=["Generate a complete final report of the task."],
    available=True
)
def final_summary(state: PipelineState, ctx: ComponentContext) -> ComponentResult:
    artifacts = state.get("artifacts", {})

    summary = {
        "pipeline_status": "terminated" if state.get("terminated") else "completed",
        "terminated_by": state.get("terminated_by"),
        "termination_reason": state.get("terminate_reason"),
        "project_manager_decision": artifacts.get("pm_decision"),
        "project_manager_notes": artifacts.get("pm_notes"),
        "ux_report": artifacts.get("ux_report"),
        "financial_report": artifacts.get("financial_report"),
        "citations": artifacts.get("citations"),
    }

    return ComponentResult(
        ok=True,
        produces={"final_report": summary},
        output=summary,
        logs=["FinalSummary: completed."]
    )
