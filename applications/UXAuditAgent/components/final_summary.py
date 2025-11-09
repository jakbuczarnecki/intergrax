# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import random
from intergrax.supervisor.supervisor_components import ComponentContext, ComponentResult, PipelineState, component

@component(
    name="Raport końcowy",
    description="Tworzy pełne podsumowanie procesu ze wszystkich artefaktów.",
    use_when="Uruchamiane zawsze na końcu (blok finally).",
    examples=["Przygotowanie finalnego raportu zadania."],
    available=True
)
def final_summary(state: PipelineState, ctx: ComponentContext) -> ComponentResult:
    arts = state.get("artifacts", {})
    summary = {
        "status_pipeline": "terminated" if state.get("terminated") else "completed",
        "terminated_by": state.get("terminated_by"),
        "terminate_reason": state.get("terminate_reason"),
        "pm_decision": arts.get("pm_decision"),
        "pm_notes": arts.get("pm_notes"),
        "ux_report": arts.get("ux_report"),
        "financial_report": arts.get("financial_report"),
        "citations": arts.get("citations"),
    }
    return ComponentResult(ok=True, produces={"final_report": summary}, output=summary, logs=["FinalSummary: gotowe."])