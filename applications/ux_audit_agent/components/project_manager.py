# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import random
from intergrax.supervisor.supervisor_components import ComponentContext, ComponentResult, PipelineState, component

@component(
    name="Project Manager",
    description="Weryfikuje raport UX i losowo akceptuje z uwagami lub odrzuca.",
    use_when="Gdy trzeba zebrać decyzję PM przed dalszymi krokami.",
    examples=["PM dokonuje sprawdzenia funkcjonalności i akceptuje albo odrzuca zmiany w projekcie."],
    available=True
)
def project_manager(state: PipelineState, ctx: ComponentContext) -> ComponentResult:
    decision = random.choices(["accepted", "rejected"], weights=[0.7, 0.3], k=1)[0]
    notes = "Proszę uwzględnić kontrast i spójność kolorów." if decision == "accepted" else "Za duże ryzyko; wrócić do projektu."
    produces = {"pm_decision": decision, "pm_notes": notes}
    meta = {}
    logs = [f"PM: decyzja = {decision}"]
    if decision == "rejected":
        meta = {"stop": True, "reason": "Project Manager odrzucił projekt"}
    return ComponentResult(ok=True, produces=produces, output={"decision": decision, "notes": notes}, logs=logs, meta=meta)