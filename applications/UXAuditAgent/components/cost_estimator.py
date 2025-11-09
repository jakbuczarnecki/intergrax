# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import random
from intergrax.supervisor.supervisor_components import ComponentContext, ComponentResult, PipelineState, component

@component(
    name="Agent kosztorysowy",
    description="Szacuje koszt zmian na podstawie raportu UX (mock).",
    use_when="Używaj, gdy trzeba policzyć szacunkowy koszt zmian wynikających z audytu UX.",
    examples=["Oszacuj koszt zmian po audycie UX."],
    available=True
)
def cost_estimator(state: PipelineState, ctx: ComponentContext) -> ComponentResult:
    ux_report = state.get("artifacts", {}).get("ux_report") or {}
    issues = ux_report.get("problemy", []) or ux_report.get("issues", [])
    base = 3000
    add_per_issue = 600
    cost = base + add_per_issue * len(issues)
    result = {"cost_estimate": float(cost), "currency": "PLN", "method": "mock: base + per_issue"}
    return ComponentResult(
        ok=True,
        produces={"cost_estimate": result["cost_estimate"], "cost_estimate_meta": result},
        output=result,
        logs=[f"Kosztorys: {result['cost_estimate']} PLN (mock)."]
    )