# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import random
from intergrax.supervisor.supervisor_components import ComponentContext, ComponentResult, PipelineState, component

@component(
    name="Cost Estimation Agent",
    description="Estimates the cost of UX-related changes based on the audit report (mock calculation).",
    use_when="Use when estimating the cost of UX updates derived from an audit.",
    examples=["Estimate the cost of required UX changes after the audit."],
    available=True
)
def cost_estimator(state: PipelineState, ctx: ComponentContext) -> ComponentResult:
    ux_report = state.get("artifacts", {}).get("ux_report") or {}
    issues = ux_report.get("problemy", []) or ux_report.get("issues", [])

    # Mock pricing model
    base = 3000
    add_per_issue = 600
    cost = base + add_per_issue * len(issues)

    result = {
        "cost_estimate": float(cost),
        "currency": "PLN",
        "method": "mock: base + per_issue"
    }

    return ComponentResult(
        ok=True,
        produces={"cost_estimate": result["cost_estimate"], "cost_estimate_meta": result},
        output=result,
        logs=[f"Cost estimation: {result['cost_estimate']} PLN (mock model)."]
    )
