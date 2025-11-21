# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import random
from intergrax.supervisor.supervisor_components import ComponentContext, ComponentResult, PipelineState, component


@component(
    name="Financial Agent",
    description="Generates a mock financial report and VAT calculations (test data).",
    use_when="Use when testing financial computations, budget constraints, or cost reports.",
    examples=["Gross value from 25k net with 23% VAT", "Budget report for previous quarter"],
    available=True
)
def financial(state: PipelineState, ctx: ComponentContext) -> ComponentResult:
    report = {
        "net": 25000.0,
        "vat_rate": 23.0,
        "vat_amount": 5750.0,
        "gross": 30750.0,
        "currency": "PLN",
        "previous_quarter_budget": 35000.0
    }

    return ComponentResult(
        ok=True,
        produces={"financial_report": report},
        output=report,
        logs=["Finance: mock report generated."]
    )
