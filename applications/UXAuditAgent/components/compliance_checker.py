# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import random
from intergrax.supervisor.supervisor_components import ComponentContext, ComponentResult, PipelineState, component


@component(
    name="Weryfikacja regulaminów i polityk prywatności",
    description="Sprawdza, czy lista zmian jest zgodna z polityką prywatności i regulaminami (mock).",
    use_when="Używaj, gdy trzeba ocenić zgodność zmian z polityką prywatności/regulaminami.",
    examples=["Zweryfikuj zgodność zmian UX z polityką prywatności."],
    available=True
)
def compliance_checker(state: PipelineState, ctx: ComponentContext) -> ComponentResult:
    ux_report = state.get("artifacts", {}).get("ux_report")
    compliant = random.choices([True, False], weights=[0.8, 0.2], k=1)[0]
    findings = {
        "compliant": compliant,
        "policy_violations": [] if compliant else [
            {"id": "privacy-link", "desc": "Brak linku do polityki prywatności w stopce formularza."}
        ],
        "requires_dpo_review": False if compliant else True,
        "notes": "Zgodne." if compliant else "Wymagana korekta formularza i konsultacja z DPO."
    }
    meta = {}
    logs = ["Compliance: analiza zmian względem polityk/regulaminów (mock)."]
    if not compliant:
        meta = {"stop": True, "reason": "Niezgodność z polityką/regulaminami (wymagana korekta/DPO)."}
    return ComponentResult(ok=True, produces={"compliance_findings": findings}, output=findings, logs=logs, meta=meta)