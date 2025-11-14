# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import random
from intergrax.supervisor.supervisor_components import ComponentContext, ComponentResult, PipelineState, component

@component(
    name="Finansowy Agent",
    description="Generuje przykładowy raport finansowy i obliczenia VAT (dane testowe).",
    use_when="Używaj do testowania kalkulacji finansowych i raportów kosztów.",
    examples=["Brutto z 25k netto, 23% VAT", "Raport budżetu Q(-1)"],
    available=True
)
def financial(state: PipelineState, ctx: ComponentContext) -> ComponentResult:
    raport = {
        "netto": 25000.0,
        "stawka_vat": 23.0,
        "kwota_vat": 5750.0,
        "brutto": 30750.0,
        "waluta": "PLN",
        "budzet_ostatni_kwartal": 35000.0
    }
    return ComponentResult(ok=True, produces={"financial_report": raport}, output=raport, logs=["Finanse: mock raport."])