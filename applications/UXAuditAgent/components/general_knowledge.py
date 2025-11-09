# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import random
from intergrax.supervisor.supervisor_components import ComponentContext, ComponentResult, PipelineState, component

@component(
    name="Generalny",
    description="Odpowiada na pytania dotyczące systemu Intergrax, jego modułów, architektury i dokumentacji.",
    use_when="Używaj, gdy użytkownik pyta o strukturę lub funkcje systemu Intergrax.",
    examples=["Jakie moduły zawiera Intergrax?", "Gdzie znajduje się polityka prywatności Intergrax?"],
    available=True
)
def general_intergrax_knowledge(state: PipelineState, ctx: ComponentContext) -> ComponentResult:
    answer = (
        "System Intergrax składa się z modułów: CRM, Projekty, Faktury, Magazyn, "
        "oraz modułów branżowych. Polityka prywatności: Ustawienia → Bezpieczeństwo."
    )
    fake_docs = [{"doc_id": "policy", "page": 1}, {"doc_id": "architecture", "page": 2}]
    return ComponentResult(
        ok=True,
        produces={"rag_context": "Dane testowe", "rag_answer": answer, "citations": fake_docs},
        output=answer,
        logs=["Generalny: zwrócono przykładową odpowiedź."]
    )