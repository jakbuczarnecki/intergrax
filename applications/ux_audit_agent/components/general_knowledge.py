# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import random
from intergrax.supervisor.supervisor_components import ComponentContext, ComponentResult, PipelineState, component


@component(
    name="General",
    description="Answers general questions about the Intergrax system, including modules, architecture, and documentation.",
    use_when="Use when the user asks about the structure, features, or configuration of the Intergrax system.",
    examples=["What modules does Intergrax include?", "Where can I find the Intergrax privacy policy?"],
    available=True
)
def general_intergrax_knowledge(state: PipelineState, ctx: ComponentContext) -> ComponentResult:
    answer = (
        "The Intergrax system consists of the following modules: CRM, Projects, Invoicing, Warehouse, "
        "and additional industry-specific modules. Privacy policy can be found under: Settings → Security."
    )

    fake_docs = [
        {"doc_id": "policy", "page": 1},
        {"doc_id": "architecture", "page": 2}
    ]

    return ComponentResult(
        ok=True,
        produces={
            "rag_context": "Mock data",
            "rag_answer": answer,
            "citations": fake_docs
        },
        output=answer,
        logs=["General: returned mock knowledge response."]
    )
