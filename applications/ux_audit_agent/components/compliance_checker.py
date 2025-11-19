# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import random
from intergrax.supervisor.supervisor_components import ComponentContext, ComponentResult, PipelineState, component


@component(
    name="Policy & Privacy Compliance Checker",
    description="Checks whether the proposed changes comply with the privacy policy and regulations (mocked validation).",
    use_when="Use when changes need to be evaluated against privacy policy or regulatory rules.",
    examples=["Verify whether the proposed UX changes comply with the privacy policy."],
    available=True
)
def compliance_checker(state: PipelineState, ctx: ComponentContext) -> ComponentResult:
    ux_report = state.get("artifacts", {}).get("ux_report")

    # Simulated compliance result (80% chance of being compliant)
    compliant = random.choices([True, False], weights=[0.8, 0.2], k=1)[0]

    findings = {
        "compliant": compliant,
        "policy_violations": [] if compliant else [
            {"id": "privacy-link", "desc": "Missing link to the Privacy Policy in the form footer."}
        ],
        "requires_dpo_review": False if compliant else True,
        "notes": "Compliant." if compliant else "Form changes required and legal/DPO consultation recommended."
    }

    meta = {}
    logs = ["Compliance: evaluating changes against policies and regulations (mock run)."]

    # If non-compliant, stop execution until corrections are made
    if not compliant:
        meta = {"stop": True, "reason": "Non-compliance with policies/regulations — correction or DPO review required."}

    return ComponentResult(
        ok=True,
        produces={"compliance_findings": findings},
        output=findings,
        logs=logs,
        meta=meta
    )
