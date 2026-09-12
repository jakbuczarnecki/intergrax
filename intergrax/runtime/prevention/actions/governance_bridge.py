# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Preventive proposal → governance → external operation admission (PREVENTIVE R7)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.external_operations.admission import (
    ExternalOperationAdmission,
    ExternalOperationAdmissionContext,
    OperationAdmissionVerdict,
)
from intergrax.contracts.external_operations.intent import ExternalOperationIntent
from intergrax.contracts.preventive.actions.admission import (
    PreventiveActionAdmissionContext,
    PreventiveActionAdmissionDecision,
    PreventiveActionAdmissionGate,
    PreventiveActionAdmissionVerdict,
)
from intergrax.contracts.preventive.actions.proposal import PreventiveActionProposal


@dataclass(frozen=True, slots=True)
class PreventiveGovernedExternalOperationDecision:
    preventive_admission: PreventiveActionAdmissionDecision
    external_intent: ExternalOperationIntent | None
    may_translate: bool
    may_execute: bool

    def __post_init__(self) -> None:
        if self.may_execute and self.preventive_admission.verdict is PreventiveActionAdmissionVerdict.DENY:
            raise ValueError("may_execute cannot be true when preventive admission is DENY")


def resolve_preventive_governance_chain(
    *,
    proposal: PreventiveActionProposal,
    preventive_gate: PreventiveActionAdmissionGate,
    preventive_context: PreventiveActionAdmissionContext,
    external_intent: ExternalOperationIntent | None,
    external_admission: ExternalOperationAdmission | None = None,
    external_context: ExternalOperationAdmissionContext | None = None,
) -> PreventiveGovernedExternalOperationDecision:
    """Single governance authority — never auto-execute on predictive confidence."""
    preventive_decision = preventive_gate.evaluate(proposal, preventive_context)
    if preventive_decision.verdict is PreventiveActionAdmissionVerdict.DENY:
        return PreventiveGovernedExternalOperationDecision(
            preventive_admission=preventive_decision,
            external_intent=None,
            may_translate=False,
            may_execute=False,
        )
    if preventive_decision.verdict is PreventiveActionAdmissionVerdict.REQUIRES_APPROVAL:
        if not (
            preventive_context.human_approval_granted and preventive_context.approval_id
        ):
            return PreventiveGovernedExternalOperationDecision(
                preventive_admission=preventive_decision,
                external_intent=None,
                may_translate=False,
                may_execute=False,
            )
    may_translate = external_intent is not None
    may_execute = False
    if external_intent is not None and external_admission is not None and external_context is not None:
        if external_intent.tenant_id != proposal.tenant_id:
            return PreventiveGovernedExternalOperationDecision(
                preventive_admission=preventive_decision,
                external_intent=external_intent,
                may_translate=True,
                may_execute=False,
            )
        ext_decision = external_admission.evaluate(external_intent, external_context)
        may_execute = ext_decision.verdict is OperationAdmissionVerdict.ALLOW
        if ext_decision.verdict is OperationAdmissionVerdict.REQUIRES_APPROVAL:
            may_execute = bool(
                external_context.human_approval_granted and external_context.approval_id,
            )
    return PreventiveGovernedExternalOperationDecision(
        preventive_admission=preventive_decision,
        external_intent=external_intent,
        may_translate=may_translate,
        may_execute=may_execute,
    )


__all__ = [
    "PreventiveGovernedExternalOperationDecision",
    "resolve_preventive_governance_chain",
]
