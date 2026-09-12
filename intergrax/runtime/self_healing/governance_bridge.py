# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Self-healing decision → governance → external operation admission (SELF-HEALING R1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.external_operations.admission import (
    ExternalOperationAdmission,
    ExternalOperationAdmissionContext,
    OperationAdmissionVerdict,
)
from intergrax.contracts.external_operations.intent import ExternalOperationIntent
from intergrax.contracts.self_healing.decision import SelfHealingDecision
from intergrax.contracts.self_healing.governance import (
    SelfHealingAdmissionContext,
    SelfHealingAdmissionDecision,
    SelfHealingAdmissionGate,
    SelfHealingAdmissionVerdict,
)


@dataclass(frozen=True, slots=True)
class SelfHealingGovernedExternalOperationDecision:
    self_healing_admission: SelfHealingAdmissionDecision
    external_intent: ExternalOperationIntent | None
    may_translate: bool
    may_execute: bool

    def __post_init__(self) -> None:
        if (
            self.may_execute
            and self.self_healing_admission.verdict is SelfHealingAdmissionVerdict.DENY
        ):
            raise ValueError("may_execute cannot be true when self-healing admission is DENY")


def resolve_self_healing_governance_chain(
    *,
    decision: SelfHealingDecision,
    admission_gate: SelfHealingAdmissionGate,
    admission_context: SelfHealingAdmissionContext,
    external_intent: ExternalOperationIntent | None,
    external_admission: ExternalOperationAdmission | None = None,
    external_context: ExternalOperationAdmissionContext | None = None,
) -> SelfHealingGovernedExternalOperationDecision:
    sh_decision = admission_gate.evaluate(decision, admission_context)
    if sh_decision.verdict is SelfHealingAdmissionVerdict.DENY:
        return SelfHealingGovernedExternalOperationDecision(
            self_healing_admission=sh_decision,
            external_intent=None,
            may_translate=False,
            may_execute=False,
        )
    if sh_decision.verdict is SelfHealingAdmissionVerdict.REQUIRES_APPROVAL:
        if not (admission_context.human_approval_granted and admission_context.approval_id):
            return SelfHealingGovernedExternalOperationDecision(
                self_healing_admission=sh_decision,
                external_intent=None,
                may_translate=False,
                may_execute=False,
            )
    may_translate = external_intent is not None
    may_execute = False
    if external_intent is not None and external_admission is not None and external_context is not None:
        if external_intent.tenant_id != admission_context.tenant_id:
            return SelfHealingGovernedExternalOperationDecision(
                self_healing_admission=sh_decision,
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
    return SelfHealingGovernedExternalOperationDecision(
        self_healing_admission=sh_decision,
        external_intent=external_intent,
        may_translate=may_translate,
        may_execute=may_execute,
    )


__all__ = [
    "SelfHealingGovernedExternalOperationDecision",
    "resolve_self_healing_governance_chain",
]
