# © Artur Czarnecki. All rights reserved.

"""Canonical bundle-backed control-plane mutation policy evaluation (CLA-04)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.control_plane_mutation import (
    ControlPlaneMutationPolicyEvaluator,
    ControlPlaneMutationRequest,
)
from intergrax.contracts.meaningful_side_effect import (
    MeaningfulSideEffectKind,
    MeaningfulSideEffectRequest,
)
from intergrax.contracts.runtime_policy import PolicyDecision
from intergrax.runtime.policy.runtime_policy_bundle_evaluator import (
    RuntimePolicyBundleEvaluator,
)


def control_plane_mutation_to_meaningful_side_effect_request(
    request: ControlPlaneMutationRequest,
) -> MeaningfulSideEffectRequest:
    """Map one control-plane mutation to PG-FIX-D ``match_action`` evaluation input."""
    principal = request.principal
    principal_id = principal.user_id or principal.auth_subject
    if not principal_id:
        raise ValueError("control_plane_mutation_requires_principal_identity")
    execution_task_id = (
        str(request.task_id) if request.task_id is not None else request.mutation_id
    )
    execution_run_id = (
        str(request.run_id) if request.run_id is not None else request.mutation_id
    )
    return MeaningfulSideEffectRequest(
        action=request.mutation_type,
        kinds=(MeaningfulSideEffectKind.MUTATION,),
        side_effect_scope_id=request.resource_scope,
        task_id=execution_task_id,
        run_id=execution_run_id,
        principal_id=principal_id,
        tenant_id=principal.tenant_id,
        resource=f"{request.resource_type}:{request.resource_id}",
        context={
            "mutation_id": request.mutation_id,
            "mutation_type": request.mutation_type,
            "current_revision": request.current_revision,
            "target_revision": request.target_revision,
            "risk_classification": request.risk_classification.value,
            "principal_type": principal.principal_type.value,
            "resource_type": request.resource_type,
            "resource_id": request.resource_id,
        },
    )


@dataclass(slots=True)
class BundleBackedControlPlaneMutationEvaluator:
    """Generic platform evaluator: ``ControlPlaneMutationRequest`` → bundle rules."""

    bundle_evaluator: RuntimePolicyBundleEvaluator

    def evaluate(self, request: ControlPlaneMutationRequest) -> PolicyDecision:
        side_effect = control_plane_mutation_to_meaningful_side_effect_request(request)
        return self.bundle_evaluator.evaluate(side_effect).decision


def bundle_backed_control_plane_mutation_evaluator(
    bundle_evaluator: RuntimePolicyBundleEvaluator,
) -> ControlPlaneMutationPolicyEvaluator:
    """Return a protocol-compatible evaluator backed by one immutable policy pack."""
    return BundleBackedControlPlaneMutationEvaluator(bundle_evaluator=bundle_evaluator)


__all__ = [
    "BundleBackedControlPlaneMutationEvaluator",
    "bundle_backed_control_plane_mutation_evaluator",
    "control_plane_mutation_to_meaningful_side_effect_request",
]
