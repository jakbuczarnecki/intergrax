# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Compose PRE_MODEL policy evaluation for :class:`InferenceExecutor` (outside execution)."""

from __future__ import annotations

from intergrax.contracts.structured_inference_pre_model_gate import StructuredInferencePreModelGate
from intergrax.runtime.governance.governance_evidence_recorder import GovernanceEvidenceRecorder
from intergrax.runtime.policy.policy_engine import PolicyEngine
from intergrax.runtime.policy.pre_model_policy_evaluation import (
    enforce_pre_model_before_structured_inference,
)


def build_structured_inference_pre_model_gate(
    policy_engine: PolicyEngine,
    governance_evidence_recorder: GovernanceEvidenceRecorder | None,
) -> StructuredInferencePreModelGate:
    """Wire PolicyEngine + evidence into the execution PRE_MODEL port."""

    def _gate(
        *,
        adapter,
        messages,
        inference_profile_id,
    ) -> None:
        enforce_pre_model_before_structured_inference(
            policy_engine,
            governance_evidence_recorder,
            adapter=adapter,
            messages=messages,
            inference_profile_id=inference_profile_id,
        )

    return _gate


__all__ = ["build_structured_inference_pre_model_gate"]
