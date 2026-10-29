# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit composition for governed structured inference (GR-10-R6 / R6-R1)."""

from __future__ import annotations

from typing import TypeVar

from intergrax.contracts.governed_execution_governance_evidence import (
    GovernanceEvidencePersistencePort,
)
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.execution.inference import InferenceExecutor
from intergrax.runtime.execution.inference_profile import InferenceProfileResolver
from intergrax.runtime.governance.governance_evidence_composition import (
    build_governance_evidence_recorder,
)
from intergrax.runtime.governance.structured_inference_pre_model_composition import (
    build_structured_inference_pre_model_gate,
)
from intergrax.runtime.policy.policy_engine import PolicyEngine

OutputT = TypeVar("OutputT")


def build_governed_inference_executor(
    adapter: LLMAdapter,
    *,
    governance_evidence_persistence: GovernanceEvidencePersistencePort,
    policy_engine: PolicyEngine | None = None,
    profile_resolver: InferenceProfileResolver | None = None,
) -> InferenceExecutor[OutputT]:
    """Wire policy + mandatory evidence persistence for structured inference."""
    engine = policy_engine if policy_engine is not None else PolicyEngine()
    recorder = build_governance_evidence_recorder(
        persistence=governance_evidence_persistence,
    )
    pre_model_gate = build_structured_inference_pre_model_gate(engine, recorder)
    return InferenceExecutor(
        adapter,
        profile_resolver=profile_resolver,
        pre_model_gate=pre_model_gate,
        governance_evidence_recorder=recorder,
    )


__all__ = ["build_governed_inference_executor"]
