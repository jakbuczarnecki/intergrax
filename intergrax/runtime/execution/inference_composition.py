# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit composition for governed structured inference (GR-10-R6 / R6-R1).

Policy and Governance evidence persistence are wired here — not inside
:class:`InferenceExecutor` consumers.

Legal production-qualified governed inference **requires** a replaceable
``GovernanceEvidencePersistencePort`` at this boundary.
"""

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
    recorder = build_governance_evidence_recorder(
        persistence=governance_evidence_persistence,
    )
    return InferenceExecutor(
        adapter,
        profile_resolver=profile_resolver,
        policy_engine=policy_engine if policy_engine is not None else PolicyEngine(),
        governance_evidence_recorder=recorder,
    )


__all__ = ["build_governed_inference_executor"]
