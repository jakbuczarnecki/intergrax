# © Artur Czarnecki. All rights reserved.

"""Test composition helpers for governed structured inference (GR-10-R2)."""

from __future__ import annotations

from contextvars import Token

from intergrax.contracts.admitted_root_governance_identity import AdmittedRootGovernanceIdentity
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.execution.inference import InferenceExecutor
from intergrax.runtime.execution.inference_profile import InferenceProfileResolver
from intergrax.runtime.execution.runtime import RootExecutionOptions
from intergrax.runtime.governance.active_execution_governance_identity import (
    ActiveExecutionGovernanceIdentity,
    bind_active_execution_governance_identity,
    reset_active_execution_governance_identity,
)
from intergrax.runtime.governance.governance_evidence_recorder import GovernanceEvidenceRecorder
from intergrax.runtime.policy.policy_engine import PolicyEngine

TEST_INFERENCE_TENANT_ID = "tenant_inference_governance"
TEST_INFERENCE_WORKSPACE_ID = "workspace_inference_governance"
TEST_INFERENCE_PRINCIPAL_ID = "principal_inference_governance"


def governed_inference_executor(
    adapter: LLMAdapter,
    *,
    policy_engine: PolicyEngine | None = None,
    governance_evidence_recorder: GovernanceEvidenceRecorder | None = None,
    profile_resolver: InferenceProfileResolver | None = None,
) -> InferenceExecutor[object]:
    return InferenceExecutor(
        adapter,
        profile_resolver=profile_resolver,
        policy_engine=policy_engine if policy_engine is not None else PolicyEngine(),
        governance_evidence_recorder=governance_evidence_recorder,
    )


def governed_root_execution_options(**overrides: object) -> RootExecutionOptions:
    base = {
        "authority": ParentExecutionAuthority.unrestricted_root(),
        "governance_identity": AdmittedRootGovernanceIdentity(
            tenant_id=TEST_INFERENCE_TENANT_ID,
            workspace_id=TEST_INFERENCE_WORKSPACE_ID,
            principal_id=TEST_INFERENCE_PRINCIPAL_ID,
        ),
    }
    base.update(overrides)
    return RootExecutionOptions(**base)


def bind_test_inference_governance_identity() -> Token:
    return bind_active_execution_governance_identity(
        ActiveExecutionGovernanceIdentity(
            tenant_id=TEST_INFERENCE_TENANT_ID,
            workspace_id=TEST_INFERENCE_WORKSPACE_ID,
            principal_id=TEST_INFERENCE_PRINCIPAL_ID,
        ),
    )


def reset_test_inference_governance_identity(token: Token) -> None:
    reset_active_execution_governance_identity(token)
