# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit composition for root execution governance admission (GR-2).

Default policy engine binding belongs here — not inside admission consumers.
"""

from __future__ import annotations

from typing import TypeVar

from intergrax.contracts.execution_intake import CanonicalExecutionIntakePort
from intergrax.contracts.root_execution_launch import RootExecutionLaunchPort
from intergrax.contracts.runtime_execution_policy_admission import (
    RootExecutionAdmissionPolicyRule,
    RuntimeExecutionPolicyAdmissionPort,
)
from intergrax.runtime.governance.root_execution_authority_admission import (
    RootExecutionAuthorityAdmissionService,
)
from intergrax.runtime.governance.runtime_execution_policy_admission import (
    AllowingRuntimeExecutionPolicyAdmission,
    RuntimeExecutionPolicyAdmissionEvaluator,
)
from intergrax.runtime.governance.default_root_execution_launcher import (
    DefaultRootExecutionLauncher,
)
from intergrax.runtime.governance.governance_evidence_recorder import GovernanceEvidenceRecorder
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine

PayloadT = TypeVar("PayloadT")
ResultT = TypeVar("ResultT")


def build_runtime_execution_policy_admission(
    *,
    policy_engine: RuntimePolicyEngine,
) -> RuntimeExecutionPolicyAdmissionPort:
    """Wire ``RuntimePolicyEngine`` behind the governance admission port."""
    return RuntimeExecutionPolicyAdmissionEvaluator(policy_engine=policy_engine)


def build_fail_closed_runtime_execution_policy_admission() -> RuntimeExecutionPolicyAdmissionPort:
    """Fail-closed evaluator — no rules until composition supplies them."""
    return build_runtime_execution_policy_admission(policy_engine=RuntimePolicyEngine())


def build_root_execution_authority_admission(
    *,
    runtime_policy_admission: RuntimeExecutionPolicyAdmissionPort,
    governance_evidence_recorder: GovernanceEvidenceRecorder | None = None,
) -> RootExecutionAuthorityAdmissionService:
    """Trusted root authority admission with an explicitly configured policy port."""
    return RootExecutionAuthorityAdmissionService(
        runtime_policy_admission=runtime_policy_admission,
        governance_evidence_recorder=governance_evidence_recorder,
    )


def build_root_execution_authority_admission_from_rules(
    *,
    root_execution_admission_rules: tuple[RootExecutionAdmissionPolicyRule, ...],
) -> RootExecutionAuthorityAdmissionService:
    """Convenience builder for hosts that configure admission via policy rules."""
    policy_engine = RuntimePolicyEngine(
        root_execution_admission_rules=root_execution_admission_rules,
    )
    return build_root_execution_authority_admission(
        runtime_policy_admission=build_runtime_execution_policy_admission(
            policy_engine=policy_engine,
        ),
    )


def build_reference_allowing_root_execution_authority_admission() -> (
    RootExecutionAuthorityAdmissionService
):
    """Reference/test composition — not for production security claims."""
    return build_root_execution_authority_admission(
        runtime_policy_admission=AllowingRuntimeExecutionPolicyAdmission(),
    )


def build_default_root_execution_launcher(
    *,
    runtime_policy_admission: RuntimeExecutionPolicyAdmissionPort,
    execution_intake: CanonicalExecutionIntakePort[PayloadT, ResultT],
) -> RootExecutionLaunchPort[PayloadT, ResultT]:
    """Composition-root launcher: policy port → admission service → intake."""
    return DefaultRootExecutionLauncher(
        root_authority_admission=build_root_execution_authority_admission(
            runtime_policy_admission=runtime_policy_admission,
        ),
        execution_intake=execution_intake,
    )


__all__ = [
    "build_default_root_execution_launcher",
    "build_reference_allowing_root_execution_authority_admission",
    "build_fail_closed_runtime_execution_policy_admission",
    "build_root_execution_authority_admission",
    "build_root_execution_authority_admission_from_rules",
    "build_runtime_execution_policy_admission",
]
