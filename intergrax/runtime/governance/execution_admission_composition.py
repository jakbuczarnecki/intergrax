# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit composition for root execution governance admission (GR-2).

Default policy engine binding belongs here — not inside admission consumers.
"""

from __future__ import annotations

from intergrax.contracts.runtime_execution_policy_admission import (
    RootExecutionAdmissionPolicyRule,
    RuntimeExecutionPolicyAdmissionPort,
)
from intergrax.runtime.governance.root_execution_authority_admission import (
    RootExecutionAuthorityAdmissionService,
)
from intergrax.runtime.governance.runtime_execution_policy_admission import (
    RuntimeExecutionPolicyAdmissionEvaluator,
)
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine


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
) -> RootExecutionAuthorityAdmissionService:
    """Trusted root authority admission with an explicitly configured policy port."""
    return RootExecutionAuthorityAdmissionService(
        runtime_policy_admission=runtime_policy_admission,
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


__all__ = [
    "build_fail_closed_runtime_execution_policy_admission",
    "build_root_execution_authority_admission",
    "build_root_execution_authority_admission_from_rules",
    "build_runtime_execution_policy_admission",
]
