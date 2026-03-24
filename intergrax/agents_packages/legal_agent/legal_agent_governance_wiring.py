# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Factory helpers: attach post-run + pre-bridge governance in one step.

Use when building :class:`~intergrax.agents_packages.legal_agent.legal_agent_config.LegalAgentConfig`
in your agent factory (HTTP layer, worker, CLI).
"""

from __future__ import annotations

from intergrax.agents_packages.legal_agent.legal_agent_config import LegalAgentConfig
from intergrax.agents_packages.legal_agent.legal_platform_policy_governance import (
    DualLegalGovernanceService,
    LegalExecutionPolicyPort,
)
from intergrax.runtime.governance.execution_guard import ExecutionGuard


def with_dual_legal_governance(
    config: LegalAgentConfig,
    *,
    guard: ExecutionGuard,
    policy: LegalExecutionPolicyPort,
) -> LegalAgentConfig:
    """
    Return a copy of ``config`` with the same :class:`DualLegalGovernanceService` on both
    ``governance_service`` and ``legal_tool_plan_governance``.

    In ``production_mode=True``, ``governance_service`` must be non-``None`` — this satisfies that.
    """
    dual = DualLegalGovernanceService(guard=guard, policy=policy)
    return config.model_copy(
        update={
            "governance_service": dual,
            "legal_tool_plan_governance": dual,
        }
    )
