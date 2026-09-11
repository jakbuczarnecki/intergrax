# © Artur Czarnecki. All rights reserved.

"""Composition-root helpers for NPSC-4 agent runtime governance."""

from __future__ import annotations

from intergrax.contracts.agent_runtime_governance import CapabilityGrant
from intergrax.runtime.agent_governance.approval_boundary import (
    AgentRuntimeApprovalBoundary,
    InMemoryApprovalStore,
)
from intergrax.runtime.agent_governance.audit import (
    GovernanceAuditRecorder,
    InMemoryGovernanceAuditSink,
)
from intergrax.runtime.agent_governance.authorization_boundary import (
    AgentRuntimeGovernanceBoundary,
)
from intergrax.runtime.agent_governance.capability_resolver import InMemoryCapabilityGrantResolver
from intergrax.runtime.agent_governance.pipeline import AgentRuntimeGovernancePipeline
from intergrax.runtime.agent_governance.policy_engine import (
    AgentRuntimePolicyEngine,
    FinancialApprovalPolicyProvider,
    HighRiskApprovalPolicyProvider,
)


def default_lab_capability_grants(tenant_id: str) -> tuple[CapabilityGrant, ...]:
    """Reference lab strict-harness grants for default echo probe agent."""
    normalized_tenant = tenant_id.strip() or "default-tenant"
    return (
        CapabilityGrant(
            agent_id="echo",
            tenant_id=normalized_tenant,
            allowed_capabilities=frozenset({"echo.basic"}),
        ),
    )


def build_agent_runtime_governance_boundary(
    *,
    capability_grants: tuple[CapabilityGrant, ...],
) -> AgentRuntimeGovernanceBoundary:
    """Shared pre-execution governance boundary for production tool invocations."""
    sink = InMemoryGovernanceAuditSink()
    pipeline = AgentRuntimeGovernancePipeline(
        capability_resolver=InMemoryCapabilityGrantResolver(capability_grants),
        policy_engine=AgentRuntimePolicyEngine(
            (
                HighRiskApprovalPolicyProvider(),
                FinancialApprovalPolicyProvider(),
            ),
        ),
        audit_recorder=GovernanceAuditRecorder(sink),
        approval_boundary=AgentRuntimeApprovalBoundary(InMemoryApprovalStore()),
    )
    return AgentRuntimeGovernanceBoundary(pipeline)
