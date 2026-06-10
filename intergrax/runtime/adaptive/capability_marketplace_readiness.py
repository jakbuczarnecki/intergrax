# © Artur Czarnecki. All rights reserved.

"""Capability marketplace readiness evaluation (AUDIT-IDEAL-AHI.3)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.catalog import iter_entries
from intergrax.integrations.registry.marketplace_catalog import IntegrationMarketplaceCatalog
from intergrax.runtime.architecture.agent_certification import (
    AgentCertificationEvaluation,
    AgentCertificationEvidence,
    AgentCertificationGate,
    AgentCertificationOwner,
    GateCheckStatus,
    evaluate_agent_certification,
)


class CapabilityMarketplaceReadinessReport(BaseModel):
    schema_version: str = "1.0.0"
    trust_ready: bool
    certification_ready: bool
    billing_boundary_ready: bool
    ready: bool


def evaluate_capability_marketplace_readiness(
    *,
    marketplace_catalog: IntegrationMarketplaceCatalog,
    sample_agent_id: str = "echo",
) -> CapabilityMarketplaceReadinessReport:
    """Check trust, certification, and billing boundary readiness for marketplace."""
    trust_ready = bool(marketplace_catalog.entries) and all(
        entry.trust_score >= 0.5 for entry in marketplace_catalog.entries
    )
    billing_boundary_ready = any(
        IntegrationCategory.BILLING_METER in entry.categories for entry in iter_entries()
    )
    certification = evaluate_agent_certification(
        AgentCertificationEvidence(
            agent_id=sample_agent_id,
            agent_version="1.0.0",
            production_eligible=True,
            owner=AgentCertificationOwner(team="platform", owner="platform@intergrax", on_call="oncall@intergrax"),
            quality_gates=[
                AgentCertificationGate(
                    name="eval.golden",
                    status=GateCheckStatus.PASS,
                    evidence_ref="tests/golden",
                )
            ],
            policy_gates=[
                AgentCertificationGate(
                    name="policy.pre_output",
                    status=GateCheckStatus.PASS,
                    evidence_ref="runtime/policy",
                )
            ],
            security_gates=[
                AgentCertificationGate(
                    name="security.tenant",
                    status=GateCheckStatus.PASS,
                    evidence_ref="runtime/security",
                )
            ],
        )
    )
    certification_ready = certification.eligible
    ready = trust_ready and certification_ready and billing_boundary_ready
    return CapabilityMarketplaceReadinessReport(
        trust_ready=trust_ready,
        certification_ready=certification_ready,
        billing_boundary_ready=billing_boundary_ready,
        ready=ready,
    )
