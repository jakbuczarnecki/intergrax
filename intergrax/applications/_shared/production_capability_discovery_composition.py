# © Artur Czarnecki. All rights reserved.

"""Canonical production capability discovery orchestration (CAPABILITY-CATALOG-1 Stage 5).

Production STRICT hosts MUST cross governed discovery before downstream selection or
lifecycle handoff. This module owns composition-time enforcement; capability catalog
core remains host-agnostic.
"""

from __future__ import annotations

from intergrax.applications._shared.production_capability_governance_evidence import (
    ProductionCapabilityGovernanceEvidenceBundle,
    project_production_capability_governance_evidence,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.capability_catalog.adapters.agent_governance import AgentTrustGovernanceEvaluator
from intergrax.capability_catalog.adapters.skill_governance import SkillProfileGovernanceEvaluator
from intergrax.capability_catalog.adapters.tool_governance import ToolPolicyGovernanceEvaluator
from intergrax.capability_catalog.discovery import discover_capability_candidates
from intergrax.capability_catalog.governed_candidate import GovernedCapabilityCandidate
from intergrax.capability_catalog.governed_result import GovernedDiscoveryResult
from intergrax.capability_catalog.governance import (
    AvailabilityPreservingGovernanceEvaluator,
    CapabilityGovernanceEvaluator,
    govern_capability_candidates,
)
from intergrax.capability_catalog.ranking import CapabilityRanker, rank_capability_candidates
from intergrax.capability_catalog.snapshot import CapabilityCatalogSnapshot
from intergrax.contracts.capability_catalog.evidence import CapabilityDiscoveryAvailabilityEvidence
from intergrax.contracts.capability_catalog.governance import (
    CapabilityAgentGovernanceEvidence,
    CapabilityGovernanceContext,
    CapabilityGovernancePosture,
)
from intergrax.contracts.capability_catalog.query import CapabilityDiscoveryQuery


def resolve_capability_governance_posture(
    environment: ApplicationEnvironmentProfile,
) -> CapabilityGovernancePosture:
    """Map host execution mode to Stage-5 governance posture at composition boundary."""
    if environment.execution_mode is ExecutionMode.STRICT:
        return CapabilityGovernancePosture.STRICT
    return CapabilityGovernancePosture.NON_STRICT


def build_production_capability_governance_evaluators() -> tuple[
    CapabilityGovernanceEvaluator,
    ...,
]:
    """Default production evaluator pipeline — constructor injection, no registry."""
    return (
        AvailabilityPreservingGovernanceEvaluator(),
        ToolPolicyGovernanceEvaluator(),
        AgentTrustGovernanceEvaluator(),
        SkillProfileGovernanceEvaluator(),
    )


def build_production_capability_governance_context(
    environment: ApplicationEnvironmentProfile,
    *,
    evidence: ProductionCapabilityGovernanceEvidenceBundle | None = None,
    agent_evidence: CapabilityAgentGovernanceEvidence | None = None,
) -> CapabilityGovernanceContext:
    """Build read-only governance context from host posture and authority projections."""
    posture = resolve_capability_governance_posture(environment)
    bundle = (
        evidence
        if evidence is not None
        else project_production_capability_governance_evidence(
            environment,
            agent_evidence=agent_evidence,
        )
    )
    resolved_agent_evidence = (
        bundle.agent_evidence
        if bundle.agent_evidence is not None
        else agent_evidence
    )
    return CapabilityGovernanceContext(
        posture=posture,
        tool_evidence=bundle.tool_evidence,
        agent_evidence=resolved_agent_evidence,
        skill_evidence=bundle.skill_evidence,
    )


def discover_rank_and_govern_capabilities(
    *,
    snapshot: CapabilityCatalogSnapshot,
    query: CapabilityDiscoveryQuery,
    availability_evidence: CapabilityDiscoveryAvailabilityEvidence,
    environment: ApplicationEnvironmentProfile,
    ranker: CapabilityRanker,
    governance_evaluators: tuple[CapabilityGovernanceEvaluator, ...] | None = None,
    governance_evidence: ProductionCapabilityGovernanceEvidenceBundle | None = None,
    agent_evidence: CapabilityAgentGovernanceEvidence | None = None,
) -> GovernedDiscoveryResult:
    """Run discovery → ranking → governance for one production composition request."""
    candidates = discover_capability_candidates(
        snapshot,
        query,
        availability_evidence=availability_evidence,
    )
    ranked = rank_capability_candidates(candidates, ranker)
    evaluators = (
        governance_evaluators
        if governance_evaluators is not None
        else build_production_capability_governance_evaluators()
    )
    context = build_production_capability_governance_context(
        environment,
        evidence=governance_evidence,
        agent_evidence=agent_evidence,
    )
    return govern_capability_candidates(
        ranked,
        evaluators=evaluators,
        context=context,
    )


def consume_governed_discovery_for_downstream(
    result: GovernedDiscoveryResult,
) -> tuple[GovernedCapabilityCandidate, ...]:
    """Production downstream boundary — allowed governed candidates only."""
    return result.allowed


__all__ = [
    "build_production_capability_governance_context",
    "build_production_capability_governance_evaluators",
    "consume_governed_discovery_for_downstream",
    "discover_rank_and_govern_capabilities",
    "resolve_capability_governance_posture",
]
