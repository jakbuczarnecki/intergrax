# © Artur Czarnecki. All rights reserved.

"""STRICT product capability graph deploy validation (APP-OPS-1 · §50.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from intergrax.applications._shared.roster_agent_contract_authority import (
    ContractAuthority,
    resolve_roster_agent_contract,
)
from intergrax.applications._shared.capability_graph_assembly_resolver import (
    CapabilityGraphAssemblyValidationResult,
    validate_environment_capability_graph,
)
from intergrax.applications._shared.capability_graph_wiring import EnvironmentCapabilityGraphView
from intergrax.applications._shared.registry_snapshot import HarnessRegistrySnapshot
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
from intergrax.applications.contracts.application_capability_projection import (
    resolve_binding_contract_id,
)
from intergrax.runtime.architecture.capability_graph_lineage import (
    CapabilityImpactReport,
    CapabilityLineageReport,
    build_capability_impact_report,
    build_capability_lineage_report,
)

if TYPE_CHECKING:
    from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile

STRICT_DEPLOY_BLOCKED_AGENT_LIFECYCLES: frozenset[AgentLifecycleState] = frozenset(
    {
        AgentLifecycleState.EXPERIMENTAL,
        AgentLifecycleState.DEVELOPMENT,
        AgentLifecycleState.CANDIDATE,
        AgentLifecycleState.DEPRECATED,
        AgentLifecycleState.RETIRED,
    }
)


@dataclass(frozen=True, slots=True)
class EnvironmentCapabilityDeployReport:
    """Environment-scoped capability graph deploy review artifact."""

    view: EnvironmentCapabilityGraphView
    lineage: CapabilityLineageReport
    impact: CapabilityImpactReport


def build_environment_capability_deploy_report(
    view: EnvironmentCapabilityGraphView,
) -> EnvironmentCapabilityDeployReport:
    """Build lineage and blast-radius reports for an environment capability graph."""
    graph = view.graph
    return EnvironmentCapabilityDeployReport(
        view=view,
        lineage=build_capability_lineage_report(graph),
        impact=build_capability_impact_report(graph),
    )


def validate_strict_capability_graph_deploy(
    view: EnvironmentCapabilityGraphView,
    snapshot: HarnessRegistrySnapshot,
    manifest: ApplicationManifest,
    env: ApplicationEnvironmentProfile,
    *,
    contract_authority: ContractAuthority | None = None,
) -> CapabilityGraphAssemblyValidationResult:
    """Validate STRICT product deploy rules for environment capability graph."""
    errors = list(validate_environment_capability_graph(view, snapshot, manifest).errors)

    if not view.graph.nodes:
        errors.append("environment capability graph must not be empty")

    deploy_report = build_environment_capability_deploy_report(view)
    if not deploy_report.impact.impacts:
        errors.append("capability impact report must include blast-radius entries")

    impact_by_node = {record.node_id: record for record in deploy_report.impact.impacts}

    for binding in manifest.enabled_agents():
        contract_id = resolve_binding_contract_id(binding)
        node_id = f"agent:{contract_id}"
        if not view.contains_node(node_id):
            errors.append(f"roster agent {contract_id!r} missing from environment capability graph")
        elif node_id not in impact_by_node:
            errors.append(f"roster agent {contract_id!r} missing from capability impact report")

    if (
        env.execution_mode is ExecutionMode.STRICT
        and env.application_profile is ApplicationProfile.PRODUCT
    ):
        if contract_authority is None:
            errors.append(
                "STRICT product deploy validation requires revision-bound agent contract authority",
            )
        for binding in manifest.enabled_agents():
            contract_id = resolve_binding_contract_id(binding)
            node_id = f"agent:{contract_id}"
            if contract_authority is None:
                continue
            contract = resolve_roster_agent_contract(
                binding,
                contract_authority=contract_authority,
                allow_compatibility_resolver=False,
            )
            if contract.lifecycle_state in STRICT_DEPLOY_BLOCKED_AGENT_LIFECYCLES:
                blast = impact_by_node.get(node_id)
                radius_size = len(blast.blast_radius_node_ids) if blast is not None else 0
                errors.append(
                    f"STRICT deploy blocks roster agent {contract_id} lifecycle "
                    f"{contract.lifecycle_state.value} (blast radius {radius_size} nodes)"
                )

    return CapabilityGraphAssemblyValidationResult(valid=not errors, errors=tuple(errors))


__all__ = [
    "EnvironmentCapabilityDeployReport",
    "STRICT_DEPLOY_BLOCKED_AGENT_LIFECYCLES",
    "build_environment_capability_deploy_report",
    "validate_strict_capability_graph_deploy",
]
