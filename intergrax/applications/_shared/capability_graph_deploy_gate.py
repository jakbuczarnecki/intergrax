# © Artur Czarnecki. All rights reserved.

"""STRICT product capability graph deploy validation (APP-OPS-1 · §50.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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
        for binding in manifest.enabled_agents():
            contract_id = resolve_binding_contract_id(binding)
            node_id = f"agent:{contract_id}"
            contract = binding.resolved_agent_type()().get_contract()
            if contract.lifecycle_state in STRICT_DEPLOY_BLOCKED_AGENT_LIFECYCLES:
                blast = impact_by_node.get(node_id)
                radius_size = len(blast.blast_radius_node_ids) if blast is not None else 0
                errors.append(
                    f"STRICT deploy blocks roster agent {contract_id} lifecycle "
                    f"{contract.lifecycle_state.value} (blast radius {radius_size} nodes)"
                )

    return CapabilityGraphAssemblyValidationResult(valid=not errors, errors=tuple(errors))


def _gate_wiring_environment(env: ApplicationEnvironmentProfile) -> ApplicationEnvironmentProfile:
    """Use lab bindings for CI wiring — avoids optional vendor drivers (e.g. Neo4j)."""
    from intergrax.applications.contracts.application_host import ApplicationProfile
    from intergrax.integrations.registry.profile import IntegrationProfile

    return env.model_copy(
        update={
            "integration_profile": IntegrationProfile.lab(),
            "application_profile": ApplicationProfile.LAB,
        }
    )


def check_strict_product_capability_graph(
    product_id: str,
    manifest: ApplicationManifest,
) -> list[str]:
    """Return deploy-gate violations for one STRICT product manifest."""
    from intergrax.applications._shared.environment_wiring import wire_application_environment

    env = manifest.resolved_environment()
    if env.execution_mode is not ExecutionMode.STRICT:
        return []
    if manifest.profile is not ApplicationProfile.PRODUCT:
        return []

    gate_env = _gate_wiring_environment(env)
    try:
        wiring = wire_application_environment(manifest, gate_env, conformance_check=False)
    except Exception as exc:  # noqa: BLE001 — gate surfaces wiring failures
        return [f"{product_id}: wire_application_environment failed: {exc}"]

    view = wiring.capability_graph
    snapshot = wiring.registry_snapshot
    if view is None:
        return [f"{product_id}: capability_graph not materialized"]
    if snapshot is None:
        return [f"{product_id}: registry_snapshot not materialized"]

    result = validate_strict_capability_graph_deploy(
        view,
        snapshot,
        manifest,
        env,
    )
    return [f"{product_id}: {error}" for error in result.errors]
