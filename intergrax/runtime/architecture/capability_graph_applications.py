# © Artur Czarnecki. All rights reserved.

"""Per-application capability graph edge mapping (Phase V-REM-CG.1)."""

from __future__ import annotations

from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.runtime.architecture.capability_graph import CapabilityEdge, CapabilityEdgeType


def application_capability_node_id(manifest: ApplicationManifest) -> str:
    """Stable application node id aligned with ``_system_nodes`` naming."""
    return f"application:{manifest.app_id}_application"


def resolve_binding_agent_contract_id(binding: AgentBinding) -> str:
    """Resolve agent contract id from binding metadata without dynamic imports."""
    if binding.contract_id:
        return binding.contract_id
    agent_type = binding.resolved_agent_type()
    instance = agent_type()
    return instance.get_contract().id


def catalog_application_manifests() -> tuple[ApplicationManifest, ...]:
    """Reference Tier-3 manifests used for harness capability graph edges."""
    from applications.lab_application.manifest import build_lab_manifest_default
    from applications.legal_application.manifest import LEGAL_APPLICATION_MANIFEST
    from applications.poc_template_application.manifest import APPLICATION_MANIFEST
    from applications.research_application.manifest import RESEARCH_APPLICATION_MANIFEST

    return (
        build_lab_manifest_default(),
        LEGAL_APPLICATION_MANIFEST,
        RESEARCH_APPLICATION_MANIFEST,
        APPLICATION_MANIFEST,
    )


def build_application_agent_edges(
    *,
    agent_node_ids: frozenset[str],
) -> list[CapabilityEdge]:
    """Map each application host to only its roster agents (not global union)."""
    edges: list[CapabilityEdge] = []
    for manifest in catalog_application_manifests():
        application_node = application_capability_node_id(manifest)
        for binding in manifest.enabled_agents():
            contract_id = resolve_binding_agent_contract_id(binding)
            agent_node = f"agent:{contract_id}"
            if agent_node not in agent_node_ids:
                continue
            edges.append(
                CapabilityEdge(
                    source_node_id=application_node,
                    target_node_id=agent_node,
                    edge_type=CapabilityEdgeType.DEPENDS_ON,
                )
            )
    return edges
