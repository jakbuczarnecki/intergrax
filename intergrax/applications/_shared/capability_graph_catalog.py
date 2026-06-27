# © Artur Czarnecki. All rights reserved.

"""Build capability graph catalog entries from application manifests."""

from __future__ import annotations

from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.applications.reference.harness_manifest_catalog import build_harness_reference_manifests
from intergrax.contracts.capability_graph_catalog import ApplicationCapabilityCatalogEntry


def resolve_binding_agent_contract_id(binding: AgentBinding) -> str:
    """Resolve agent contract id from binding metadata without dynamic imports."""
    if binding.contract_id:
        return binding.contract_id
    agent_type = binding.resolved_agent_type()
    instance = agent_type()
    return instance.get_contract().id


def catalog_entry_from_manifest(manifest: ApplicationManifest) -> ApplicationCapabilityCatalogEntry:
    """Map one application manifest to a neutral catalog entry."""
    return ApplicationCapabilityCatalogEntry(
        app_id=manifest.app_id,
        agent_contract_ids=[
            resolve_binding_agent_contract_id(binding) for binding in manifest.enabled_agents()
        ],
    )


def build_harness_capability_catalog() -> tuple[ApplicationCapabilityCatalogEntry, ...]:
    """Reference catalog for harness capability graph edges."""
    return tuple(catalog_entry_from_manifest(m) for m in build_harness_reference_manifests())
