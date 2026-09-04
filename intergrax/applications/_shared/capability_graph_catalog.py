# © Artur Czarnecki. All rights reserved.

"""Build capability graph catalog entries from application manifests."""

from __future__ import annotations

from intergrax.applications.contracts.application_capability_projection import (
    application_capability_descriptor_from_manifest,
)
from intergrax.contracts.application_capability_metadata import ApplicationCapabilityDescriptor
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.applications.reference.harness_manifest_catalog import build_harness_reference_manifests
from intergrax.contracts.capability_graph_catalog import ApplicationCapabilityCatalogEntry


def catalog_entry_from_descriptor(
    descriptor: ApplicationCapabilityDescriptor,
) -> ApplicationCapabilityCatalogEntry:
    """Map one application descriptor to a neutral catalog entry."""
    return ApplicationCapabilityCatalogEntry(
        app_id=descriptor.application_id,
        agent_contract_ids=list(descriptor.agent_contract_ids),
    )


def catalog_entry_from_manifest(
    manifest: ApplicationManifest,
) -> ApplicationCapabilityCatalogEntry:
    """Map one application manifest to a neutral catalog entry."""
    return catalog_entry_from_descriptor(application_capability_descriptor_from_manifest(manifest))


def build_harness_capability_catalog() -> tuple[ApplicationCapabilityCatalogEntry, ...]:
    """Reference catalog for harness capability graph edges."""
    return tuple(
        catalog_entry_from_manifest(manifest) for manifest in build_harness_reference_manifests()
    )
