# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tier-3 application composition contracts (Phase N)."""

from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.factory import AgentFactory
from intergrax.applications.contracts.manifest import (
    AgentBinding,
    ApplicationFeatures,
    ApplicationManifest,
    ApplicationProfile,
)
from intergrax.applications._shared.wiring import (
    AgentImportError,
    ApplicationManifestConformanceError,
    build_agent_from_binding,
    build_application_registry,
    build_registry_from_manifest,
    load_agent_from_binding,
    validate_manifest_wiring,
)

__all__ = [
    "AgentBinding",
    "AgentFactory",
    "AgentImportError",
    "ApplicationBuildContext",
    "ApplicationFeatures",
    "ApplicationManifest",
    "ApplicationManifestConformanceError",
    "ApplicationProfile",
    "build_agent_from_binding",
    "build_application_registry",
    "build_registry_from_manifest",
    "load_agent_from_binding",
    "validate_manifest_wiring",
]
