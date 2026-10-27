# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.


from intergrax.contracts.agent_binding_errors import AgentImportError

__all__ = ["AgentImportError", "ApplicationManifestConformanceError"]


class ApplicationManifestConformanceError(ValueError):
    """Manifest cannot be wired into a runtime registry."""
