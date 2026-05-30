# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.


class AgentImportError(ImportError):
    """Failed to import or instantiate an agent from a binding."""


class ApplicationManifestConformanceError(ValueError):
    """Manifest cannot be wired into a runtime registry."""
