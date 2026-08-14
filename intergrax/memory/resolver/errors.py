# © Artur Czarnecki. All rights reserved.

"""Memory store plugin resolution errors (ENTERPRISE-5 / BLOCK D)."""


class MemoryStorePluginResolutionError(RuntimeError):
    """Fail-closed error for explicit external Memory store selection."""
