# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Agent discovery and registration."""

from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead

__all__ = [
    "AgentRegistry",
    "AgentRegistryRead",
]
