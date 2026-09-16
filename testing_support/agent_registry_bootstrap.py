# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Thin re-export shim — canonical owner: ``intergrax.dev_support.agent_registry_bootstrap``."""

from intergrax.dev_support.agent_registry_bootstrap import (
    AgentRegistryBootstrapIdentityError,
    bootstrap_agent_registry_from_agents,
    build_harness_registry,
    build_legal_registry,
    build_organization_worker_registry,
    build_research_registry,
)

__all__ = [
    "AgentRegistryBootstrapIdentityError",
    "bootstrap_agent_registry_from_agents",
    "build_harness_registry",
    "build_legal_registry",
    "build_organization_worker_registry",
    "build_research_registry",
]
