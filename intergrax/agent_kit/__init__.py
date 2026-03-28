# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Tier-1 scaffolding for **agent products** (Tier-2).

Agent products (domain + host + UI) live **outside** the ``intergrax`` package —
typically under ``applications/<product>/`` in this monorepo, or in separate repos.
They **import** ``intergrax.*`` only; ``intergrax`` **must not** import Tier-2 code.

Shared building blocks (runtime, Nexus, FastAPI core, agents framework) remain in
``intergrax``. Use this namespace for small cross-product helpers and contracts
that stay free of any single product's domain.

Submodules:

- :mod:`intergrax.agent_kit.tiers` — deployment layer labels (:class:`DeploymentTier`).
- :mod:`intergrax.agent_kit.metadata` — :class:`AgentProductMetadata` for Tier-2 hosts.
"""

from intergrax.agent_kit.metadata import AgentProductMetadata
from intergrax.agent_kit.tiers import (
    TIER_FRAMEWORK,
    TIER_PLATFORM,
    TIER_PRODUCT,
    DeploymentTier,
)

__all__ = [
    "AgentProductMetadata",
    "DeploymentTier",
    "TIER_FRAMEWORK",
    "TIER_PLATFORM",
    "TIER_PRODUCT",
]
