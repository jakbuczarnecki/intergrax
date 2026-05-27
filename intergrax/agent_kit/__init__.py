# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Tier-1 scaffolding helpers for **Tier-2 agents** and **Tier-3 applications**.

Concrete agents live under ``agents/<name>/`` (Tier-2).

Ready-made environments live under ``applications/<name>/`` (Tier-3).

They **import** ``intergrax.*`` only; ``intergrax`` **must not** import Tier-2 or Tier-3 code.

Shared building blocks (runtime, Nexus, FastAPI core, agents framework) remain in
``intergrax``. Use this namespace for small cross-product helpers and contracts
that stay free of any single product's domain.

Submodules:

- :mod:`intergrax.agent_kit.tiers` — platform tier labels (:class:`DeploymentTier`).
- :mod:`intergrax.agent_kit.metadata` — :class:`AgentProductMetadata` for agents and applications.
"""

from intergrax.agent_kit.metadata import AgentProductMetadata
from intergrax.agent_kit.tiers import (
    TIER_AGENT,
    TIER_APPLICATION,
    TIER_FRAMEWORK,
    TIER_PLATFORM,
    TIER_PRODUCT,
    DeploymentTier,
)

__all__ = [
    "AgentProductMetadata",
    "DeploymentTier",
    "TIER_AGENT",
    "TIER_APPLICATION",
    "TIER_FRAMEWORK",
    "TIER_PLATFORM",
    "TIER_PRODUCT",
]
