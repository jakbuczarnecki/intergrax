# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Deployment layer distinction (monorepo and SaaS)."""

from __future__ import annotations

from enum import IntEnum
from typing import Final


class DeploymentTier(IntEnum):
    """
    Deployment layer label—for logs, metrics, and product configuration.

    - ``PLATFORM``: core engine (execution, trace storage, etc.) — Tier-0.
    - ``FRAMEWORK``: Nexus, FastAPI core, agent scaffolding — Tier-1.
    - ``PRODUCT``: concrete agent / HTTP host / UI — Tier-2 (outside the ``intergrax`` package).
    """

    PLATFORM = 0
    FRAMEWORK = 1
    PRODUCT = 2


TIER_PLATFORM: Final[DeploymentTier] = DeploymentTier.PLATFORM
TIER_FRAMEWORK: Final[DeploymentTier] = DeploymentTier.FRAMEWORK
TIER_PRODUCT: Final[DeploymentTier] = DeploymentTier.PRODUCT

__all__ = [
    "DeploymentTier",
    "TIER_FRAMEWORK",
    "TIER_PLATFORM",
    "TIER_PRODUCT",
]
