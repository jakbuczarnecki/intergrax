# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Deployment layer distinction (monorepo and SaaS)."""

from __future__ import annotations

from enum import IntEnum
from typing import Final


class DeploymentTier(IntEnum):
    """
    Platform tier label—for logs, metrics, and product configuration.

    Aligned with ``docs/project/architecture/intergrax_runtime_architecture.md`` §5.1:

    - ``PLATFORM`` (0): Tier-0 — universal components (LLM, memory, adapters, …).
    - ``FRAMEWORK`` (1): Tier-1 — Nexus Agent OS (orchestration, registry, …).
    - ``AGENT`` (2): Tier-2 — concrete agent capability modules under ``agents/``.
    - ``APPLICATION`` (3): Tier-3 — ready-made environments under ``applications/``.

    ``PRODUCT`` is a deprecated alias for ``AGENT`` (legacy metadata).
    """

    PLATFORM = 0
    FRAMEWORK = 1
    AGENT = 2
    APPLICATION = 3
    PRODUCT = 2


TIER_PLATFORM: Final[DeploymentTier] = DeploymentTier.PLATFORM
TIER_FRAMEWORK: Final[DeploymentTier] = DeploymentTier.FRAMEWORK
TIER_AGENT: Final[DeploymentTier] = DeploymentTier.AGENT
TIER_APPLICATION: Final[DeploymentTier] = DeploymentTier.APPLICATION
TIER_PRODUCT: Final[DeploymentTier] = DeploymentTier.PRODUCT

__all__ = [
    "DeploymentTier",
    "TIER_AGENT",
    "TIER_APPLICATION",
    "TIER_FRAMEWORK",
    "TIER_PLATFORM",
    "TIER_PRODUCT",
]
