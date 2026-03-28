# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Minimal metadata for an agent product (Tier-2), with no domain coupling.

Products may attach this shape to telemetry, health checks, or host documentation.
"""

from __future__ import annotations

from typing import NotRequired, TypedDict

from intergrax.agent_kit.tiers import DeploymentTier


class AgentProductMetadata(TypedDict, total=False):
    """Optional fields identifying a Tier-2 product deployment."""

    product_id: str
    """Stable product identifier (e.g. ``legal_agent``)."""

    display_name: str
    """Human-readable name for UI or logs."""

    deployment_tier: DeploymentTier
    """For Tier-2 hosts, set to :attr:`DeploymentTier.PRODUCT`."""


__all__ = ["AgentProductMetadata"]
