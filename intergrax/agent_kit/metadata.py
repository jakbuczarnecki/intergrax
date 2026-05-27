# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Minimal metadata for an agent or application deployment, with no domain coupling.

Products may attach this shape to telemetry, health checks, or host documentation.
"""

from __future__ import annotations

from typing import NotRequired, TypedDict

from intergrax.agent_kit.tiers import DeploymentTier


class AgentProductMetadata(TypedDict, total=False):
    """Optional fields identifying a Tier-2 agent or Tier-3 application deployment."""

    product_id: str
    """Stable product identifier (e.g. ``legal``)."""

    display_name: str
    """Human-readable name for UI or logs."""

    deployment_tier: DeploymentTier
    """For Tier-2 agents use :attr:`DeploymentTier.AGENT`; for Tier-3 hosts use :attr:`DeploymentTier.APPLICATION`."""


__all__ = ["AgentProductMetadata"]
