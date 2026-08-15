"""Contribution-driven composition root for Vendor Knowledge Live."""

from __future__ import annotations

from intergrax.runtime.vendor_knowledge.contribution_catalog import (
    build_default_vendor_knowledge_contribution_catalog,
    build_vendor_knowledge_live_registration_registry as compose_live_registry,
)
from intergrax.runtime.vendor_knowledge.live.registration import (
    VendorKnowledgeLiveRegistrationRegistry,
)


def build_vendor_knowledge_live_registration_registry(
    *,
    discover_entry_points: bool = False,
) -> VendorKnowledgeLiveRegistrationRegistry:
    """Build the provider-neutral Live registry from the contribution catalog."""
    catalog = build_default_vendor_knowledge_contribution_catalog(
        discover_entry_points=discover_entry_points,
    )
    return compose_live_registry(catalog)


__all__ = ["build_vendor_knowledge_live_registration_registry"]
