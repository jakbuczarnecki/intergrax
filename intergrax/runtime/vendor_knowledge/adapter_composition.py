# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical contribution-driven composition of Vendor Knowledge adapters."""

from __future__ import annotations

from intergrax.runtime.vendor_knowledge.contribution_catalog import (
    build_default_vendor_knowledge_contribution_catalog,
    build_vendor_knowledge_adapter_registry,
)
from intergrax.runtime.vendor_knowledge.registry import KnowledgeAdapterRegistry


def build_default_vendor_knowledge_adapter_registry(
    *,
    discover_entry_points: bool = False,
) -> KnowledgeAdapterRegistry:
    """Build the deterministic registry from the canonical contribution catalog."""
    catalog = build_default_vendor_knowledge_contribution_catalog(
        discover_entry_points=discover_entry_points,
    )
    return build_vendor_knowledge_adapter_registry(catalog)


__all__ = ["build_default_vendor_knowledge_adapter_registry"]
