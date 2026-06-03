# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Optional smoke helpers for integration catalog registration (Phase P-Ext.1.4)."""

from __future__ import annotations

from intergrax.integrations.registry.catalog import catalog_snapshot


def integration_registered(slug: str) -> bool:
    """Return True when ``slug`` is present in the in-memory integration catalog."""
    normalized = slug.strip().lower()
    return normalized in catalog_snapshot()


def ping_integration(slug: str) -> bool:
    """Alias for catalog presence check (no live backend probe in harness tests)."""
    return integration_registered(slug)
