# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog slug conflict resolution for Tier-0 bootstrap (Phase P-Ext.4.3)."""

from __future__ import annotations

import logging
from typing import Literal

from intergrax.core.plugins.discovery import ConflictPolicy

logger = logging.getLogger(__name__)

CatalogSlugConflictPolicy = Literal["error", "override", "warn_override", "skip"]


def entry_point_conflict_policy(on_conflict: ConflictPolicy) -> ConflictPolicy:
    """Map bootstrap ``on_conflict`` to entry-point loader policy."""
    if on_conflict == "warn_override":
        return "override"
    return on_conflict


def catalog_registration_override(
    *,
    slug: str,
    slug_registered: bool,
    on_conflict: ConflictPolicy,
    catalog_kind: str,
    plugin_type: type,
) -> bool:
    """
    Decide whether to pass ``override=True`` to catalog register functions.

    Raises ``ValueError`` when ``on_conflict`` is ``error`` and the slug exists.
    Returns ``False`` without registering when ``on_conflict`` is ``skip`` and slug exists.
    """
    if not slug_registered:
        return False
    if on_conflict == "skip":
        logger.warning(
            "Skipping %s plugin %s: catalog slug %r already registered",
            catalog_kind,
            plugin_type.__qualname__,
            slug,
        )
        return False
    if on_conflict == "warn_override":
        logger.warning(
            "Overriding catalog slug %r with %s plugin %s",
            slug,
            catalog_kind,
            plugin_type.__qualname__,
        )
        return True
    if on_conflict == "override":
        return True
    raise ValueError(
        f"Catalog slug {slug!r} is already registered; "
        f"cannot register {plugin_type.__qualname__} with on_conflict='error'"
    )


def should_skip_catalog_registration(
    *,
    slug_registered: bool,
    on_conflict: ConflictPolicy,
) -> bool:
    """True when registration should be skipped (``skip`` policy + existing slug)."""
    return slug_registered and on_conflict == "skip"
