# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Marketplace → domain lifecycle handoff intent (ME-RB4)."""

from __future__ import annotations

from enum import StrEnum


class MarketplaceLifecycleHandoffIntent(StrEnum):
    """
    High-level marketplace intent — not install/activate/execute semantics.

    Domain authorities interpret intent within their own lifecycle contracts.
    """

    REQUEST_LIFECYCLE = "request_lifecycle"
    REQUEST_AVAILABILITY = "request_availability"
    PREPARE = "prepare"


__all__ = ["MarketplaceLifecycleHandoffIntent"]
