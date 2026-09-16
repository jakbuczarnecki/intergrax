# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Marketplace lifecycle handoff orchestration errors (ME-RB4)."""


class MarketplaceLifecycleHandoffError(Exception):
    """Base error for handoff orchestration."""


class MarketplaceLifecycleHandoffValidationError(MarketplaceLifecycleHandoffError):
    """Invalid handoff request before domain delegation."""


class MarketplaceLifecycleHandlerError(MarketplaceLifecycleHandoffError):
    """Expected handler failure surfaced as HANDLER_FAILED outcome."""


__all__ = [
    "MarketplaceLifecycleHandoffError",
    "MarketplaceLifecycleHandoffValidationError",
    "MarketplaceLifecycleHandlerError",
]
