# © Artur Czarnecki. All rights reserved.

"""Backward-compatible re-export — canonical bridge lives in ``intergrax.marketplace``."""

from intergrax.marketplace.handoff.adapters.tool_acquisition_bridge import (
    ToolMarketplaceAcquisitionBridge,
)

__all__ = ["ToolMarketplaceAcquisitionBridge"]
