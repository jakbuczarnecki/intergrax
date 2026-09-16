# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""ME-10 marketplace handoff traceability errors."""


class MarketplaceHandoffTraceabilityError(Exception):
    """Base error for discovery → handoff orchestration."""


class MarketplaceHandoffSelectionError(MarketplaceHandoffTraceabilityError):
    """Explicit selection is not admissible in the governed visible candidate set."""


class MarketplaceHandoffTenantConsistencyError(MarketplaceHandoffTraceabilityError):
    """Tenant scope on handoff facts is inconsistent with caller context."""


__all__ = [
    "MarketplaceHandoffSelectionError",
    "MarketplaceHandoffTenantConsistencyError",
    "MarketplaceHandoffTraceabilityError",
]
