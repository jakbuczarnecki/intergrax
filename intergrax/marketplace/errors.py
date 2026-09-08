# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Marketplace catalog configuration errors (Stage 11)."""


class MarketplaceCatalogError(ValueError):
    """Base marketplace catalog error."""


class MarketplaceCatalogConfigurationError(MarketplaceCatalogError):
    """Invalid marketplace source or snapshot configuration."""
