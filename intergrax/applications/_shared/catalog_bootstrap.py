# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Re-export Tier-0 catalog bootstrap from ``intergrax.core`` (avoids heavy ``applications`` import chain)."""

from intergrax.core.catalog_bootstrap import (
    CatalogBootstrapResult,
    bootstrap_catalogs,
    reset_tier0_catalog_bootstrap_for_tests,
)

__all__ = [
    "CatalogBootstrapResult",
    "bootstrap_catalogs",
    "reset_tier0_catalog_bootstrap_for_tests",
]
