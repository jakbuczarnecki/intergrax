# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Federated capability catalog read model (CAPABILITY-CATALOG-1 Stage 2)."""

from __future__ import annotations

from intergrax.capability_catalog.entry import (
    SCHEMA_CAPABILITY_CATALOG_ENTRY_V1,
    CapabilityCatalogEntry,
)
from intergrax.capability_catalog.errors import (
    CapabilityCatalogConfigurationError,
    CapabilityCatalogError,
    CapabilityCatalogIdentityConflict,
    CapabilityCatalogSourceFailure,
)
from intergrax.capability_catalog.federation import (
    FederatedCapabilityCatalog,
    merge_capability_catalog_entries,
)
from intergrax.capability_catalog.snapshot import (
    SCHEMA_CAPABILITY_CATALOG_SNAPSHOT_V1,
    CapabilityCatalogSnapshot,
)
from intergrax.capability_catalog.source import CapabilityCatalogSource

__all__ = [
    "CapabilityCatalogConfigurationError",
    "CapabilityCatalogEntry",
    "CapabilityCatalogError",
    "CapabilityCatalogIdentityConflict",
    "CapabilityCatalogSnapshot",
    "CapabilityCatalogSource",
    "CapabilityCatalogSourceFailure",
    "FederatedCapabilityCatalog",
    "SCHEMA_CAPABILITY_CATALOG_ENTRY_V1",
    "SCHEMA_CAPABILITY_CATALOG_SNAPSHOT_V1",
    "merge_capability_catalog_entries",
]
