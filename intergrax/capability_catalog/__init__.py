# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Federated capability catalog read model (CAPABILITY-CATALOG-1 Stage 2–4)."""

from __future__ import annotations

from intergrax.capability_catalog.candidate import (
    SCHEMA_CAPABILITY_DISCOVERY_CANDIDATE_V1,
    CapabilityDiscoveryCandidate,
)
from intergrax.capability_catalog.discovery import discover_capability_candidates
from intergrax.capability_catalog.entry import (
    SCHEMA_CAPABILITY_CATALOG_ENTRY_V1,
    CapabilityCatalogEntry,
)
from intergrax.capability_catalog.errors import (
    CapabilityCatalogConfigurationError,
    CapabilityCatalogDiscoveryError,
    CapabilityCatalogError,
    CapabilityCatalogIdentityConflict,
    CapabilityCatalogSourceFailure,
    CapabilityRankingError,
)
from intergrax.capability_catalog.federation import (
    FederatedCapabilityCatalog,
    merge_capability_catalog_entries,
)
from intergrax.capability_catalog.ranked_candidate import (
    SCHEMA_RANKED_CAPABILITY_CANDIDATE_V1,
    RankedCapabilityCandidate,
)
from intergrax.capability_catalog.ranking import (
    STABLE_IDENTITY_RANKER_ID,
    CapabilityRanker,
    StableIdentityRanker,
    rank_capability_candidates,
)
from intergrax.capability_catalog.snapshot import (
    SCHEMA_CAPABILITY_CATALOG_SNAPSHOT_V1,
    CapabilityCatalogSnapshot,
)
from intergrax.capability_catalog.source import CapabilityCatalogSource

__all__ = [
    "CapabilityCatalogConfigurationError",
    "CapabilityCatalogDiscoveryError",
    "CapabilityCatalogEntry",
    "CapabilityCatalogError",
    "CapabilityCatalogIdentityConflict",
    "CapabilityCatalogSnapshot",
    "CapabilityCatalogSource",
    "CapabilityCatalogSourceFailure",
    "CapabilityDiscoveryCandidate",
    "CapabilityRanker",
    "CapabilityRankingError",
    "FederatedCapabilityCatalog",
    "RankedCapabilityCandidate",
    "SCHEMA_CAPABILITY_CATALOG_ENTRY_V1",
    "SCHEMA_CAPABILITY_CATALOG_SNAPSHOT_V1",
    "SCHEMA_CAPABILITY_DISCOVERY_CANDIDATE_V1",
    "SCHEMA_RANKED_CAPABILITY_CANDIDATE_V1",
    "STABLE_IDENTITY_RANKER_ID",
    "StableIdentityRanker",
    "discover_capability_candidates",
    "rank_capability_candidates",
    "merge_capability_catalog_entries",
]
