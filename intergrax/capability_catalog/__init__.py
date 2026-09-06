# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Federated capability catalog read model (CAPABILITY-CATALOG-1 Stage 2–5)."""

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
    CapabilityGovernanceError,
    CapabilityRankingError,
)
from intergrax.capability_catalog.federation import (
    FederatedCapabilityCatalog,
    merge_capability_catalog_entries,
)
from intergrax.capability_catalog.governed_candidate import (
    SCHEMA_BLOCKED_CAPABILITY_CANDIDATE_V1,
    SCHEMA_GOVERNED_CAPABILITY_CANDIDATE_V1,
    BlockedCapabilityCandidate,
    GovernedCapabilityCandidate,
)
from intergrax.capability_catalog.governed_result import (
    SCHEMA_GOVERNED_DISCOVERY_RESULT_V1,
    GovernedDiscoveryResult,
)
from intergrax.capability_catalog.governance import (
    AVAILABILITY_PRESERVING_GOVERNANCE_EVALUATOR_ID,
    AvailabilityPreservingGovernanceEvaluator,
    CapabilityGovernanceDecision,
    CapabilityGovernanceEvaluator,
    govern_capability_candidates,
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
    "AvailabilityPreservingGovernanceEvaluator",
    "AVAILABILITY_PRESERVING_GOVERNANCE_EVALUATOR_ID",
    "BlockedCapabilityCandidate",
    "CapabilityCatalogConfigurationError",
    "CapabilityCatalogDiscoveryError",
    "CapabilityCatalogEntry",
    "CapabilityCatalogError",
    "CapabilityCatalogIdentityConflict",
    "CapabilityCatalogSnapshot",
    "CapabilityCatalogSource",
    "CapabilityCatalogSourceFailure",
    "CapabilityDiscoveryCandidate",
    "CapabilityGovernanceDecision",
    "CapabilityGovernanceError",
    "CapabilityGovernanceEvaluator",
    "CapabilityRanker",
    "CapabilityRankingError",
    "FederatedCapabilityCatalog",
    "GovernedCapabilityCandidate",
    "GovernedDiscoveryResult",
    "RankedCapabilityCandidate",
    "SCHEMA_BLOCKED_CAPABILITY_CANDIDATE_V1",
    "SCHEMA_CAPABILITY_CATALOG_ENTRY_V1",
    "SCHEMA_CAPABILITY_CATALOG_SNAPSHOT_V1",
    "SCHEMA_CAPABILITY_DISCOVERY_CANDIDATE_V1",
    "SCHEMA_GOVERNED_CAPABILITY_CANDIDATE_V1",
    "SCHEMA_GOVERNED_DISCOVERY_RESULT_V1",
    "SCHEMA_RANKED_CAPABILITY_CANDIDATE_V1",
    "STABLE_IDENTITY_RANKER_ID",
    "StableIdentityRanker",
    "discover_capability_candidates",
    "govern_capability_candidates",
    "merge_capability_catalog_entries",
    "rank_capability_candidates",
]
