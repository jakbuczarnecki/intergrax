# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Cross-domain Capability Catalog discovery vocabulary (CAPABILITY-CATALOG-1 Stage 1)."""

from __future__ import annotations

from intergrax.contracts.capability_catalog.identity import (
    SCHEMA_CAPABILITY_DISCOVERY_IDENTITY_V1,
    SCHEMA_CAPABILITY_LOGICAL_IDENTITY_V1,
    SCHEMA_CAPABILITY_SOURCE_IDENTITY_V1,
    CapabilityCatalogContractError,
    CapabilityDiscoveryIdentity,
    CapabilityDiscoveryIdentityConflict,
    CapabilityLogicalIdentity,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
    discovery_identity_sort_key,
    normalize_discovery_identity_set,
)
from intergrax.contracts.capability_catalog.kind import (
    V1_CAPABILITY_KINDS,
    CapabilityKind,
)
from intergrax.contracts.capability_catalog.provenance import (
    SCHEMA_CAPABILITY_PROVENANCE_V1,
    CapabilityProvenance,
)
from intergrax.contracts.capability_catalog.vocabulary import (
    NORMATIVE_CAPABILITY_STAGE_VOCABULARY,
    CapabilityStageVocabulary,
)

__all__ = [
    "CapabilityCatalogContractError",
    "CapabilityDiscoveryIdentity",
    "CapabilityDiscoveryIdentityConflict",
    "CapabilityKind",
    "CapabilityLogicalIdentity",
    "CapabilityProvenance",
    "CapabilitySourceIdentity",
    "CapabilitySourceKind",
    "CapabilityStageVocabulary",
    "NORMATIVE_CAPABILITY_STAGE_VOCABULARY",
    "SCHEMA_CAPABILITY_DISCOVERY_IDENTITY_V1",
    "SCHEMA_CAPABILITY_LOGICAL_IDENTITY_V1",
    "SCHEMA_CAPABILITY_PROVENANCE_V1",
    "SCHEMA_CAPABILITY_SOURCE_IDENTITY_V1",
    "V1_CAPABILITY_KINDS",
    "discovery_identity_sort_key",
    "normalize_discovery_identity_set",
]
