# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Cross-domain Capability Catalog discovery vocabulary (CAPABILITY-CATALOG-1 Stage 1)."""

from __future__ import annotations

from intergrax.contracts.capability_catalog.availability import (
    AvailabilityDisposition,
    NORMATIVE_AVAILABILITY_DISPOSITIONS,
)
from intergrax.contracts.capability_catalog.evidence import (
    SCHEMA_CAPABILITY_DISCOVERY_AVAILABILITY_EVIDENCE_V1,
    CapabilityDiscoveryAvailabilityEvidence,
)
from intergrax.contracts.capability_catalog.identity import (
    SCHEMA_CAPABILITY_DISCOVERY_IDENTITY_V1,
    SCHEMA_CAPABILITY_LOGICAL_IDENTITY_V1,
    SCHEMA_CAPABILITY_SOURCE_IDENTITY_V1,
    CapabilityCatalogContractError,
    CapabilityDiscoveryIdentity,
    CapabilityLogicalIdentity,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)
from intergrax.contracts.capability_catalog.identity_key import (
    SCHEMA_CAPABILITY_IDENTITY_KEY_V1,
    CapabilityIdentityKey,
)
from intergrax.contracts.capability_catalog.kind import (
    V1_CAPABILITY_KINDS,
    CapabilityKind,
)
from intergrax.contracts.capability_catalog.query import (
    SCHEMA_CAPABILITY_DISCOVERY_QUERY_V1,
    CapabilityDiscoveryQuery,
    LogicalIdentityFilter,
    SourceFilter,
)
from intergrax.contracts.capability_catalog.provenance import (
    SCHEMA_CAPABILITY_PROVENANCE_V1,
    CapabilityProvenance,
)
from intergrax.contracts.capability_catalog.scope import (
    SCHEMA_CAPABILITY_DISCOVERY_SCOPE_V1,
    CapabilityDiscoveryScope,
    CapabilityDiscoveryScopeMode,
)
from intergrax.contracts.capability_catalog.vocabulary import (
    NORMATIVE_CAPABILITY_STAGE_VOCABULARY,
    CapabilityStageVocabulary,
)

__all__ = [
    "AvailabilityDisposition",
    "CapabilityCatalogContractError",
    "CapabilityDiscoveryAvailabilityEvidence",
    "CapabilityDiscoveryIdentity",
    "CapabilityDiscoveryQuery",
    "CapabilityDiscoveryScope",
    "CapabilityDiscoveryScopeMode",
    "CapabilityIdentityKey",
    "CapabilityKind",
    "CapabilityLogicalIdentity",
    "LogicalIdentityFilter",
    "NORMATIVE_AVAILABILITY_DISPOSITIONS",
    "CapabilityProvenance",
    "CapabilitySourceIdentity",
    "CapabilitySourceKind",
    "CapabilityStageVocabulary",
    "NORMATIVE_CAPABILITY_STAGE_VOCABULARY",
    "SCHEMA_CAPABILITY_DISCOVERY_AVAILABILITY_EVIDENCE_V1",
    "SCHEMA_CAPABILITY_DISCOVERY_IDENTITY_V1",
    "SCHEMA_CAPABILITY_DISCOVERY_QUERY_V1",
    "SCHEMA_CAPABILITY_DISCOVERY_SCOPE_V1",
    "SCHEMA_CAPABILITY_IDENTITY_KEY_V1",
    "SourceFilter",
    "SCHEMA_CAPABILITY_LOGICAL_IDENTITY_V1",
    "SCHEMA_CAPABILITY_PROVENANCE_V1",
    "SCHEMA_CAPABILITY_SOURCE_IDENTITY_V1",
    "V1_CAPABILITY_KINDS",
]
