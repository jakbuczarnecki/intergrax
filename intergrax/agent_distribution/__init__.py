# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tier-0 Agent Distribution domain contracts and store ports (AP-3)."""

from intergrax.agent_distribution.binding import (
    AgentBindingFactoryReference,
    AgentBindingPolicyOverrides,
    ApplicationAgentBinding,
)
from intergrax.agent_distribution.catalog import (
    AgentCatalogEntry,
    AgentCatalogVersionChannelRef,
    CatalogEntryFilters,
    CatalogPackageResolution,
    CatalogProviderKind,
    CatalogSourceIdentity,
    CatalogSourceProvider,
    ProviderHealth,
    ProviderHealthStatus,
)
from intergrax.agent_distribution.dependency import (
    CandidateDependencySpecification,
    DependencyResolverInput,
    InstalledAgentPackageRequirement,
    InstalledAgentRequirementSet,
    MaterializedAgentClosureEntry,
    MaterializedLockPackage,
    MaterializedLockReproducibilityEvidence,
    MaterializedLockRollbackEvidence,
    MaterializedRuntimeLock,
    PolicyDependencyConstraint,
    RepositoryDependencyDeclaration,
)
from intergrax.agent_distribution.identity import AgentPackageCandidate, AgentPackageIdentity
from intergrax.agent_distribution.installation import (
    AgentInstallationRecord,
    InstallationState,
    installation_state_is_installed,
)
from intergrax.agent_distribution.materialization import MaterializationInput, MaterializationOutput
from intergrax.agent_distribution.roster import EffectiveRoster, EffectiveRosterEntry
from intergrax.agent_distribution.runtime_graph import (
    CandidateApplicationRuntimeGraph,
    RuntimeGraphAgentRef,
    RuntimeGraphThirdPartyRef,
    RuntimeGraphTierViolation,
)
from intergrax.agent_distribution.runtime_revision import (
    MaterializationTopology,
    RuntimeRevision,
    RuntimeRevisionState,
)
from intergrax.agent_distribution.stores import (
    AgentArtifactMetadata,
    AgentArtifactMetadataStore,
    AgentInstallationStore,
    ApplicationAgentBindingStore,
    MaterializedRuntimeLockStore,
    RuntimeRevisionStore,
)
from intergrax.agent_distribution.trust import (
    AgentDeliverySource,
    AgentInstallationTrustRecord,
    AgentPackageQualificationResult,
    AgentPublisherIdentity,
    AgentQualificationEvidence,
    AgentQualificationEvidenceKind,
    AgentQualificationStatus,
    AgentTrustEvidenceRef,
)

__all__ = [
    "AgentArtifactMetadata",
    "AgentArtifactMetadataStore",
    "AgentBindingFactoryReference",
    "AgentBindingPolicyOverrides",
    "AgentCatalogEntry",
    "AgentCatalogVersionChannelRef",
    "AgentDeliverySource",
    "AgentInstallationRecord",
    "AgentInstallationStore",
    "AgentInstallationTrustRecord",
    "AgentPackageCandidate",
    "AgentPackageIdentity",
    "AgentPackageQualificationResult",
    "AgentPublisherIdentity",
    "AgentQualificationEvidence",
    "AgentQualificationEvidenceKind",
    "AgentQualificationStatus",
    "AgentTrustEvidenceRef",
    "ApplicationAgentBinding",
    "ApplicationAgentBindingStore",
    "CandidateApplicationRuntimeGraph",
    "CandidateDependencySpecification",
    "CatalogEntryFilters",
    "CatalogPackageResolution",
    "CatalogProviderKind",
    "CatalogSourceIdentity",
    "CatalogSourceProvider",
    "DependencyResolverInput",
    "EffectiveRoster",
    "EffectiveRosterEntry",
    "InstalledAgentPackageRequirement",
    "InstalledAgentRequirementSet",
    "InstallationState",
    "MaterializationInput",
    "MaterializationOutput",
    "MaterializationTopology",
    "MaterializedAgentClosureEntry",
    "MaterializedLockPackage",
    "MaterializedLockReproducibilityEvidence",
    "MaterializedLockRollbackEvidence",
    "MaterializedRuntimeLock",
    "MaterializedRuntimeLockStore",
    "PolicyDependencyConstraint",
    "ProviderHealth",
    "ProviderHealthStatus",
    "RepositoryDependencyDeclaration",
    "RuntimeGraphAgentRef",
    "RuntimeGraphThirdPartyRef",
    "RuntimeGraphTierViolation",
    "RuntimeRevision",
    "RuntimeRevisionState",
    "RuntimeRevisionStore",
    "installation_state_is_installed",
]
