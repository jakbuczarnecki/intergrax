# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tier-0 Agent Distribution domain contracts, stores, and services (AP-3/AP-4)."""

from intergrax.agent_distribution.binding import (
    AgentBindingFactoryReference,
    AgentBindingPolicyOverrides,
    ApplicationAgentBinding,
)
from intergrax.agent_distribution.binding_service import BindingService
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
from intergrax.agent_distribution.errors import (
    AgentDistributionError,
    AgentDistributionNotFoundError,
    AgentPackageTrustError,
    BindingLifecycleError,
    BindingRevisionConflict,
    InstallationLifecycleError,
    InstallationSlotConflict,
    RuntimeRevisionConflict,
    RuntimeRevisionLifecycleError,
)
from intergrax.agent_distribution.events import AgentDistributionEvent, TransitionResult
from intergrax.agent_distribution.in_memory_stores import (
    AgentDistributionStoreState,
    InMemoryAgentArtifactMetadataStore,
    InMemoryAgentInstallationStore,
    InMemoryApplicationAgentBindingStore,
    InMemoryRuntimeRevisionStore,
)
from intergrax.agent_distribution.installation_service import InstallationService
from intergrax.agent_distribution.installation import (
    AgentInstallationRecord,
    InstallationState,
    installation_state_is_installed,
)
from intergrax.agent_distribution.package_trust import (
    AgentPackageTrustCoordinator,
    assert_installation_trust_record_acceptable,
)
from intergrax.agent_distribution.materialization import MaterializationInput, MaterializationOutput
from intergrax.agent_distribution.roster import EffectiveRoster, EffectiveRosterEntry
from intergrax.agent_distribution.runtime_graph import (
    CandidateApplicationRuntimeGraph,
    RuntimeGraphAgentRef,
    RuntimeGraphThirdPartyRef,
    RuntimeGraphTierViolation,
)
from intergrax.agent_distribution.identity import AgentPackageCandidate, AgentPackageIdentity
from intergrax.agent_distribution.runtime_revision import (
    MaterializationTopology,
    RuntimeRevision,
    RuntimeRevisionState,
)
from intergrax.agent_distribution.runtime_revision_service import RuntimeRevisionService
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
    AgentPackageTrustDecision,
    AgentPackageTrustOutcome,
    AgentPackageTrustPolicy,
    AgentPackageTrustPosture,
    AgentPackageTrustReasonCode,
    AgentPackageTrustRevocationState,
    AgentPublisherIdentity,
    AgentQualificationEvidence,
    AgentQualificationEvidenceKind,
    AgentQualificationStatus,
    AgentTrustEvidenceRef,
    qualification_status_satisfies,
)

__all__ = [
    "AgentArtifactMetadata",
    "AgentArtifactMetadataStore",
    "AgentBindingFactoryReference",
    "AgentBindingPolicyOverrides",
    "AgentCatalogEntry",
    "AgentCatalogVersionChannelRef",
    "AgentDeliverySource",
    "AgentDistributionError",
    "AgentDistributionEvent",
    "AgentDistributionNotFoundError",
    "AgentDistributionStoreState",
    "AgentInstallationRecord",
    "AgentInstallationStore",
    "AgentInstallationTrustRecord",
    "AgentPackageCandidate",
    "AgentPackageIdentity",
    "AgentPackageQualificationResult",
    "AgentPackageTrustCoordinator",
    "AgentPackageTrustDecision",
    "AgentPackageTrustError",
    "AgentPackageTrustOutcome",
    "AgentPackageTrustPolicy",
    "AgentPackageTrustPosture",
    "AgentPackageTrustReasonCode",
    "AgentPackageTrustRevocationState",
    "AgentPublisherIdentity",
    "AgentQualificationEvidence",
    "AgentQualificationEvidenceKind",
    "AgentQualificationStatus",
    "AgentTrustEvidenceRef",
    "ApplicationAgentBinding",
    "ApplicationAgentBindingStore",
    "BindingLifecycleError",
    "BindingRevisionConflict",
    "BindingService",
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
    "InMemoryAgentArtifactMetadataStore",
    "InMemoryAgentInstallationStore",
    "InMemoryApplicationAgentBindingStore",
    "InMemoryRuntimeRevisionStore",
    "InstalledAgentPackageRequirement",
    "InstalledAgentRequirementSet",
    "InstallationLifecycleError",
    "InstallationService",
    "InstallationSlotConflict",
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
    "RuntimeRevisionConflict",
    "RuntimeRevisionLifecycleError",
    "RuntimeRevisionService",
    "RuntimeRevisionState",
    "RuntimeRevisionStore",
    "TransitionResult",
    "assert_installation_trust_record_acceptable",
    "installation_state_is_installed",
    "qualification_status_satisfies",
]
