# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tier-0 Agent Distribution domain contracts, stores, and services (AP-3/AP-4)."""

from intergrax.agent_distribution.agent_project_metadata import (
    AgentProjectMetadata,
    AgentProjectMetadataProvider,
)
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
from intergrax.agent_distribution.dependency_specification import (
    build_candidate_dependency_specification,
)
from intergrax.agent_distribution.effective_roster import (
    EffectiveRosterBuilder,
    InstalledAgentRequirementSetBuilder,
)
from intergrax.agent_distribution.errors import (
    AgentDistributionError,
    AgentDistributionNotFoundError,
    AgentPackageTrustError,
    BindingLifecycleError,
    BindingRevisionConflict,
    DependencySpecificationError,
    DependencyResolutionError,
    EffectiveRosterConflict,
    InstallationLifecycleError,
    InstallationSlotConflict,
    MaterializedRuntimeLockConflict,
    MaterializedRuntimeLockError,
    MaterializationError,
    MaterializationInputConflict,
    MaterializationUnsupportedTopology,
    CandidateRuntimeGraphError,
    RuntimeRevisionConflict,
    RuntimeRevisionLifecycleError,
)
from intergrax.agent_distribution.events import AgentDistributionEvent, TransitionResult
from intergrax.agent_distribution.in_memory_stores import (
    AgentDistributionStoreState,
    InMemoryAgentArtifactMetadataStore,
    InMemoryAgentInstallationStore,
    InMemoryApplicationAgentBindingStore,
    InMemoryMaterializedRuntimeLockStore,
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
from intergrax.agent_distribution.materialization_adapters import (
    OciImageMaterializationAdapter,
    RuntimeMaterializationAdapter,
    FakeRuntimeMaterializationAdapter,
    UnsupportedVenvBundleMaterializationAdapter,
    default_materialization_adapters,
)
from intergrax.agent_distribution.materialization_service import RuntimeMaterializationService
from intergrax.agent_distribution.roster import (
    EffectiveRoster,
    EffectiveRosterEntry,
    ManifestDefaultAgentDeclaration,
)
from intergrax.agent_distribution.runtime_graph import (
    CandidateApplicationRuntimeGraph,
    RuntimeGraphAgentRef,
    RuntimeGraphThirdPartyRef,
    RuntimeGraphTierViolation,
)
from intergrax.agent_distribution.identity import (
    AgentPackageCandidate,
    AgentPackageIdentity,
)
from intergrax.agent_distribution.runtime_revision import (
    MaterializationTopology,
    RuntimeRevision,
    RuntimeRevisionState,
)
from intergrax.agent_distribution.resolver import (
    CallableDependencyResolver,
    DependencyResolver,
    ResolvedDependencyClosure,
)
from intergrax.agent_distribution.runtime_graph_service import (
    CandidateRuntimeGraphBuilder,
    CandidateRuntimeGraphValidator,
)
from intergrax.agent_distribution.runtime_lock import (
    MaterializedRuntimeLockProducer,
    MaterializedRuntimeLockService,
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
    "AgentProjectMetadata",
    "AgentProjectMetadataProvider",
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
    "CallableDependencyResolver",
    "CandidateApplicationRuntimeGraph",
    "CandidateRuntimeGraphBuilder",
    "CandidateRuntimeGraphError",
    "CandidateRuntimeGraphValidator",
    "CandidateDependencySpecification",
    "CatalogEntryFilters",
    "CatalogPackageResolution",
    "CatalogProviderKind",
    "CatalogSourceIdentity",
    "CatalogSourceProvider",
    "DependencySpecificationError",
    "DependencyResolutionError",
    "DependencyResolver",
    "DependencyResolverInput",
    "EffectiveRosterBuilder",
    "EffectiveRosterConflict",
    "EffectiveRoster",
    "EffectiveRosterEntry",
    "InMemoryAgentArtifactMetadataStore",
    "InMemoryAgentInstallationStore",
    "InMemoryApplicationAgentBindingStore",
    "InMemoryMaterializedRuntimeLockStore",
    "InMemoryRuntimeRevisionStore",
    "InstalledAgentPackageRequirement",
    "InstalledAgentRequirementSet",
    "InstalledAgentRequirementSetBuilder",
    "InstallationLifecycleError",
    "InstallationService",
    "InstallationSlotConflict",
    "InstallationState",
    "ManifestDefaultAgentDeclaration",
    "ApplicationBuildContext",
    "MaterializationError",
    "MaterializationInputConflict",
    "MaterializationUnsupportedTopology",
    "MaterializationInput",
    "MaterializationOutput",
    "OciImageMaterializationAdapter",
    "RuntimeMaterializationAdapter",
    "RuntimeMaterializationService",
    "FakeRuntimeMaterializationAdapter",
    "UnsupportedVenvBundleMaterializationAdapter",
    "default_materialization_adapters",
    "MaterializationTopology",
    "MaterializedAgentClosureEntry",
    "MaterializedLockPackage",
    "MaterializedLockReproducibilityEvidence",
    "MaterializedLockRollbackEvidence",
    "MaterializedRuntimeLock",
    "MaterializedRuntimeLockConflict",
    "MaterializedRuntimeLockError",
    "MaterializedRuntimeLockProducer",
    "MaterializedRuntimeLockService",
    "MaterializedRuntimeLockStore",
    "PolicyDependencyConstraint",
    "ProviderHealth",
    "ProviderHealthStatus",
    "RepositoryDependencyDeclaration",
    "ResolvedDependencyClosure",
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
    "build_candidate_dependency_specification",
    "installation_state_is_installed",
    "qualification_status_satisfies",
]
