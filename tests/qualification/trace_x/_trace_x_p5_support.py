# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-P0 policy / profile / configuration provenance baseline (mechanical SSOT)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from tests.qualification.trace_x._trace_x_p5_discovery import (
    discover_configuration_provenance_surfaces,
    discover_policy_provenance_surfaces,
    discover_profile_revision_surfaces,
)
from tests.qualification.trace_x._trace_x_p5_registry_types import (
    ClassifiedDiscoveryCandidate,
    ConfigurationProvenanceSurfaceKind,
    DiscoveryCandidateDisposition,
    PolicyProvenanceSurfaceKind,
    ProfileRevisionSurfaceKind,
    ProvenanceDisposition,
    ProvenanceGap,
    ProvenanceJoin,
    RegisteredConfigurationProvenanceSurface,
    RegisteredPolicyProvenanceSurface,
    RegisteredProfileRevisionSurface,
    SurfaceParityResult,
    compare_discovered_to_registry,
)

TRACE_X_P5_P0_START_HEAD: Final[str] = "0102eeabc6d1d59efbecff52737491c96b1d3f0c"
TRACE_X_P5_P0_R1_START_HEAD: Final[str] = "9d48ca424e025888f7ac8ea61ed463f4284a0d29"
TRACE_X_P5_P0_R1_R1_START_HEAD: Final[str] = "a9fd88da1d5efa3fc61d668289c13f78e4d52b1d"

_POLICY_DISCOVERY_CLASSIFICATIONS: Final[tuple[ClassifiedDiscoveryCandidate, ...]] = (
    ClassifiedDiscoveryCandidate(
        "intergrax/contracts/evaluated_policy_decision.py",
        "EvaluatedPolicyDecision",
        DiscoveryCandidateDisposition.NOT_PROVENANCE,
        "Policy evaluation snapshot — boundary authority remains PolicyDecisionSection",
    ),
    ClassifiedDiscoveryCandidate(
        "intergrax/runtime/evidence/obligation_derivation.py",
        "_CanonicalSerializedRuleV1",
        DiscoveryCandidateDisposition.NOT_PROVENANCE,
        "Internal derivation serializer — not a policy provenance surface",
    ),
)

_PROFILE_DISCOVERY_CLASSIFICATIONS: Final[tuple[ClassifiedDiscoveryCandidate, ...]] = (
    ClassifiedDiscoveryCandidate(
        "intergrax/applications/_shared/profile_resolution/activation_store.py",
        "decode_active_effective_profile_revision_binding",
        DiscoveryCandidateDisposition.NOT_PROVENANCE,
        "Persistence codec helper — not a revision provenance authority surface",
    ),
    ClassifiedDiscoveryCandidate(
        "intergrax/applications/contracts/runtime_inspection/safe_views.py",
        "SafeEffectiveProfileRevisionView",
        DiscoveryCandidateDisposition.NOT_PROVENANCE,
        "Inspection projection — not revision provenance authority",
    ),
    ClassifiedDiscoveryCandidate(
        "intergrax/context/provider_lifecycle.py",
        "ContextProviderExecutionPinningStore",
        DiscoveryCandidateDisposition.NOT_PROVENANCE,
        "Context provider lifecycle pinning — outside effective profile SSOT",
    ),
    ClassifiedDiscoveryCandidate(
        "intergrax/context/provider_lifecycle.py",
        "InMemoryContextProviderExecutionPinningStore",
        DiscoveryCandidateDisposition.NOT_PROVENANCE,
        "Context provider lifecycle pinning — outside effective profile SSOT",
    ),
    ClassifiedDiscoveryCandidate(
        "intergrax/skills/execution_binding.py",
        "SkillExecutionPinningStore",
        DiscoveryCandidateDisposition.NOT_PROVENANCE,
        "Skills execution pinning — outside effective profile revision SSOT",
    ),
    ClassifiedDiscoveryCandidate(
        "intergrax/skills/execution_binding.py",
        "InMemorySkillExecutionPinningStore",
        DiscoveryCandidateDisposition.NOT_PROVENANCE,
        "Skills execution pinning — outside effective profile revision SSOT",
    ),
)

_CONFIGURATION_DISCOVERY_CLASSIFICATIONS: Final[tuple[ClassifiedDiscoveryCandidate, ...]] = (
    ClassifiedDiscoveryCandidate(
        "intergrax/integrations/contracts/existing_capability_configuration.py",
        "ExistingCapabilityConfigurationRealizationStrategy",
        DiscoveryCandidateDisposition.NOT_PROVENANCE,
        "Abstract strategy protocol — concrete strategies are inventoried separately",
    ),
    ClassifiedDiscoveryCandidate(
        "intergrax/integrations/contracts/scoped_integration_adaptation.py",
        "ScopedIntegrationAdaptationTarget",
        DiscoveryCandidateDisposition.NOT_PROVENANCE,
        "Scoped adaptation identity — not configured capability provenance",
    ),
    ClassifiedDiscoveryCandidate(
        "intergrax/integrations/contracts/existing_capability_configuration_opportunity.py",
        "ExistingCapabilityConfigurationOpportunityFacts",
        DiscoveryCandidateDisposition.NOT_PROVENANCE,
        "Configuration opportunity facts — pre-realization discovery, not provenance slice",
    ),
    ClassifiedDiscoveryCandidate(
        "intergrax/integrations/contracts/existing_capability_configuration_opportunity.py",
        "ExistingCapabilityConfigurationOpportunity",
        DiscoveryCandidateDisposition.NOT_PROVENANCE,
        "Configuration opportunity DTO — could-be-realized, not execution provenance",
    ),
)


def _discovery_keys_for_parity(
    discovered: frozenset[tuple[str, str]],
    classifications: tuple[ClassifiedDiscoveryCandidate, ...],
) -> frozenset[tuple[str, str]]:
    excluded = {row.key for row in classifications}
    return frozenset(key for key in discovered if key not in excluded)


POLICY_PROVENANCE_SURFACE_REGISTRY: Final[tuple[RegisteredPolicyProvenanceSurface, ...]] = (
    RegisteredPolicyProvenanceSurface(
        "intergrax/contracts/execution_evidence/boundary_event.py",
        "PolicyDecisionSection",
        PolicyProvenanceSurfaceKind.CONTRACT_DEFINITION,
        "Governance / execution boundary evidence",
        "PolicyDecisionSection",
        "ExecutionBoundaryEvent.task_id+run_id → policy.bundle_id/bundle_version/bundle_digest/decision_ref",
        "test_txp5p0_q02_policy_discovery_registry_parity",
    ),
    RegisteredPolicyProvenanceSurface(
        "intergrax/contracts/execution_evidence/boundary_event.py",
        "GovernanceEvidenceSection",
        PolicyProvenanceSurfaceKind.CONTRACT_DEFINITION,
        "Governance evidence persistence",
        "GovernanceEvidenceSection",
        "ExecutionBoundaryEvent → governance_evidence.evidence_id",
        "test_txp5p0_q02_policy_discovery_registry_parity",
    ),
    RegisteredPolicyProvenanceSurface(
        "intergrax/contracts/governed_proof.py",
        "GovernanceEvidenceRef",
        PolicyProvenanceSurfaceKind.CONTRACT_DEFINITION,
        "Governance evidence plane",
        "GovernanceEvidenceRef",
        "GovernedProofProfile → governance_evidence.evidence_id",
        "test_txp5p0_q02_policy_discovery_registry_parity",
    ),
    RegisteredPolicyProvenanceSurface(
        "intergrax/runtime/evidence/obligation_derivation_contracts.py",
        "PolicyRevisionReferenceV1",
        PolicyProvenanceSurfaceKind.OBLIGATION_DERIVATION,
        "Obligation derivation",
        "PolicyRevisionReferenceV1",
        "PolicyEvidenceBasisV1 → policy_document_id+revision_id",
        "test_txp5p0_q02_policy_discovery_registry_parity",
    ),
    RegisteredPolicyProvenanceSurface(
        "intergrax/runtime/evidence/obligation_derivation_contracts.py",
        "PolicyEvidenceBasisV1",
        PolicyProvenanceSurfaceKind.OBLIGATION_DERIVATION,
        "Obligation derivation",
        "PolicyEvidenceBasisV1",
        "run/plan basis snapshot → policy_revisions[]",
        "test_txp5p0_q02_policy_discovery_registry_parity",
    ),
    RegisteredPolicyProvenanceSurface(
        "intergrax/runtime/evidence/obligation_derivation_contracts.py",
        "RequirementOriginV1",
        PolicyProvenanceSurfaceKind.OBLIGATION_DERIVATION,
        "Obligation derivation",
        "RequirementOriginV1",
        "obligation → policy_document_id+revision_id+rule_id",
        "test_txp5p0_q02_policy_discovery_registry_parity",
    ),
    RegisteredPolicyProvenanceSurface(
        "intergrax/runtime/evidence/obligation_derivation_contracts.py",
        "RequireIndexedEvidencePolicyRuleV1",
        PolicyProvenanceSurfaceKind.OBLIGATION_DERIVATION,
        "Obligation derivation",
        "ResolvedPolicyRuleV1",
        "resolved rules → policy_document_id+revision_id",
        "test_txp5p0_q02_policy_discovery_registry_parity",
    ),
    RegisteredPolicyProvenanceSurface(
        "intergrax/runtime/evidence/obligation_derivation_contracts.py",
        "RequireLiveEvidencePolicyRuleV1",
        PolicyProvenanceSurfaceKind.OBLIGATION_DERIVATION,
        "Obligation derivation",
        "ResolvedPolicyRuleV1",
        "resolved rules → policy_document_id+revision_id",
        "test_txp5p0_q02_policy_discovery_registry_parity",
    ),
    RegisteredPolicyProvenanceSurface(
        "intergrax/runtime/execution_evidence/compose.py",
        "compose_execution_boundary_event",
        PolicyProvenanceSurfaceKind.EVIDENCE_COMPOSITION,
        "Execution evidence composition",
        "ExecutionBoundaryEvent",
        "boundary event persistence → ExecutionId via task/run correlation",
        "test_txp5p0_q02_policy_discovery_registry_parity",
    ),
    RegisteredPolicyProvenanceSurface(
        "intergrax/runtime/execution_evidence/compose.py",
        "compose_execution_boundary_event_v2_from_result",
        PolicyProvenanceSurfaceKind.EVIDENCE_COMPOSITION,
        "Execution evidence composition",
        "ExecutionBoundaryEventV2",
        "boundary event persistence → ExecutionId via task/run correlation",
        "test_txp5p0_q02_policy_discovery_registry_parity",
    ),
    RegisteredPolicyProvenanceSurface(
        "intergrax/runtime/governance/governance_policy_decision_evidence_recording.py",
        "record_governance_policy_decision_evidence",
        PolicyProvenanceSurfaceKind.GOVERNANCE_RECORDING,
        "Governance evidence recorder",
        "GovernanceDecisionEvidenceFact",
        "governance_evidence_id → policy decision material",
        "test_txp5p0_q02_policy_discovery_registry_parity",
    ),
    RegisteredPolicyProvenanceSurface(
        "intergrax/runtime/governance/governance_policy_decision_evidence_recording.py",
        "record_governance_policy_decision_evidence_for_active_identity",
        PolicyProvenanceSurfaceKind.GOVERNANCE_RECORDING,
        "Governance evidence recorder",
        "GovernanceDecisionEvidenceFact",
        "governance_evidence_id → policy decision material",
        "test_txp5p0_q02_policy_discovery_registry_parity",
    ),
    RegisteredPolicyProvenanceSurface(
        "intergrax/runtime/runtime_inspection/adapters/governance_read.py",
        "GovernanceAuditInspectionAdapter.read_governance_decisions",
        PolicyProvenanceSurfaceKind.INSPECTION_PROJECTION,
        "Runtime inspection (read-only)",
        "RuntimeInspectionGovernanceSection",
        "projection only — not provenance authority",
        "test_txp5p0_q02_policy_discovery_registry_parity",
    ),
)

PROFILE_REVISION_SURFACE_REGISTRY: Final[tuple[RegisteredProfileRevisionSurface, ...]] = (
    RegisteredProfileRevisionSurface(
        "intergrax/applications/contracts/profile_resolution/revision.py",
        "EffectiveProfileRevision",
        ProfileRevisionSurfaceKind.CONTRACT_DEFINITION,
        "Profile resolution",
        "EffectiveProfileRevision",
        "revision_id + fingerprint",
        "test_txp5p0_q03_profile_discovery_registry_parity",
    ),
    RegisteredProfileRevisionSurface(
        "intergrax/applications/contracts/profile_resolution/revision.py",
        "EffectiveProfileRevisionScope",
        ProfileRevisionSurfaceKind.CONTRACT_DEFINITION,
        "Profile resolution",
        "EffectiveProfileRevisionScope",
        "tenant_id + application_id scope",
        "test_txp5p0_q03_profile_discovery_registry_parity",
    ),
    RegisteredProfileRevisionSurface(
        "intergrax/applications/contracts/profile_resolution/revision_id.py",
        "EffectiveProfileRevisionId",
        ProfileRevisionSurfaceKind.CONTRACT_DEFINITION,
        "Profile resolution",
        "EffectiveProfileRevisionId",
        "typed revision identity",
        "test_txp5p0_q03_profile_discovery_registry_parity",
    ),
    RegisteredProfileRevisionSurface(
        "intergrax/contracts/effective_profile_revision_provenance_ref.py",
        "EffectiveProfileRevisionProvenanceRef",
        ProfileRevisionSurfaceKind.CONTRACT_DEFINITION,
        "Evidence reconstruction (neutral read reference)",
        "EffectiveProfileRevisionProvenanceRef",
        "Profile Resolution-owned revision reference — not identity authority",
        "test_txp5p0_q03_profile_discovery_registry_parity",
    ),
    RegisteredProfileRevisionSurface(
        "intergrax/contracts/execution_effective_profile_provenance.py",
        "ExecutionEffectiveProfileProvenance",
        ProfileRevisionSurfaceKind.EXECUTION_PINNING,
        "Evidence reconstruction",
        "ExecutionEffectiveProfileProvenance",
        "tenant_id + ExecutionId → revision_ref + fingerprint (read projection)",
        "test_txp5p0_q03_profile_discovery_registry_parity",
    ),
    RegisteredProfileRevisionSurface(
        "intergrax/contracts/execution_effective_profile_provenance.py",
        "ExecutionEffectiveProfileProvenanceReader",
        ProfileRevisionSurfaceKind.EXECUTION_PINNING,
        "Evidence reconstruction port",
        "ExecutionEffectiveProfileProvenanceReader",
        "read-only pinning projection port",
        "test_txp5p0_q03_profile_discovery_registry_parity",
    ),
    RegisteredProfileRevisionSurface(
        "intergrax/applications/_shared/profile_resolution/execution_effective_profile_provenance_reader.py",
        "PinningStoreExecutionEffectiveProfileProvenanceReader",
        ProfileRevisionSurfaceKind.EXECUTION_PINNING,
        "Profile resolution adapter",
        "ExecutionEffectiveProfileProvenanceReader",
        "EffectiveProfileExecutionPinningStore.get → neutral provenance",
        "test_txp5p0_q03_profile_discovery_registry_parity",
    ),
    RegisteredProfileRevisionSurface(
        "intergrax/applications/contracts/profile_resolution/execution_binding.py",
        "EffectiveProfileExecutionBinding",
        ProfileRevisionSurfaceKind.EXECUTION_PINNING,
        "Profile resolution / execution pinning",
        "EffectiveProfileExecutionBinding",
        "ExecutionId + tenant_id → revision_id + fingerprint",
        "test_txp5p0_q03_profile_discovery_registry_parity",
    ),
    RegisteredProfileRevisionSurface(
        "intergrax/applications/contracts/profile_resolution/execution_binding.py",
        "EffectiveProfileRevisionCheckpointEvidence",
        ProfileRevisionSurfaceKind.EXECUTION_PINNING,
        "Profile resolution",
        "EffectiveProfileRevisionCheckpointEvidence",
        "TaskCheckpoint metadata — not revision authority",
        "test_txp5p0_q03_profile_discovery_registry_parity",
    ),
    RegisteredProfileRevisionSurface(
        "intergrax/applications/contracts/profile_resolution/activation.py",
        "ActiveEffectiveProfileRevisionBinding",
        ProfileRevisionSurfaceKind.ACTIVATION,
        "Profile resolution / activation",
        "ActiveEffectiveProfileRevisionBinding",
        "scope → active revision_id",
        "test_txp5p0_q03_profile_discovery_registry_parity",
    ),
    RegisteredProfileRevisionSurface(
        "intergrax/applications/contracts/profile_resolution/store.py",
        "EffectiveProfileRevisionStore",
        ProfileRevisionSurfaceKind.PERSISTENCE,
        "Profile resolution persistence",
        "EffectiveProfileRevisionStore",
        "revision_id → EffectiveProfileRevision",
        "test_txp5p0_q03_profile_discovery_registry_parity",
    ),
    RegisteredProfileRevisionSurface(
        "intergrax/applications/contracts/profile_resolution/execution_binding.py",
        "EffectiveProfileExecutionPinningStore",
        ProfileRevisionSurfaceKind.PERSISTENCE,
        "Profile resolution persistence",
        "EffectiveProfileExecutionPinningStore",
        "ExecutionId → binding",
        "test_txp5p0_q03_profile_discovery_registry_parity",
    ),
    RegisteredProfileRevisionSurface(
        "intergrax/applications/_shared/profile_resolution/store.py",
        "InMemoryEffectiveProfileRevisionStore",
        ProfileRevisionSurfaceKind.PERSISTENCE,
        "Profile resolution persistence",
        "EffectiveProfileRevisionStore",
        "in-memory revision materialization store",
        "test_txp5p0_q03_profile_discovery_registry_parity",
    ),
    RegisteredProfileRevisionSurface(
        "intergrax/applications/_shared/profile_resolution/execution_pinning.py",
        "InMemoryEffectiveProfileExecutionPinningStore",
        ProfileRevisionSurfaceKind.PERSISTENCE,
        "Profile resolution persistence",
        "EffectiveProfileExecutionPinningStore",
        "in-memory execution pinning store",
        "test_txp5p0_q03_profile_discovery_registry_parity",
    ),
    RegisteredProfileRevisionSurface(
        "intergrax/applications/_shared/profile_resolution/persistence.py",
        "DocumentStoreEffectiveProfileRevisionStore",
        ProfileRevisionSurfaceKind.PERSISTENCE,
        "Profile resolution persistence",
        "EffectiveProfileRevisionStore",
        "durable document-store revision materialization",
        "test_txp5p0_q03_profile_discovery_registry_parity",
    ),
    RegisteredProfileRevisionSurface(
        "intergrax/applications/_shared/profile_resolution/persistence.py",
        "KvEffectiveProfileRevisionStore",
        ProfileRevisionSurfaceKind.PERSISTENCE,
        "Profile resolution persistence",
        "EffectiveProfileRevisionStore",
        "durable KV revision materialization",
        "test_txp5p0_q03_profile_discovery_registry_parity",
    ),
    RegisteredProfileRevisionSurface(
        "intergrax/applications/_shared/profile_resolution/persistence.py",
        "DocumentStoreEffectiveProfileExecutionPinningStore",
        ProfileRevisionSurfaceKind.PERSISTENCE,
        "Profile resolution persistence",
        "EffectiveProfileExecutionPinningStore",
        "durable document-store execution pinning",
        "test_txp5p0_q03_profile_discovery_registry_parity",
    ),
    RegisteredProfileRevisionSurface(
        "intergrax/applications/_shared/profile_resolution/persistence.py",
        "KvEffectiveProfileExecutionPinningStore",
        ProfileRevisionSurfaceKind.PERSISTENCE,
        "Profile resolution persistence",
        "EffectiveProfileExecutionPinningStore",
        "durable KV execution pinning",
        "test_txp5p0_q03_profile_discovery_registry_parity",
    ),
    RegisteredProfileRevisionSurface(
        "intergrax/applications/_shared/profile_resolution/materialize.py",
        "materialize_effective_profile_revision",
        ProfileRevisionSurfaceKind.MATERIALIZATION,
        "Profile resolution",
        "EffectiveProfileRevision",
        "configured inputs → materialized revision",
        "test_txp5p0_q03_profile_discovery_registry_parity",
    ),
    RegisteredProfileRevisionSurface(
        "intergrax/applications/_shared/profile_resolution/activation_service.py",
        "resolve_active_effective_profile_revision",
        ProfileRevisionSurfaceKind.ACTIVATION,
        "Profile resolution / activation",
        "ActiveEffectiveProfileRevisionBinding",
        "scope → active revision at admission time",
        "test_txp5p0_q03_profile_discovery_registry_parity",
    ),
    RegisteredProfileRevisionSurface(
        "intergrax/applications/_shared/profile_resolution/execution_pinning.py",
        "pin_effective_profile_revision_for_execution",
        ProfileRevisionSurfaceKind.EXECUTION_PINNING,
        "Profile resolution / execution pinning",
        "EffectiveProfileExecutionBinding",
        "ExecutionId pin at admission",
        "test_txp5p0_q03_profile_discovery_registry_parity",
    ),
    RegisteredProfileRevisionSurface(
        "intergrax/applications/_shared/profile_resolution/execution_admission.py",
        "EffectiveProfileRevisionAdmission",
        ProfileRevisionSurfaceKind.EXECUTION_ADMISSION,
        "Profile resolution / host admission",
        "EffectiveProfileRevisionAdmission",
        "host task admission gate",
        "test_txp5p0_q03_profile_discovery_registry_parity",
    ),
    RegisteredProfileRevisionSurface(
        "intergrax/applications/_shared/profile_resolution/execution_admission.py",
        "EffectiveProfileRevisionAdmission.admit_root_execution",
        ProfileRevisionSurfaceKind.EXECUTION_ADMISSION,
        "Profile resolution / host admission",
        "EffectiveProfileRevisionAdmissionPort",
        "ExecutionId → pinned revision",
        "test_txp5p0_q03_profile_discovery_registry_parity",
    ),
    RegisteredProfileRevisionSurface(
        "intergrax/runtime/execution/effective_profile_revision_admission.py",
        "EffectiveProfileRevisionAdmissionPort",
        ProfileRevisionSurfaceKind.EXECUTION_ADMISSION,
        "Execution host boundary",
        "EffectiveProfileRevisionAdmissionPort",
        "optional host wiring — not global default",
        "test_txp5p0_q03_profile_discovery_registry_parity",
    ),
    RegisteredProfileRevisionSurface(
        "intergrax/contracts/execution_environment_isolation.py",
        "EffectiveProfileRevisionIsolationView",
        ProfileRevisionSurfaceKind.INSPECTION_PROJECTION,
        "Execution environment isolation",
        "EffectiveProfileRevisionIsolationView",
        "read-only isolation view",
        "test_txp5p0_q03_profile_discovery_registry_parity",
    ),
)

CONFIGURATION_PROVENANCE_SURFACE_REGISTRY: Final[
    tuple[RegisteredConfigurationProvenanceSurface, ...]
] = (
    RegisteredConfigurationProvenanceSurface(
        "intergrax/integrations/contracts/existing_capability_configuration.py",
        "ConfiguredCapabilityBinding",
        ConfigurationProvenanceSurfaceKind.CONTRACT_DEFINITION,
        "Integrations / configuration realization",
        "ConfiguredCapabilityBinding",
        "configuration_fingerprint + configuration_version",
        "test_txp5p0_q04_config_discovery_registry_parity",
    ),
    RegisteredConfigurationProvenanceSurface(
        "intergrax/integrations/contracts/existing_capability_configuration.py",
        "ExistingCapabilityConfigurationRealizationRequest",
        ConfigurationProvenanceSurfaceKind.CONTRACT_DEFINITION,
        "Integrations / configuration realization",
        "ExistingCapabilityConfigurationRealizationRequest",
        "request fingerprint + tenant + optional task_id/run_id",
        "test_txp5p0_q04_config_discovery_registry_parity",
    ),
    RegisteredConfigurationProvenanceSurface(
        "intergrax/integrations/contracts/existing_capability_configuration.py",
        "ExistingCapabilityConfigurationRealizationResult",
        ConfigurationProvenanceSurfaceKind.CONTRACT_DEFINITION,
        "Integrations / configuration realization",
        "ExistingCapabilityConfigurationRealizationResult",
        "configured_binding + authorization_evidence",
        "test_txp5p0_q04_config_discovery_registry_parity",
    ),
    RegisteredConfigurationProvenanceSurface(
        "intergrax/integrations/contracts/existing_capability_configuration.py",
        "ExistingCapabilityIntegrationTarget",
        ConfigurationProvenanceSurfaceKind.CONTRACT_DEFINITION,
        "Integrations / configuration realization",
        "ExistingCapabilityIntegrationTarget",
        "current_revision continuity",
        "test_txp5p0_q04_config_discovery_registry_parity",
    ),
    RegisteredConfigurationProvenanceSurface(
        "intergrax/integrations/contracts/existing_capability_configuration.py",
        "validate_realization_request_invariants",
        ConfigurationProvenanceSurfaceKind.INVARIANT_VALIDATION,
        "Integrations / configuration realization",
        "validate_realization_request_invariants",
        "configured fingerprint must match payload — fail closed",
        "test_txp5p0_q04_config_discovery_registry_parity",
    ),
    RegisteredConfigurationProvenanceSurface(
        "intergrax/integrations/existing_capability_configuration_service.py",
        "ExistingCapabilityConfigurationRealizationService",
        ConfigurationProvenanceSurfaceKind.REALIZATION_CORE,
        "Integrations / configuration realization",
        "ExistingCapabilityConfigurationRealizationService",
        "realize_admitted → configured_binding",
        "test_txp5p0_q04_config_discovery_registry_parity",
    ),
    RegisteredConfigurationProvenanceSurface(
        "intergrax/integrations/existing_capability_configuration_service.py",
        "ExistingCapabilityConfigurationRealizationService.realize_admitted",
        ConfigurationProvenanceSurfaceKind.REALIZATION_CORE,
        "Integrations / configuration realization",
        "ExistingCapabilityConfigurationRealizationService",
        "pure realization core — no execution evidence minting",
        "test_txp5p0_q04_config_discovery_registry_parity",
    ),
    RegisteredConfigurationProvenanceSurface(
        "intergrax/integrations/providers/relational_store/sqlite/configuration_realization.py",
        "SQLiteRelationalStoreConfigurationRealizationStrategy",
        ConfigurationProvenanceSurfaceKind.STRATEGY_OUTPUT,
        "Integrations strategy",
        "ConfiguredCapabilityBinding",
        "strategy-emitted configured binding",
        "test_txp5p0_q04_config_discovery_registry_parity",
    ),
    RegisteredConfigurationProvenanceSurface(
        "intergrax/contracts/execution_integration_configuration_provenance.py",
        "ConfiguredIntegrationProvenanceSlice",
        ConfigurationProvenanceSurfaceKind.CONTRACT_DEFINITION,
        "Neutral execution integration configuration provenance (TRACE-X-P5-R2-P1)",
        "ConfiguredIntegrationProvenanceSlice",
        "configured fingerprint projection — not effective provider proof",
        "test_txp5p0_q04_config_discovery_registry_parity",
    ),
)


def compare_policy_surfaces_to_registry(
    discovered: frozenset[tuple[str, str]],
) -> SurfaceParityResult:
    parity_keys = _discovery_keys_for_parity(discovered, _POLICY_DISCOVERY_CLASSIFICATIONS)
    return compare_discovered_to_registry(parity_keys, POLICY_PROVENANCE_SURFACE_REGISTRY)


def compare_profile_surfaces_to_registry(
    discovered: frozenset[tuple[str, str]],
) -> SurfaceParityResult:
    parity_keys = _discovery_keys_for_parity(discovered, _PROFILE_DISCOVERY_CLASSIFICATIONS)
    return compare_discovered_to_registry(parity_keys, PROFILE_REVISION_SURFACE_REGISTRY)


def compare_configuration_surfaces_to_registry(
    discovered: frozenset[tuple[str, str]],
) -> SurfaceParityResult:
    parity_keys = _discovery_keys_for_parity(discovered, _CONFIGURATION_DISCOVERY_CLASSIFICATIONS)
    return compare_discovered_to_registry(parity_keys, CONFIGURATION_PROVENANCE_SURFACE_REGISTRY)


PROVENANCE_JOINS: Final[tuple[ProvenanceJoin, ...]] = (
    ProvenanceJoin(
        "policy",
        "P5-J-POL-01",
        "ExecutionBoundaryEvent.task_id + ExecutionBoundaryEvent.run_id",
        "PolicyDecisionSection.bundle_id + bundle_version + bundle_digest (+ decision_ref)",
        "ExecutionBoundaryEvent / PolicyDecisionSection",
        "Execution evidence composition",
    ),
    ProvenanceJoin(
        "policy",
        "P5-J-POL-02",
        "GovernanceEvidenceRef.evidence_id",
        "GovernanceDecisionEvidenceFact (persisted)",
        "GovernanceEvidenceRef",
        "Governance evidence persistence",
    ),
    ProvenanceJoin(
        "profile_revision",
        "P5-J-PRF-01",
        "EffectiveProfileExecutionBinding.execution_id + tenant_id",
        "EffectiveProfileExecutionBinding.revision_id + fingerprint",
        "EffectiveProfileExecutionBinding",
        "Profile resolution / execution pinning",
    ),
    ProvenanceJoin(
        "profile_revision",
        "P5-J-PRF-02",
        "EffectiveProfileRevisionId",
        "EffectiveProfileRevision (materialized)",
        "EffectiveProfileRevision",
        "Profile resolution revision store",
    ),
    ProvenanceJoin(
        "configuration",
        "P5-J-CFG-01",
        "ExistingCapabilityConfigurationRealizationRequest.request_id",
        "ConfiguredCapabilityBinding.configuration_fingerprint",
        "ConfiguredCapabilityBinding",
        "Integrations realization (scoped)",
    ),
    ProvenanceJoin(
        "configuration",
        "P5-J-CFG-02",
        "optional RunId + TaskId on realization request",
        "configured_binding — correlation only within INT-CONFIG scope",
        "ExistingCapabilityConfigurationRealizationRequest",
        "Integrations realization",
    ),
)

PROVENANCE_GAPS: Final[tuple[ProvenanceGap, ...]] = (
    ProvenanceGap(
        "P5-GAP-01",
        "policy",
        ("FRZ-TRC-07",),
        "IN-SCOPE BLOCKER",
        "ExecutionReconstructor does not join policy revision fields; global execution→policy revision "
        "reconstruction is not certified on all paths.",
        "TRACE-X-P5-R1",
    ),
    ProvenanceGap(
        "P5-GAP-02",
        "profile_revision",
        ("FRZ-TRC-08",),
        "IN-SCOPE BLOCKER",
        "EffectiveProfileRevisionAdmissionPort is optional on HostTask; executions without admission "
        "wiring lack pinned revision evidence.",
        "TRACE-X-P5-R1",
    ),
    ProvenanceGap(
        "P5-GAP-03",
        "profile_revision",
        ("FRZ-TRC-08",),
        "TRACKED FREEZE DEBT",
        "TX-B01 superseded at contract level by profile-resolution SSOT; global TRACE-X closure still "
        "requires execution-wide pinning + reconstruction (see P5-GAP-02).",
        None,
    ),
    ProvenanceGap(
        "P5-GAP-04",
        "configuration",
        ("FRZ-TRC-11",),
        "IN-SCOPE BLOCKER",
        "No canonical global chain from ConfiguredCapabilityBinding to ExecutionId / RuntimeEvent "
        "outside INT-CONFIG realization scope.",
        "TRACE-X-P5-R2",
    ),
    ProvenanceGap(
        "P5-GAP-05",
        "policy",
        ("FRZ-TRC-07",),
        "TRACKED FREEZE DEBT",
        "TXP1R1-Q02 / TXP1R1-Q23 documentation vs HEAD expectations remain qualification debt "
        "(non-blocking for P5-P0 inventory).",
        None,
    ),
)

FRZ_TRC_P5_DISPOSITION: Final[dict[str, ProvenanceDisposition]] = {
    "FRZ-TRC-07": ProvenanceDisposition.PARTIAL_CURRENT_HEAD,
    "FRZ-TRC-08": ProvenanceDisposition.PARTIAL_CURRENT_HEAD,
    "FRZ-TRC-11": ProvenanceDisposition.PARTIAL_CURRENT_HEAD,
}


@dataclass(frozen=True, slots=True)
class P5GateEvidence:
    gate_id: str
    description: str
    nodeid: str


P5_P0_GATE_REGISTRY: Final[tuple[P5GateEvidence, ...]] = (
    P5GateEvidence("TXP5P0-Q01", "START_HEAD ancestry", "test_txp5p0_q01_start_head_ancestry"),
    P5GateEvidence(
        "TXP5P0-Q02",
        "Policy provenance discovery/registry parity",
        "test_txp5p0_q02_policy_discovery_registry_parity",
    ),
    P5GateEvidence(
        "TXP5P0-Q03",
        "Profile revision discovery/registry parity",
        "test_txp5p0_q03_profile_discovery_registry_parity",
    ),
    P5GateEvidence(
        "TXP5P0-Q04",
        "Configuration provenance discovery/registry parity",
        "test_txp5p0_q04_config_discovery_registry_parity",
    ),
    P5GateEvidence(
        "TXP5P0-Q05",
        "Exactly-one semantic owner per join axis",
        "test_txp5p0_q05_provenance_joins_non_heuristic",
    ),
    P5GateEvidence(
        "TXP5P0-Q06",
        "Synthetic unregistered policy surface fails parity",
        "test_txp5p0_q06_synthetic_policy_surface_negative",
    ),
    P5GateEvidence(
        "TXP5P0-Q07",
        "Synthetic unregistered profile surface fails parity",
        "test_txp5p0_q07_synthetic_profile_surface_negative",
    ),
    P5GateEvidence(
        "TXP5P0-Q08",
        "Synthetic unregistered configuration surface fails parity",
        "test_txp5p0_q08_synthetic_configuration_surface_negative",
    ),
    P5GateEvidence(
        "TXP5P0-Q09",
        "Configured fingerprint mismatch fail-closed",
        "test_txp5p0_q09_configured_fingerprint_mismatch_fail_closed",
    ),
    P5GateEvidence(
        "TXP5P0-Q10",
        "Cross-tenant configuration realization rejected",
        "test_txp5p0_q10_configuration_tenant_mismatch_fail_closed",
    ),
    P5GateEvidence(
        "TXP5P0-Q11",
        "Profile execution binding tenant mismatch rejected",
        "test_txp5p0_q11_profile_binding_tenant_mismatch_fail_closed",
    ),
    P5GateEvidence(
        "TXP5P0-Q12",
        "FRZ P5 dispositions remain OPEN (no PASS promotion)",
        "test_txp5p0_q12_frz_dispositions_not_pass",
    ),
    P5GateEvidence(
        "TXP5P0-R1-Q01",
        "R1 START_HEAD ancestry",
        "test_txp5p0_r1_q01_start_head_ancestry",
    ),
    P5GateEvidence(
        "TXP5P0-R1-Q02",
        "Policy sentinel discovered (structural AST)",
        "test_txp5p0_r1_q02_policy_sentinel_discovered_without_registry_union",
    ),
    P5GateEvidence(
        "TXP5P0-R1-Q03",
        "Profile sentinel discovered (structural AST)",
        "test_txp5p0_r1_q03_profile_sentinel_discovered_without_registry_union",
    ),
    P5GateEvidence(
        "TXP5P0-R1-Q04",
        "Configuration sentinel discovered (structural AST)",
        "test_txp5p0_r1_q04_config_sentinel_discovered_without_registry_union",
    ),
    P5GateEvidence(
        "TXP5P0-R1-Q05",
        "Renamed policy sentinel still discovered",
        "test_txp5p0_r1_q05_renamed_policy_sentinel_still_discovered",
    ),
    P5GateEvidence(
        "TXP5P0-R1-Q06",
        "Registry row removal does not alter discovery",
        "test_txp5p0_r1_q06_registry_removal_does_not_change_discovery",
    ),
    P5GateEvidence(
        "TXP5P0-R1-Q07",
        "Registry static — not discovery-derived",
        "test_txp5p0_r1_q07_discovery_registry_is_static_not_discovery_derived",
    ),
    P5GateEvidence(
        "TXP5P0-R1-Q08",
        "Explicit typed discovery classifications",
        "test_txp5p0_r1_q08_classifications_are_explicit_typed",
    ),
    P5GateEvidence(
        "TXP5P0-R1-R1-Q01",
        "R1-R1 START_HEAD ancestry",
        "test_txp5p0_r1_r1_q01_start_head_ancestry",
    ),
    P5GateEvidence(
        "TXP5P0-R1-R1-Q02",
        "P5 sentinel fixtures outside distributable intergrax package",
        "test_txp5p0_r1_r1_q02_qualification_sentinel_package_isolation",
    ),
)
