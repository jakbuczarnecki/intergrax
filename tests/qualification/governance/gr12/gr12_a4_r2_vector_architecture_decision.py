# © Artur Czarnecki. All rights reserved.

"""GR-12-A4-R2 vector administration architecture decision SSOT (qualification)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from tests.qualification.governance.gr12.catalog import (
    GR12_A4_R2_VECTOR_ADR_PATH,
    GR12_CANONICAL_BOUNDARY_CLASS,
    GR12_CANONICAL_POLICY_PORT,
    Gr12Applicability,
)
from tests.qualification.governance.gr12.gr12_a4_r2_r0_vector_revision_semantics import (
    GR12_VECTOR_CONFIGURATION_DIGEST_EXCLUDED_RUNTIME_FIELDS,
    GR12_VECTOR_CONFIGURATION_DIGEST_INCLUDED_FIELDS,
    GR12_VECTOR_CONFIGURATION_PROJECTION_SCHEMA,
    GR12_VECTOR_REVISION_DIGEST_INVARIANT,
    Gr12VectorAbsentIndexRevisionState,
    Gr12VectorAlreadyCompatibleRevisionSemantics,
    Gr12VectorIdentityIntrinsicValidation,
    Gr12VectorLiveOperatorIdentityValidation,
    Gr12VectorLivePrepareAuthorizationTiming,
    Gr12VectorProviderCasAvailability,
    Gr12VectorStaleAuthorizationBehavior,
    Gr12VectorStaleRetryFreshAuthorizationPolicy,
)


class Gr12VectorPrepareIndexGovernanceDecision(StrEnum):
    """§14 — prepare_index is not always a live control-plane mutation."""

    OPTION_B_CONDITIONAL_LIVE_OPERATOR_ONLY = "OPTION_B_CONDITIONAL_LIVE_OPERATOR_ONLY"


class Gr12VectorArchitecturePhase(StrEnum):
    ARCHITECTURE_DECISION_CLOSED = "ARCHITECTURE_DECISION_CLOSED"
    ARCHITECTURE_DECISION_RECONCILED = "ARCHITECTURE_DECISION_RECONCILED"


class Gr12VectorIndexOperationClass(StrEnum):
    READ_ONLY = "READ_ONLY"
    CONDITIONAL_MUTATION = "CONDITIONAL_MUTATION"
    MUTATION = "MUTATION"
    LIFECYCLE_ONLY = "LIFECYCLE_ONLY"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


@dataclass(frozen=True, slots=True)
class Gr12VectorIndexOperationClassification:
    operation: str
    operation_class: Gr12VectorIndexOperationClass
    consequential_for_control_plane: bool
    gr12_live_operator_applicable: bool
    reason: str


MUTATION_TYPE_VECTOR_INDEX_PREPARE: Final[str] = "vector_index.prepare"
VECTOR_INDEX_RESOURCE_TYPE: Final[str] = "vector_index"
VECTOR_INDEX_RESOURCE_SCOPE_TEMPLATE: Final[str] = "vector_index.tenant/{tenant_id}"
VECTOR_INDEX_RESOURCE_ID_TEMPLATE: Final[str] = "{tenant_id}/{logical_name}"

GR12_VECTOR_CANONICAL_PORT: Final[str] = (
    "intergrax.integrations.contracts.vector_index_administration.VectorIndexAdministration"
)

GR12_VECTOR_COMPOSITION_ROOT: Final[str] = (
    "intergrax.integrations.providers.vector_store.qdrant.opens."
    "open_qdrant_vector_index_administration"
)

GR12_VECTOR_PREFERRED_GOVERNANCE_MODEL: Final[str] = (
    "operator RequestIdentity → VectorIndexAdminService (applications) → "
    f"{GR12_CANONICAL_BOUNDARY_CLASS} → ControlPlaneMutationPolicyEvaluator → "
    "VectorIndexAdministration (integrations port) → provider adapter"
)

GR12_VECTOR_REVISION_STRATEGY: Final[str] = (
    f"{GR12_VECTOR_CONFIGURATION_PROJECTION_SCHEMA}: one logical schema for current "
    "(VectorIndexDescription) and target (VectorIndexSpec); "
    f"{GR12_VECTOR_REVISION_DIGEST_INVARIANT}; absent current token "
    f"{Gr12VectorAbsentIndexRevisionState.ABSENT.value}"
)

GR12_VECTOR_TOCTOU_STRATEGY: Final[str] = (
    "Optimistic stale-state detection: re-read describe_index before prepare_index; "
    "stale state invalidates prior authorization; "
    f"{Gr12VectorStaleAuthorizationBehavior.STALE_INVALIDATES_PRIOR_AUTHORIZATION_ABORT.value} "
    "or bounded fresh CLA-04; provider CAS unavailable (no fake CAS)"
)

GR12_VECTOR_NEXT_BOUNDED_TASK: Final[str] = (
    "GR-12-A4-R2-R1 — Governed Vector Index Operator Service & CLA-04 Enforcement"
)

GR12_VECTOR_OPERATION_CLASSIFICATIONS: tuple[Gr12VectorIndexOperationClassification, ...] = (
    Gr12VectorIndexOperationClassification(
        operation="probe",
        operation_class=Gr12VectorIndexOperationClass.READ_ONLY,
        consequential_for_control_plane=False,
        gr12_live_operator_applicable=False,
        reason="Provider reachability check; no authoritative index configuration change.",
    ),
    Gr12VectorIndexOperationClassification(
        operation="describe_index",
        operation_class=Gr12VectorIndexOperationClass.READ_ONLY,
        consequential_for_control_plane=False,
        gr12_live_operator_applicable=False,
        reason="Projects persisted index shape; does not mutate provider state.",
    ),
    Gr12VectorIndexOperationClassification(
        operation="prepare_index",
        operation_class=Gr12VectorIndexOperationClass.CONDITIONAL_MUTATION,
        consequential_for_control_plane=True,
        gr12_live_operator_applicable=True,
        reason=(
            "Live operator prepare_index is governed before invocation (may mutate). "
            "CREATED mutates provider index; ALREADY_COMPATIBLE is idempotent no-op "
            "but remains an authorized operator action."
        ),
    ),
    Gr12VectorIndexOperationClassification(
        operation="close",
        operation_class=Gr12VectorIndexOperationClass.LIFECYCLE_ONLY,
        consequential_for_control_plane=False,
        gr12_live_operator_applicable=False,
        reason="Closes local adapter/client; not a control-plane mutation.",
    ),
)

GR12_VECTOR_REJECTED_ALTERNATIVES: Final[tuple[str, ...]] = (
    "GovernedVectorIndexAdministration decorator as public authority layer",
    "Provider-layer CLA-04 inside QdrantVectorIndexAdministration",
    "Second vector admin port or provider-specific governance contract",
    "Provider physical collection name as CLA-04 resource_id authority",
    "Treating bootstrap prepare_index as live operator control-plane mutation",
)

@dataclass(frozen=True, slots=True)
class Gr12A4R2VectorArchitectureDecision:
    architecture_phase: Gr12VectorArchitecturePhase
    cla04_applicability: Gr12Applicability
    prepare_index_decision: Gr12VectorPrepareIndexGovernanceDecision
    live_operator_surface_exists: bool
    destructive_ops_on_neutral_port: bool
    adr_path: str
    canonical_port: str
    composition_root: str
    preferred_governance_model: str
    mutation_type_prepare: str
    resource_type: str
    resource_id_template: str
    resource_scope_template: str
    revision_strategy: str
    toctou_strategy: str
    identity_intrinsic_validation: Gr12VectorIdentityIntrinsicValidation
    live_operator_identity_validation: Gr12VectorLiveOperatorIdentityValidation
    configuration_projection_schema: str
    configuration_digest_included_fields: tuple[str, ...]
    excluded_revision_fields: tuple[str, ...]
    absent_state_semantics: Gr12VectorAbsentIndexRevisionState
    already_compatible_revision_semantics: Gr12VectorAlreadyCompatibleRevisionSemantics
    live_prepare_authorization_timing: Gr12VectorLivePrepareAuthorizationTiming
    stale_authorization_behavior: Gr12VectorStaleAuthorizationBehavior
    stale_retry_fresh_authorization_policy: Gr12VectorStaleRetryFreshAuthorizationPolicy
    provider_cas_available: Gr12VectorProviderCasAvailability
    policy_evaluator_port: str
    next_bounded_task: str
    rejected_alternatives: tuple[str, ...]


GR12_A4_R2_VECTOR_ARCHITECTURE_DECISION: Gr12A4R2VectorArchitectureDecision = (
    Gr12A4R2VectorArchitectureDecision(
        architecture_phase=Gr12VectorArchitecturePhase.ARCHITECTURE_DECISION_RECONCILED,
        cla04_applicability=Gr12Applicability.APPLICABLE,
        prepare_index_decision=(
            Gr12VectorPrepareIndexGovernanceDecision.OPTION_B_CONDITIONAL_LIVE_OPERATOR_ONLY
        ),
        live_operator_surface_exists=False,
        destructive_ops_on_neutral_port=False,
        adr_path=GR12_A4_R2_VECTOR_ADR_PATH,
        canonical_port=GR12_VECTOR_CANONICAL_PORT,
        composition_root=GR12_VECTOR_COMPOSITION_ROOT,
        preferred_governance_model=GR12_VECTOR_PREFERRED_GOVERNANCE_MODEL,
        mutation_type_prepare=MUTATION_TYPE_VECTOR_INDEX_PREPARE,
        resource_type=VECTOR_INDEX_RESOURCE_TYPE,
        resource_id_template=VECTOR_INDEX_RESOURCE_ID_TEMPLATE,
        resource_scope_template=VECTOR_INDEX_RESOURCE_SCOPE_TEMPLATE,
        revision_strategy=GR12_VECTOR_REVISION_STRATEGY,
        toctou_strategy=GR12_VECTOR_TOCTOU_STRATEGY,
        identity_intrinsic_validation=Gr12VectorIdentityIntrinsicValidation.NONE,
        live_operator_identity_validation=(
            Gr12VectorLiveOperatorIdentityValidation.REQUIRED_NON_EMPTY_LOGICAL_NAME_AND_TENANT_ID
        ),
        configuration_projection_schema=GR12_VECTOR_CONFIGURATION_PROJECTION_SCHEMA,
        configuration_digest_included_fields=tuple(
            field.value for field in GR12_VECTOR_CONFIGURATION_DIGEST_INCLUDED_FIELDS
        ),
        excluded_revision_fields=GR12_VECTOR_CONFIGURATION_DIGEST_EXCLUDED_RUNTIME_FIELDS,
        absent_state_semantics=Gr12VectorAbsentIndexRevisionState.ABSENT,
        already_compatible_revision_semantics=(
            Gr12VectorAlreadyCompatibleRevisionSemantics.NOT_REVISION_EQUALITY
        ),
        live_prepare_authorization_timing=(
            Gr12VectorLivePrepareAuthorizationTiming.BEFORE_PREPARE_INDEX_INVOCATION
        ),
        stale_authorization_behavior=(
            Gr12VectorStaleAuthorizationBehavior.STALE_INVALIDATES_PRIOR_AUTHORIZATION_ABORT
        ),
        stale_retry_fresh_authorization_policy=(
            Gr12VectorStaleRetryFreshAuthorizationPolicy.ONE_EXPLICIT_REEVALUATION_OR_ABORT
        ),
        provider_cas_available=Gr12VectorProviderCasAvailability.UNAVAILABLE,
        policy_evaluator_port=GR12_CANONICAL_POLICY_PORT,
        next_bounded_task=GR12_VECTOR_NEXT_BOUNDED_TASK,
        rejected_alternatives=GR12_VECTOR_REJECTED_ALTERNATIVES,
    )
)
