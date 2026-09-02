# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Canonical qualification primitives."""

from intergrax.core.qualification.evidence import QualificationEvidence
from intergrax.core.qualification.provider import (
    ProviderQualificationEnvironmentMetadata,
    ProviderQualificationEvidenceKind,
    ProviderQualificationExecutor,
    ProviderQualificationResultSummary,
    ProviderQualificationRun,
    ProviderQualificationSubject,
)
from intergrax.core.qualification.status import (
    QualificationStatus,
    qualification_status_satisfies,
)
from intergrax.core.qualification.validity import (
    ProviderQualificationValidityContext,
    QualificationEvidenceValidity,
    QualificationRunId,
    QualificationValidityInterpretation,
    QualificationValidityRecord,
    ValidityEvaluationId,
    new_qualification_run_id,
    new_validity_evaluation_id,
    validate_qualification_run_id,
    validate_validity_evaluation_id,
)
from intergrax.core.qualification.execution import (
    ProviderQualificationExecutionCausality,
    ProviderQualificationExecutionConflictError,
    ProviderQualificationExecutionDependencies,
    ProviderQualificationExecutionError,
    ProviderQualificationExecutionRequest,
    ProviderQualificationMaterializationError,
    ProviderQualificationPersistenceExecutionError,
    ProviderQualificationRequestIncompatibleError,
    ProviderQualificationResolutionError,
    ProviderQualificationRunIdentityError,
    ProviderQualificationSubjectMismatchError,
    ProviderQualificationSuiteIdentityMismatchError,
    ProviderQualificationSuiteInfrastructureError,
    causality_from_requalification_identity,
    execute_provider_qualification,
    resolve_integration_provider_id,
)
from intergrax.core.qualification.observability import (
    NoOpProviderQualificationExecutionObservability,
    ProviderQualificationExecutionObservabilityPort,
    RecordingProviderQualificationExecutionObservability,
)
from intergrax.core.qualification.suite import (
    ProviderQualificationDomainBinding,
    ProviderQualificationMaterializationHandle,
    ProviderQualificationSuite,
    ProviderQualificationSuiteIdentity,
    ProviderQualificationSuiteOutcome,
)
from intergrax.core.qualification.requalification import (
    ProviderRequalificationDecision,
    ProviderRequalificationPreparationError,
    ProviderRequalificationRunIdentity,
    determine_provider_requalification_requirement,
    establish_provider_requalification_requirement,
    prepare_provider_requalification_run_identity,
)
from intergrax.core.qualification.validity_evaluation import (
    evaluate_provider_qualification_validity,
    establish_current_qualification_validity,
    get_current_qualification_validity,
    interpret_latest_qualification_validity,
    record_provider_qualification_validity_revocation,
    resolve_latest_qualification_validity,
    validity_context_from_run,
)
from intergrax.core.qualification.functional_qualification_identity import (
    FunctionalQualificationPluginId,
    MODEL_ROUTING_PLUGIN_ID,
    QualificationGateId,
    RAG_PLUGIN_ID,
    TOOL_SELECTION_PLUGIN_ID,
    WEB_SEARCH_PLUGIN_ID,
)
from intergrax.core.qualification.functional_qualification_plugin import (
    FunctionalQualificationPlugin,
    QualificationPluginDescriptor,
)
from intergrax.core.qualification.functional_qualification_plan import QualificationPlan
from intergrax.core.qualification.functional_qualification_registry import (
    QualificationPluginRegistry,
    QualificationPluginRegistryError,
)
from intergrax.core.qualification.functional_qualification_runner import (
    QualificationRunnerInfrastructureError,
    run_qualification_plan,
)
from intergrax.core.qualification.functional_qualification_verdict import QualificationVerdict

__all__ = [
    "ProviderQualificationExecutionCausality",
    "ProviderQualificationExecutionConflictError",
    "ProviderQualificationExecutionDependencies",
    "ProviderQualificationExecutionError",
    "ProviderQualificationExecutionRequest",
    "ProviderQualificationExecutionObservabilityPort",
    "NoOpProviderQualificationExecutionObservability",
    "RecordingProviderQualificationExecutionObservability",
    "ProviderQualificationMaterializationError",
    "ProviderQualificationPersistenceExecutionError",
    "ProviderQualificationRequestIncompatibleError",
    "ProviderQualificationResolutionError",
    "ProviderQualificationRunIdentityError",
    "ProviderQualificationSubjectMismatchError",
    "ProviderQualificationSuiteIdentityMismatchError",
    "ProviderQualificationSuiteInfrastructureError",
    "ProviderQualificationDomainBinding",
    "ProviderQualificationMaterializationHandle",
    "ProviderQualificationSuite",
    "ProviderQualificationSuiteIdentity",
    "ProviderQualificationSuiteOutcome",
    "causality_from_requalification_identity",
    "execute_provider_qualification",
    "resolve_integration_provider_id",
    "ProviderRequalificationDecision",
    "ProviderRequalificationPreparationError",
    "ProviderRequalificationRunIdentity",
    "ProviderQualificationValidityContext",
    "QualificationEvidence",
    "QualificationEvidenceValidity",
    "QualificationRunId",
    "QualificationStatus",
    "QualificationValidityInterpretation",
    "QualificationValidityRecord",
    "ValidityEvaluationId",
    "ProviderQualificationEnvironmentMetadata",
    "ProviderQualificationEvidenceKind",
    "ProviderQualificationExecutor",
    "ProviderQualificationResultSummary",
    "ProviderQualificationRun",
    "ProviderQualificationSubject",
    "determine_provider_requalification_requirement",
    "evaluate_provider_qualification_validity",
    "establish_current_qualification_validity",
    "establish_provider_requalification_requirement",
    "get_current_qualification_validity",
    "interpret_latest_qualification_validity",
    "new_qualification_run_id",
    "prepare_provider_requalification_run_identity",
    "new_validity_evaluation_id",
    "qualification_status_satisfies",
    "record_provider_qualification_validity_revocation",
    "resolve_latest_qualification_validity",
    "validate_qualification_run_id",
    "validate_validity_evaluation_id",
    "validity_context_from_run",
    "FunctionalQualificationPlugin",
    "FunctionalQualificationPluginId",
    "MODEL_ROUTING_PLUGIN_ID",
    "QualificationGateId",
    "RAG_PLUGIN_ID",
    "TOOL_SELECTION_PLUGIN_ID",
    "WEB_SEARCH_PLUGIN_ID",
    "QualificationPlan",
    "QualificationPluginDescriptor",
    "QualificationPluginRegistry",
    "QualificationPluginRegistryError",
    "QualificationRunnerInfrastructureError",
    "QualificationVerdict",
    "run_qualification_plan",
]
