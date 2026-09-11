# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Enterprise external operation admission contracts (LLM-EXTERNAL-OPERATION-ADMISSION R1)."""

from intergrax.contracts.external_operations.admission import (
    ExternalOperationAdmission,
    ExternalOperationAdmissionContext,
    OperationAdmissionDecision,
    OperationAdmissionVerdict,
)
from intergrax.contracts.external_operations.attempt import (
    ExternalOperationAttempt,
    ExternalOperationAttemptLifecycle,
    ExternalOperationAttemptTransitionError,
)
from intergrax.contracts.external_operations.audit import ExternalOperationAuditRecord
from intergrax.contracts.external_operations.evidence import (
    ExecutionFailureEvidence,
    ExternalOperationEvidence,
    ExternalOperationFailed,
    ExternalOperationEvidenceKind,
    ProviderFailureEvidence,
)
from intergrax.contracts.external_operations.governance import (
    ExternalOperationGovernanceContext,
    ExternalOperationGovernanceDecision,
)
from intergrax.contracts.external_operations.intent import (
    ExternalOperationIntent,
    ExternalOperationType,
)
from intergrax.contracts.external_operations.provider import (
    ExternalOperationProvider,
    ProviderPayloadBounds,
    ProviderRiskProfile,
)
from intergrax.contracts.external_operations.safety import (
    ExternalOperationAdmissionDeniedError,
    ExternalOperationApprovalRequiredError,
    ExternalOperationExecutionForbiddenError,
    assert_no_secrets_in_audit_payload,
    sanitize_external_operation_text,
)

__all__ = [
    "ExecutionFailureEvidence",
    "ExternalOperationAdmission",
    "ExternalOperationAdmissionContext",
    "ExternalOperationAdmissionDeniedError",
    "ExternalOperationApprovalRequiredError",
    "ExternalOperationAttempt",
    "ExternalOperationAttemptLifecycle",
    "ExternalOperationAttemptTransitionError",
    "ExternalOperationAuditRecord",
    "ExternalOperationEvidence",
    "ExternalOperationEvidenceKind",
    "ExternalOperationExecutionForbiddenError",
    "ExternalOperationFailed",
    "ExternalOperationGovernanceContext",
    "ExternalOperationGovernanceDecision",
    "ExternalOperationIntent",
    "ExternalOperationProvider",
    "ExternalOperationType",
    "OperationAdmissionDecision",
    "OperationAdmissionVerdict",
    "ProviderFailureEvidence",
    "ProviderPayloadBounds",
    "ProviderRiskProfile",
    "assert_no_secrets_in_audit_payload",
    "sanitize_external_operation_text",
]
