# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Neutral UCA capability qualification contracts (UCA-4)."""

from __future__ import annotations

from intergrax.contracts.capability_qualification.audit_record import (
    SCHEMA_CAPABILITY_QUALIFICATION_AUDIT_RECORD_V1,
    CapabilityQualificationAuditRecord,
    build_qualification_audit_record,
)
from intergrax.contracts.capability_qualification.errors import (
    CapabilityQualificationConfigurationError,
    CapabilityQualificationError,
    CapabilityQualificationIntegrityError,
)
from intergrax.contracts.capability_qualification.lifecycle_decision import (
    SCHEMA_CAPABILITY_QUALIFICATION_LIFECYCLE_DECISION_V1,
    CapabilityQualificationLifecycleDecision,
    CapabilityQualificationLifecycleOutcome,
    CapabilityQualificationLifecyclePolicy,
    CapabilityQualificationLifecycleReasonCode,
)
from intergrax.contracts.capability_qualification.provider import (
    CapabilityQualificationProvider,
)
from intergrax.contracts.capability_qualification.provider_descriptor import (
    SCHEMA_CAPABILITY_QUALIFICATION_PROVIDER_DESCRIPTOR_V1,
    CapabilityQualificationProviderDescriptor,
)
from intergrax.contracts.capability_qualification.provider_selection import (
    SCHEMA_CAPABILITY_QUALIFICATION_PROVIDER_SELECTION_V1,
    CapabilityQualificationProviderSelection,
    CapabilityQualificationProviderSelectionOutcome,
    CapabilityQualificationProviderSelectionPolicy,
)
from intergrax.contracts.capability_qualification.qualification_decision import (
    SCHEMA_CAPABILITY_QUALIFICATION_DECISION_V1,
    CapabilityQualificationDecision,
)
from intergrax.contracts.capability_qualification.qualification_evidence import (
    SCHEMA_CAPABILITY_QUALIFICATION_EVIDENCE_V1,
    CapabilityQualificationEvidence,
)
from intergrax.contracts.capability_qualification.qualification_outcome import (
    NORMATIVE_CAPABILITY_QUALIFICATION_OUTCOMES,
    CapabilityQualificationOutcome,
)
from intergrax.contracts.capability_qualification.qualification_reason_code import (
    NORMATIVE_CAPABILITY_QUALIFICATION_REASON_CODES,
    CapabilityQualificationReasonCode,
)
from intergrax.contracts.capability_qualification.qualification_request import (
    SCHEMA_CAPABILITY_QUALIFICATION_REQUEST_V1,
    CapabilityQualificationRequest,
    derive_capability_qualification_request_id,
)
from intergrax.contracts.capability_qualification.qualification_result import (
    SCHEMA_CAPABILITY_QUALIFICATION_RESULT_V1,
    CapabilityQualificationResult,
)

__all__ = [
    "CapabilityQualificationAuditRecord",
    "CapabilityQualificationConfigurationError",
    "CapabilityQualificationDecision",
    "CapabilityQualificationError",
    "CapabilityQualificationEvidence",
    "CapabilityQualificationIntegrityError",
    "CapabilityQualificationLifecycleDecision",
    "CapabilityQualificationLifecycleOutcome",
    "CapabilityQualificationLifecyclePolicy",
    "CapabilityQualificationLifecycleReasonCode",
    "CapabilityQualificationOutcome",
    "CapabilityQualificationProvider",
    "CapabilityQualificationProviderDescriptor",
    "CapabilityQualificationProviderSelection",
    "CapabilityQualificationProviderSelectionOutcome",
    "CapabilityQualificationProviderSelectionPolicy",
    "CapabilityQualificationReasonCode",
    "CapabilityQualificationRequest",
    "CapabilityQualificationResult",
    "NORMATIVE_CAPABILITY_QUALIFICATION_OUTCOMES",
    "NORMATIVE_CAPABILITY_QUALIFICATION_REASON_CODES",
    "SCHEMA_CAPABILITY_QUALIFICATION_AUDIT_RECORD_V1",
    "SCHEMA_CAPABILITY_QUALIFICATION_DECISION_V1",
    "SCHEMA_CAPABILITY_QUALIFICATION_EVIDENCE_V1",
    "SCHEMA_CAPABILITY_QUALIFICATION_LIFECYCLE_DECISION_V1",
    "SCHEMA_CAPABILITY_QUALIFICATION_PROVIDER_DESCRIPTOR_V1",
    "SCHEMA_CAPABILITY_QUALIFICATION_PROVIDER_SELECTION_V1",
    "SCHEMA_CAPABILITY_QUALIFICATION_REQUEST_V1",
    "SCHEMA_CAPABILITY_QUALIFICATION_RESULT_V1",
    "build_qualification_audit_record",
    "derive_capability_qualification_request_id",
]
