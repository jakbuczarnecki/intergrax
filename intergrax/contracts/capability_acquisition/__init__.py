# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Neutral UCA capability realization (UCA-2) and acquisition (UCA-3) contracts."""

from __future__ import annotations

from intergrax.contracts.capability_acquisition.acquisition_authorization import (
    CapabilityAcquisitionAuthorizationOutcome,
    CapabilityAcquisitionAuthorizationPort,
    CapabilityAcquisitionAuthorizationResult,
)
from intergrax.contracts.capability_acquisition.acquisition_evidence import (
    SCHEMA_CAPABILITY_ACQUISITION_EVIDENCE_V1,
    CapabilityAcquisitionEvidence,
)
from intergrax.contracts.capability_acquisition.acquisition_outcome import (
    NORMATIVE_CAPABILITY_ACQUISITION_OUTCOMES,
    CapabilityAcquisitionOutcome,
)
from intergrax.contracts.capability_acquisition.acquisition_reason_code import (
    NORMATIVE_CAPABILITY_ACQUISITION_REASON_CODES,
    CapabilityAcquisitionReasonCode,
)
from intergrax.contracts.capability_acquisition.acquisition_request import (
    SCHEMA_CAPABILITY_ACQUISITION_REQUEST_V1,
    CapabilityAcquisitionRequest,
    derive_capability_acquisition_request_id,
)
from intergrax.contracts.capability_acquisition.acquisition_result import (
    SCHEMA_CAPABILITY_ACQUISITION_RESULT_V1,
    CapabilityAcquisitionResult,
)
from intergrax.contracts.capability_acquisition.acquisition_strategy import (
    CapabilityAcquisitionStrategy,
)
from intergrax.contracts.capability_acquisition.strategy_descriptor import (
    SCHEMA_CAPABILITY_ACQUISITION_STRATEGY_DESCRIPTOR_V1,
    CapabilityAcquisitionStrategyDescriptor,
)
from intergrax.contracts.capability_acquisition.strategy_selection import (
    CapabilityAcquisitionGovernanceContext,
    CapabilityAcquisitionStrategySelection,
    CapabilityAcquisitionStrategySelectionOutcome,
    CapabilityAcquisitionStrategySelectionPolicy,
    SCHEMA_CAPABILITY_ACQUISITION_STRATEGY_SELECTION_V1,
)
from intergrax.contracts.capability_acquisition.errors import (
    CapabilityAcquisitionConfigurationError,
    CapabilityAcquisitionError,
    CapabilityAcquisitionIntegrityError,
    CapabilityRealizationConfigurationError,
    CapabilityRealizationError,
    CapabilityRealizationIntegrityError,
)
from intergrax.contracts.capability_acquisition.evidence import (
    SCHEMA_CAPABILITY_REALIZATION_EVIDENCE_V1,
    CapabilityRealizationEvidence,
)
from intergrax.contracts.capability_acquisition.outcome import (
    NORMATIVE_CAPABILITY_REALIZATION_OUTCOMES,
    CapabilityRealizationOutcome,
)
from intergrax.contracts.capability_acquisition.provider import (
    CapabilityRealizationProvider,
)
from intergrax.contracts.capability_acquisition.reason_code import (
    NORMATIVE_CAPABILITY_REALIZATION_REASON_CODES,
    CapabilityRealizationReasonCode,
)
from intergrax.contracts.capability_acquisition.request import (
    SCHEMA_CAPABILITY_REALIZATION_REQUEST_V1,
    CapabilityRealizationRequest,
    derive_capability_realization_request_id,
)
from intergrax.contracts.capability_acquisition.result import (
    SCHEMA_CAPABILITY_REALIZATION_RESULT_V1,
    CapabilityRealizationResult,
)

__all__ = [
    "CapabilityAcquisitionAuthorizationOutcome",
    "CapabilityAcquisitionAuthorizationPort",
    "CapabilityAcquisitionAuthorizationResult",
    "CapabilityAcquisitionConfigurationError",
    "CapabilityAcquisitionError",
    "CapabilityAcquisitionEvidence",
    "CapabilityAcquisitionGovernanceContext",
    "CapabilityAcquisitionIntegrityError",
    "CapabilityAcquisitionOutcome",
    "CapabilityAcquisitionReasonCode",
    "CapabilityAcquisitionRequest",
    "CapabilityAcquisitionResult",
    "CapabilityAcquisitionStrategy",
    "CapabilityAcquisitionStrategyDescriptor",
    "CapabilityAcquisitionStrategySelection",
    "CapabilityAcquisitionStrategySelectionOutcome",
    "CapabilityAcquisitionStrategySelectionPolicy",
    "CapabilityRealizationConfigurationError",
    "CapabilityRealizationError",
    "CapabilityRealizationEvidence",
    "CapabilityRealizationIntegrityError",
    "CapabilityRealizationOutcome",
    "CapabilityRealizationProvider",
    "CapabilityRealizationReasonCode",
    "CapabilityRealizationRequest",
    "CapabilityRealizationResult",
    "NORMATIVE_CAPABILITY_ACQUISITION_OUTCOMES",
    "NORMATIVE_CAPABILITY_ACQUISITION_REASON_CODES",
    "NORMATIVE_CAPABILITY_REALIZATION_OUTCOMES",
    "NORMATIVE_CAPABILITY_REALIZATION_REASON_CODES",
    "SCHEMA_CAPABILITY_ACQUISITION_EVIDENCE_V1",
    "SCHEMA_CAPABILITY_ACQUISITION_REQUEST_V1",
    "SCHEMA_CAPABILITY_ACQUISITION_RESULT_V1",
    "SCHEMA_CAPABILITY_ACQUISITION_STRATEGY_DESCRIPTOR_V1",
    "SCHEMA_CAPABILITY_ACQUISITION_STRATEGY_SELECTION_V1",
    "SCHEMA_CAPABILITY_REALIZATION_EVIDENCE_V1",
    "SCHEMA_CAPABILITY_REALIZATION_REQUEST_V1",
    "SCHEMA_CAPABILITY_REALIZATION_RESULT_V1",
    "derive_capability_acquisition_request_id",
    "derive_capability_realization_request_id",
]
