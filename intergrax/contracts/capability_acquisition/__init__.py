# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Neutral UCA capability realization contracts (UCA-2)."""

from __future__ import annotations

from intergrax.contracts.capability_acquisition.errors import (
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
    "CapabilityRealizationConfigurationError",
    "CapabilityRealizationError",
    "CapabilityRealizationEvidence",
    "CapabilityRealizationIntegrityError",
    "CapabilityRealizationOutcome",
    "CapabilityRealizationProvider",
    "CapabilityRealizationReasonCode",
    "CapabilityRealizationRequest",
    "CapabilityRealizationResult",
    "NORMATIVE_CAPABILITY_REALIZATION_OUTCOMES",
    "NORMATIVE_CAPABILITY_REALIZATION_REASON_CODES",
    "SCHEMA_CAPABILITY_REALIZATION_EVIDENCE_V1",
    "SCHEMA_CAPABILITY_REALIZATION_REQUEST_V1",
    "SCHEMA_CAPABILITY_REALIZATION_RESULT_V1",
    "derive_capability_realization_request_id",
]
