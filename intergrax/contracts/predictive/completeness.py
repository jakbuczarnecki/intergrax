# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Context completeness model (PREDICTIVE R4)."""

from __future__ import annotations

from enum import Enum


class PredictiveContextCompleteness(str, Enum):
    """
    Whether analyzers may treat absent signals as zero risk vs missing data.

    COMPLETE — all registered providers succeeded.
    PARTIAL — some providers failed or timed out; bounded data remains.
    LIMITED — scope-only or minimal diagnostic refs.
    UNAVAILABLE — no usable provider data (analyzers must not infer absence of risk).
    """

    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    LIMITED = "LIMITED"
    UNAVAILABLE = "UNAVAILABLE"


__all__ = ["PredictiveContextCompleteness"]
