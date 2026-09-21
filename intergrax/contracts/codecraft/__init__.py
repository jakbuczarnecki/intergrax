# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""CodeCraft public contracts for governed capability synthesis (UCA-6A)."""

from __future__ import annotations

from intergrax.contracts.codecraft.gap_synthesis import (
    SCHEMA_CODECRAFT_GAP_SYNTHESIS_REQUEST_V1,
    SCHEMA_CODECRAFT_GAP_SYNTHESIS_RESULT_V1,
    CodeCraftGapSynthesisOutcome,
    CodeCraftGapSynthesisPort,
    CodeCraftGapSynthesisRequest,
    CodeCraftGapSynthesisResult,
)

__all__ = [
    "CodeCraftGapSynthesisOutcome",
    "CodeCraftGapSynthesisPort",
    "CodeCraftGapSynthesisRequest",
    "CodeCraftGapSynthesisResult",
    "SCHEMA_CODECRAFT_GAP_SYNTHESIS_REQUEST_V1",
    "SCHEMA_CODECRAFT_GAP_SYNTHESIS_RESULT_V1",
]
