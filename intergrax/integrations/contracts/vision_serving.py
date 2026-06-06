# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Vision serving integration contract (Phase M.6 P6)."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, Field


class VisionPrediction(BaseModel):
    """Single inference output row."""

    label: str = ""
    score: float = 0.0
    metadata: dict[str, str] = Field(default_factory=dict)


class VisionInferenceResult(BaseModel):
    """Remote CV inference response."""

    model_name: str = ""
    predictions: Sequence[VisionPrediction] = Field(default_factory=list)


@runtime_checkable
class VisionServingBackend(Protocol):
    """Remote computer-vision inference host (Triton, …)."""

    def predict(self, model_name: str, *, input_uri: str) -> VisionInferenceResult:
        """Run inference against a hosted vision model."""
