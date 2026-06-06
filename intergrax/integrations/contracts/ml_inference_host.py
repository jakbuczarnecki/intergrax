# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Managed ML inference host integration contract (Phase M.6 P6)."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class InferencePrediction(BaseModel):
    """Generic hosted model prediction payload."""

    output: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)


@runtime_checkable
class MlInferenceHostBackend(Protocol):
    """Managed model endpoint without local GPU (Replicate, …)."""

    def predict(
        self,
        model_ref: str,
        *,
        inputs: Mapping[str, Any],
    ) -> InferencePrediction:
        """Run a hosted model prediction."""
