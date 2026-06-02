# © Artur Czarnecki. All rights reserved.

"""Remote serving placeholder adapters (Phase W-ML.4)."""

from __future__ import annotations

from intergrax.model_inference.adapters.stub_vision import StubVisionInferenceAdapter


class TritonVisionServingAdapter(StubVisionInferenceAdapter):
    """Placeholder for Triton / remote vision serving integration."""

    slug = "vision_serving"


class MlInferenceHostAdapter(StubVisionInferenceAdapter):
    """Placeholder for generic ML inference host remote path."""

    slug = "ml_inference_host"
