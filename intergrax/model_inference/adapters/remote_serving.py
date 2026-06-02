# © Artur Czarnecki. All rights reserved.

"""Remote serving adapters — re-exports for backward compatibility."""

from intergrax.model_inference.adapters.huggingface_inference_vision import HuggingFaceInferenceVisionAdapter
from intergrax.model_inference.adapters.stub_ml import StubModelInferenceAdapter
from intergrax.model_inference.adapters.triton_vision import TritonVisionServingAdapter

# Legacy slug used in early harness registry wiring.
class MlInferenceHostAdapter(StubModelInferenceAdapter):
    """Placeholder slug for remote classical ML hosts (delegates to stub classifier)."""

    slug = "ml_inference_host"
