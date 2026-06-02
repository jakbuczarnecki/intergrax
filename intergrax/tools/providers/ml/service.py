# © Artur Czarnecki. All rights reserved.

import uuid

from intergrax.model_inference.contracts import InferenceRequest
from intergrax.tools.providers.speech.backends import MODEL_INFERENCE_REGISTRY_EXTRA_KEY
from intergrax.tools.providers.ml.contracts import MlPredictInput, MlPredictOutput
from intergrax.tools.registry.wiring import ToolWiringContext

ML_PREDICT_TOOL_ID = "ml.predict"


def ml_predict(ctx: ToolWiringContext, payload: MlPredictInput) -> MlPredictOutput:
    registry = ctx.extras.get(MODEL_INFERENCE_REGISTRY_EXTRA_KEY)
    if registry is None:
        from intergrax.model_inference.bootstrap import build_harness_model_inference_registry

        registry = build_harness_model_inference_registry()
    artifact = registry.get_artifact(payload.artifact_id)
    adapter = registry.get_ml_adapter(payload.adapter_slug)
    request_id = uuid.uuid4().hex
    result = adapter.predict(
        InferenceRequest(
            request_id=request_id,
            artifact_id=artifact.artifact_id,
            features=dict(payload.features),
        ),
        artifact=artifact,
    )
    return MlPredictOutput(request_id=result.request_id, predictions=result.predictions)
