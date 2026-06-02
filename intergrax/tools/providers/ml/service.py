# © Artur Czarnecki. All rights reserved.

import uuid

from intergrax.model_inference.contracts import InferenceRequest
from intergrax.tools.providers.ml.contracts import MlExplainInput, MlExplainOutput, MlPredictInput, MlPredictOutput
from intergrax.tools.providers.speech.backends import MODEL_INFERENCE_REGISTRY_EXTRA_KEY
from intergrax.tools.registry.wiring import ToolWiringContext

ML_PREDICT_TOOL_ID = "ml.predict"
ML_EXPLAIN_TOOL_ID = "ml.explain"


def _resolve_registry(ctx: ToolWiringContext):
    registry = ctx.extras.get(MODEL_INFERENCE_REGISTRY_EXTRA_KEY)
    if registry is None:
        from intergrax.model_inference.bootstrap import build_harness_model_inference_registry

        return build_harness_model_inference_registry()
    return registry


def ml_predict(ctx: ToolWiringContext, payload: MlPredictInput) -> MlPredictOutput:
    registry = _resolve_registry(ctx)
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


def ml_explain(ctx: ToolWiringContext, payload: MlExplainInput) -> MlExplainOutput:
    registry = _resolve_registry(ctx)
    artifact = registry.get_artifact(payload.artifact_id)
    adapter = registry.get_ml_adapter(payload.adapter_slug)
    request_id = uuid.uuid4().hex
    result = adapter.explain(
        InferenceRequest(
            request_id=request_id,
            artifact_id=artifact.artifact_id,
            features=dict(payload.features),
        ),
        artifact=artifact,
    )
    return MlExplainOutput(
        request_id=result.request_id,
        predictions=result.predictions,
        feature_importance=result.feature_importance,
    )
