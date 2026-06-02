# © Artur Czarnecki. All rights reserved.

import uuid

from intergrax.model_inference.contracts import InferenceRequest
from intergrax.model_inference.execution import build_modality_inference_executor
from intergrax.model_inference.execution.executor import ModalityInferenceExecutor
from intergrax.model_inference.execution.factory import MODALITY_EXECUTOR_EXTRA_KEY
from intergrax.tools.providers.ml.contracts import (
    MlBatchPredictInput,
    MlBatchPredictOutput,
    MlExplainInput,
    MlExplainOutput,
    MlPredictInput,
    MlPredictOutput,
)
from intergrax.tools.providers.speech.backends import MODEL_INFERENCE_REGISTRY_EXTRA_KEY
from intergrax.tools.registry.wiring import ToolWiringContext

ML_PREDICT_TOOL_ID = "ml.predict"
ML_EXPLAIN_TOOL_ID = "ml.explain"
ML_BATCH_PREDICT_TOOL_ID = "ml.batch_predict"


def _resolve_registry(ctx: ToolWiringContext):
    registry = ctx.extras.get(MODEL_INFERENCE_REGISTRY_EXTRA_KEY)
    if registry is None:
        from intergrax.model_inference.bootstrap import build_harness_model_inference_registry

        return build_harness_model_inference_registry()
    return registry


def _resolve_executor(ctx: ToolWiringContext) -> ModalityInferenceExecutor:
    executor = ctx.extras.get(MODALITY_EXECUTOR_EXTRA_KEY)
    if isinstance(executor, ModalityInferenceExecutor):
        return executor
    return build_modality_inference_executor()


def ml_predict(ctx: ToolWiringContext, payload: MlPredictInput) -> MlPredictOutput:
    registry = _resolve_registry(ctx)
    executor = _resolve_executor(ctx)
    artifact = registry.get_artifact(payload.artifact_id)
    adapter = registry.get_ml_adapter(payload.adapter_slug)
    request_id = uuid.uuid4().hex
    result = executor.run_predict(
        registry=registry,
        adapter=adapter,
        artifact=artifact,
        request=InferenceRequest(
            request_id=request_id,
            artifact_id=artifact.artifact_id,
            features=dict(payload.features),
        ),
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


def ml_batch_predict(ctx: ToolWiringContext, payload: MlBatchPredictInput) -> MlBatchPredictOutput:
    registry = _resolve_registry(ctx)
    executor = _resolve_executor(ctx)
    artifact = registry.get_artifact(payload.artifact_id)
    adapter = registry.get_ml_adapter(payload.adapter_slug)
    request_ids: list[str] = []
    predictions: list[dict[str, float]] = []
    for row in payload.feature_rows:
        request_id = uuid.uuid4().hex
        result = executor.run_predict(
            registry=registry,
            adapter=adapter,
            artifact=artifact,
            request=InferenceRequest(
                request_id=request_id,
                artifact_id=artifact.artifact_id,
                features=dict(row),
            ),
        )
        request_ids.append(result.request_id)
        predictions.append(dict(result.predictions))
    return MlBatchPredictOutput(request_ids=request_ids, predictions=predictions)
