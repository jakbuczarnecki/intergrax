from __future__ import annotations

from intergrax.model_inference.bootstrap import build_harness_model_inference_registry
from intergrax.model_inference.execution import build_modality_inference_executor
from intergrax.model_inference.execution.factory import MODALITY_EXECUTOR_EXTRA_KEY
from intergrax.tools.providers.ml.contracts import MlBatchPredictInput
from intergrax.tools.providers.ml.service import ml_batch_predict
from intergrax.tools.providers.speech.backends import MODEL_INFERENCE_REGISTRY_EXTRA_KEY
from intergrax.tools.registry.wiring import ToolWiringContext


def test_ml_batch_predict_returns_row_predictions() -> None:
    ctx = ToolWiringContext(
        extras={
            MODEL_INFERENCE_REGISTRY_EXTRA_KEY: build_harness_model_inference_registry(),
            MODALITY_EXECUTOR_EXTRA_KEY: build_modality_inference_executor(),
        }
    )
    output = ml_batch_predict(
        ctx,
        MlBatchPredictInput(
            feature_rows=[{"x": 1.0}, {"y": 2.0, "z": 1.0}],
        ),
    )
    assert len(output.request_ids) == 2
    assert len(output.predictions) == 2
