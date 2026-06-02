# © Artur Czarnecki. All rights reserved.

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.ml.contracts import (
    MlBatchPredictInput,
    MlBatchPredictOutput,
    MlExplainInput,
    MlExplainOutput,
    MlPredictInput,
    MlPredictOutput,
)
from intergrax.tools.providers.ml.service import ml_batch_predict, ml_explain, ml_predict


class MlPredictHandler(ServiceToolHandler[MlPredictInput, MlPredictOutput]):
    _service = ml_predict


class MlExplainHandler(ServiceToolHandler[MlExplainInput, MlExplainOutput]):
    _service = ml_explain


class MlBatchPredictHandler(ServiceToolHandler[MlBatchPredictInput, MlBatchPredictOutput]):
    _service = ml_batch_predict
