# © Artur Czarnecki. All rights reserved.

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.ml.contracts import MlPredictInput, MlPredictOutput
from intergrax.tools.providers.ml.service import ml_predict


class MlPredictHandler(ServiceToolHandler[MlPredictInput, MlPredictOutput]):
    _service = ml_predict
