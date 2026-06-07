# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.eval.contracts import (
    EvalListObservationsInput,
    EvalListObservationsOutput,
    EvalRecordObservationInput,
    EvalRecordObservationOutput,
    EvalSummarizeReleaseInput,
    EvalSummarizeReleaseOutput,
)
from intergrax.tools.providers.eval.service import (
    eval_list_observations,
    eval_record_observation,
    eval_summarize_release,
)


class EvalRecordObservationHandler(ServiceToolHandler[EvalRecordObservationInput, EvalRecordObservationOutput]):
    _service = eval_record_observation


class EvalListObservationsHandler(ServiceToolHandler[EvalListObservationsInput, EvalListObservationsOutput]):
    _service = eval_list_observations


class EvalSummarizeReleaseHandler(ServiceToolHandler[EvalSummarizeReleaseInput, EvalSummarizeReleaseOutput]):
    _service = eval_summarize_release
