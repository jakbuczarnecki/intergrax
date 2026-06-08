# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.eval.contracts import (
    EvalCompareReleasesInput,
    EvalCompareReleasesOutput,
    EvalExportObservationsInput,
    EvalExportObservationsOutput,
    EvalJudgeInput,
    EvalJudgeOutput,
    EvalListObservationsInput,
    EvalListObservationsOutput,
    EvalRecordObservationInput,
    EvalRecordObservationOutput,
    EvalSummarizeReleaseInput,
    EvalSummarizeReleaseOutput,
    EvalTrajectoryInput,
    EvalTrajectoryOutput,
)
from intergrax.tools.providers.eval.judge import eval_judge
from intergrax.tools.providers.eval.trajectory import eval_trajectory
from intergrax.tools.providers.eval.service import (
    eval_compare_releases,
    eval_export_observations,
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


class EvalCompareReleasesHandler(ServiceToolHandler[EvalCompareReleasesInput, EvalCompareReleasesOutput]):
    _service = eval_compare_releases


class EvalExportObservationsHandler(
    ServiceToolHandler[EvalExportObservationsInput, EvalExportObservationsOutput]
):
    _service = eval_export_observations


class EvalJudgeHandler(ServiceToolHandler[EvalJudgeInput, EvalJudgeOutput]):
    _service = eval_judge


class EvalTrajectoryHandler(ServiceToolHandler[EvalTrajectoryInput, EvalTrajectoryOutput]):
    _service = eval_trajectory
