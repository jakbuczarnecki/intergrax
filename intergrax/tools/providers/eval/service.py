# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.runtime.architecture.online_evaluation_models import OnlineEvaluationMode, OnlineEvaluationObservation
from intergrax.tools.providers.eval.contracts import (
    EvalListObservationsInput,
    EvalListObservationsOutput,
    EvalObservationOutput,
    EvalRecordObservationInput,
    EvalRecordObservationOutput,
    EvalSummarizeReleaseInput,
    EvalSummarizeReleaseOutput,
)
from intergrax.tools.registry.runtime_bindings import OnlineEvaluationRegistryBinding
from intergrax.tools.registry.wiring import ToolWiringContext

EVAL_RECORD_OBSERVATION_TOOL_ID = "eval.record_observation"
EVAL_LIST_OBSERVATIONS_TOOL_ID = "eval.list_observations"
EVAL_SUMMARIZE_RELEASE_TOOL_ID = "eval.summarize_release"


def _require_eval_registry(ctx: ToolWiringContext) -> OnlineEvaluationRegistryBinding:
    registry = ctx.evaluation_registry
    if registry is None:
        raise RuntimeError("evaluation_registry_not_configured")
    return registry


def _observation_output(item: OnlineEvaluationObservation) -> EvalObservationOutput:
    return EvalObservationOutput(
        observation_id=item.observation_id,
        run_id=item.run_id,
        agent_id=item.agent_id,
        mode=item.mode.value,
        scenario_id=item.scenario_id,
        passed=item.passed,
        score=item.score,
        candidate_profile_version_id=item.candidate_profile_version_id,
        recorded_at=item.recorded_at.isoformat(),
    )


def eval_record_observation(
    ctx: ToolWiringContext,
    params: EvalRecordObservationInput,
) -> EvalRecordObservationOutput:
    observation = OnlineEvaluationObservation(
        observation_id=params.observation_id.strip(),
        run_id=params.run_id.strip(),
        agent_id=params.agent_id.strip(),
        mode=OnlineEvaluationMode(params.mode),
        scenario_id=params.scenario_id.strip(),
        passed=params.passed,
        score=params.score,
        candidate_profile_version_id=params.candidate_profile_version_id,
    )
    _require_eval_registry(ctx).append(observation)
    return EvalRecordObservationOutput(recorded=True, observation_id=observation.observation_id)


def eval_list_observations(ctx: ToolWiringContext, params: EvalListObservationsInput) -> EvalListObservationsOutput:
    raw = _require_eval_registry(ctx).list_observations()
    observations: list[EvalObservationOutput] = []
    for item in raw[: params.limit]:
        if not isinstance(item, OnlineEvaluationObservation):
            continue
        observations.append(_observation_output(item))
    passed_count = sum(1 for item in observations if item.passed)
    total = len(observations)
    pass_rate = float(passed_count) / float(total) if total else 0.0
    average_score = sum(item.score for item in observations) / float(total) if total else 0.0
    return EvalListObservationsOutput(
        observations=observations,
        total=total,
        pass_rate=pass_rate,
        average_score=average_score,
    )


def eval_summarize_release(ctx: ToolWiringContext, params: EvalSummarizeReleaseInput) -> EvalSummarizeReleaseOutput:
    release_id = params.release_id.strip()
    raw = _require_eval_registry(ctx).list_observations()
    matched: list[OnlineEvaluationObservation] = []
    for item in raw:
        if not isinstance(item, OnlineEvaluationObservation):
            continue
        if item.scenario_id.startswith(release_id) or item.run_id.startswith(release_id):
            matched.append(item)
    if not matched:
        matched = [item for item in raw if isinstance(item, OnlineEvaluationObservation)]
    observations = [_observation_output(item) for item in matched]
    passed_count = sum(1 for item in observations if item.passed)
    total = len(observations)
    pass_rate = float(passed_count) / float(total) if total else 0.0
    average_score = sum(item.score for item in observations) / float(total) if total else 0.0
    return EvalSummarizeReleaseOutput(
        release_id=release_id,
        observation_count=total,
        pass_rate=pass_rate,
        average_score=average_score,
        passed_count=passed_count,
        failed_count=total - passed_count,
    )
