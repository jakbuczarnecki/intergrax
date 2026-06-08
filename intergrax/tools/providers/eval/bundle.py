# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
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
from intergrax.tools.providers.eval.handlers import (
    EvalCompareReleasesHandler,
    EvalExportObservationsHandler,
    EvalJudgeHandler,
    EvalListObservationsHandler,
    EvalRecordObservationHandler,
    EvalSummarizeReleaseHandler,
    EvalTrajectoryHandler,
)
from intergrax.tools.providers.eval.judge import EVAL_JUDGE_TOOL_ID
from intergrax.tools.providers.eval.service import (
    EVAL_COMPARE_RELEASES_TOOL_ID,
    EVAL_EXPORT_OBSERVATIONS_TOOL_ID,
    EVAL_LIST_OBSERVATIONS_TOOL_ID,
    EVAL_RECORD_OBSERVATION_TOOL_ID,
    EVAL_SUMMARIZE_RELEASE_TOOL_ID,
)
from intergrax.tools.providers.eval.trajectory import EVAL_TRAJECTORY_TOOL_ID
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

EVAL_BUNDLE_ID = "eval"
EVAL_TOOL_IDS: tuple[str, ...] = (
    EVAL_RECORD_OBSERVATION_TOOL_ID,
    EVAL_LIST_OBSERVATIONS_TOOL_ID,
    EVAL_SUMMARIZE_RELEASE_TOOL_ID,
    EVAL_COMPARE_RELEASES_TOOL_ID,
    EVAL_EXPORT_OBSERVATIONS_TOOL_ID,
    EVAL_JUDGE_TOOL_ID,
    EVAL_TRAJECTORY_TOOL_ID,
)


def register_eval_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=EVAL_RECORD_OBSERVATION_TOOL_ID,
            name=EVAL_RECORD_OBSERVATION_TOOL_ID,
            description="Record a harness online/shadow evaluation observation (V-EVAL registry).",
            description_short="Record eval observation.",
            input_schema=EvalRecordObservationInput,
            output_schema=EvalRecordObservationOutput,
            error_mapping={},
            side_effects=True,
            category="eval",
            risk_level=ToolRiskLevel.LOW,
            tags=("eval", "harness", "observability"),
        ),
        EvalRecordObservationHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=EVAL_LIST_OBSERVATIONS_TOOL_ID,
            name=EVAL_LIST_OBSERVATIONS_TOOL_ID,
            description="List recorded harness evaluation observations with pass rate and average score.",
            description_short="List eval observations.",
            input_schema=EvalListObservationsInput,
            output_schema=EvalListObservationsOutput,
            error_mapping={},
            side_effects=False,
            category="eval",
            risk_level=ToolRiskLevel.LOW,
            tags=("eval", "harness", "observability"),
        ),
        EvalListObservationsHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=EVAL_SUMMARIZE_RELEASE_TOOL_ID,
            name=EVAL_SUMMARIZE_RELEASE_TOOL_ID,
            description="Summarize evaluation observations for a release id label (promote gate input).",
            description_short="Summarize release eval.",
            input_schema=EvalSummarizeReleaseInput,
            output_schema=EvalSummarizeReleaseOutput,
            error_mapping={},
            side_effects=False,
            category="eval",
            risk_level=ToolRiskLevel.LOW,
            tags=("eval", "harness", "release"),
        ),
        EvalSummarizeReleaseHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=EVAL_COMPARE_RELEASES_TOOL_ID,
            name=EVAL_COMPARE_RELEASES_TOOL_ID,
            description="Compare evaluation pass rate and average score between two release labels.",
            description_short="Compare release eval.",
            input_schema=EvalCompareReleasesInput,
            output_schema=EvalCompareReleasesOutput,
            error_mapping={},
            side_effects=False,
            category="eval",
            risk_level=ToolRiskLevel.LOW,
            tags=("eval", "harness", "release"),
        ),
        EvalCompareReleasesHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=EVAL_EXPORT_OBSERVATIONS_TOOL_ID,
            name=EVAL_EXPORT_OBSERVATIONS_TOOL_ID,
            description="Export harness evaluation observations as JSON for offline analysis.",
            description_short="Export eval observations.",
            input_schema=EvalExportObservationsInput,
            output_schema=EvalExportObservationsOutput,
            error_mapping={},
            side_effects=False,
            category="eval",
            risk_level=ToolRiskLevel.LOW,
            tags=("eval", "harness", "export"),
        ),
        EvalExportObservationsHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=EVAL_JUDGE_TOOL_ID,
            name=EVAL_JUDGE_TOOL_ID,
            description="Semantic LLM-as-judge scoring for agent output against a rubric (CRIT-V).",
            description_short="LLM judge rubric score.",
            input_schema=EvalJudgeInput,
            output_schema=EvalJudgeOutput,
            error_mapping={},
            side_effects=True,
            category="eval",
            risk_level=ToolRiskLevel.MEDIUM,
            tags=("eval", "critic", "judge"),
        ),
        EvalJudgeHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=EVAL_TRAJECTORY_TOOL_ID,
            name=EVAL_TRAJECTORY_TOOL_ID,
            description="Trajectory/process evaluation from persisted run trace (CRIT-V).",
            description_short="Evaluate run trajectory.",
            input_schema=EvalTrajectoryInput,
            output_schema=EvalTrajectoryOutput,
            error_mapping={},
            side_effects=False,
            category="eval",
            risk_level=ToolRiskLevel.LOW,
            tags=("eval", "critic", "trajectory"),
        ),
        EvalTrajectoryHandler(ctx),
    )
