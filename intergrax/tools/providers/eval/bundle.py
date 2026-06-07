# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.eval.contracts import (
    EvalListObservationsInput,
    EvalListObservationsOutput,
    EvalRecordObservationInput,
    EvalRecordObservationOutput,
    EvalSummarizeReleaseInput,
    EvalSummarizeReleaseOutput,
)
from intergrax.tools.providers.eval.handlers import (
    EvalListObservationsHandler,
    EvalRecordObservationHandler,
    EvalSummarizeReleaseHandler,
)
from intergrax.tools.providers.eval.service import (
    EVAL_LIST_OBSERVATIONS_TOOL_ID,
    EVAL_RECORD_OBSERVATION_TOOL_ID,
    EVAL_SUMMARIZE_RELEASE_TOOL_ID,
)
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

EVAL_BUNDLE_ID = "eval"
EVAL_TOOL_IDS: tuple[str, ...] = (
    EVAL_RECORD_OBSERVATION_TOOL_ID,
    EVAL_LIST_OBSERVATIONS_TOOL_ID,
    EVAL_SUMMARIZE_RELEASE_TOOL_ID,
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
