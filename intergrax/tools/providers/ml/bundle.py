# © Artur Czarnecki. All rights reserved.

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.ml.contracts import (
    MlBatchPredictInput,
    MlBatchPredictOutput,
    MlExplainInput,
    MlExplainOutput,
    MlPredictInput,
    MlPredictOutput,
)
from intergrax.tools.providers.ml.handlers import MlBatchPredictHandler, MlExplainHandler, MlPredictHandler
from intergrax.tools.providers.ml.service import ML_BATCH_PREDICT_TOOL_ID, ML_EXPLAIN_TOOL_ID, ML_PREDICT_TOOL_ID
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

ML_BUNDLE_ID = "ml"


def register_ml_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=ML_PREDICT_TOOL_ID,
            name=ML_PREDICT_TOOL_ID,
            description="Run classical ML inference via registered ModelArtifact.",
            description_short="Classical ML predict.",
            input_schema=MlPredictInput,
            output_schema=MlPredictOutput,
            error_mapping={},
            side_effects=False,
            category="ml",
            risk_level=ToolRiskLevel.LOW,
            tags=("ml", "modality"),
        ),
        MlPredictHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=ML_EXPLAIN_TOOL_ID,
            name=ML_EXPLAIN_TOOL_ID,
            description="Explain classical ML predictions with feature importance scores.",
            description_short="Classical ML explain.",
            input_schema=MlExplainInput,
            output_schema=MlExplainOutput,
            error_mapping={},
            side_effects=False,
            category="ml",
            risk_level=ToolRiskLevel.MEDIUM,
            tags=("ml", "modality", "explain"),
        ),
        MlExplainHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=ML_BATCH_PREDICT_TOOL_ID,
            name=ML_BATCH_PREDICT_TOOL_ID,
            description="Run classical ML inference for multiple feature rows.",
            description_short="Classical ML batch predict.",
            input_schema=MlBatchPredictInput,
            output_schema=MlBatchPredictOutput,
            error_mapping={},
            side_effects=False,
            category="ml",
            risk_level=ToolRiskLevel.LOW,
            tags=("ml", "modality", "batch"),
        ),
        MlBatchPredictHandler(ctx),
    )
