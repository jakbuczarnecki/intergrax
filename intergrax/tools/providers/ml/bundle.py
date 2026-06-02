# © Artur Czarnecki. All rights reserved.

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.ml.contracts import MlPredictInput, MlPredictOutput
from intergrax.tools.providers.ml.handlers import MlPredictHandler
from intergrax.tools.providers.ml.service import ML_PREDICT_TOOL_ID
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
