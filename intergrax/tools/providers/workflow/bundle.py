# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.workflow.contracts import (
    WorkflowFetchLogsInput,
    WorkflowFetchLogsOutput,
    WorkflowPollInput,
    WorkflowPollOutput,
    WorkflowTriggerInput,
    WorkflowTriggerOutput,
)
from intergrax.tools.providers.workflow.handlers import (
    WorkflowFetchLogsHandler,
    WorkflowPollHandler,
    WorkflowTriggerHandler,
)
from intergrax.tools.providers.workflow.service import (
    WORKFLOW_FETCH_LOGS_TOOL_ID,
    WORKFLOW_POLL_TOOL_ID,
    WORKFLOW_TRIGGER_TOOL_ID,
)
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

WORKFLOW_BUNDLE_ID = "workflow"
WORKFLOW_TOOL_IDS: tuple[str, ...] = (
    WORKFLOW_TRIGGER_TOOL_ID,
    WORKFLOW_POLL_TOOL_ID,
    WORKFLOW_FETCH_LOGS_TOOL_ID,
)


def register_workflow_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=WORKFLOW_TRIGGER_TOOL_ID,
            name=WORKFLOW_TRIGGER_TOOL_ID,
            description="Trigger a batch workflow/deployment run (Prefect/Airflow).",
            description_short="Trigger workflow run.",
            input_schema=WorkflowTriggerInput,
            output_schema=WorkflowTriggerOutput,
            error_mapping={},
            side_effects=True,
            category="workflow",
            risk_level=ToolRiskLevel.MEDIUM,
            tags=("workflow", "prefect", "airflow"),
        ),
        WorkflowTriggerHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=WORKFLOW_POLL_TOOL_ID,
            name=WORKFLOW_POLL_TOOL_ID,
            description="Poll workflow run status.",
            description_short="Poll workflow status.",
            input_schema=WorkflowPollInput,
            output_schema=WorkflowPollOutput,
            error_mapping={},
            side_effects=False,
            category="workflow",
            risk_level=ToolRiskLevel.LOW,
            tags=("workflow", "status"),
        ),
        WorkflowPollHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=WORKFLOW_FETCH_LOGS_TOOL_ID,
            name=WORKFLOW_FETCH_LOGS_TOOL_ID,
            description="Fetch recent logs for a workflow run.",
            description_short="Fetch workflow logs.",
            input_schema=WorkflowFetchLogsInput,
            output_schema=WorkflowFetchLogsOutput,
            error_mapping={},
            side_effects=False,
            category="workflow",
            risk_level=ToolRiskLevel.LOW,
            tags=("workflow", "logs"),
        ),
        WorkflowFetchLogsHandler(ctx),
    )
