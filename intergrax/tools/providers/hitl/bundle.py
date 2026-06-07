# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.hitl.contracts import (
    HitlGetDecisionInput,
    HitlGetDecisionOutput,
    HitlListForTaskInput,
    HitlListForTaskOutput,
    HitlListPendingInput,
    HitlListPendingOutput,
    HitlSubmitResponseInput,
    HitlSubmitResponseOutput,
    HitlSummarizeQueueInput,
    HitlSummarizeQueueOutput,
)
from intergrax.tools.providers.hitl.handlers import (
    HitlGetDecisionHandler,
    HitlListForTaskHandler,
    HitlListPendingHandler,
    HitlSubmitResponseHandler,
    HitlSummarizeQueueHandler,
)
from intergrax.tools.providers.hitl.service import (
    HITL_GET_DECISION_TOOL_ID,
    HITL_LIST_FOR_TASK_TOOL_ID,
    HITL_LIST_PENDING_TOOL_ID,
    HITL_SUBMIT_RESPONSE_TOOL_ID,
    HITL_SUMMARIZE_QUEUE_TOOL_ID,
)
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

HITL_BUNDLE_ID = "hitl"
HITL_TOOL_IDS: tuple[str, ...] = (
    HITL_LIST_PENDING_TOOL_ID,
    HITL_GET_DECISION_TOOL_ID,
    HITL_SUMMARIZE_QUEUE_TOOL_ID,
    HITL_SUBMIT_RESPONSE_TOOL_ID,
    HITL_LIST_FOR_TASK_TOOL_ID,
)


def register_hitl_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=HITL_LIST_PENDING_TOOL_ID,
            name=HITL_LIST_PENDING_TOOL_ID,
            description="List pending HITL escalation decisions for a tenant (read-only).",
            description_short="List pending HITL escalations.",
            input_schema=HitlListPendingInput,
            output_schema=HitlListPendingOutput,
            error_mapping={},
            side_effects=False,
            category="hitl",
            risk_level=ToolRiskLevel.LOW,
            tags=("hitl", "governance", "read_only"),
        ),
        HitlListPendingHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=HITL_GET_DECISION_TOOL_ID,
            name=HITL_GET_DECISION_TOOL_ID,
            description="Fetch a single persisted human decision record by id (read-only).",
            description_short="Get HITL decision.",
            input_schema=HitlGetDecisionInput,
            output_schema=HitlGetDecisionOutput,
            error_mapping={},
            side_effects=False,
            category="hitl",
            risk_level=ToolRiskLevel.LOW,
            tags=("hitl", "governance", "read_only"),
        ),
        HitlGetDecisionHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=HITL_SUMMARIZE_QUEUE_TOOL_ID,
            name=HITL_SUMMARIZE_QUEUE_TOOL_ID,
            description="Summarize human decision counts by verdict for a tenant (read-only).",
            description_short="Summarize HITL queue.",
            input_schema=HitlSummarizeQueueInput,
            output_schema=HitlSummarizeQueueOutput,
            error_mapping={},
            side_effects=False,
            category="hitl",
            risk_level=ToolRiskLevel.LOW,
            tags=("hitl", "governance", "read_only"),
        ),
        HitlSummarizeQueueHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=HITL_SUBMIT_RESPONSE_TOOL_ID,
            name=HITL_SUBMIT_RESPONSE_TOOL_ID,
            description="Persist a human decision response for a task (policy-gated write path).",
            description_short="Submit HITL response.",
            input_schema=HitlSubmitResponseInput,
            output_schema=HitlSubmitResponseOutput,
            error_mapping={},
            side_effects=True,
            category="hitl",
            risk_level=ToolRiskLevel.HIGH,
            tags=("hitl", "governance", "write"),
        ),
        HitlSubmitResponseHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=HITL_LIST_FOR_TASK_TOOL_ID,
            name=HITL_LIST_FOR_TASK_TOOL_ID,
            description="List persisted human decisions for a task (read-only).",
            description_short="List HITL decisions for task.",
            input_schema=HitlListForTaskInput,
            output_schema=HitlListForTaskOutput,
            error_mapping={},
            side_effects=False,
            category="hitl",
            risk_level=ToolRiskLevel.LOW,
            tags=("hitl", "governance", "read_only"),
        ),
        HitlListForTaskHandler(ctx),
    )
