# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.pagerduty.contracts import PagerDutyTriggerIncidentInput, PagerDutyTriggerIncidentOutput
from intergrax.tools.providers.pagerduty.handlers import PagerDutyTriggerIncidentHandler
from intergrax.tools.providers.pagerduty.service import PAGERDUTY_TRIGGER_INCIDENT_TOOL_ID
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

PAGERDUTY_BUNDLE_ID = "pagerduty"
PAGERDUTY_TOOL_IDS: tuple[str, ...] = (PAGERDUTY_TRIGGER_INCIDENT_TOOL_ID,)


def register_pagerduty_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=PAGERDUTY_TRIGGER_INCIDENT_TOOL_ID,
            name=PAGERDUTY_TRIGGER_INCIDENT_TOOL_ID,
            description="Trigger a PagerDuty incident via Events API v2.",
            description_short="Trigger PagerDuty incident.",
            input_schema=PagerDutyTriggerIncidentInput,
            output_schema=PagerDutyTriggerIncidentOutput,
            error_mapping={},
            side_effects=True,
            category="notification",
            risk_level=ToolRiskLevel.HIGH,
            tags=("pagerduty", "notification", "escalation"),
        ),
        PagerDutyTriggerIncidentHandler(ctx),
    )
