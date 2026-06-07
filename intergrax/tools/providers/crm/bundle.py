# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.crm.contracts import (
    CrmGetAccountInput,
    CrmGetAccountOutput,
    CrmListContactsInput,
    CrmListContactsOutput,
    CrmListTicketsInput,
    CrmListTicketsOutput,
)
from intergrax.tools.providers.crm.handlers import CrmGetAccountHandler, CrmListContactsHandler, CrmListTicketsHandler
from intergrax.tools.providers.crm.service import (
    CRM_GET_ACCOUNT_TOOL_ID,
    CRM_LIST_CONTACTS_TOOL_ID,
    CRM_LIST_TICKETS_TOOL_ID,
)
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

CRM_BUNDLE_ID = "crm"
CRM_TOOL_IDS: tuple[str, ...] = (
    CRM_GET_ACCOUNT_TOOL_ID,
    CRM_LIST_CONTACTS_TOOL_ID,
    CRM_LIST_TICKETS_TOOL_ID,
)


def register_crm_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=CRM_GET_ACCOUNT_TOOL_ID,
            name=CRM_GET_ACCOUNT_TOOL_ID,
            description="Fetch CRM account profile by id (read-only support context).",
            description_short="Get CRM account.",
            input_schema=CrmGetAccountInput,
            output_schema=CrmGetAccountOutput,
            error_mapping={},
            side_effects=False,
            category="crm",
            risk_level=ToolRiskLevel.LOW,
            tags=("crm", "support"),
        ),
        CrmGetAccountHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=CRM_LIST_CONTACTS_TOOL_ID,
            name=CRM_LIST_CONTACTS_TOOL_ID,
            description="List CRM contacts for an account id.",
            description_short="List CRM contacts.",
            input_schema=CrmListContactsInput,
            output_schema=CrmListContactsOutput,
            error_mapping={},
            side_effects=False,
            category="crm",
            risk_level=ToolRiskLevel.LOW,
            tags=("crm", "support"),
        ),
        CrmListContactsHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=CRM_LIST_TICKETS_TOOL_ID,
            name=CRM_LIST_TICKETS_TOOL_ID,
            description="List CRM support tickets for an account id.",
            description_short="List CRM tickets.",
            input_schema=CrmListTicketsInput,
            output_schema=CrmListTicketsOutput,
            error_mapping={},
            side_effects=False,
            category="crm",
            risk_level=ToolRiskLevel.LOW,
            tags=("crm", "support"),
        ),
        CrmListTicketsHandler(ctx),
    )
