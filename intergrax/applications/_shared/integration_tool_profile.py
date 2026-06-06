# © Artur Czarnecki. All rights reserved.

"""Enable Tier-1 catalog tools when IntegrationProfile P6 slots are configured."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.tools.providers.security.service import SECURITY_SCAN_TOOL_ID
from intergrax.tools.providers.workflow.service import (
    WORKFLOW_FETCH_LOGS_TOOL_ID,
    WORKFLOW_POLL_TOOL_ID,
    WORKFLOW_TRIGGER_TOOL_ID,
)
from intergrax.tools.registry.profile import ToolProfile

_CATEGORY_TOOL_IDS: dict[IntegrationCategory, tuple[str, ...]] = {
    IntegrationCategory.SECURITY_SCANNER: (SECURITY_SCAN_TOOL_ID,),
    IntegrationCategory.SANDBOX_HOST: ("sandbox.exec",),
    IntegrationCategory.WORKFLOW_ORCHESTRATOR: (
        WORKFLOW_TRIGGER_TOOL_ID,
        WORKFLOW_POLL_TOOL_ID,
        WORKFLOW_FETCH_LOGS_TOOL_ID,
    ),
}


def integration_category_configured(
    integration_profile: IntegrationProfile,
    category: IntegrationCategory,
) -> bool:
    """Return whether ``category`` has a slug binding or pre-built instance."""
    if integration_profile.instance_for_category(category) is not None:
        return True
    return integration_profile.slug_for_category(category) is not None


def extend_tool_profile_for_integration(
    tool_profile: ToolProfile,
    integration_profile: IntegrationProfile | None,
) -> ToolProfile:
    """Append P6 integration-backed tool_ids when matching categories are configured."""
    if integration_profile is None:
        return tool_profile

    additions: list[str] = []
    for category, tool_ids in _CATEGORY_TOOL_IDS.items():
        if integration_category_configured(integration_profile, category):
            additions.extend(tool_ids)

    if not additions:
        return tool_profile

    enabled = list(tool_profile.enabled)
    for tool_id in additions:
        if tool_id not in enabled:
            enabled.append(tool_id)
    return tool_profile.model_copy(update={"enabled": enabled})
