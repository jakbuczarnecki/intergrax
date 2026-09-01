"""Trusted workflow policy configuration for order assistance."""

from __future__ import annotations

from enum import StrEnum

from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    PolicyRulesProfile,
)
from intergrax.applications.contracts.environment_profile.sub_profiles import (
    ApplicationSecurityProfile,
)
from intergrax.runtime.policy.rules.evaluation import PolicyEnforcementMode
from intergrax.tools.registry.profile import ToolProfile

from platform_proofs.scenarios.indirect_prompt_injection.application.tools import (
    SCENARIO_TOOL_IDS,
    TOOL_ORDER_UPDATE_SHIPPING_ADDRESS,
)

READ_ONLY_DENY_RULE_ID = "deny_order_update_shipping_address_read_only"


class WorkflowKind(StrEnum):
    SAFE_READ = "SAFE-READ"
    AUTHORIZED_WRITE = "AUTHORIZED-WRITE"


class ControlKind(StrEnum):
    SAFE_READ = "SAFE-READ"
    AUTHORIZED_WRITE = "AUTHORIZED-WRITE"


def read_only_deny_rule() -> dict[str, str]:
    return {
        "rule_id": READ_ONLY_DENY_RULE_ID,
        "handler_id": "deny_tool",
        "resource_kind": "tool",
        "resource_id": TOOL_ORDER_UPDATE_SHIPPING_ADDRESS,
        "action": "deny",
    }


def build_scenario_environment_profile(
    workflow: WorkflowKind,
) -> ApplicationEnvironmentProfile:
    env = ApplicationEnvironmentProfile.lab_defaults(
        profile_id=f"scenario.indirect_prompt_injection.{workflow.value.lower()}",
        harness_tools=False,
    )
    env.tool_profile = ToolProfile(enabled=list(SCENARIO_TOOL_IDS))
    env.security_profile = ApplicationSecurityProfile(
        prompt_defense_enabled=False,
        tool_injection_defense_enabled=False,
        retrieval_poisoning_defense_enabled=False,
    )
    if workflow is WorkflowKind.SAFE_READ:
        env.policy_rules = PolicyRulesProfile(
            inline_rules=[read_only_deny_rule()],
            policy_enforcement_mode=PolicyEnforcementMode.ENFORCE,
        )
    else:
        env.policy_rules = PolicyRulesProfile(
            inline_rules=[],
            policy_enforcement_mode=PolicyEnforcementMode.ENFORCE,
        )
    return env
