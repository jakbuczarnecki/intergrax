# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations
from typing import Dict

from intergrax.runtime.replay.policy_config import ExecutionPolicyConfig


class PolicyRegistry:
    """
    Provides policy configurations for execution governance.

    Source of truth for agent behavior rules.
    """

    def __init__(self, policies: Dict[str, ExecutionPolicyConfig]) -> None:
        self._policies = policies

    def get_policy(
        self,
        agent_id: str,
        environment: str,
    ) -> ExecutionPolicyConfig:

        key = f"{environment}:{agent_id}"

        if key in self._policies:
            return self._policies[key]

        if environment in self._policies:
            return self._policies[environment]

        return ExecutionPolicyConfig()
