# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations
from typing import Dict
import yaml

from intergrax.runtime.replay.policy_config import ExecutionPolicyConfig


class YamlPolicyLoader:
    """
    Loads execution policies from YAML file.
    """

    def load(self, path: str) -> Dict[str, ExecutionPolicyConfig]:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        policies: Dict[str, ExecutionPolicyConfig] = {}

        for key, cfg in raw.items():
            policies[key] = ExecutionPolicyConfig(
                max_total_tokens=cfg.get("max_total_tokens"),
                max_llm_call_delta=cfg.get("max_llm_call_delta"),
                min_tool_calls=cfg.get("min_tool_calls"),
                max_steps=cfg.get("max_steps"),
                fail_on_answer_change=cfg.get("fail_on_answer_change", False),
            )

        return policies
