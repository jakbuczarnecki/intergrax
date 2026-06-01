# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.applications._shared.runtime_config_bridge import apply_policy_bundle_to_runtime_config
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle

pytestmark = pytest.mark.gate


def test_apply_policy_bundle_sets_runtime_config_fields() -> None:
    budget = object()
    plan_loop = object()
    bundle = RuntimePolicyBundle(budget=budget, plan_loop=plan_loop)
    config = RuntimeConfig(llm_adapter=MagicMock())
    updated = apply_policy_bundle_to_runtime_config(config, bundle)
    assert updated.policy_bundle is bundle
    assert updated.budget_policy is budget
    assert updated.plan_loop_policy is plan_loop
