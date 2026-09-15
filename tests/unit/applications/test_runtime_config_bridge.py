# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.applications._shared.runtime_config_bridge import apply_policy_bundle_to_runtime_config
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.policy.tool_policy_resolution_adapter import (
    resolve_allowed_tools_from_runtime_config,
)
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.runtime.tools.scope_policy import StaticToolScopePolicy

pytestmark = pytest.mark.gate


def test_apply_policy_bundle_sets_runtime_config_fields() -> None:
    bundle = RuntimePolicyBundle()
    config = RuntimeConfig(llm_adapter=MagicMock())
    updated = apply_policy_bundle_to_runtime_config(config, bundle)
    assert updated.policy_bundle is bundle


def test_apply_policy_bundle_sets_tool_scope_when_present() -> None:
    scope = StaticToolScopePolicy(allowed_tools={"websearch.query"})
    bundle = RuntimePolicyBundle(tool_access=scope)
    config = RuntimeConfig(llm_adapter=MagicMock())
    updated = apply_policy_bundle_to_runtime_config(config, bundle)
    assert updated.tool_scope_policy is scope


def test_nexus_adapter_delegates_to_neutral_resolver() -> None:
    scope = StaticToolScopePolicy(allowed_tools={"a.tool"})
    config = RuntimeConfig(
        llm_adapter=MagicMock(),
        policy_bundle=RuntimePolicyBundle(tool_access=scope),
    )
    assert resolve_allowed_tools_from_runtime_config(config) == ["a.tool"]
