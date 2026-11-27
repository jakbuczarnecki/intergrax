# © Artur Czarnecki. All rights reserved.

"""Neutral cross-application policy isolation witnesses (tier-3 tests)."""

from __future__ import annotations

import pytest
from intergrax.applications._shared.lab_environment_profile import (
    _LAB_POLICY_RULES,
    build_lab_environment_profile,
)

from lab_application.host.settings import LabApplicationSettings
from local_workspace_application.host.environment_profile import (
    build_local_workspace_environment_profile,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_lab_environment_does_not_inherit_lkw_policy_rules() -> None:
    lab_env = build_lab_environment_profile(LabApplicationSettings.from_env())
    lkw_env = build_local_workspace_environment_profile()
    assert lkw_env.policy_rules is not None
    assert lkw_env.policy_rules.rules_path is not None
    assert "local_workspace_application" in str(lkw_env.policy_rules.rules_path)
    if lab_env.policy_rules is not None and lab_env.policy_rules.rules_path is not None:
        assert lab_env.policy_rules.rules_path == _LAB_POLICY_RULES
        assert lab_env.policy_rules.rules_path != lkw_env.policy_rules.rules_path
