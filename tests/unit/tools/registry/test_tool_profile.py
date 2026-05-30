# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

import pytest

from intergrax.tools.registry.profile import ToolProfile

pytestmark = pytest.mark.unit


def test_tool_profile_lab_is_empty() -> None:
    profile = ToolProfile.lab()
    assert profile.enabled == []
    assert profile.enabled_bundles == []
    assert profile.register_all_catalog_bundles is False


def test_tool_profile_all_catalog_flag() -> None:
    profile = ToolProfile.all_catalog()
    assert profile.register_all_catalog_bundles is True


def test_should_register_bundle_by_explicit_tool_id() -> None:
    profile = ToolProfile(enabled=["echo.ping"])
    assert profile.should_register_bundle("echo", tool_ids=("echo.ping", "echo.other")) is True


def test_should_register_bundle_by_bundle_id() -> None:
    profile = ToolProfile(enabled_bundles=["jira"])
    assert profile.should_register_bundle("jira", tool_ids=("jira.get_issue",)) is True
    assert profile.should_register_bundle("other", tool_ids=("other.tool",)) is False
