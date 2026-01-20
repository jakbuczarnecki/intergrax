# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest

from intergrax.chat_agent import ChatRouterConfig


pytestmark = pytest.mark.unit


def test_ensure_prompts_loads_from_yaml() -> None:
    cfg = ChatRouterConfig(
        tools_description="",
        general_description="",
    )

    cfg.ensure_prompts()

    assert cfg.general_description
    assert cfg.tools_description


def test_ensure_prompts_preserves_existing_values() -> None:
    cfg = ChatRouterConfig(
        tools_description="CUSTOM_TOOLS",
        general_description="CUSTOM_GENERAL",
    )

    cfg.ensure_prompts()

    assert cfg.tools_description == "CUSTOM_TOOLS"
    assert cfg.general_description == "CUSTOM_GENERAL"
