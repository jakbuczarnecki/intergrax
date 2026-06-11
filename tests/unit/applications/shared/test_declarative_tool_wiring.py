# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.applications._shared.declarative_tool_wiring import (
    build_declarative_invoker_from_tool_wiring,
)
from intergrax.applications._shared.tool_wiring import ApplicationToolWiring
from intergrax.tools.registry import ToolProfile, ToolRegistry, ToolWiringContext

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_build_declarative_invoker_returns_none_when_tools_disabled() -> None:
    wiring = ApplicationToolWiring(
        profile=ToolProfile(enabled=False),
        wiring_context=ToolWiringContext(),
        registry=ToolRegistry(),
    )
    assert build_declarative_invoker_from_tool_wiring(wiring) is None
