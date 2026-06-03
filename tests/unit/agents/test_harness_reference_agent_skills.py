# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from agents.echo.echo_agent import EchoAgent
from signoff_probe.contract import build_agent_contract


pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_echo_agent_declares_harness_tool_smoke_skill() -> None:
    contract = EchoAgent().get_contract()
    assert [s.skill_id for s in contract.skills] == ["harness.tool_smoke"]


def test_signoff_probe_declares_harness_tool_smoke_skill() -> None:
    contract = build_agent_contract()
    assert [s.skill_id for s in contract.skills] == ["harness.tool_smoke"]
