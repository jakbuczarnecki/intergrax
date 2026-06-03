# © Artur Czarnecki. All rights reserved.

"""Lab graph spec validation and echo roster (Phase H-APP.3.6)."""

from __future__ import annotations

import pytest

from intergrax.applications.contracts.graph_spec import ApplicationGraphSpec, GraphNode
from echo.echo_agent import EchoAgent
from intergrax.applications.contracts.manifest import AgentBinding
from lab_application.manifest import build_lab_manifest_default

pytestmark = [pytest.mark.integration, pytest.mark.gate]


def test_lab_graph_spec_validates_echo_roster() -> None:
    manifest = build_lab_manifest_default()
    manifest = manifest.model_copy(
        update={
            "agents": [
                AgentBinding.mount(EchoAgent, capabilities=["echo.basic"]),
            ]
        }
    )
    spec = ApplicationGraphSpec(
        nodes=[GraphNode(agent_id="EchoAgent")],
        edges=[],
    )
    spec.validate_against_roster(manifest.agents)
