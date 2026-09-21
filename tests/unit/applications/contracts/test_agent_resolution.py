# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.applications._shared.agent_resolution import (
    resolve_agent_type,
    resolve_agent_type_from_binding,
)
from intergrax.applications.contracts.errors import AgentImportError
from intergrax.applications.contracts.manifest import AgentBinding
from intergrax.contracts.tier2_agent import Tier2Agent
from echo.echo_agent import EchoAgent

pytestmark = pytest.mark.unit


def test_resolve_agent_type_from_serialized_import_path() -> None:
    resolved = resolve_agent_type(
        agent_type=None,
        import_path="echo.echo_agent.EchoAgent",
    )
    assert resolved is EchoAgent
    assert issubclass(resolved, Tier2Agent)


def test_resolve_agent_type_from_binding_mount() -> None:
    binding = AgentBinding.mount(EchoAgent, contract_id="echo")
    assert resolve_agent_type_from_binding(binding) is EchoAgent


def test_invalid_import_path_raises_agent_import_error() -> None:
    with pytest.raises(AgentImportError, match="Cannot import module"):
        resolve_agent_type(agent_type=None, import_path="no.such.module.Agent")
