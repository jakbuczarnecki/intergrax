# © Artur Czarnecki. All rights reserved.

"""RuntimeContext production composition requires agent runtime governance (U3 / EP-13)."""

from __future__ import annotations

import pytest

from intergrax.runtime.agent_governance.authorization_boundary import (
    AgentRuntimeGovernanceBoundary,
)
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.runtime.wiring.agent_runtime_governance_factory import (
    build_agent_runtime_governance_boundary,
    default_lab_capability_grants,
)
from intergrax.runtime.wiring.harness_governance import create_lab_allow_governance_service
from testing_support.builder import FakeLLMAdapter

pytestmark = pytest.mark.unit


def test_runtime_context_requires_agent_runtime_governance_in_production_mode() -> None:
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        enable_rag=False,
        enable_websearch=False,
        production_mode=True,
        trace_db_path="/tmp/trace.db",
    )
    session_manager = SessionManager(storage=InMemorySessionStorage())
    governance_service = create_lab_allow_governance_service()

    with pytest.raises(ValueError, match="agent_runtime_governance is required"):
        RuntimeContext.build(
            config=config,
            session_manager=session_manager,
            governance_service=governance_service,
        )


def test_runtime_context_builds_when_agent_runtime_governance_present() -> None:
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        enable_rag=False,
        enable_websearch=False,
        production_mode=True,
        trace_db_path="/tmp/trace.db",
        agent_runtime_governance=build_agent_runtime_governance_boundary(
            capability_grants=default_lab_capability_grants("tenant-a"),
        ),
    )
    session_manager = SessionManager(storage=InMemorySessionStorage())
    ctx = RuntimeContext.build(
        config=config,
        session_manager=session_manager,
        governance_service=create_lab_allow_governance_service(),
    )
    invoker = ctx.config.tool_invoker
    assert invoker is not None
    assert isinstance(config.agent_runtime_governance, AgentRuntimeGovernanceBoundary)
