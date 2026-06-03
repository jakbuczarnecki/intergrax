# © Artur Czarnecki. All rights reserved.

"""Reliability profile drives Nexus long-running flags (Phase H-APP.4.7)."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.nexus_factory import build_nexus_loop_from_environment
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ReliabilityProfile,
)
from intergrax.runtime.registry.agent_registry import AgentRegistry
from echo.echo_agent import EchoAgent

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_long_running_enabled_via_environment_only() -> None:
    from intergrax.applications.contracts.environment_profile import OrchestrationProfile

    registry = AgentRegistry()
    registry.register(EchoAgent())
    env = ApplicationEnvironmentProfile.lab_defaults().model_copy(
        update={
            "reliability_profile": ReliabilityProfile(long_running_scheduler_enabled=True),
            "orchestration_profile": OrchestrationProfile(long_running_enabled=True),
        }
    )
    import tempfile
    from pathlib import Path

    from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore

    checkpoint = SQLiteTaskCheckpointStore(db_path=Path(tempfile.mkdtemp()) / "ckpt.db")
    loop = build_nexus_loop_from_environment(
        registry,
        env=env,
        checkpoint_store=checkpoint,
    )
    assert loop._checkpoint_store is checkpoint  # noqa: SLF001
