# © Artur Czarnecki. All rights reserved.

"""GraphExecutor public configuration API used by NexusLoop."""

from __future__ import annotations

import pytest

from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.nexus.validation.validation_engine import NexusValidationEngine
from intergrax.runtime.registry.agent_registry import AgentRegistry

pytestmark = pytest.mark.unit


def test_apply_validation_engine_replaces_graph_executor_dependency() -> None:
    executor = GraphExecutor(AgentRegistry())
    replacement = NexusValidationEngine()
    executor.apply_validation_engine(replacement)
    assert executor._validation_engine is replacement  # noqa: SLF001
