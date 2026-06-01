# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.llm_adapters.tracking.metrics import set_metrics_enabled
from intergrax.rag.tracking.metrics import set_rag_metrics_enabled
from intergrax.applications._shared.platform_wiring import bootstrap_nexus_platform
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry

pytestmark = [pytest.mark.integration, pytest.mark.gate]


def _subscription_ids(bus) -> set[str]:
    ids: set[str] = set()
    for sid, _, _ in getattr(bus, "_wildcard", []):
        ids.add(sid)
    for handlers in getattr(bus, "_handlers", {}).values():
        for sid, _, _ in handlers:
            ids.add(sid)
    return ids


def test_bootstrap_registers_llm_and_rag_observability_plugins() -> None:
    set_metrics_enabled(True)
    set_rag_metrics_enabled(True)
    registry = AgentRegistry()
    nexus = NexusLoop(registry)
    bootstrap_nexus_platform(nexus, trace_store=MagicMock())
    ids = _subscription_ids(nexus.event_bus)
    assert "plugin.llm_metrics_export" in ids
    assert "plugin.rag_metrics_export" in ids
