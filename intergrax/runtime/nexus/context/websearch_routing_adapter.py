# © Artur Czarnecki. All rights reserved.

"""Nexus-internal adapter: WebSearch routing snapshot sync port."""

from __future__ import annotations

from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.routing_snapshot_sync import sync_routing_before_llm_call
from intergrax.websearch.contracts.routing_snapshot_sync import WebSearchLlmRoutingSnapshotSync


class NexusWebSearchLlmRoutingSnapshotSync:
    """Implements ``WebSearchLlmRoutingSnapshotSync`` using Nexus ``RuntimeConfig``."""

    __slots__ = ("_config",)

    def __init__(self, config: RuntimeConfig) -> None:
        self._config = config

    def sync_before_llm_call(self, *, run_id: str | None) -> None:
        sync_routing_before_llm_call(self._config, run_id=run_id)


def websearch_routing_sync_from_runtime_config(
    config: RuntimeConfig,
) -> WebSearchLlmRoutingSnapshotSync:
    return NexusWebSearchLlmRoutingSnapshotSync(config)
