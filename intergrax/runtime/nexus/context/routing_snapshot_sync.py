# © Artur Czarnecki. All rights reserved.

"""Tier-1 hooks for live LLM routing snapshot refresh (M-LLM-X.12.3)."""

from __future__ import annotations

from typing import Any

from intergrax.llm_adapters.routing.runtime_sync import refresh_config_routing_snapshot
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.task.task import Task


def sync_routing_before_llm_call(
    config: RuntimeConfig,
    *,
    run_id: str | None = None,
    step_index: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Refresh routing snapshot on config immediately before an LLM invocation."""
    if config.llm_routing_snapshot is None and config.llm_routing_context is None:
        return
    refresh_config_routing_snapshot(
        config,
        tenant_id=config.tenant_id,
        run_id=run_id,
        step_index=step_index,
        metadata=metadata,
    )


def sync_routing_for_graph_task(
    task: Task,
    *,
    step_index: int | None = None,
    runtime_config: RuntimeConfig | None = None,
) -> None:
    """Refresh routing snapshot for graph execution when a runtime config is available."""
    from intergrax.runtime.nexus.config import RuntimeConfig as RC

    cfg = runtime_config
    if cfg is None:
        raw = task.metadata.get("nexus_runtime_config")
        if isinstance(raw, RC):
            cfg = raw
    if cfg is None:
        return
    degrade_raw = task.metadata.get("budget_degrade_active")
    degrade = bool(degrade_raw) if isinstance(degrade_raw, bool) else None
    refresh_config_routing_snapshot(
        cfg,
        tenant_id=task.tenant_id,
        step_index=step_index,
        metadata=dict(task.metadata),
        run_id=task.task_id,
        budget_degrade_active=degrade,
    )
