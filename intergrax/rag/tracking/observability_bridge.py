# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Bridge RAG retrieval metrics into Nexus runtime observability."""

from __future__ import annotations

import logging
from typing import Optional

from intergrax.rag.tracking.metrics import get_rag_metrics_collector, is_rag_metrics_enabled
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.hooks.hook_registry import HookRegistry
from intergrax.runtime.plugins.contract import PolicyEngineLike, RuntimeEventBusLike, RuntimePlugin

logger = logging.getLogger(__name__)


def make_rag_metrics_runtime_plugin() -> RuntimePlugin:
    """Runtime plugin — log/export RAG metrics when a task completes."""

    def _register(
        event_bus: RuntimeEventBusLike,
        _hook_registry: HookRegistry,
        _policy_engine: PolicyEngineLike,
    ) -> None:
        if not is_rag_metrics_enabled():
            return

        async def _export_rag_metrics(event: RuntimeEvent) -> None:
            if event.event_type != RuntimeEventType.TASK_COMPLETED:
                return
            tenant = event.tenant_id or "_platform"
            collector = get_rag_metrics_collector()
            per_tenant = collector.snapshot_for_tenant(tenant)
            if not per_tenant:
                return
            logger.info(
                "rag_metrics_export tenant=%s run_id=%s task_id=%s",
                tenant,
                event.run_id,
                event.task_id,
                extra={
                    "run_id": event.run_id,
                    "task_id": event.task_id,
                    "tenant_id": tenant,
                    "rag_metrics": per_tenant,
                },
            )

        event_bus.subscribe(
            _export_rag_metrics,
            event_types={RuntimeEventType.TASK_COMPLETED},
            subscription_id="plugin.rag_metrics_export",
        )

    return RuntimePlugin(
        plugin_id="runtime.rag_metrics_export",
        version="1.0.0",
        register=_register,
    )


def register_rag_observability_plugin(
    plugins: list[RuntimePlugin],
    *,
    enabled: Optional[bool] = None,
) -> list[RuntimePlugin]:
    """Append RAG metrics plugin to an existing plugin list."""
    if enabled is False:
        return plugins
    if enabled is None and not is_rag_metrics_enabled():
        return plugins
    plugins.append(make_rag_metrics_runtime_plugin())
    return plugins
